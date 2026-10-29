from __future__ import annotations

import torch

from triattention.kernels.prefill import I4, I8

WORDS = {I8: 2048, I4: 1024}
K8_WORDS, K4_WORDS = 2048, 1024
NEAR_V8 = 16


class DecodeCache:
    def __init__(self, module, max_tail=256):
        self.module, self.max_tail = module, max_tail
        self.layers, self.formats, self.step = {}, {}, None

    def codes(self, rows):
        if rows not in self.formats:
            j = torch.arange(rows)
            codes = torch.where((rows - 1 - j < NEAR_V8) | (j == 0), I8, I4)
            self.formats[rows] = codes, frozenset(codes.unique().tolist())
        return self.formats[rows]

    def store(self, layer, prompt, kp, ka, kstat, vstat, packed, codes, bands, k4, ka4):
        tiles = (prompt + 63) // 64
        hk, device = ka.numel() // (tiles * 4), ka.device
        bands = bands.to(torch.int64)
        j = torch.arange(tiles)
        w8, w4, d = bands[:, :1], bands[:, 1:], (tiles - 1) - j[None, :]
        state = torch.where((j[None, :] == 0) | (d < w8), 8, torch.where(d < w4, 4, 0))
        state = state.view(hk, bands.shape[0] // hk, tiles)
        kv = state.amax(1)
        rowmask = ((state > 0).to(torch.int64) << torch.arange(state.shape[1])[None, :, None]).sum(1)
        count = (kv > 0).sum(1)
        maxn = max(1, int(count.max()))
        order = torch.argsort((kv == 0).to(torch.int8), dim=1, stable=True)[:, :maxn]
        valid = torch.arange(maxn)[None, :] < count[:, None]
        is4 = (kv.gather(1, order) == 4) & valid
        is8 = (kv.gather(1, order) == 8) & valid
        rows = torch.arange(hk)[:, None] * tiles + order
        ragged = lambda select: torch.cumsum(select.flatten(), 0).view(hk, maxn) - 1
        kslot = torch.where(is4, ragged(is4), torch.where(is8, ragged(is8), 0))
        kinfo = (rowmask.gather(1, order) & 0xFF) | (is4.to(torch.int64) << 8) | (kslot << 9)
        kinfo = torch.where(valid, kinfo, 0)
        vcode = codes[order]
        vslot, values = torch.zeros((hk, maxn), dtype=torch.int64), {}
        for code in (I8, I4):
            select = valid & (vcode == code)
            vslot = torch.where(select, ragged(select), vslot)
            if bool(select.any()):
                values[code] = packed[code].view(hk * tiles, WORDS[code]).index_select(0, rows[select].to(device))
            else:
                values[code] = torch.zeros((1, WORDS[code]), dtype=torch.uint32, device=device)
        wave = torch.cuda.get_device_properties(device).multi_processor_count * self.module.decode_occupancy()
        per = max(1, -(-int(count.sum()) // wave))
        to = lambda x: x.to(torch.int32).to(device)
        has4 = bool(is4.any())
        if self.step is None:
            self.step = torch.zeros(1, dtype=torch.int64, device=device)
        self.step.zero_()
        tail_k = torch.zeros((1, hk, self.max_tail, 128), dtype=torch.bfloat16, device=device)
        self.layers[layer] = dict(
            prompt=prompt, maxn=maxn, per=per, fmt=to(codes - 1), slot=to(vslot), entry=to(order), kinfo=to(kinfo),
            count=to(count), k8=kp.view(hk * tiles, K8_WORDS).index_select(0, rows[is8].to(device)),
            k4=k4.view(hk * tiles, K4_WORDS).index_select(0, rows[is4].to(device)) if has4 else None,
            ka=ka, ka4=ka4 if has4 else None, kstat=kstat, vstat=vstat, v=values,
            tail_k=tail_k, tail_v=torch.zeros_like(tail_k))

    def __call__(self, layer, q, k, v, scale):
        e = self.layers[layer]
        hq = q.shape[1]
        splits = max(1, -(-e["maxn"] // e["per"]))
        if "partials" not in e:
            e["partials"] = (torch.empty(hq * splits, device=q.device), torch.empty(hq * splits, device=q.device),
                             torch.empty(hq * splits * 128, device=q.device))
        out = torch.empty_like(q)
        self.module.decode(q, e["k8"], e["k4"], e["ka"], e["ka4"], e["entry"], e["kinfo"], e["count"], e["fmt"],
                           e["slot"], e["v"][I8], e["v"][I4], e["vstat"], e["kstat"], e["tail_k"], e["tail_v"], k, v,
                           self.step, out, *e["partials"], e["prompt"], e["per"], scale)
        return out

    def advance(self):
        self.step.add_(1)

    def stored(self, layer):
        return layer in self.layers
