from __future__ import annotations

from dataclasses import dataclass

import torch

SKIP, F16, I8, I4, Q4 = 0, 1, 2, 3, 4


@dataclass
class BandTable:
    order: torch.Tensor
    bands: torch.Tensor
    formats: frozenset
    host: torch.Tensor

    @classmethod
    def build(cls, bands, rows, kv_heads, device):
        b = torch.as_tensor(bands, dtype=torch.int64).cpu()
        w8, w4 = b[:, :1], b[:, 1:]
        qb = torch.arange(rows)[None, :]
        n8 = (qb - torch.clamp(qb - w8 + 1, min=1)).clamp(min=0)
        n4 = (qb - w8 - torch.clamp(qb - w4 + 1, min=1) + 1).clamp(min=0)
        heads = torch.arange(b.shape[0]).view(kv_heads, -1)
        order = (heads[:, None, :] * rows + qb.flip(1)[0][None, :, None]).flatten()
        formats = {F16} | ({I8} if bool((n8 > 0).any()) else set()) | ({Q4} if bool((n4 > 0).any()) else set())
        host = b.to(torch.int32).contiguous()
        return cls(order.to(torch.int32).to(device), host.to(device), frozenset(formats), host)


class Prefill:
    def __init__(self, module, bands_fn, decode_cache=None):
        self.module, self.bands_fn, self.decode_cache = module, bands_fn, decode_cache
        self.tables, self.buffers, self.views, self.layer_buffers = {}, {}, {}, {}

    def table(self, layer, heads, kv_heads, rows, device):
        key = (None if getattr(self.bands_fn, "layer_independent", False) else layer, heads, rows)
        if key not in self.tables:
            self.tables[key] = BandTable.build(self.bands_fn(layer, heads), rows, kv_heads, device)
        return self.tables[key]

    def buffer(self, name, numel, dtype, device):
        current = self.buffers.get(name)
        if current is None or current.numel() < numel:
            current = self.buffers[name] = torch.empty(numel, dtype=dtype, device=device)
            self.views.clear()
        return current[:numel]

    def kept(self, layer, hk, rows, int4, device):
        key = (layer, hk, rows, int4)
        if key not in self.layer_buffers:
            self.layer_buffers[key] = (
                torch.empty(hk * rows * 4, dtype=torch.float32, device=device),
                torch.empty(hk * 3 * 128, dtype=torch.float32, device=device),
                torch.empty((1, hk, 3, 128), dtype=torch.float32, device=device),
                torch.empty(hk * rows * 4, dtype=torch.float32, device=device) if int4 else None,
            )
        return self.layer_buffers[key]

    def operands(self, hq, hk, t, int4, formats, device):
        key = (hq, hk, t, int4, formats)
        if key in self.views:
            return self.views[key]
        rows = (t + 63) // 64
        buf = lambda name, n, dtype: self.buffer(name, n, dtype, device)
        ops = dict(
            q8=buf("q8", hq * t * 128, torch.int8).view(1, hq, t, 128),
            qa=buf("qa", hq * t, torch.float32),
            kp=buf("k", hk * rows * 2048, torch.uint32),
            ka=buf("ka", hk * rows * 4, torch.float32),
            partial=buf("partial", hk * 64 * 3 * 128, torch.float32),
            kstat=buf("kstat", hk * 3 * 128, torch.float32),
            vstat=buf("vstat", hk * 3 * 128, torch.float32).view(1, hk, 3, 128),
            q4=buf("q4", hq * t * 64, torch.uint8) if int4 else None,
            qa4=buf("qa4", hq * t, torch.float32) if int4 else None,
            k4=buf("k4", hk * rows * 1024, torch.uint32) if int4 else None,
            ka4=buf("ka4", hk * rows * 4, torch.float32) if int4 else None,
        )
        packed = {code: buf(f"v{code}", hk * rows * words, torch.uint32) if code in formats else None
                  for code, words in ((F16, 4096), (I8, 2048), (I4, 1024))}
        if int4 and packed[I8] is None:
            packed[I8] = buf(f"v{I8}", hk * rows * 2048, torch.uint32)
        ops["packed"] = packed
        self.views[key] = ops
        return ops

    def __call__(self, layer, q, k, v, scale):
        _, hq, t, _ = q.shape
        hk, rows, device = k.shape[1], (t + 63) // 64, q.device
        table = self.table(layer, hq, hk, rows, device)
        int4 = Q4 in table.formats
        decode = self.decode_cache.codes(rows) if self.decode_cache is not None else None
        formats = table.formats | decode[1] if decode is not None else table.formats
        ops = self.operands(hq, hk, t, int4, formats, device)
        ka, kstat, vstat, ka4 = ops["ka"], ops["kstat"], ops["vstat"], ops["ka4"]
        if decode is not None:
            ka, kstat, vstat, ka4 = self.kept(layer, hk, rows, int4, device)
        packed = ops["packed"]
        self.module.prepare_qk(q, k, ops["q8"], ops["qa"], ops["kp"], ka, ops["partial"], kstat,
                               ops["q4"], ops["qa4"], ops["k4"], ka4)
        self.module.pack_v(v, ops["partial"], vstat, packed[F16], packed[I8], packed[I4])
        out = torch.empty((1, t, hq, 128), dtype=q.dtype, device=device).transpose(1, 2)
        self.module.attention(table.order, table.bands, None, None, packed[F16], packed[I8], packed[I4], vstat,
                              ops["q8"], ops["kp"], ops["qa"], ka, ops["q4"], ops["k4"], ops["qa4"], ka4, out, scale)
        if decode is not None:
            self.decode_cache.store(layer, t, ops["kp"], ka, kstat, vstat, packed, decode[0], table.host,
                                    ops["k4"], ka4)
        return out
