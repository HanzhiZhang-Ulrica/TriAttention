from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from triattention.calibrate import load_samples
from triattention.kernels.extension import load_extension
from triattention.kernels.prefill import Prefill
from triattention.maps import from_bands
from triattention.models import exact_attention, load_model, load_tokenizer, use_attention

KL_LIMIT = 0.002
NLL_TOLERANCE = 0.01


def length_group(tokens):
    return "128K" if tokens > 80000 else "64K" if tokens > 40000 else "32K"


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Select the cheapest candidate map that passes the quality gates.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--response-tokens", type=int, default=256)
    args = parser.parse_args()
    tokenizer, model = load_tokenizer(args.model), load_model(args.model)
    samples = load_samples(args.samples, tokenizer, args.response_tokens)
    candidates = json.loads(Path(args.candidates).read_text())
    module, shared, backends = load_extension(), {}, {"bf16": None}
    for point, entry in sorted(candidates["maps"].items(), key=lambda item: float(item[0])):
        backends[point] = Prefill(module, from_bands(entry["bands"], model))
        backends[point].buffers = shared

    reference, nll, kl = [], {}, {}
    for name, prefill in backends.items():
        if prefill is None:
            use_attention(model, "triattention_select", lambda m, q, k, v, s: exact_attention(q, k, v, s))
        else:
            use_attention(model, "triattention_select", lambda m, q, k, v, s, p=prefill: p(m.layer_idx, q, k, v, s))
        nll[name], kl[name] = [], []
        for n, (ids, response) in enumerate(samples):
            x = torch.tensor([ids], device="cuda")
            logp = model(input_ids=x, use_cache=False, logits_to_keep=response + 1).logits[0, :-1].float().log_softmax(-1)
            nll[name].append(float(F.nll_loss(logp, x[0, -response:])))
            if prefill is None:
                reference.append(logp.cpu())
            else:
                ref = reference[n].to(logp.device)
                kl[name].append(float((ref.exp() * (ref - logp)).sum(-1).mean()))
        print(f"{name}: nll {sum(nll[name]) / len(samples):.4f}" +
              (f", kl {sum(kl[name]) / len(samples):.5f}" if kl[name] else ""), flush=True)

    groups = {}
    for n, (ids, _) in enumerate(samples):
        groups.setdefault(length_group(len(ids)), []).append(n)

    def passes(point):
        for idx in groups.values():
            delta = sum(nll[point][n] - nll["bf16"][n] for n in idx) / len(idx)
            if delta > NLL_TOLERANCE or sum(kl[point][n] for n in idx) / len(idx) > KL_LIMIT:
                return False
        return True

    eligible = [p for p in candidates["maps"] if passes(p)]
    if not eligible:
        raise SystemExit("no candidate passes the quality gates")
    point = min(eligible, key=lambda p: candidates["maps"][p]["cost"])
    entry = candidates["maps"][point]
    result = dict(format="triattention-map-v1", model_type=candidates["model_type"], length=candidates["length"],
                  c4=candidates["c4"], point=float(point), cost=entry["cost"], bands=entry["bands"])
    Path(args.output).write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print(f"selected {point} (cost {entry['cost']:.3f})")


if __name__ == "__main__":
    main()
