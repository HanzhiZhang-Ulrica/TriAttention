from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from triattention.kernels.extension import load_extension
from triattention.kernels.prefill import Prefill
from triattention.maps import INF, uniform
from triattention.models import exact_attention, load_model, load_tokenizer, render_prompt, use_attention

W8 = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, INF)
W4 = (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, INF)


def grid():
    return [(w8, w4) for w8 in W8 for w4 in sorted({w8, *W4}) if w4 >= w8 and not (w8 == INF and w4 != INF)]


def load_samples(path, tokenizer, response_tokens):
    samples = []
    for line in Path(path).read_text().splitlines():
        record = json.loads(line)
        prompt = tokenizer(render_prompt(tokenizer, record["prompt"]), add_special_tokens=False).input_ids
        response = tokenizer(record["response"], add_special_tokens=False).input_ids[:response_tokens]
        samples.append((prompt + response, len(response)))
    return samples


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Measure the per-head error of every distance-band configuration.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--response-tokens", type=int, default=256)
    args = parser.parse_args()
    tokenizer, model = load_tokenizer(args.model), load_model(args.model)
    samples = load_samples(args.samples, tokenizer, args.response_tokens)
    configs = grid()
    module, shared = load_extension(), {}
    executors = []
    for w8, w4 in configs:
        executors.append(Prefill(module, uniform(w8, w4)))
        executors[-1].buffers = shared
    errors = {}

    def attention(mod, query, key, value, scaling):
        layer, heads, t = mod.layer_idx, query.shape[1], query.shape[2]
        rows = slice(max(0, t - args.rows), t)
        ref = exact_attention(query, key, value, scaling)
        w = mod.o_proj.weight.float()
        wh = w.view(w.shape[0], heads, 128).permute(1, 0, 2)
        gram = wh.transpose(1, 2) @ wh
        exact = ref[0, :, rows].float()
        energy = float((exact.transpose(0, 1).reshape(-1, heads * 128) @ w.T).square().sum())
        total = errors.setdefault(layer, torch.zeros(heads, len(configs), dtype=torch.float64))
        for i, prefill in enumerate(executors):
            delta = prefill(layer, query, key, value, scaling)[0, :, rows].float() - exact
            total[:, i] += torch.einsum("htd,hde,hte->h", delta, gram, delta).double().cpu() / energy
        return ref

    use_attention(model, "triattention_calibration", attention)
    for n, (ids, _) in enumerate(samples):
        model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False, logits_to_keep=1)
        print(f"sample {n + 1}/{len(samples)}: {len(ids)} tokens", flush=True)
    result = dict(format="triattention-calibration-v1", model_type=model.config.model_type, rows=args.rows,
                  samples=len(samples), configs=configs,
                  errors={layer: (e / len(samples)).tolist() for layer, e in sorted(errors.items())})
    Path(args.output).write_text(json.dumps(result))


if __name__ == "__main__":
    main()
