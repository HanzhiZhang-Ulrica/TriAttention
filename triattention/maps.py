from __future__ import annotations

import json
from pathlib import Path

import torch

INF = 1 << 20
DATA = Path(__file__).resolve().parent / "data"
PRESETS = {"llama": "llama-3.2-3b-instruct", "qwen2": "qwen2-7b-instruct", "mistral3": "ministral-3-8b-instruct-2512"}


def preset(model):
    name = PRESETS.get(model.config.model_type)
    if name is None:
        raise ValueError(f"no bundled map for model type {model.config.model_type!r}; pass --map")
    return DATA / f"{name}.json"


def load_map(path, model=None):
    return from_bands(json.loads(Path(path).read_text())["bands"], model)


def from_bands(bands, model=None):
    if model is not None:
        text = getattr(model.config, "text_config", model.config)
        if len(bands) != text.num_hidden_layers or len(bands["0"]) != text.num_attention_heads:
            raise ValueError("the map does not match the model's layers and heads")

    def bands_fn(layer, heads):
        return torch.tensor(bands[str(layer)], dtype=torch.int32)

    return bands_fn


def uniform(w8, w4):
    def bands_fn(layer, heads):
        return torch.tensor([[w8, w4]] * heads, dtype=torch.int32)

    bands_fn.layer_independent = True
    return bands_fn
