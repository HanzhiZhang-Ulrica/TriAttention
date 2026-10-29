from __future__ import annotations

import torch
import torch.nn.functional as F
import transformers
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoConfig, AutoTokenizer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from triattention.kernels.decode import DecodeCache
from triattention.kernels.extension import load_extension
from triattention.kernels.prefill import Prefill
from triattention.maps import load_map, preset


def load_tokenizer(path):
    mistral = AutoConfig.from_pretrained(path).model_type == "mistral3"
    return AutoTokenizer.from_pretrained(path, **({"fix_mistral_regex": True} if mistral else {}))


def load_model(path):
    config = AutoConfig.from_pretrained(path)
    if config.model_type == "qwen2":
        theta = config.rope_parameters["rope_theta"]
        config.rope_parameters = dict(rope_type="yarn", factor=4.0, original_max_position_embeddings=32768,
                                      rope_theta=theta)
        config.max_position_embeddings = 4 * 32768
    if hasattr(config, "vision_config"):
        loader = transformers.AutoModelForImageTextToText
    else:
        loader = transformers.AutoModelForCausalLM
    return loader.from_pretrained(path, config=config, dtype=torch.bfloat16).cuda().eval()


def render_prompt(tokenizer, prompt):
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                         add_generation_prompt=True, date_string="30 Jul 2026")


def eos_ids(tokenizer, model):
    as_set = lambda ids: set(ids if isinstance(ids, list) else [ids])
    return as_set(tokenizer.eos_token_id) | as_set(model.generation_config.eos_token_id)


def exact_attention(query, key, value, scaling):
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        return F.scaled_dot_product_attention(query, key, value, is_causal=True, scale=scaling, enable_gqa=True)


def use_attention(model, name, fn):
    def attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        return fn(module, query, key, value, scaling).transpose(1, 2).contiguous(), None

    ALL_ATTENTION_FUNCTIONS[name] = attention
    model.set_attn_implementation(name)


class TriAttention:
    def __init__(self, model, qk_map=None, max_new_tokens=256):
        self.model = model
        module = load_extension()
        self.decode_cache = DecodeCache(module, max_tail=max_new_tokens)
        self.prefill = Prefill(module, load_map(qk_map or preset(model), model), self.decode_cache)

        def attention(mod, query, key, value, scaling):
            if query.shape[2] == 1 and self.decode_cache.stored(mod.layer_idx):
                return self.decode_cache(mod.layer_idx, query, key, value, scaling)
            return self.prefill(mod.layer_idx, query, key, value, scaling)

        use_attention(model, "triattention", attention)

    @torch.inference_mode()
    def generate(self, ids, max_new_tokens, eos):
        out = self.model(input_ids=ids, use_cache=False, logits_to_keep=1)
        tokens = []
        for i in range(max_new_tokens):
            token = int(out.logits[0, -1].argmax())
            if token in eos:
                break
            tokens.append(token)
            position = torch.tensor([[ids.shape[1] + i]], device=ids.device)
            out = self.model(input_ids=torch.tensor([[token]], device=ids.device), position_ids=position,
                             use_cache=False)
            self.decode_cache.advance()
        return tokens
