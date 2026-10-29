from __future__ import annotations

import argparse

from triattention.models import TriAttention, eos_ids, load_model, load_tokenizer, render_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Run greedy generation with TriAttention.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--map")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    tokenizer, model = load_tokenizer(args.model), load_model(args.model)
    ids = tokenizer(render_prompt(tokenizer, args.prompt), return_tensors="pt").input_ids.cuda()
    tokens = TriAttention(model, args.map, args.max_new_tokens).generate(ids, args.max_new_tokens, eos_ids(tokenizer, model))
    print(tokenizer.decode(tokens, skip_special_tokens=True))


if __name__ == "__main__":
    main()
