# TriAttention

Requirements: Python 3.10+, an NVIDIA GPU with compute capability 8.0 or newer, and a CUDA toolkit with `nvcc`.

```bash
pip install -e .
```

## Inference

```bash
python -m triattention.infer --model /path/to/model --prompt "$(cat prompt.txt)" --max-new-tokens 128
```

`--model` is a local Hugging Face checkpoint of Llama-3.2-3B-Instruct, Qwen2-7B-Instruct or Ministral-3-8B-Instruct-2512-BF16. The map of each model is in `triattention/data/` and is chosen automatically. The CUDA extension is compiled on the first run.

## Calibration

```bash
python -m triattention.calibrate --model /path/to/model --samples calibration.jsonl --output calibration.json
python -m triattention.allocate --calibration calibration.json --output candidates.json
python -m triattention.select --model /path/to/model --candidates candidates.json --samples selection.jsonl --output map.json
python -m triattention.infer --model /path/to/model --map map.json --prompt "$(cat prompt.txt)"
```

Each line of a samples file is a JSON object with `prompt` and `response` strings.
