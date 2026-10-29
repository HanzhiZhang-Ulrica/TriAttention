from __future__ import annotations

import functools
import hashlib
import os
import tempfile
from pathlib import Path

import torch

CUDA_DIR = Path(__file__).resolve().parent / "cuda"


def arch() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("TriAttention requires a CUDA GPU.")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) < (8, 0):
        raise RuntimeError("TriAttention requires compute capability 8.0 or newer.")
    return f"sm{major}{minor}"


def source(name: str) -> Path:
    specific = CUDA_DIR / f"prefill_{name}.cu"
    return specific if specific.is_file() else CUDA_DIR / "prefill_sm80.cu"


@functools.cache
def load_extension():
    from torch.utils.cpp_extension import load

    name = arch()
    src = source(name)
    files = [src, CUDA_DIR / "common_device.cuh", CUDA_DIR / "common_host.cuh"]
    digest = hashlib.sha256(
        b"".join(f.read_bytes() for f in files) + torch.__version__.encode() + str(torch.version.cuda).encode()
    ).hexdigest()[:12]
    module = f"triattention_{name}_{digest}"
    root = Path(os.environ.get("TORCH_EXTENSIONS_DIR", Path(tempfile.gettempdir()) / "triattention-extensions"))
    build = root / module
    build.mkdir(parents=True, exist_ok=True)
    if src.name == "prefill_sm90.cu" and name == "sm90":
        gencode = "-gencode=arch=compute_90a,code=sm_90a"
    else:
        gencode = f"-gencode=arch=compute_{name[2:]},code=sm_{name[2:]}"
    return load(
        name=module,
        sources=[str(src)],
        build_directory=str(build),
        extra_include_paths=[str(CUDA_DIR)],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17", "--use_fast_math", "-lineinfo", gencode],
        with_cuda=True,
    )
