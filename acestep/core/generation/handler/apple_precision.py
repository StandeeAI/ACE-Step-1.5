"""Precision and memory-release helpers for non-CUDA devices (mainly Apple Silicon).

On Apple Silicon the handler runs in float32 by default, so the PyTorch weights take twice the space of the bf16
checkpoint and the native MLX copy of the DiT decoder adds more. These helpers make that configurable without
changing any default: every switch is an opt-in environment variable.
"""

import gc
import os
from typing import Mapping, Optional

import torch

_TRUTHY = {"1", "true", "yes", "on"}


def env_flag(name: str, env: Optional[Mapping[str, str]] = None) -> bool:
    """Return True when environment variable ``name`` is set to a truthy value (default off)."""
    source = os.environ if env is None else env
    return str(source.get(name, "")).strip().lower() in _TRUTHY


def non_cuda_dtype(device: str, env: Optional[Mapping[str, str]] = None) -> torch.dtype:
    """Return the model dtype for a device that is neither CUDA nor ROCm.

    XPU uses bfloat16. MPS stays on float32 unless ``ACESTEP_MPS_BF16`` is set, which loads the PyTorch parts
    in bfloat16 (about half the memory). Everything else (CPU) uses float32.
    """
    if device == "xpu":
        return torch.bfloat16
    if device == "mps" and env_flag("ACESTEP_MPS_BF16", env):
        return torch.bfloat16
    return torch.float32


def release_unused_device_memory(device: str) -> None:
    """Hand cached-but-unused GPU memory back after loading (MPS only; a no-op elsewhere).

    Converting weights leaves freed buffers in the allocator caches (6 GB seen with the MLX DiT in bf16),
    which counts against the process until the caches are emptied.
    """
    if device != "mps":
        return
    gc.collect()
    try:
        torch.mps.empty_cache()
    except (AttributeError, RuntimeError):
        pass
    try:
        import mlx.core as mx

        clear = getattr(mx, "clear_cache", None) or mx.metal.clear_cache
        clear()
    except (ImportError, AttributeError):
        pass
