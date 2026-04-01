"""Runtime bootstrap helpers for vLLM entrypoints."""

from __future__ import annotations

import ctypes
import os


def ensure_cuda_runtime_libs() -> bool:
    """Return whether the expected CUDA runtime lib is visible."""
    try:
        ctypes.CDLL("libcusparseLt.so.0")
        return True
    except OSError:
        # Some environments can still import torch successfully because
        # wheels resolve this dependency via internal loader paths.
        try:
            import torch  # noqa: F401

            return True
        except Exception:
            return False


def ensure_vllm_mode() -> None:
    """Optionally force vLLM mode via WAVESLICE_VLLM_MODE.

    WAVESLICE_VLLM_MODE:
    - "v0": force VLLM_USE_V1=0
    - "v1": force VLLM_USE_V1=1
    - "auto": keep vLLM default behavior
    - unset/other: default to v0 for backward compatibility
    """
    mode = os.environ.get("WAVESLICE_VLLM_MODE", "").strip().lower()
    if mode == "v0":
        os.environ["VLLM_USE_V1"] = "0"
    elif mode == "v1":
        os.environ["VLLM_USE_V1"] = "1"
    elif mode == "auto":
        return
    elif "VLLM_USE_V1" not in os.environ:
        os.environ["VLLM_USE_V1"] = "0"


def ensure_vllm_cuda_platform() -> bool:
    """Best-effort fix for environments where vLLM platform auto-detect is empty."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
    except Exception:
        return False

    try:
        import vllm.platforms as platforms

        if getattr(platforms.current_platform, "device_type", ""):
            return True

        from vllm.platforms.cuda import CudaPlatform

        platforms._current_platform = CudaPlatform()  # type: ignore[attr-defined]
        return True
    except Exception:
        return False


def bootstrap_vllm_runtime() -> bool:
    """Apply environment/runtime tweaks needed by this project before vLLM import."""
    ensure_vllm_mode()
    ok = ensure_cuda_runtime_libs()
    ensure_vllm_cuda_platform()
    return ok
