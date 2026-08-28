"""Runtime bootstrap and shutdown helpers for vLLM V1."""

from __future__ import annotations

import os
from typing import Any


def ensure_v1_runtime() -> None:
    """Select vLLM V1 and reject an explicitly requested legacy engine."""

    configured = os.environ.get("VLLM_USE_V1")
    if configured not in (None, "1"):
        raise RuntimeError("WaveSlice requires the vLLM V1 engine")
    os.environ["VLLM_USE_V1"] = "1"


def bootstrap_vllm_runtime() -> None:
    """Prepare the process before importing or constructing a vLLM engine."""

    ensure_v1_runtime()


def shutdown_vllm_engine(engine: Any | None) -> None:
    """Shut down a vLLM V1 engine created by the experiment harness."""

    if engine is None:
        return
    if type(engine).__module__ != "vllm.v1.engine.llm_engine":
        engine_type = f"{type(engine).__module__}.{type(engine).__name__}"
        raise TypeError(f"unexpected vLLM engine type: {engine_type}")
    engine.engine_core.shutdown()


__all__ = ["bootstrap_vllm_runtime", "ensure_v1_runtime", "shutdown_vllm_engine"]
