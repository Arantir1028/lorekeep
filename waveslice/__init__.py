"""Public API for the WaveSlice vLLM plugin."""

from __future__ import annotations

from typing import Any

from waveslice.config import WaveSliceConfig
from waveslice.policy import WaveSlicePolicy
from waveslice.scheduling import SlicePlan, WaveBaseSlicer, WaveScheduler
from waveslice.vllm.integration import get_wave_slice_metrics, reset_wave_slice_metrics


def __getattr__(name: str) -> Any:
    if name in {"EngineArgs", "WaveSliceEngineArgs"}:
        from waveslice.engine_args import EngineArgs, WaveSliceEngineArgs

        return {
            "EngineArgs": EngineArgs,
            "WaveSliceEngineArgs": WaveSliceEngineArgs,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EngineArgs",
    "SlicePlan",
    "WaveBaseSlicer",
    "WaveScheduler",
    "WaveSliceConfig",
    "WaveSliceEngineArgs",
    "WaveSlicePolicy",
    "get_wave_slice_metrics",
    "reset_wave_slice_metrics",
]
