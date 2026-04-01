"""Public package API for Wave-Slice."""

from engine.vllm_hijacker import (
    WaveSlicePolicy,
    get_wave_slice_metrics,
    inject_wave_slice,
    is_wave_slice_injected,
    reset_wave_slice_metrics,
    uninject_wave_slice,
    wave_slice_session,
)
from scheduler.wave_scheduler import WaveScheduler

__all__ = [
    "WaveScheduler",
    "WaveSlicePolicy",
    "get_wave_slice_metrics",
    "inject_wave_slice",
    "uninject_wave_slice",
    "is_wave_slice_injected",
    "reset_wave_slice_metrics",
    "wave_slice_session",
]
