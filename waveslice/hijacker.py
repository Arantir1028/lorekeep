"""Compatibility imports for the legacy imperative WaveSlice API.

New applications should configure :class:`waveslice.EngineArgs` instead.
"""

from waveslice.policy import WaveSlicePolicy
from waveslice.vllm.integration import (
    get_wave_slice_metrics,
    inject_wave_slice,
    is_wave_slice_injected,
    reset_wave_slice_metrics,
    uninject_wave_slice,
    wave_slice_session,
)

__all__ = [
    "WaveSlicePolicy",
    "get_wave_slice_metrics",
    "inject_wave_slice",
    "is_wave_slice_injected",
    "reset_wave_slice_metrics",
    "uninject_wave_slice",
    "wave_slice_session",
]
