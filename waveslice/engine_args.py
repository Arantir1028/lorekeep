"""Drop-in vLLM EngineArgs with WaveSlice activation fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from waveslice.config import WAVESLICE_ADDITIONAL_CONFIG_KEY, WaveSliceConfig
from waveslice.vllm.bootstrap import bootstrap_vllm_runtime

bootstrap_vllm_runtime()

from vllm.engine.arg_utils import EngineArgs as VllmEngineArgs  # noqa: E402

_VLLM_V1_SCHEDULER_PATH = "vllm.v1.core.sched.scheduler.Scheduler"


@dataclass
class WaveSliceEngineArgs(VllmEngineArgs):
    """A vLLM ``EngineArgs`` subclass that activates WaveSlice declaratively."""

    enable_wave_slice: bool = False
    wave_slice_config: WaveSliceConfig | dict[str, Any] | None = None

    def __post_init__(self) -> None:
        bootstrap_vllm_runtime()
        super().__post_init__()
        self.wave_slice_config = WaveSliceConfig.from_value(self.wave_slice_config)
        self._waveslice_native_scheduler_cls = self.scheduler_cls

    def create_engine_config(self, *args: Any, **kwargs: Any):
        bootstrap_vllm_runtime()
        settings = WaveSliceConfig.from_value(self.wave_slice_config)
        additional_config = dict(self.additional_config or {})

        if self.enable_wave_slice:
            from waveslice.vllm.scheduler import WaveSliceScheduler

            native_scheduler_cls = self._waveslice_native_scheduler_cls
            allowed_native_schedulers = (
                VllmEngineArgs.scheduler_cls,
                _VLLM_V1_SCHEDULER_PATH,
                WaveSliceScheduler.__mro__[1],
            )
            if native_scheduler_cls not in allowed_native_schedulers or self.scheduler_cls not in (
                native_scheduler_cls,
                WaveSliceScheduler,
            ):
                raise ValueError(
                    "enable_wave_slice cannot be combined with another custom scheduler_cls"
                )
            self.scheduler_cls = WaveSliceScheduler
            additional_config[WAVESLICE_ADDITIONAL_CONFIG_KEY] = settings.to_vllm_config(self.model)
        else:
            from waveslice.vllm.integration import deactivate_wave_slice

            deactivate_wave_slice()
            self.scheduler_cls = self._waveslice_native_scheduler_cls
            additional_config.pop(WAVESLICE_ADDITIONAL_CONFIG_KEY, None)

        self.additional_config = additional_config
        vllm_config = super().create_engine_config(*args, **kwargs)

        if self.enable_wave_slice:
            from waveslice.vllm.integration import activate_wave_slice
            from waveslice.vllm.scheduler import WaveSliceScheduler

            activate_wave_slice(
                settings.resolve_lut_model(self.model),
                gamma=settings.gamma,
                policy=settings.policy,
                scheduler_cls=WaveSliceScheduler,
            )
        return vllm_config


EngineArgs = WaveSliceEngineArgs

__all__ = ["EngineArgs", "WaveSliceEngineArgs"]
