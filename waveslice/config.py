"""Public configuration objects for WaveSlice."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from waveslice.policy import WaveSlicePolicy

WAVESLICE_ADDITIONAL_CONFIG_KEY = "waveslice"


@dataclass(frozen=True)
class WaveSliceConfig:
    """Settings required to activate WaveSlice for one vLLM engine."""

    lut_model: str | None = None
    gamma: float = 2.0
    policy: WaveSlicePolicy = field(default_factory=WaveSlicePolicy)

    @classmethod
    def from_value(
        cls,
        value: WaveSliceConfig | Mapping[str, Any] | None,
    ) -> WaveSliceConfig:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("wave_slice_config must be a WaveSliceConfig or mapping")

        values = dict(value)
        policy = values.get("policy")
        if isinstance(policy, Mapping):
            values["policy"] = WaveSlicePolicy(**dict(policy))
        return cls(**values)

    def resolve_lut_model(self, engine_model: str) -> str:
        configured = (self.lut_model or "").strip()
        if configured:
            return configured
        inferred = str(engine_model).rstrip("/").rsplit("/", 1)[-1]
        if not inferred:
            raise ValueError("WaveSlice could not infer a LUT model from EngineArgs.model")
        return inferred

    def to_vllm_config(self, engine_model: str) -> dict[str, Any]:
        return {
            "enabled": True,
            "lut_model": self.resolve_lut_model(engine_model),
            "gamma": float(self.gamma),
            "policy": asdict(self.policy),
        }


__all__ = [
    "WAVESLICE_ADDITIONAL_CONFIG_KEY",
    "WaveSliceConfig",
    "WaveSlicePolicy",
]
