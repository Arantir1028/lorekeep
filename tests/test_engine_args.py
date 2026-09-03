"""Contracts for the declarative WaveSlice vLLM entry point."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest

from tests.eval_config import build_wave_slice_config
from waveslice import WaveSliceConfig, WaveSlicePolicy
from waveslice.config import WAVESLICE_ADDITIONAL_CONFIG_KEY
from waveslice.lut.config import resolve_model_name
from waveslice.vllm import integration
from waveslice.vllm.bootstrap import ensure_v1_runtime


@dataclass
class _NativeEngineArgs:
    model: str
    scheduler_cls: object = "vllm.v1.core.sched.scheduler.Scheduler"
    additional_config: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.native_post_init_called = True

    def create_engine_config(self, *_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            scheduler_cls=self.scheduler_cls,
            additional_config=dict(self.additional_config),
        )


class _NativeScheduler:
    def schedule(self) -> None:
        return None

    def add_request(self) -> None:
        return None

    def _update_after_schedule(self) -> None:
        return None


def _package(name: str) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = []
    return module


@pytest.fixture
def engine_args_module(monkeypatch: pytest.MonkeyPatch):
    modules = {
        "vllm": _package("vllm"),
        "vllm.engine": _package("vllm.engine"),
        "vllm.engine.arg_utils": ModuleType("vllm.engine.arg_utils"),
        "vllm.v1": _package("vllm.v1"),
        "vllm.v1.core": _package("vllm.v1.core"),
        "vllm.v1.core.sched": _package("vllm.v1.core.sched"),
        "vllm.v1.core.sched.scheduler": ModuleType("vllm.v1.core.sched.scheduler"),
    }
    modules["vllm.engine.arg_utils"].EngineArgs = _NativeEngineArgs
    modules["vllm.v1.core.sched.scheduler"].Scheduler = _NativeScheduler

    monkeypatch.delenv("VLLM_USE_V1", raising=False)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    for name in ("waveslice.engine_args", "waveslice.vllm.scheduler"):
        sys.modules.pop(name, None)

    module = importlib.import_module("waveslice.engine_args")
    yield module

    for name in ("waveslice.engine_args", "waveslice.vllm.scheduler"):
        sys.modules.pop(name, None)


def test_config_accepts_mapping_and_infers_lut_name() -> None:
    config = WaveSliceConfig.from_value(
        {
            "gamma": 1.5,
            "policy": {"enable_phase2_scheduler": False},
        }
    )

    assert config.gamma == 1.5
    assert config.policy.enable_phase2_scheduler is False
    assert config.resolve_lut_model("/models/Mistral-7B-v0.1/") == "Mistral-7B-v0.1"
    assert (
        config.resolve_lut_model("mistralai/Mistral-7B-Instruct-v0.2")
        == "mistralai--Mistral-7B-Instruct-v0.2"
    )
    assert (
        config.resolve_lut_model(
            "/cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/revision"
        )
        == "Qwen--Qwen2.5-7B-Instruct"
    )


def test_lut_resolution_prefers_an_exact_generated_profile() -> None:
    assert (
        resolve_model_name("mistralai/Mistral-7B-Instruct-v0.2")
        == "mistralai--Mistral-7B-Instruct-v0.2"
    )
    assert (
        resolve_model_name("Mistral-7B-Instruct-v0.2")
        == "mistralai--Mistral-7B-Instruct-v0.2"
    )
    assert resolve_model_name("Mistral-7B-v0.10") == "Mistral-7B-v0.10"


def test_evaluator_leaves_lut_selection_to_the_engine_model() -> None:
    config = build_wave_slice_config(mode="phase1_only", phase1_gamma=1.5)

    assert config is not None
    assert config.lut_model is None
    assert config.gamma == 1.5


def test_disabled_engine_args_preserve_native_scheduler(engine_args_module) -> None:
    args = engine_args_module.EngineArgs(
        model="/models/Mistral-7B-v0.1",
        enable_wave_slice=False,
        additional_config={"other_plugin": {"enabled": True}},
    )

    with (
        mock.patch.object(integration, "activate_wave_slice") as activate,
        mock.patch.object(integration, "deactivate_wave_slice") as deactivate,
    ):
        config = args.create_engine_config()

    assert args.native_post_init_called
    assert config.scheduler_cls == "vllm.v1.core.sched.scheduler.Scheduler"
    assert config.additional_config == {"other_plugin": {"enabled": True}}
    activate.assert_not_called()
    deactivate.assert_called_once_with()


def test_enabled_engine_args_select_adapter_and_publish_config(engine_args_module) -> None:
    policy = WaveSlicePolicy(enable_phase2_scheduler=False)
    args = engine_args_module.EngineArgs(
        model="/models/Mistral-7B-v0.1",
        enable_wave_slice=True,
        wave_slice_config=WaveSliceConfig(
            lut_model="Mistral-7B-v0.1",
            gamma=1.25,
            policy=policy,
        ),
    )

    with mock.patch.object(integration, "activate_wave_slice") as activate:
        config = args.create_engine_config()

    assert issubclass(config.scheduler_cls, _NativeScheduler)
    payload = config.additional_config[WAVESLICE_ADDITIONAL_CONFIG_KEY]
    assert payload["enabled"] is True
    assert payload["lut_model"] == "Mistral-7B-v0.1"
    assert payload["gamma"] == 1.25
    assert payload["policy"]["enable_phase2_scheduler"] is False
    activate.assert_called_once_with(
        "Mistral-7B-v0.1",
        gamma=1.25,
        policy=policy,
        scheduler_cls=config.scheduler_cls,
    )


def test_engine_args_can_toggle_back_to_native(engine_args_module) -> None:
    args = engine_args_module.EngineArgs(
        model="/models/Mistral-7B-v0.1",
        enable_wave_slice=True,
    )

    with (
        mock.patch.object(integration, "activate_wave_slice"),
        mock.patch.object(integration, "deactivate_wave_slice") as deactivate,
    ):
        args.create_engine_config()
        args.enable_wave_slice = False
        config = args.create_engine_config()

    assert config.scheduler_cls == "vllm.v1.core.sched.scheduler.Scheduler"
    assert WAVESLICE_ADDITIONAL_CONFIG_KEY not in config.additional_config
    deactivate.assert_called_once_with()


def test_engine_args_reject_another_custom_scheduler(engine_args_module) -> None:
    class OtherScheduler:
        pass

    args = engine_args_module.EngineArgs(
        model="/models/Mistral-7B-v0.1",
        scheduler_cls=OtherScheduler,
        enable_wave_slice=True,
    )

    with pytest.raises(ValueError, match="another custom scheduler_cls"):
        args.create_engine_config()


def test_wave_slice_rejects_an_explicit_legacy_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_USE_V1", "0")

    with pytest.raises(RuntimeError, match="requires the vLLM V1 engine"):
        ensure_v1_runtime()
