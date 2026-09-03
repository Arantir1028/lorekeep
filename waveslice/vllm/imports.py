"""Lazy imports for the vLLM V1 integration boundary."""

from __future__ import annotations

import importlib

from waveslice.vllm.bootstrap import bootstrap_vllm_runtime

_V1_CLASSES = {
    "scheduler": ("vllm.v1.core.sched.scheduler", "Scheduler"),
    "engine": ("vllm.v1.engine.llm_engine", "LLMEngine"),
    "processor": ("vllm.v1.engine.processor", "Processor"),
    "engine_core": ("vllm.v1.engine.core", "EngineCore"),
    "request": ("vllm.v1.request", "Request"),
}


def _load(role: str) -> type:
    bootstrap_vllm_runtime()
    module_name, class_name = _V1_CLASSES[role]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def load_scheduler_target(scheduler_cls: type | None = None) -> tuple[type, str]:
    """Return the V1 scheduler class and its scheduling method name."""

    return scheduler_cls or _load("scheduler"), "schedule"


def load_llm_engine_cls() -> type:
    return _load("engine")


def load_v1_processor_cls() -> type:
    return _load("processor")


def load_v1_engine_core_cls() -> type:
    return _load("engine_core")


def load_v1_request_cls() -> type:
    return _load("request")


__all__ = [
    "load_llm_engine_cls",
    "load_scheduler_target",
    "load_v1_engine_core_cls",
    "load_v1_processor_cls",
    "load_v1_request_cls",
]
