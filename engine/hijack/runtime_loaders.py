from __future__ import annotations

import importlib
import os
from typing import Optional

from engine.runtime_bootstrap import bootstrap_vllm_runtime


def load_scheduler_target() -> tuple[type, str]:
    bootstrap_vllm_runtime()
    force_v1 = os.environ.get("VLLM_USE_V1", "").strip() == "1"

    candidates: list[tuple[str, str]] = []
    if force_v1:
        candidates.append(("vllm.v1.core.sched.scheduler", "schedule"))
        candidates.append(("vllm.core.scheduler", "_schedule"))
    else:
        candidates.append(("vllm.core.scheduler", "_schedule"))
        candidates.append(("vllm.v1.core.sched.scheduler", "schedule"))

    last_exc: Optional[Exception] = None
    for module_name, method_name in candidates:
        try:
            mod = importlib.import_module(module_name)
            cls = getattr(mod, "Scheduler", None)
            if cls is None:
                continue
            method = getattr(cls, method_name, None)
            if callable(method):
                return cls, method_name
        except Exception as exc:
            last_exc = exc
            continue
    raise RuntimeError("vLLM Scheduler class/method not found.") from last_exc


def load_model_runner_cls() -> type:
    bootstrap_vllm_runtime()
    force_v1 = os.environ.get("VLLM_USE_V1", "").strip() == "1"
    candidates = (
        [("vllm.v1.worker.gpu_model_runner", "GPUModelRunner"), ("vllm.worker.model_runner", "ModelRunner")]
        if force_v1
        else [("vllm.worker.model_runner", "ModelRunner"), ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner")]
    )
    last_exc: Optional[Exception] = None
    for mod_name, cls_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, cls_name, None)
            if cls is not None:
                return cls
        except Exception as exc:
            last_exc = exc
    raise RuntimeError("vLLM ModelRunner/GPUModelRunner class not found.") from last_exc


def load_llm_engine_cls() -> type:
    bootstrap_vllm_runtime()
    force_v1 = os.environ.get("VLLM_USE_V1", "").strip() == "1"
    candidates = (
        [("vllm.v1.engine.llm_engine", "LLMEngine"), ("vllm.engine.llm_engine", "LLMEngine")]
        if force_v1
        else [("vllm.engine.llm_engine", "LLMEngine"), ("vllm.v1.engine.llm_engine", "LLMEngine")]
    )
    last_exc: Optional[Exception] = None
    for mod_name, cls_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, cls_name, None)
            if cls is not None:
                return cls
        except Exception as exc:
            last_exc = exc
    raise RuntimeError("vLLM LLMEngine class not found.") from last_exc


def load_v1_output_processor_cls() -> type:
    bootstrap_vllm_runtime()
    mod = importlib.import_module("vllm.v1.engine.output_processor")
    cls = getattr(mod, "OutputProcessor", None)
    if cls is None:
        raise RuntimeError("vLLM v1 OutputProcessor class not found.")
    return cls


def load_v1_processor_cls() -> type:
    bootstrap_vllm_runtime()
    mod = importlib.import_module("vllm.v1.engine.processor")
    cls = getattr(mod, "Processor", None)
    if cls is None:
        raise RuntimeError("vLLM v1 Processor class not found.")
    return cls


def load_v1_engine_core_cls() -> type:
    bootstrap_vllm_runtime()
    mod = importlib.import_module("vllm.v1.engine.core")
    cls = getattr(mod, "EngineCore", None)
    if cls is None:
        raise RuntimeError("vLLM v1 EngineCore class not found.")
    return cls


def load_sequence_data_cls() -> type:
    bootstrap_vllm_runtime()
    mod = importlib.import_module("vllm.sequence")
    cls = getattr(mod, "SequenceData", None)
    if cls is None:
        raise RuntimeError("vLLM SequenceData class not found.")
    return cls


def load_v1_request_cls() -> type:
    bootstrap_vllm_runtime()
    mod = importlib.import_module("vllm.v1.request")
    cls = getattr(mod, "Request", None)
    if cls is None:
        raise RuntimeError("vLLM v1 Request class not found.")
    return cls


def load_logits_processor_lora_cls() -> type:
    bootstrap_vllm_runtime()
    mod = importlib.import_module("vllm.lora.layers")
    cls = getattr(mod, "LogitsProcessorWithLoRA", None)
    if cls is None:
        raise RuntimeError("vLLM LogitsProcessorWithLoRA class not found.")
    return cls
