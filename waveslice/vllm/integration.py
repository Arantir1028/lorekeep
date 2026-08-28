"""WaveSlice activation and compatibility boundary for vLLM V1."""

from __future__ import annotations

import contextlib
import importlib
import json
import logging
import os
import threading
from collections.abc import Callable, Iterator
from typing import Any

from waveslice.metrics import WaveSliceMetrics
from waveslice.policy import WaveSlicePolicy
from waveslice.scheduling.scheduler import WaveScheduler
from waveslice.vllm.common import estimate_prompt_tokens, estimate_solo_us
from waveslice.vllm.engine_hooks import (
    build_add_request_hook,
    build_step_hook,
    build_v1_engine_core_add_request_hook,
    build_v1_process_inputs_hook,
)
from waveslice.vllm.imports import (
    load_llm_engine_cls,
    load_scheduler_target,
    load_v1_engine_core_cls,
    load_v1_processor_cls,
    load_v1_request_cls,
)
from waveslice.vllm.phase1_state import phase1_maybe_seed_ingress_virtual
from waveslice.vllm.request_hooks import (
    _build_v1_request_num_tokens_hook,
    _build_v1_request_num_tokens_with_spec_hook,
    _build_v1_scheduler_add_request_hook,
    _build_v1_scheduler_update_after_schedule_hook,
    _lookup_engine_prompt_tokens,
)
from waveslice.vllm.runtime import _build_scheduler_hook
from waveslice.vllm.state import RuntimeState
from waveslice.vllm.subprocess import (
    RUNTIME_ENV_ENABLED,
    RUNTIME_ENV_GAMMA,
    RUNTIME_ENV_MODEL,
    RUNTIME_ENV_POLICY,
    RUNTIME_ENV_SCHEDULER,
    clear_runtime_environment,
    publish_runtime_environment,
    read_cross_process_metrics,
    reset_cross_process_metrics_file,
)

logger = logging.getLogger("waveslice")
logger.addHandler(logging.NullHandler())

_runtime_lock = threading.RLock()
_runtime_state: RuntimeState | None = None


def _scheduler_class_from_environment() -> type | None:
    qualified_name = os.environ.get(RUNTIME_ENV_SCHEDULER, "").strip()
    if not qualified_name:
        return None

    module_name, separator, attribute_path = qualified_name.partition(":")
    if not separator or not module_name or not attribute_path:
        raise ValueError(f"invalid WaveSlice scheduler path: {qualified_name!r}")

    value: Any = importlib.import_module(module_name)
    for attribute in attribute_path.split("."):
        value = getattr(value, attribute)
    if not isinstance(value, type):
        raise TypeError(f"WaveSlice scheduler is not a class: {qualified_name!r}")
    return value


def activate_from_environment() -> None:
    """Activate WaveSlice in a vLLM child process configured by its parent."""

    if os.environ.get(RUNTIME_ENV_ENABLED, "").strip() != "1":
        return
    with _runtime_lock:
        if _runtime_state is not None:
            return

    model_name = os.environ.get(RUNTIME_ENV_MODEL, "").strip()
    if not model_name:
        return

    gamma = float(os.environ.get(RUNTIME_ENV_GAMMA, "2.0"))
    policy = WaveSlicePolicy(**json.loads(os.environ.get(RUNTIME_ENV_POLICY, "{}")))
    activate_wave_slice(
        model_name=model_name,
        gamma=gamma,
        policy=policy,
        scheduler_cls=_scheduler_class_from_environment(),
        force=False,
    )
    logger.info("[WaveSlice] activated child process for model=%s", model_name)


def _required_method(owner: type, name: str) -> Callable[..., Any]:
    value = getattr(owner, name)
    if not callable(value):
        raise TypeError(f"{owner.__name__}.{name} must be callable")
    return value


def _required_property(owner: type, name: str) -> property:
    value = getattr(owner, name)
    if not isinstance(value, property):
        raise TypeError(f"{owner.__name__}.{name} must be a property")
    return value


def _install_scheduler_hooks(state: RuntimeState) -> None:
    scheduler = state.scheduler_cls
    request = state.v1_request_cls = load_v1_request_cls()
    state.original_v1_request_num_tokens = _required_property(request, "num_tokens")
    state.original_v1_request_num_tokens_with_spec = _required_property(
        request, "num_tokens_with_spec"
    )
    request.num_tokens = _build_v1_request_num_tokens_hook(state)
    request.num_tokens_with_spec = _build_v1_request_num_tokens_with_spec_hook(state)

    state.original_scheduler_add_request = _required_method(scheduler, "add_request")
    state.original_scheduler_update_after_schedule = _required_method(
        scheduler, "_update_after_schedule"
    )
    scheduler.add_request = _build_v1_scheduler_add_request_hook(state)
    scheduler._update_after_schedule = _build_v1_scheduler_update_after_schedule_hook(state)


def _install_metrics_hooks(state: RuntimeState) -> None:
    engine = state.llm_engine_cls = load_llm_engine_cls()
    state.original_add_request = _required_method(engine, "add_request")
    state.original_step = _required_method(engine, "step")
    engine.add_request = build_add_request_hook(
        state,
        estimate_prompt_tokens=estimate_prompt_tokens,
        estimate_solo_us=estimate_solo_us,
        lookup_engine_prompt_tokens=_lookup_engine_prompt_tokens,
        phase1_maybe_seed_ingress_virtual=phase1_maybe_seed_ingress_virtual,
    )
    engine.step = build_step_hook(state)

    processor = state.v1_processor_cls = load_v1_processor_cls()
    core = state.v1_engine_core_cls = load_v1_engine_core_cls()
    state.original_v1_processor_process_inputs = _required_method(processor, "process_inputs")
    state.original_v1_engine_core_add_request = _required_method(core, "add_request")
    processor.process_inputs = build_v1_process_inputs_hook(
        state,
        estimate_solo_us=estimate_solo_us,
        phase1_maybe_seed_ingress_virtual=phase1_maybe_seed_ingress_virtual,
    )
    core.add_request = build_v1_engine_core_add_request_hook(
        state,
        estimate_solo_us=estimate_solo_us,
        phase1_maybe_seed_ingress_virtual=phase1_maybe_seed_ingress_virtual,
    )


def _restore_patches(state: RuntimeState) -> None:
    for owner, attribute, original in (
        (state.scheduler_cls, "schedule", state.original_schedule),
        (state.scheduler_cls, "add_request", state.original_scheduler_add_request),
        (
            state.scheduler_cls,
            "_update_after_schedule",
            state.original_scheduler_update_after_schedule,
        ),
        (state.v1_request_cls, "num_tokens", state.original_v1_request_num_tokens),
        (
            state.v1_request_cls,
            "num_tokens_with_spec",
            state.original_v1_request_num_tokens_with_spec,
        ),
        (state.llm_engine_cls, "add_request", state.original_add_request),
        (state.llm_engine_cls, "step", state.original_step),
        (state.v1_processor_cls, "process_inputs", state.original_v1_processor_process_inputs),
        (state.v1_engine_core_cls, "add_request", state.original_v1_engine_core_add_request),
    ):
        if owner is not None and original is not None:
            setattr(owner, attribute, original)


def activate_wave_slice(
    model_name: str,
    *,
    gamma: float = 2.0,
    policy: WaveSlicePolicy | None = None,
    force: bool = False,
    scheduler_cls: type | None = None,
) -> None:
    """Activate WaveSlice for the selected V1 scheduler class."""

    global _runtime_state
    chosen_policy = policy or WaveSlicePolicy()
    with _runtime_lock:
        if _runtime_state is not None and not force:
            same = (
                _runtime_state.model_name == model_name
                and float(_runtime_state.brain.gamma) == float(gamma)
                and _runtime_state.policy == chosen_policy
                and (scheduler_cls is None or _runtime_state.scheduler_cls is scheduler_cls)
            )
            if not same:
                raise RuntimeError(
                    "WaveSlice is already active with a different model, policy, or scheduler"
                )
            return

        if _runtime_state is not None:
            deactivate_wave_slice()

        scheduler_cls, _ = load_scheduler_target(scheduler_cls)
        current_schedule = _required_method(scheduler_cls, "schedule")
        objective_mode = str(chosen_policy.scheduler_objective_mode).strip().lower()
        if objective_mode not in {"fair_escape", "pure_gain"}:
            raise ValueError(
                f"unknown scheduler objective: {chosen_policy.scheduler_objective_mode}"
            )

        state = RuntimeState(
            scheduler_cls=scheduler_cls,
            original_schedule=current_schedule,
            brain=WaveScheduler(
                model_name=model_name,
                gamma=gamma,
                objective_mode=objective_mode,
            ),
            policy=chosen_policy,
            model_name=model_name,
            metrics=WaveSliceMetrics(
                short_threshold_tokens=chosen_policy.metrics_short_request_tokens
            ),
        )
        try:
            _install_scheduler_hooks(state)
            scheduler_cls.schedule = _build_scheduler_hook(state)
            if chosen_policy.enable_metrics_hook:
                _install_metrics_hooks(state)
            publish_runtime_environment(model_name, gamma, chosen_policy, scheduler_cls)
        except Exception:
            _restore_patches(state)
            clear_runtime_environment()
            raise

        _runtime_state = state
        logger.info(
            "[WaveSlice] activated model=%s scheduler=%s.schedule phase2=%s metrics=%s",
            model_name,
            scheduler_cls.__module__,
            str(chosen_policy.enable_phase2_scheduler),
            str(chosen_policy.enable_metrics_hook),
        )


def deactivate_wave_slice() -> None:
    """Deactivate WaveSlice and restore all patched vLLM methods."""

    global _runtime_state
    with _runtime_lock:
        if _runtime_state is None:
            clear_runtime_environment()
            return

        state = _runtime_state
        _restore_patches(state)
        _runtime_state = None
        clear_runtime_environment()
        logger.info("[WaveSlice] deactivated from the vLLM runtime")


def is_wave_slice_active() -> bool:
    with _runtime_lock:
        return _runtime_state is not None


def get_wave_slice_metrics(*, reset: bool = False) -> dict[str, Any]:
    with _runtime_lock:
        state = _runtime_state
    if state is None:
        return {}

    report = state.metrics.summary(read_cross_process_metrics())
    if reset:
        reset_cross_process_metrics_file()
        state.metrics.reset()
    return report


def reset_wave_slice_metrics() -> None:
    reset_cross_process_metrics_file()
    with _runtime_lock:
        state = _runtime_state
    if state is not None:
        state.metrics.reset()


@contextlib.contextmanager
def wave_slice_session(
    model_name: str,
    *,
    gamma: float = 2.0,
    policy: WaveSlicePolicy | None = None,
    force: bool = False,
) -> Iterator[None]:
    """Compatibility context manager for the former imperative API."""

    chosen_policy = policy or WaveSlicePolicy()
    with _runtime_lock:
        active = _runtime_state is not None
        if active and force:
            raise RuntimeError("force=True cannot replace an active WaveSlice session")

    activate_wave_slice(model_name=model_name, gamma=gamma, policy=chosen_policy, force=False)
    try:
        yield
    finally:
        if not active:
            deactivate_wave_slice()


def inject_wave_slice(
    model_name: str,
    *,
    gamma: float = 2.0,
    policy: WaveSlicePolicy | None = None,
    force: bool = False,
    scheduler_cls: type | None = None,
) -> None:
    """Compatibility wrapper for the former imperative API."""

    activate_wave_slice(
        model_name,
        gamma=gamma,
        policy=policy,
        force=force,
        scheduler_cls=scheduler_cls,
    )


def uninject_wave_slice() -> None:
    """Compatibility wrapper for the former imperative API."""

    deactivate_wave_slice()


def is_wave_slice_injected() -> bool:
    """Compatibility wrapper for the former imperative API."""

    return is_wave_slice_active()


__all__ = [
    "activate_from_environment",
    "activate_wave_slice",
    "deactivate_wave_slice",
    "get_wave_slice_metrics",
    "is_wave_slice_active",
    "reset_wave_slice_metrics",
]
