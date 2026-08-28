"""V1 token-cap hooks for returning early from a long prefill."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from waveslice.vllm.common import safe_request_id, safe_total_tokens
from waveslice.vllm.phase1_math import phase1_authoritative_chunk
from waveslice.vllm.state import RuntimeState


def _request_fields(request: Any) -> tuple[str, int, int, int]:
    return (
        str(getattr(request, "request_id", "") or ""),
        int(getattr(request, "num_prompt_tokens", 0) or 0),
        int(getattr(request, "num_output_tokens", 0) or 0),
        int(getattr(request, "num_computed_tokens", 0) or 0),
    )


def _prefill_cap(state: Any, request: Any, computed: int | None = None) -> int | None:
    request_id, prompt, output, native_computed = _request_fields(request)
    computed = native_computed if computed is None else max(native_computed, int(computed))
    if not request_id or output > 0 or computed >= max(1, prompt):
        return None
    return state.phase1_virtual_token_caps.get(request_id)


def _cap_v1_request_total_tokens(state: Any, request: Any, total: Any) -> int:
    total = int(total)
    if getattr(request, "__wave_slice_skip_v1_cap__", False):
        return total
    effective = max(
        int(getattr(request, "num_computed_tokens", 0) or 0),
        int(getattr(request, "__wave_slice_effective_computed_tokens__", 0) or 0),
    )
    cap = _prefill_cap(state, request, effective)
    if cap is None:
        return total
    state.metrics.record_phase1_virtual_cap_probe(
        helper_called=True, prefill_call=True, target_hit=True
    )
    capped = min(total, effective + max(0, int(cap)))
    applied = effective < capped < total
    state.metrics.record_phase1_virtual_cap(
        old_total_tokens=total, new_total_tokens=capped if applied else total, applied=applied
    )
    return capped if applied else total


def _property_hook(state: Any, original: property) -> property:
    @functools.wraps(original.fget)
    def value(request: Any):
        return _cap_v1_request_total_tokens(state, request, original.fget(request))

    return property(value)


def _build_v1_request_num_tokens_hook(state: RuntimeState) -> property:
    return _property_hook(state, state.original_v1_request_num_tokens)


def _build_v1_request_num_tokens_with_spec_hook(state: RuntimeState) -> property:
    return _property_hook(state, state.original_v1_request_num_tokens_with_spec)


def _install_v1_waiting_cap_runtime_hooks(state: RuntimeState, scheduler: Any):
    cleanup: list[Any] = []
    restore: list[tuple[Any, str, Any]] = []
    manager = getattr(scheduler, "kv_cache_manager", None)
    original = getattr(manager, "get_computed_blocks", None)
    if callable(original):

        @functools.wraps(original)
        def get_blocks(request: Any, *args: Any, _original=original, **kwargs: Any):
            request.__wave_slice_skip_v1_cap__ = True
            try:
                result = _original(request, *args, **kwargs)
            finally:
                request.__wave_slice_skip_v1_cap__ = False
            request.__wave_slice_effective_computed_tokens__ = max(0, int(result[1]))
            cleanup.append(request)
            return result

        restore.append((manager, "get_computed_blocks", original))
        manager.get_computed_blocks = get_blocks
    connector = getattr(scheduler, "connector", None)
    original = getattr(connector, "get_num_new_matched_tokens", None)
    if callable(original):

        @functools.wraps(original)
        def get_matched(request: Any, local: int, *args: Any, _original=original, **kwargs: Any):
            result = _original(request, local, *args, **kwargs)
            effective = int(
                getattr(request, "__wave_slice_effective_computed_tokens__", local) or 0
            )
            request.__wave_slice_effective_computed_tokens__ = max(0, effective) + max(
                0, int(result[0])
            )
            cleanup.append(request)
            return result

        restore.append((connector, "get_num_new_matched_tokens", original))
        connector.get_num_new_matched_tokens = get_matched
    return cleanup, restore


def _restore_v1_waiting_cap_runtime_hooks(
    cleanup: list[Any], restore: list[tuple[Any, str, Any]]
) -> None:
    for owner, name, original in reversed(restore):
        setattr(owner, name, original)
    for request in cleanup:
        for name in ("__wave_slice_skip_v1_cap__", "__wave_slice_effective_computed_tokens__"):
            if hasattr(request, name):
                delattr(request, name)


def _lookup_engine_prompt_tokens(engine: Any, *, request_id: str) -> int | None:
    schedulers = getattr(engine, "scheduler", ())
    schedulers = schedulers if isinstance(schedulers, (list, tuple)) else [schedulers]
    for scheduler in schedulers:
        for name in ("waiting", "running", "swapped"):
            for group in getattr(scheduler, name, ()) or ():
                if safe_request_id(group) == request_id and (total := safe_total_tokens(group)):
                    return int(total)
        request = (getattr(scheduler, "requests", {}) or {}).get(request_id)
        if request is not None and (prompt := int(getattr(request, "num_prompt_tokens", 0) or 0)):
            return prompt
    return None


def _maybe_apply_v1_scheduler_provisional_cap(state: Any, request: Any) -> None:
    if not state.policy.phase1_ingress_direct_authoritative:
        return
    request_id, prompt, output, computed = _request_fields(request)
    floor = max(1, state.policy.min_long_seq, state.policy.metrics_short_request_tokens)
    if not request_id or prompt <= floor or output or computed:
        return
    upper = prompt - 1
    cap = phase1_authoritative_chunk(
        state.policy,
        state.slicer,
        target=state.policy.phase1_ingress_target_chunk,
        short_len=min(state.policy.metrics_short_request_tokens, upper),
        upper=upper,
    )
    if getattr(request, "lora_request", None) is not None:
        cap = min(
            cap,
            max(
                state.policy.phase1_force_min_chunk,
                int(prompt * state.policy.phase1_target_long_fraction),
            ),
        )
    cap = max(1, min(cap, upper))
    if (
        request_id not in state.phase1_virtual_token_caps
        or state.phase1_virtual_token_caps[request_id] > cap
    ):
        state.phase1_virtual_token_caps[request_id] = cap
        state.metrics.record_phase1_virtual_cap_probe(target_set=True)


def _build_v1_scheduler_add_request_hook(state: RuntimeState) -> Callable[..., Any]:
    original = state.original_scheduler_add_request

    @functools.wraps(original)
    def add_request(scheduler: Any, request: Any, *args: Any, **kwargs: Any):
        _maybe_apply_v1_scheduler_provisional_cap(state, request)
        return original(scheduler, request, *args, **kwargs)

    add_request.__wave_slice_phase1_add_request_hook__ = True
    return add_request


def _clamp_v1_scheduler_output_before_state_advance(
    state: Any, scheduler: Any, output: Any
) -> None:
    scheduled = getattr(output, "num_scheduled_tokens", None)
    if (
        not output
        or not state.policy.phase1_ingress_direct_authoritative
        or not isinstance(scheduled, dict)
    ):
        return
    delta = 0
    for request_id, old in list(scheduled.items()):
        request = getattr(scheduler, "requests", {}).get(str(request_id))
        cap = _prefill_cap(state, request) if request is not None else None
        if cap is None or (new := max(1, min(int(cap), int(old)))) >= int(old):
            continue
        scheduled[request_id], delta = new, delta + int(old) - new
    if delta:
        output.total_num_scheduled_tokens = max(0, int(output.total_num_scheduled_tokens) - delta)


def _build_v1_scheduler_update_after_schedule_hook(state: RuntimeState) -> Callable[..., Any]:
    original = state.original_scheduler_update_after_schedule

    @functools.wraps(original)
    def update(scheduler: Any, output: Any, *args: Any, **kwargs: Any):
        _clamp_v1_scheduler_output_before_state_advance(state, scheduler, output)
        return original(scheduler, output, *args, **kwargs)

    update.__wave_slice_phase1_update_after_schedule_hook__ = True
    return update
