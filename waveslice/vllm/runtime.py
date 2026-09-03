"""Compose the WaveSlice scheduler hook for vLLM V1.

Phase I planning, Phase I state mutation, and Phase II cashout live in focused
modules. This module only coordinates those components around the native
scheduler call.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterable
from typing import Any

from waveslice.vllm.common import (
    estimate_solo_us as _estimate_solo_us,
    has_running_waiting_queues as _has_running_waiting_queues,
    phase2_scheduler_cashout_enabled as _phase2_scheduler_cashout_enabled,
    safe_request_id as _safe_request_id,
    safe_total_tokens as _safe_total_tokens,
)
from waveslice.vllm.phase1_runtime import (
    _phase1_reorder_scheduler_queues,
    _phase1_rewrite_scheduler_outputs,
    _phase1_schedule_scope,
    _restore_hidden,
)
from waveslice.vllm.phase2_runtime import (
    _phase12_apply_scheduler_cashout_to_queues,
    _phase12_scheduler_cashout_rewrite,
    _phase12_tick_recent_phase1,
    _phase12_tick_recent_phase2,
)
from waveslice.vllm.state import RuntimeState


def _lora_enabled(scheduler_obj: Any) -> bool:
    if getattr(scheduler_obj, "lora_config", None) is not None:
        return True
    if any(
        bool(getattr(scheduler_obj, name, False))
        for name in ("enable_lora", "lora_enabled", "has_lora", "_enable_lora")
    ):
        return True
    config = getattr(scheduler_obj, "scheduler_config", None)
    return any(bool(getattr(config, name, False)) for name in ("enable_lora", "lora_enabled"))


def _observe_scheduler_requests(state: RuntimeState, queues: Iterable[Any]) -> None:
    for seq_group in queues:
        request_id = _safe_request_id(seq_group)
        if not request_id:
            continue
        total_tokens = _safe_total_tokens(seq_group)
        state.metrics.observe_scheduler_request(
            request_id,
            total_tokens=total_tokens,
            solo_us=_estimate_solo_us(state.brain, total_tokens),
            is_short=total_tokens is not None
            and total_tokens <= state.policy.metrics_short_request_tokens,
        )


def _post_schedule_cashout(state: RuntimeState, outputs: Any) -> Any:
    if _phase2_scheduler_cashout_enabled(state.policy):
        (outputs, _) = _phase12_scheduler_cashout_rewrite(state=state, scheduler_outputs=outputs)
    return outputs


def _run_native_schedule(
    state: RuntimeState,
    scheduler_obj: Any,
    schedule_impl: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Run the native scheduler while temporarily hiding Phase-II candidates."""
    hidden_running: list[Any] = []
    hidden_waiting: list[Any] = []
    if _phase2_scheduler_cashout_enabled(state.policy) and _has_running_waiting_queues(
        scheduler_obj
    ):
        (scheduler_obj.running, scheduler_obj.waiting, hidden_running, hidden_waiting, _) = (
            _phase12_apply_scheduler_cashout_to_queues(
                state=state,
                running=scheduler_obj.running,
                waiting=scheduler_obj.waiting,
            )
        )
    try:
        outputs = schedule_impl(scheduler_obj, *args, **kwargs)
    finally:
        _restore_hidden(scheduler_obj, "running", hidden_running)
        _restore_hidden(scheduler_obj, "waiting", hidden_waiting)
    return _post_schedule_cashout(state, outputs)


def _build_scheduler_hook(state: RuntimeState) -> Callable[..., Any]:
    original_schedule = state.original_schedule
    schedule_impl = original_schedule

    @functools.wraps(original_schedule)
    def _wave_schedule_hook(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not _has_running_waiting_queues(self) or self.running is None or self.waiting is None:
            return _run_native_schedule(state, self, schedule_impl, args, kwargs)
        _observe_scheduler_requests(state, list(self.waiting) + list(self.running))
        _phase12_tick_recent_phase1(state)
        _phase12_tick_recent_phase2(state)
        if not state.policy.enable_phase1_scheduler:
            state.metrics.record_scheduler_decision(False)
            return _run_native_schedule(state, self, schedule_impl, args, kwargs)
        lora_enabled = _lora_enabled(self)
        can_threshold = state.policy.enable_phase1_dynamic_threshold and (
            not lora_enabled
            or state.policy.allow_phase1_with_lora
            or state.policy.allow_phase1_threshold_with_lora
        )
        can_budget = state.policy.enable_phase1_budget_guidance and (
            not lora_enabled
            or state.policy.allow_phase1_with_lora
            or state.policy.allow_phase1_budget_with_lora
        )
        if lora_enabled and (not (can_threshold or can_budget)):
            state.metrics.record_scheduler_decision(False)
            return _run_native_schedule(state, self, schedule_impl, args, kwargs)
        _phase1_reorder_scheduler_queues(state, self)
        scheduler_config = getattr(self, "scheduler_config", None)
        original_budget = getattr(scheduler_config, "max_num_batched_tokens", None)
        original_threshold = getattr(scheduler_config, "long_prefill_token_threshold", None)
        with _phase1_schedule_scope(
            state=state,
            scheduler_obj=self,
            lora_enabled=lora_enabled,
            can_threshold=can_threshold,
            can_budget=can_budget,
            scheduler_config=scheduler_config,
            original_budget=original_budget,
            original_threshold=original_threshold,
        ) as decision:
            if decision is None:
                return _run_native_schedule(state, self, schedule_impl, args, kwargs)
            outputs = _post_schedule_cashout(state, schedule_impl(self, *args, **kwargs))
            outputs, _, groups, old_sum, new_sum, delta = _phase1_rewrite_scheduler_outputs(
                outputs=outputs,
                request_id=decision.cohort.long_req_id,
                target_chunk=decision.best_chunk,
            )
            state.metrics.record_phase1_rewrite(
                rewritten_groups=groups,
                old_chunk_sum=old_sum,
                new_chunk_sum=new_sum,
                token_delta_sum=delta,
            )
            return outputs

    _wave_schedule_hook.__wave_slice_hook__ = True
    return _wave_schedule_hook


__all__ = ["_build_scheduler_hook"]
