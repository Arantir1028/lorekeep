from __future__ import annotations

import functools
import inspect
import logging
import re
import textwrap
from typing import Any, Callable, Optional

from engine.hijack.common import (
    safe_first_seq as _safe_first_seq,
    safe_request_id as _safe_request_id,
    safe_total_tokens as _safe_total_tokens,
)
from engine.hijack.phase1_math import (
    phase1_authoritative_chunk as _phase1_authoritative_chunk,
    phase1_cohort_target_len as _phase1_cohort_target_len,
)
from engine.hijack.types import _PatchState, _Phase1CohortStats

logger = logging.getLogger("WaveSlice")


def _build_sequence_data_get_len_hook(state: _PatchState) -> Callable[..., Any]:
    original_get_len = state.original_sequence_data_get_len
    if original_get_len is None:
        raise RuntimeError("sequence-data hook requested without original get_len")

    @functools.wraps(original_get_len)
    def _wave_sequence_data_get_len(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            shadow_len = state.phase1_shadow_seq_lens.get(id(self))
            if shadow_len is not None and int(shadow_len) > 0:
                original_len = int(original_get_len(self, *args, **kwargs))
                return min(original_len, int(shadow_len))
        except Exception:
            pass
        return original_get_len(self, *args, **kwargs)

    _wave_sequence_data_get_len.__wave_slice_seq_len_hook__ = True  # type: ignore[attr-defined]
    return _wave_sequence_data_get_len


def _build_get_num_new_uncached_and_cached_tokens_hook(
    state: _PatchState,
) -> Callable[..., Any]:
    original_impl = state.original_get_new_uncached_and_cached_tokens
    if original_impl is None:
        raise RuntimeError("virtual-cap hook requested without original scheduler helper")

    @functools.wraps(original_impl)
    def _wave_get_num_new_uncached_and_cached_tokens(
        self: Any,
        seq_group: Any,
        status: Any,
        enable_chunking: bool = False,
        budget: Any = None,
        *args: Any,
        **kwargs: Any,
        ) -> Any:
        uncached, cached = original_impl(
            self,
            seq_group,
            status,
            enable_chunking,
            budget,
            *args,
            **kwargs,
        )
        state.metrics.record_phase1_virtual_cap_probe(helper_called=True)

        try:
            request_id = _safe_request_id(seq_group)
            target_chunk = state.phase1_virtual_token_caps.get(request_id) if request_id else None
            is_prefill = bool(seq_group.is_prefill()) if seq_group is not None else False
        except Exception:
            target_chunk = None
            is_prefill = False

        if (
            target_chunk is None
            and is_prefill
            and request_id
            and bool(state.policy.phase1_ingress_direct_authoritative)
        ):
            ingress_virtual = state.phase1_ingress_virtuals.get(str(request_id))
            if ingress_virtual is not None:
                try:
                    seq = _safe_first_seq(seq_group)
                    total_len = int(seq.get_len()) if seq is not None else int(ingress_virtual.original_long_len)
                    computed = int(seq.data.get_num_computed_tokens()) if seq is not None else 0
                    remaining = max(1, total_len - computed)
                except Exception:
                    remaining = max(1, int(ingress_virtual.original_long_len))
                fallback_cohort = _Phase1CohortStats(
                    representative_short_len=max(1, int(ingress_virtual.representative_short_len)),
                    short_count=max(1, int(ingress_virtual.short_count)),
                    short_token_mass=max(1, int(ingress_virtual.short_token_mass)),
                    short_lengths=[int(v) for v in ingress_virtual.short_lengths]
                    or [max(1, int(ingress_virtual.representative_short_len))],
                    long_len=max(1, int(remaining)),
                    long_req_id=str(request_id),
                    total_count=max(2, int(ingress_virtual.active_count)),
                )
                fallback_target = int(
                    _phase1_cohort_target_len(fallback_cohort, state.policy)
                )
                fallback_target = min(max(1, int(fallback_target)), max(1, int(remaining) - 1))
                if fallback_target > 0:
                    target_chunk = _phase1_authoritative_chunk(
                        state.policy,
                        state.slicer,
                        target=int(fallback_target),
                        short_len=int(fallback_cohort.representative_short_len),
                        upper=max(1, int(remaining) - 1),
                    )

        if is_prefill:
            state.metrics.record_phase1_virtual_cap_probe(prefill_call=True)
        if target_chunk is not None:
            state.metrics.record_phase1_virtual_cap_probe(target_hit=True)
        if request_id:
            computed_tokens = None
            try:
                seq = _safe_first_seq(seq_group)
                if seq is not None:
                    computed_tokens = int(seq.data.get_num_computed_tokens())
            except Exception:
                computed_tokens = None
            state.metrics.record_phase1_step_trace(
                request_id=str(request_id),
                event="virtual_cap_helper",
                is_prefill=is_prefill,
                num_computed_tokens=computed_tokens,
                uncached=int(uncached),
                cached=int(cached),
                target_chunk=int(target_chunk) if target_chunk is not None else None,
            )

        if not is_prefill or target_chunk is None or int(target_chunk) <= 0:
            return uncached, cached

        try:
            old_uncached = int(uncached)
            old_cached = int(cached)
            old_total = old_uncached + old_cached
            new_uncached = min(max(0, int(target_chunk)), old_uncached)
            new_cached = old_cached
            new_total = new_uncached + new_cached
            if new_uncached <= 0 or new_uncached >= old_uncached:
                state.metrics.record_phase1_virtual_cap(
                    old_total_tokens=old_total,
                    new_total_tokens=old_total,
                    applied=False,
                )
                return uncached, cached
            state.metrics.record_phase1_virtual_cap(
                old_total_tokens=old_total,
                new_total_tokens=new_uncached + new_cached,
                applied=True,
            )
            logger.info(
                "[Wave-Slice][P1-virtual-cap] req=%s old_total=%d uncached=%d cached=%d target=%d new_total=%d",
                str(request_id),
                old_total,
                int(uncached),
                int(cached),
                int(target_chunk),
                int(new_uncached + new_cached),
            )
            state.metrics.record_phase1_step_trace(
                request_id=str(request_id),
                event="virtual_cap_applied",
                is_prefill=is_prefill,
                num_computed_tokens=computed_tokens,
                uncached=int(new_uncached),
                cached=int(new_cached),
            )
            return new_uncached, new_cached
        except Exception:
            return uncached, cached

    _wave_get_num_new_uncached_and_cached_tokens.__wave_slice_virtual_cap_hook__ = True  # type: ignore[attr-defined]
    return _wave_get_num_new_uncached_and_cached_tokens


def _cap_v1_request_total_tokens(
    state: _PatchState,
    request: Any,
    original_total_tokens: Any,
) -> Any:
    try:
        total_tokens = int(original_total_tokens)
    except Exception:
        return original_total_tokens

    request_id = str(getattr(request, "request_id", "") or "")
    if not request_id:
        return total_tokens
    if bool(getattr(request, "__wave_slice_skip_v1_cap__", False)):
        return total_tokens
    target_chunk = state.phase1_virtual_token_caps.get(request_id)
    if target_chunk is None:
        return total_tokens

    try:
        num_output_tokens = int(getattr(request, "num_output_tokens", 0) or 0)
        num_prompt_tokens = int(getattr(request, "num_prompt_tokens", 0) or 0)
        num_computed_tokens = int(getattr(request, "num_computed_tokens", 0) or 0)
        effective_computed_tokens = int(
            getattr(
                request,
                "__wave_slice_effective_computed_tokens__",
                num_computed_tokens,
            )
            or 0
        )
    except Exception:
        return total_tokens
    effective_computed_tokens = max(num_computed_tokens, effective_computed_tokens)

    is_prefill = num_output_tokens <= 0 and effective_computed_tokens < max(1, num_prompt_tokens)
    if not is_prefill:
        return total_tokens

    state.metrics.record_phase1_virtual_cap_probe(helper_called=True)
    state.metrics.record_phase1_virtual_cap_probe(prefill_call=True)
    state.metrics.record_phase1_virtual_cap_probe(target_hit=True)
    state.metrics.record_phase1_step_trace(
        request_id=request_id,
        event="v1_virtual_cap_property",
        is_prefill=True,
        num_computed_tokens=effective_computed_tokens,
        uncached=max(0, total_tokens - effective_computed_tokens),
        cached=0,
        target_chunk=int(target_chunk),
    )

    capped_total_tokens = min(
        total_tokens,
        effective_computed_tokens + max(0, int(target_chunk)),
    )
    if capped_total_tokens >= total_tokens or capped_total_tokens <= effective_computed_tokens:
        state.metrics.record_phase1_virtual_cap(
            old_total_tokens=total_tokens,
            new_total_tokens=total_tokens,
            applied=False,
        )
        return total_tokens

    state.metrics.record_phase1_virtual_cap(
        old_total_tokens=total_tokens,
        new_total_tokens=capped_total_tokens,
        applied=True,
    )
    logger.info(
        "[Wave-Slice][P1-v1-virtual-cap] req=%s old_total=%d computed=%d target=%d new_total=%d",
        request_id,
        total_tokens,
        effective_computed_tokens,
        int(target_chunk),
        capped_total_tokens,
    )
    return capped_total_tokens


def _install_v1_waiting_cap_runtime_hooks(state: _PatchState, scheduler_obj: Any) -> tuple[list[Any], list[tuple[Any, str, Any]]]:
    cleanup_requests: list[Any] = []
    restore_specs: list[tuple[Any, str, Any]] = []

    kv_cache_manager = getattr(scheduler_obj, "kv_cache_manager", None)
    original_get_computed_blocks = getattr(kv_cache_manager, "get_computed_blocks", None)
    if callable(original_get_computed_blocks):
        @functools.wraps(original_get_computed_blocks)
        def _wave_get_computed_blocks(request: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                setattr(request, "__wave_slice_skip_v1_cap__", True)
                result = original_get_computed_blocks(request, *args, **kwargs)
            finally:
                try:
                    setattr(request, "__wave_slice_skip_v1_cap__", False)
                except Exception:
                    pass
            try:
                _blocks, num_new_computed_tokens = result
                setattr(
                    request,
                    "__wave_slice_effective_computed_tokens__",
                    int(max(0, num_new_computed_tokens)),
                )
                cleanup_requests.append(request)
            except Exception:
                pass
            return result

        restore_specs.append((kv_cache_manager, "get_computed_blocks", original_get_computed_blocks))
        setattr(kv_cache_manager, "get_computed_blocks", _wave_get_computed_blocks)

    connector = getattr(scheduler_obj, "connector", None)
    original_get_num_new_matched_tokens = getattr(connector, "get_num_new_matched_tokens", None)
    if callable(original_get_num_new_matched_tokens):
        @functools.wraps(original_get_num_new_matched_tokens)
        def _wave_get_num_new_matched_tokens(
            request: Any,
            num_new_local_computed_tokens: int,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            result = original_get_num_new_matched_tokens(
                request,
                num_new_local_computed_tokens,
                *args,
                **kwargs,
            )
            try:
                num_external_computed_tokens, _load_kv_async = result
                local_tokens = int(
                    getattr(
                        request,
                        "__wave_slice_effective_computed_tokens__",
                        max(0, int(num_new_local_computed_tokens)),
                    )
                    or 0
                )
                setattr(
                    request,
                    "__wave_slice_effective_computed_tokens__",
                    max(0, local_tokens) + max(0, int(num_external_computed_tokens)),
                )
                cleanup_requests.append(request)
            except Exception:
                pass
            return result

        restore_specs.append((connector, "get_num_new_matched_tokens", original_get_num_new_matched_tokens))
        setattr(connector, "get_num_new_matched_tokens", _wave_get_num_new_matched_tokens)

    return cleanup_requests, restore_specs


def _restore_v1_waiting_cap_runtime_hooks(
    cleanup_requests: list[Any],
    restore_specs: list[tuple[Any, str, Any]],
) -> None:
    for owner, attr, original in reversed(restore_specs):
        try:
            setattr(owner, attr, original)
        except Exception:
            pass
    for request in cleanup_requests:
        try:
            delattr(request, "__wave_slice_skip_v1_cap__")
        except Exception:
            pass
        try:
            delattr(request, "__wave_slice_effective_computed_tokens__")
        except Exception:
            pass


def _lookup_engine_prompt_tokens(
    engine_self: Any,
    *,
    request_id: str,
) -> Optional[int]:
    req_id = str(request_id or "")
    if not req_id:
        return None
    schedulers = getattr(engine_self, "scheduler", None)
    if schedulers is None:
        return None
    try:
        scheduler_items = list(schedulers) if isinstance(schedulers, (list, tuple)) else [schedulers]
    except Exception:
        scheduler_items = [schedulers]

    for scheduler_obj in scheduler_items:
        for attr in ("waiting", "running", "swapped"):
            queue_like = getattr(scheduler_obj, attr, None)
            if queue_like is None:
                continue
            try:
                seq_groups = list(queue_like)
            except Exception:
                continue
            for seq_group in seq_groups:
                if _safe_request_id(seq_group) != req_id:
                    continue
                total = _safe_total_tokens(seq_group)
                if total is not None and int(total) > 0:
                    return int(total)
        requests = getattr(scheduler_obj, "requests", None)
        if isinstance(requests, dict):
            request_obj = requests.get(req_id)
            if request_obj is None:
                continue
            try:
                prompt_tokens = int(getattr(request_obj, "num_prompt_tokens", 0) or 0)
            except Exception:
                prompt_tokens = 0
            if prompt_tokens > 0:
                return prompt_tokens
    return None


def _build_v1_request_num_tokens_hook(state: _PatchState) -> property:
    original_prop = state.original_v1_request_num_tokens
    if not isinstance(original_prop, property) or original_prop.fget is None:
        raise RuntimeError("v1 Request.num_tokens hook requested without original property")

    @functools.wraps(original_prop.fget)
    def _wave_v1_num_tokens(self: Any) -> Any:
        original_total = original_prop.fget(self)
        return _cap_v1_request_total_tokens(state, self, original_total)

    return property(_wave_v1_num_tokens)


def _build_v1_request_num_tokens_with_spec_hook(state: _PatchState) -> property:
    original_prop = state.original_v1_request_num_tokens_with_spec
    if not isinstance(original_prop, property) or original_prop.fget is None:
        raise RuntimeError("v1 Request.num_tokens_with_spec hook requested without original property")

    @functools.wraps(original_prop.fget)
    def _wave_v1_num_tokens_with_spec(self: Any) -> Any:
        original_total = original_prop.fget(self)
        return _cap_v1_request_total_tokens(state, self, original_total)

    return property(_wave_v1_num_tokens_with_spec)


def _maybe_apply_v1_scheduler_provisional_cap(
    state: _PatchState,
    request: Any,
) -> None:
    if not bool(getattr(state.policy, "phase1_ingress_direct_authoritative", False)):
        return

    request_id = str(getattr(request, "request_id", "") or "")
    if not request_id:
        return

    try:
        num_prompt_tokens = int(getattr(request, "num_prompt_tokens", 0) or 0)
        num_output_tokens = int(getattr(request, "num_output_tokens", 0) or 0)
        num_computed_tokens = int(getattr(request, "num_computed_tokens", 0) or 0)
    except Exception:
        return
    if num_prompt_tokens <= 0 or num_output_tokens > 0 or num_computed_tokens > 0:
        return

    long_floor = max(
        int(max(1, state.policy.min_long_seq)),
        int(max(1, state.policy.metrics_short_request_tokens)),
    )
    if num_prompt_tokens <= long_floor:
        return

    upper = max(1, int(num_prompt_tokens) - 1)
    provisional_target = max(1, int(state.policy.phase1_ingress_target_chunk))
    provisional_chunk = _phase1_authoritative_chunk(
        state.policy,
        state.slicer,
        target=int(provisional_target),
        short_len=int(min(max(1, state.policy.metrics_short_request_tokens), upper)),
        upper=int(upper),
    )
    lora_request = getattr(request, "lora_request", None)
    if lora_request is not None:
        # LoRA mixed-adapter workloads need a much earlier cut than the global
        # ingress floor, otherwise same-adapter short requests stay glued to a
        # moderate long prefill until after the first large chunk.
        lora_fraction_target = max(
            int(max(1, state.policy.phase1_force_min_chunk)),
            int(max(1.0, float(num_prompt_tokens) * float(state.policy.phase1_target_long_fraction))),
        )
        lora_fraction_target = max(1, min(int(lora_fraction_target), int(upper)))
        provisional_chunk = min(int(provisional_chunk), int(lora_fraction_target))
    provisional_chunk = max(1, min(int(provisional_chunk), int(upper)))
    if provisional_chunk <= 0:
        return

    existing_cap = state.phase1_virtual_token_caps.get(request_id)
    if existing_cap is None or int(existing_cap) > int(provisional_chunk):
        state.phase1_virtual_token_caps[request_id] = int(provisional_chunk)
        state.metrics.record_phase1_virtual_cap_probe(target_set=True)
        state.metrics.record_phase1_step_trace(
            request_id=request_id,
            event="phase1_v1_scheduler_add_request_provisional_cap",
            is_prefill=True,
            num_computed_tokens=int(num_computed_tokens),
            uncached=int(num_prompt_tokens),
            cached=0,
            target_chunk=int(provisional_chunk),
        )


def _build_v1_scheduler_add_request_hook(state: _PatchState) -> Callable[..., Any]:
    original_add_request = state.original_scheduler_add_request
    if original_add_request is None:
        raise RuntimeError("v1 Scheduler.add_request hook requested without original method")

    @functools.wraps(original_add_request)
    def _wave_v1_scheduler_add_request(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            _maybe_apply_v1_scheduler_provisional_cap(state, request)
        except Exception:
            logger.exception("[Wave-Slice] failed to apply provisional v1 scheduler add_request cap.")
        return original_add_request(self, request, *args, **kwargs)

    _wave_v1_scheduler_add_request.__wave_slice_phase1_add_request_hook__ = True  # type: ignore[attr-defined]
    return _wave_v1_scheduler_add_request


def _clamp_v1_scheduler_output_before_state_advance(
    state: _PatchState,
    scheduler_obj: Any,
    scheduler_output: Any,
) -> None:
    if scheduler_output is None or not bool(getattr(state.policy, "phase1_ingress_direct_authoritative", False)):
        return

    num_scheduled_tokens = getattr(scheduler_output, "num_scheduled_tokens", None)
    if not isinstance(num_scheduled_tokens, dict) or not num_scheduled_tokens:
        return

    request_map = getattr(scheduler_obj, "requests", None)
    request_map = request_map if isinstance(request_map, dict) else {}

    total_delta = 0
    clamped_any = False
    for request_id, scheduled_tokens in list(num_scheduled_tokens.items()):
        req_id = str(request_id or "")
        if not req_id:
            continue
        target_chunk = state.phase1_virtual_token_caps.get(req_id)
        if target_chunk is None:
            continue

        request = request_map.get(req_id)
        try:
            num_output_tokens = int(getattr(request, "num_output_tokens", 0) or 0)
            num_prompt_tokens = int(getattr(request, "num_prompt_tokens", 0) or 0)
            num_computed_tokens = int(getattr(request, "num_computed_tokens", 0) or 0)
        except Exception:
            num_output_tokens = 0
            num_prompt_tokens = 0
            num_computed_tokens = 0

        is_prefill = num_output_tokens <= 0 and num_computed_tokens < max(1, num_prompt_tokens)
        if not is_prefill:
            continue

        old_scheduled_tokens = int(max(0, scheduled_tokens))
        new_scheduled_tokens = max(1, min(int(target_chunk), old_scheduled_tokens))
        if new_scheduled_tokens >= old_scheduled_tokens:
            continue

        num_scheduled_tokens[req_id] = int(new_scheduled_tokens)
        total_delta += int(old_scheduled_tokens - new_scheduled_tokens)
        clamped_any = True
        state.metrics.record_phase1_step_trace(
            request_id=req_id,
            event="phase1_v1_update_after_schedule_clamp",
            is_prefill=True,
            num_computed_tokens=int(num_computed_tokens),
            uncached=int(old_scheduled_tokens),
            cached=0,
            target_chunk=int(new_scheduled_tokens),
        )

    if not clamped_any:
        return

    try:
        total_num_scheduled_tokens = int(getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0)
        scheduler_output.total_num_scheduled_tokens = max(0, total_num_scheduled_tokens - total_delta)
    except Exception:
        pass


def _build_v1_scheduler_update_after_schedule_hook(state: _PatchState) -> Callable[..., Any]:
    original_update_after_schedule = state.original_scheduler_update_after_schedule
    if original_update_after_schedule is None:
        raise RuntimeError("v1 Scheduler._update_after_schedule hook requested without original method")

    @functools.wraps(original_update_after_schedule)
    def _wave_v1_scheduler_update_after_schedule(self: Any, scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            _clamp_v1_scheduler_output_before_state_advance(
                state=state,
                scheduler_obj=self,
                scheduler_output=scheduler_output,
            )
        except Exception:
            logger.exception("[Wave-Slice] failed to clamp v1 scheduler output before state advance.")
        return original_update_after_schedule(self, scheduler_output, *args, **kwargs)

    _wave_v1_scheduler_update_after_schedule.__wave_slice_phase1_update_after_schedule_hook__ = True  # type: ignore[attr-defined]
    return _wave_v1_scheduler_update_after_schedule


def _compute_v1_waiting_num_new_tokens(
    state: _PatchState,
    scheduler_obj: Any,
    request: Any,
    num_computed_tokens: Any,
    token_budget: Any,
) -> int:
    del scheduler_obj, token_budget
    try:
        num_computed = max(0, int(num_computed_tokens or 0))
    except Exception:
        num_computed = 0

    try:
        base_num_new_tokens = int(getattr(request, "num_tokens", 0) or 0) - num_computed
    except Exception:
        return 0
    if base_num_new_tokens <= 0:
        return base_num_new_tokens

    request_id = str(getattr(request, "request_id", "") or "")
    if not request_id:
        return base_num_new_tokens

    target_chunk = state.phase1_virtual_token_caps.get(request_id)
    if target_chunk is None:
        return base_num_new_tokens

    try:
        num_prompt_tokens = int(getattr(request, "num_prompt_tokens", 0) or 0)
        num_output_tokens = int(getattr(request, "num_output_tokens", 0) or 0)
    except Exception:
        return base_num_new_tokens

    if num_prompt_tokens <= 0 or num_output_tokens > 0 or num_computed >= num_prompt_tokens:
        return base_num_new_tokens

    state.metrics.record_phase1_step_trace(
        request_id=request_id,
        event="phase1_v1_schedule_waiting_num_new_tokens_seen",
        is_prefill=True,
        num_computed_tokens=int(num_computed),
        uncached=int(base_num_new_tokens),
        cached=0,
        target_chunk=int(target_chunk),
    )

    clamped_num_new_tokens = max(1, min(int(target_chunk), int(base_num_new_tokens)))
    if clamped_num_new_tokens >= int(base_num_new_tokens):
        return base_num_new_tokens

    state.metrics.record_phase1_step_trace(
        request_id=request_id,
        event="phase1_v1_schedule_waiting_num_new_tokens_clamp",
        is_prefill=True,
        num_computed_tokens=int(num_computed),
        uncached=int(base_num_new_tokens),
        cached=0,
        target_chunk=int(clamped_num_new_tokens),
    )
    return clamped_num_new_tokens


def _compute_v1_running_num_new_tokens(
    state: _PatchState,
    scheduler_obj: Any,
    request: Any,
) -> int:
    del scheduler_obj
    try:
        num_computed = max(0, int(getattr(request, "num_computed_tokens", 0) or 0))
        total_with_spec = int(getattr(request, "num_tokens_with_spec", 0) or 0)
        num_output_placeholders = int(getattr(request, "num_output_placeholders", 0) or 0)
    except Exception:
        return 0

    base_num_new_tokens = total_with_spec + num_output_placeholders - num_computed
    if base_num_new_tokens <= 0:
        return base_num_new_tokens

    request_id = str(getattr(request, "request_id", "") or "")
    if not request_id:
        return base_num_new_tokens

    target_chunk = state.phase1_virtual_token_caps.get(request_id)
    if target_chunk is None:
        return base_num_new_tokens

    try:
        num_prompt_tokens = int(getattr(request, "num_prompt_tokens", 0) or 0)
        num_output_tokens = int(getattr(request, "num_output_tokens", 0) or 0)
    except Exception:
        return base_num_new_tokens

    if num_prompt_tokens <= 0 or num_output_tokens > 0 or num_computed >= num_prompt_tokens:
        return base_num_new_tokens

    state.metrics.record_phase1_step_trace(
        request_id=request_id,
        event="phase1_v1_schedule_running_num_new_tokens_seen",
        is_prefill=True,
        num_computed_tokens=int(num_computed),
        uncached=int(base_num_new_tokens),
        cached=0,
        target_chunk=int(target_chunk),
    )

    clamped_num_new_tokens = max(1, min(int(target_chunk), int(base_num_new_tokens)))
    if clamped_num_new_tokens >= int(base_num_new_tokens):
        return base_num_new_tokens

    state.metrics.record_phase1_step_trace(
        request_id=request_id,
        event="phase1_v1_schedule_running_num_new_tokens_clamp",
        is_prefill=True,
        num_computed_tokens=int(num_computed),
        uncached=int(base_num_new_tokens),
        cached=0,
        target_chunk=int(clamped_num_new_tokens),
    )
    return clamped_num_new_tokens


def _build_v1_schedule_waiting_patch(
    state: _PatchState,
    original_schedule: Callable[..., Any],
) -> Callable[..., Any]:
    try:
        source = textwrap.dedent(inspect.getsource(original_schedule))
    except Exception:
        logger.exception("[Wave-Slice] failed to read v1 Scheduler.schedule source for waiting patch.")
        return original_schedule

    waiting_needle = "num_new_tokens = request.num_tokens - num_computed_tokens"
    waiting_replacement = (
        "num_new_tokens = __wave_slice_compute_v1_waiting_num_new_tokens(\n"
        "    self,\n"
        "    request,\n"
        "    num_computed_tokens,\n"
        "    token_budget,\n"
        ")"
    )
    running_pattern = re.compile(
        r"num_new_tokens = \(request\.num_tokens_with_spec \+\n\s+request\.num_output_placeholders -\n\s+request\.num_computed_tokens\)"
    )
    running_replacement = (
        "num_new_tokens = __wave_slice_compute_v1_running_num_new_tokens(\n"
        "    self,\n"
        "    request,\n"
        ")"
    )
    patched_source, running_count = running_pattern.subn(running_replacement, source, count=1)
    waiting_count = patched_source.count(waiting_needle)
    if waiting_count > 0:
        patched_source = patched_source.replace(waiting_needle, waiting_replacement, 1)
    if running_count <= 0 and waiting_count <= 0:
        logger.warning("[Wave-Slice] v1 Scheduler.schedule patch needles not found; fallback to original schedule.")
        return original_schedule

    patched_globals = dict(getattr(original_schedule, "__globals__", {}))
    patched_globals["__wave_slice_compute_v1_waiting_num_new_tokens"] = functools.partial(
        _compute_v1_waiting_num_new_tokens,
        state,
    )
    patched_globals["__wave_slice_compute_v1_running_num_new_tokens"] = functools.partial(
        _compute_v1_running_num_new_tokens,
        state,
    )
    namespace: dict[str, Any] = {}
    try:
        exec(patched_source, patched_globals, namespace)
        patched_schedule = namespace.get(getattr(original_schedule, "__name__", "schedule"))
    except Exception:
        logger.exception("[Wave-Slice] failed to compile patched v1 Scheduler.schedule.")
        return original_schedule

    if not callable(patched_schedule):
        logger.warning("[Wave-Slice] compiled v1 Scheduler.schedule patch missing callable; fallback to original schedule.")
        return original_schedule

    patched_schedule = functools.wraps(original_schedule)(patched_schedule)
    patched_schedule.__wave_slice_v1_waiting_patch__ = True  # type: ignore[attr-defined]
    return patched_schedule
