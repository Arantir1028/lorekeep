"""Wave-Slice runtime hijacker for vLLM.

This module implements:
- Phase I: scheduler-side chunk-size control that returns to the
  scheduling boundary earlier for long prefills.
- Phase II: execution escape / priority promotion that reshapes the next
  scheduled window without relying on the abandoned true-unbind path.
- Runtime metrics hooks: TTFT / P99 / slowdown accounting.

All changes are applied via monkey patching and can be fully reverted.
"""

from __future__ import annotations

import contextlib
import functools
import json
import logging
import math
import os
import sys
import threading
import time
from typing import Any, Callable, Iterable, Optional

from scheduler.wave_scheduler import WaveScheduler
from engine.hijack.autoinject import (
    AUTO_ENV_ENABLED as _AUTO_ENV_ENABLED,
    AUTO_ENV_GAMMA as _AUTO_ENV_GAMMA,
    AUTO_ENV_MODEL as _AUTO_ENV_MODEL,
    AUTO_ENV_POLICY as _AUTO_ENV_POLICY,
    clear_autoinject_env as _clear_autoinject_env,
    merge_cross_process_metrics as _merge_cross_process_metrics,
    publish_autoinject_env as _publish_autoinject_env,
    reset_cross_process_metrics_file as _reset_cross_process_metrics_file,
)
from engine.hijack.engine_hooks import (
    build_add_processed_request_hook as _build_add_processed_request_hook_impl,
    build_add_request_hook as _build_add_request_hook_impl,
    build_step_hook as _build_step_hook_impl,
    build_v1_engine_core_add_request_hook as _build_v1_engine_core_add_request_hook_impl,
    build_v1_process_inputs_hook as _build_v1_process_inputs_hook_impl,
)
from engine.hijack.common import (
    collect_live_lengths as _collect_live_lengths,
    collect_live_snapshot as _collect_live_snapshot,
    compute_long_prefill_threshold as _compute_long_prefill_threshold,
    estimate_prompt_tokens as _estimate_prompt_tokens,
    estimate_solo_us as _estimate_solo_us,
    infer_lora_rank as _infer_lora_rank,
    is_phase2_strict as _is_phase2_strict,
    phase12_expected_chunk_tokens as _phase12_expected_chunk_tokens,
    queue_reorder_key as _queue_reorder_key,
    rebuild_queue_like as _rebuild_queue_like,
    reorder_queue as _reorder_queue,
    safe_first_seq as _safe_first_seq,
    safe_lora_path as _safe_lora_path,
    safe_prefill_uncomputed_tokens as _safe_prefill_uncomputed_tokens,
    safe_remaining_tokens as _safe_remaining_tokens,
    safe_request_id as _safe_request_id,
    safe_total_tokens as _safe_total_tokens,
    safe_wait_us as _safe_wait_us,
)
from engine.hijack.v1_lifecycle import (
    build_output_processor_add_request_hook as _build_output_processor_add_request_hook_impl,
    build_output_processor_process_outputs_hook as _build_output_processor_process_outputs_hook_impl,
    build_v1_scheduler_finish_requests_hook as _build_v1_scheduler_finish_requests_hook_impl,
    build_v1_scheduler_update_from_output_hook as _build_v1_scheduler_update_from_output_hook_impl,
    maybe_install_v1_runtime_lifecycle_hooks as _maybe_install_v1_runtime_lifecycle_hooks_impl,
)
from engine.hijack.phase1_math import (
    compute_budget as _compute_budget,
    compute_explicit_plan_budget as _compute_explicit_plan_budget,
    maybe_force_phase1_chunk as _maybe_force_phase1_chunk,
    need_wave_slice as _need_wave_slice,
    phase1_adjusted_queue_len as _phase1_adjusted_queue_len,
    phase1_authoritative_chunk as _phase1_authoritative_chunk,
    phase1_authoritative_short_floor as _phase1_authoritative_short_floor,
    phase1_baseline_chunk_proxy as _phase1_baseline_chunk_proxy,
    phase1_cohort_target_len as _phase1_cohort_target_len,
    phase1_effective_ingress_min_chunk as _phase1_effective_ingress_min_chunk,
    phase1_effective_short_token_mass as _phase1_effective_short_token_mass,
    phase1_runtime_adapt_policy as _phase1_runtime_adapt_policy,
    phase1_runtime_pressure_meta as _phase1_runtime_pressure_meta,
)
from engine.hijack.phase1_stateful import (
    phase1_apply_sticky_chunk as _phase1_apply_sticky_chunk,
    phase1_build_direct_explicit_plans as _phase1_build_direct_explicit_plans,
    phase1_direct_chunk_candidate as _phase1_direct_chunk_candidate,
    phase1_explicit_chunk_from_plan as _phase1_explicit_chunk_from_plan,
    phase1_maybe_seed_ingress_virtual as _phase1_maybe_seed_ingress_virtual,
    phase1_update_sticky_chunk as _phase1_update_sticky_chunk,
)
from engine.hijack.phase12_beneficiary import (
    phase12_beneficiary_signal as _phase12_beneficiary_signal,
    phase12_collect_prefill_lora_state as _phase12_collect_prefill_lora_state,
    phase12_collect_scheduled_req_infos as _phase12_collect_scheduled_req_infos,
    phase12_collect_snapshot_req_infos as _phase12_collect_snapshot_req_infos,
)
from engine.hijack.phase12_cashout import (
    phase12_scheduler_cashout_cooldown_for_grade as _phase12_scheduler_cashout_cooldown_for_grade,
    phase12_scheduler_cashout_grade as _phase12_scheduler_cashout_grade,
    phase12_scheduler_cashout_value_signal as _phase12_scheduler_cashout_value_signal,
)
from engine.hijack.phase12_priority import (
    phase12_priority_bubble_waiting_queue as _phase12_priority_bubble_waiting_queue,
)
from engine.hijack.public_metadata import (
    build_public_seq_group_metadata as _build_public_seq_group_metadata,
)
from engine.hijack.queue_ops import (
    capture_queue_pair as _capture_queue_pair,
    restore_hidden_queue_items as _restore_hidden_queue_items,
    restore_queue_pair as _restore_queue_pair,
)
from engine.hijack.hook_guards import (
    has_running_waiting_queues as _has_running_waiting_queues,
    phase2_modelrunner_passthrough as _phase2_modelrunner_passthrough,
    phase2_scheduler_cashout_enabled as _phase2_scheduler_cashout_enabled,
)
from engine.hijack.v1_split import (
    v1_execution_escape_req_ids as _v1_execution_escape_req_ids,
)
from engine.hijack.v1_waiting_cap import (
    _build_get_num_new_uncached_and_cached_tokens_hook,
    _build_sequence_data_get_len_hook,
    _build_v1_request_num_tokens_hook,
    _build_v1_request_num_tokens_with_spec_hook,
    _build_v1_schedule_waiting_patch,
    _build_v1_scheduler_add_request_hook,
    _build_v1_scheduler_update_after_schedule_hook,
    _compute_v1_running_num_new_tokens,
    _compute_v1_waiting_num_new_tokens,
    _install_v1_waiting_cap_runtime_hooks,
    _lookup_engine_prompt_tokens,
    _restore_v1_waiting_cap_runtime_hooks,
)
from engine.hijack.phase1_selection import (
    phase1_basic_cohort as _phase1_basic_cohort,
    phase1_build_cohort as _phase1_build_cohort,
    phase1_find_ingress_virtual_candidate as _phase1_find_ingress_virtual_candidate,
    phase1_find_seq_group_by_request_id as _phase1_find_seq_group_by_request_id,
    phase1_live_cohort_from_snapshot as _phase1_live_cohort_from_snapshot,
    phase1_prune_explicit_plans as _phase1_prune_explicit_plans,
)
from engine.hijack.phase2_gates import (
    phase2_has_lora_heterogeneity as _phase2_has_lora_heterogeneity,
    phase2_mixed_escape_ok as _phase2_mixed_escape_ok,
    phase2_pressure_ratio as _phase2_pressure_ratio,
    phase2_rank_ratio as _phase2_rank_ratio,
    phase2_selective_gate as _phase2_selective_gate,
)
from engine.hijack.runtime_loaders import (
    load_v1_engine_core_cls as _load_v1_engine_core_cls,
    load_llm_engine_cls as _load_llm_engine_cls,
    load_logits_processor_lora_cls as _load_logits_processor_lora_cls,
    load_model_runner_cls as _load_model_runner_cls,
    load_scheduler_target as _load_scheduler_target,
    load_v1_processor_cls as _load_v1_processor_cls,
    load_sequence_data_cls as _load_sequence_data_cls,
    load_v1_request_cls as _load_v1_request_cls,
    load_v1_output_processor_cls as _load_v1_output_processor_cls,
)
from engine.hijack.runtime_state import WaveSliceMetrics, WaveSlicePolicy
from engine.hijack.types import (
    _PatchState,
    _Phase12BeneficiarySignal,
    _Phase1CohortStats,
    _Phase1IngressVirtualSlice,
    _ScheduledReqInfo,
)

logger = logging.getLogger("WaveSlice")
logger.addHandler(logging.NullHandler())


_PATCH_LOCK = threading.RLock()
_PATCH_STATE: Optional[_PatchState] = None


def _phase1_prune_ingress_virtual_caps(state: _PatchState) -> dict[str, int]:
    active_request_ids = {
        str(req_id)
        for req_id, prompt_tokens in getattr(state, "phase1_active_prompt_tokens", {}).items()
        if str(req_id) and int(prompt_tokens or 0) > 0
    }
    ingress_request_ids = {
        str(req_id)
        for req_id in getattr(state, "phase1_ingress_virtuals", {}).keys()
        if str(req_id)
    }
    keep_ids = active_request_ids | ingress_request_ids
    return {
        str(req_id): int(chunk)
        for req_id, chunk in state.phase1_virtual_token_caps.items()
        if str(req_id) in keep_ids and int(chunk) > 0
    }


def _phase1_rewrite_scheduler_outputs(
    *,
    outputs: Any,
    request_id: Optional[str],
    target_chunk: int,
) -> tuple[Any, bool, int, int, int, int]:
    if outputs is None or not request_id or target_chunk <= 0:
        return outputs, False, 0, 0, 0, 0
    scheduled = getattr(outputs, "scheduled_seq_groups", None)
    if not isinstance(scheduled, list):
        return outputs, False, 0, 0, 0, 0

    rewritten = False
    total_delta = 0
    rewritten_groups = 0
    old_chunk_sum = 0
    new_chunk_sum = 0
    for group in scheduled:
        seq_group = getattr(group, "seq_group", None)
        if seq_group is None:
            continue
        if _safe_request_id(seq_group) != request_id:
            continue
        try:
            is_prefill = bool(seq_group.is_prefill())
        except Exception:
            is_prefill = False
        if not is_prefill:
            continue
        old_chunk = int(getattr(group, "token_chunk_size", 0) or 0)
        if old_chunk <= 0:
            continue
        new_chunk = max(1, min(int(target_chunk), old_chunk))
        if new_chunk >= old_chunk:
            continue
        try:
            group.token_chunk_size = new_chunk
            total_delta += old_chunk - new_chunk
            old_chunk_sum += old_chunk
            new_chunk_sum += new_chunk
            rewritten_groups += 1
            rewritten = True
        except Exception:
            continue

    if rewritten and total_delta > 0:
        try:
            outputs.num_batched_tokens = max(
                0, int(getattr(outputs, "num_batched_tokens", 0)) - int(total_delta)
            )
        except Exception:
            pass
    return outputs, rewritten, rewritten_groups, old_chunk_sum, new_chunk_sum, total_delta


def _phase1_rewrite_metadata_list(
    *,
    seq_group_metadata_list: Any,
    request_id: Optional[str],
    target_chunk: int,
) -> tuple[Any, bool]:
    if not isinstance(seq_group_metadata_list, list) or not request_id or target_chunk <= 0:
        return seq_group_metadata_list, False
    rewritten = False
    for meta in seq_group_metadata_list:
        try:
            if str(getattr(meta, "request_id", "")) != request_id:
                continue
            old_chunk = int(getattr(meta, "token_chunk_size", 0) or 0)
            if old_chunk <= 0:
                continue
            new_chunk = max(1, min(int(target_chunk), old_chunk))
            if new_chunk >= old_chunk:
                continue
            meta.token_chunk_size = new_chunk
            try:
                is_prompt = bool(getattr(meta, "is_prompt", False))
            except Exception:
                is_prompt = False
            if is_prompt:
                try:
                    setattr(meta, "do_sample", False)
                except Exception:
                    pass
            rewritten = True
        except Exception:
            continue
    return seq_group_metadata_list, rewritten


def _phase1_force_public_schedule_rewrite(
    *,
    state: _PatchState,
    scheduler_obj: Any,
    scheduler_outputs: Any,
) -> tuple[Any, bool]:
    if bool(state.policy.phase1_ingress_direct_authoritative):
        return scheduler_outputs, False
    scheduled = getattr(scheduler_outputs, "scheduled_seq_groups", None)
    if not isinstance(scheduled, list) or not scheduled:
        return scheduler_outputs, False
    if state.phase1_public_skip_rewrite_requests:
        for group in scheduled:
            seq_group = getattr(group, "seq_group", None)
            req_id = _safe_request_id(seq_group) if seq_group is not None else None
            if req_id and str(req_id) in state.phase1_public_skip_rewrite_requests:
                return scheduler_outputs, False

    waiting = list(getattr(scheduler_obj, "waiting", []) or [])
    running = list(getattr(scheduler_obj, "running", []) or [])
    snapshot, _ = _collect_live_snapshot(waiting, running)
    if len(snapshot) <= 1:
        snapshot = []
        for group in scheduled:
            seq_group = getattr(group, "seq_group", None)
            if seq_group is None:
                continue
            try:
                is_prefill = bool(seq_group.is_prefill())
            except Exception:
                is_prefill = False
            if not is_prefill:
                continue
            remaining = _safe_remaining_tokens(seq_group) or 0
            if remaining > 1:
                snapshot.append((seq_group, int(remaining)))

    lengths = [rem for _, rem in snapshot]
    if not _need_wave_slice(lengths, state.policy):
        return scheduler_outputs, False

    if state.policy.phase1_enable_cohort_mode:
        cohort = _phase1_build_cohort(
            snapshot,
            state.policy,
            request_id_getter=_safe_request_id,
        )
    else:
        cohort = _phase1_basic_cohort(snapshot, request_id_getter=_safe_request_id)
    if cohort is None or not cohort.long_req_id:
        return scheduler_outputs, False

    long_seq_group = _phase1_find_seq_group_by_request_id(
        snapshot,
        cohort.long_req_id,
        request_id_getter=_safe_request_id,
    )
    if long_seq_group is None:
        return scheduler_outputs, False

    total_len = _safe_total_tokens(long_seq_group)
    remaining_len = _safe_remaining_tokens(long_seq_group)
    if total_len is None or remaining_len is None or remaining_len <= 1:
        return scheduler_outputs, False

    baseline_chunk = None
    long_present_in_scheduled = False
    for group in scheduled:
        seq_group = getattr(group, "seq_group", None)
        if seq_group is None or _safe_request_id(seq_group) != cohort.long_req_id:
            continue
        long_present_in_scheduled = True
        old_chunk = int(getattr(group, "token_chunk_size", 0) or 0)
        if old_chunk > 0:
            baseline_chunk = old_chunk
            break
    if not long_present_in_scheduled:
        return scheduler_outputs, False

    queue_length = max(
        1,
        len(getattr(scheduler_obj, "running", []) or [])
        + len(getattr(scheduler_obj, "waiting", []) or []),
    )
    best_chunk = state.slicer.choose_dynamic_chunk(
        short_len=int(cohort.representative_short_len),
        long_len=int(remaining_len),
        scheduler=state.brain,
        t_wait_us=0.0,
        queue_length=queue_length,
        baseline_chunk=baseline_chunk,
    )
    direct_plans = _phase1_build_direct_explicit_plans(
        state=state,
        cohort=cohort,
        total_len=int(total_len),
        done_offset=max(0, int(total_len) - int(remaining_len)),
        remaining_len=int(remaining_len),
        baseline_chunk=baseline_chunk,
    )
    explicit_plan = False
    if direct_plans:
        if bool(state.policy.phase1_ingress_direct_authoritative) and cohort.long_req_id in state.phase1_ingress_virtuals:
            best_chunk = int(direct_plans[0].chunk_len)
        else:
            best_chunk = min(best_chunk, int(direct_plans[0].chunk_len))
        explicit_plan = True

    upper = int(remaining_len) - 1
    if baseline_chunk is not None and int(baseline_chunk) > 0:
        upper = min(upper, int(baseline_chunk) - 1)
    if upper <= int(cohort.representative_short_len):
        return scheduler_outputs, False
    best_chunk = max(int(cohort.representative_short_len), min(int(best_chunk), upper))
    if baseline_chunk is not None and best_chunk >= int(baseline_chunk):
        return scheduler_outputs, False

    state.metrics.record_phase1_choice(
        chosen_chunk=best_chunk,
        baseline_chunk=baseline_chunk,
        explicit_plan=explicit_plan,
    )
    scheduler_outputs, rewritten, rewritten_groups, old_chunk_sum, new_chunk_sum, token_delta_sum = _phase1_rewrite_scheduler_outputs(
        outputs=scheduler_outputs,
        request_id=cohort.long_req_id,
        target_chunk=best_chunk,
    )
    state.metrics.record_phase1_rewrite(
        rewritten_groups=rewritten_groups,
        old_chunk_sum=old_chunk_sum,
        new_chunk_sum=new_chunk_sum,
        token_delta_sum=token_delta_sum,
    )
    if rewritten:
        WaveSliceMetrics._emit_cross_process_event(
            "phase1_public_rewrite_applied",
            {
                "request_id": str(cohort.long_req_id),
                "best_chunk": int(best_chunk),
                "rewritten_groups": int(rewritten_groups),
            },
        )
        logger.info(
            "[Wave-Slice][P1-public-force] req=%s baseline_chunk=%s chunk=%d explicit_plan=%s groups=%d",
            str(cohort.long_req_id),
            str(baseline_chunk) if baseline_chunk is not None else "none",
            int(best_chunk),
            str(explicit_plan),
            int(rewritten_groups),
        )
    return scheduler_outputs, rewritten


def _phase1_tick_hide_keep_ceiling(
    *,
    best_chunk: int,
    cohort: _Phase1CohortStats,
) -> int:
    short_candidates = [
        int(v)
        for v in list(getattr(cohort, "short_lengths", []) or [])
        if int(v) > 0
    ]
    short_candidates.append(max(1, int(getattr(cohort, "representative_short_len", 1) or 1)))
    return max(max(1, int(best_chunk)), max(short_candidates))


def _phase12_should_force_lora_tick_hide(
    *,
    state: _PatchState,
    lora_mode_enabled: bool,
    waiting_short_count: int,
) -> bool:
    if waiting_short_count <= 0:
        return False
    if not lora_mode_enabled:
        return False
    if not bool(getattr(state.policy, "phase12_joint_coordination", False)):
        return False
    if not bool(getattr(state.policy, "enable_phase2_modelrunner", False)):
        return False
    return True


def _phase1_lora_cohort_key(seq_group: Any) -> Optional[str]:
    lora_request = getattr(seq_group, "lora_request", None)
    if lora_request is None:
        seq = _safe_first_seq(seq_group)
        lora_request = getattr(seq, "lora_request", None) if seq is not None else None
    path = _safe_lora_path(lora_request)
    if path:
        return f"path:{path}"
    rank = int(_infer_lora_rank(lora_request) or 0)
    if rank > 0:
        return f"rank:{rank}"
    return None


def _phase1_filter_snapshot_for_lora_cohort(
    snapshot: list[tuple[Any, int]],
    *,
    preferred_request_id: Optional[str] = None,
) -> list[tuple[Any, int]]:
    if len(snapshot) < 2:
        return snapshot

    long_pair: Optional[tuple[Any, int]] = None
    if preferred_request_id:
        for seq_group, rem in snapshot:
            if _safe_request_id(seq_group) == preferred_request_id:
                long_pair = (seq_group, int(rem))
                break
    if long_pair is None:
        try:
            long_pair = max(snapshot, key=lambda item: int(item[1]))
        except Exception:
            return snapshot
    if long_pair is None:
        return snapshot

    long_key = _phase1_lora_cohort_key(long_pair[0])
    if not long_key:
        return snapshot

    filtered = [
        (seq_group, int(rem))
        for seq_group, rem in snapshot
        if _phase1_lora_cohort_key(seq_group) == long_key
    ]
    if len(filtered) < 2:
        return snapshot
    return filtered


def _phase1_waiting_short_count(
    waiting: Iterable[Any],
    *,
    short_threshold_tokens: int,
) -> int:
    count = 0
    threshold = max(1, int(short_threshold_tokens))
    for seq_group in list(waiting):
        remaining = _safe_prefill_uncomputed_tokens(seq_group) or 0
        if 0 < int(remaining) <= threshold:
            count += 1
    return count


def _phase1_build_ingress_fallback_cohort(
    state: _PatchState,
    snapshot: list[tuple[Any, int]],
) -> Optional[_Phase1CohortStats]:
    if not snapshot or not getattr(state, "phase1_ingress_virtuals", None):
        return None
    try:
        long_seq_group, remaining = max(snapshot, key=lambda item: int(item[1]))
    except Exception:
        return None
    request_id = _safe_request_id(long_seq_group)
    if not request_id:
        return None
    ingress_virtual = getattr(state, "phase1_ingress_virtuals", {}).get(str(request_id))
    if ingress_virtual is None:
        return None
    short_lengths = [
        max(1, int(v))
        for v in list(getattr(ingress_virtual, "short_lengths", []) or [])
        if int(v) > 0
    ]
    representative_short_len = max(
        1,
        int(getattr(ingress_virtual, "representative_short_len", 1) or 1),
    )
    if not short_lengths:
        short_lengths = [representative_short_len]
    original_long_len = max(
        1,
        int(getattr(ingress_virtual, "original_long_len", remaining) or remaining),
    )
    if not _need_wave_slice(short_lengths + [original_long_len], state.policy):
        return None
    return _Phase1CohortStats(
        representative_short_len=representative_short_len,
        short_count=max(1, int(getattr(ingress_virtual, "short_count", len(short_lengths)) or len(short_lengths))),
        short_token_mass=max(
            representative_short_len,
            int(getattr(ingress_virtual, "short_token_mass", sum(short_lengths)) or sum(short_lengths)),
        ),
        short_lengths=short_lengths,
        long_len=max(1, int(remaining)),
        long_req_id=str(request_id),
        total_count=max(
            2,
            len(snapshot),
            int(getattr(ingress_virtual, "active_count", len(short_lengths) + 1) or (len(short_lengths) + 1)),
        ),
    )


def _phase1_build_global_activity_cohort(
    state: _PatchState,
    snapshot: list[tuple[Any, int]],
) -> Optional[_Phase1CohortStats]:
    if not snapshot:
        return None

    live_remaining: dict[str, int] = {}
    for seq_group, rem in snapshot:
        req_id = _safe_request_id(seq_group)
        if not req_id:
            continue
        rem_i = int(rem)
        if rem_i <= 0:
            continue
        existing = live_remaining.get(str(req_id))
        if existing is None or rem_i > existing:
            live_remaining[str(req_id)] = rem_i
    if not live_remaining:
        return None

    active_prompt_tokens = {
        str(req_id): int(prompt_tokens)
        for req_id, prompt_tokens in getattr(state, "phase1_active_prompt_tokens", {}).items()
        if str(req_id) and int(prompt_tokens or 0) > 0
    }
    ingress_virtuals = dict(getattr(state, "phase1_ingress_virtuals", {}) or {})
    if not active_prompt_tokens and not ingress_virtuals:
        return None

    best: Optional[_Phase1CohortStats] = None
    best_score: Optional[tuple[int, int, int]] = None
    for request_id, remaining in live_remaining.items():
        ingress_virtual = ingress_virtuals.get(str(request_id))
        original_long_len = max(
            int(remaining),
            int(
                getattr(ingress_virtual, "original_long_len", active_prompt_tokens.get(str(request_id), remaining))
                or active_prompt_tokens.get(str(request_id), remaining)
                or remaining
            ),
        )
        short_lengths = [
            max(1, int(v))
            for v in list(getattr(ingress_virtual, "short_lengths", []) or [])
            if int(v) > 0
        ]
        if not short_lengths:
            short_lengths = sorted(
                max(1, int(tok))
                for rid, tok in active_prompt_tokens.items()
                if str(rid) != str(request_id) and int(tok) > 0
            )
        if not short_lengths:
            continue
        if not _need_wave_slice(short_lengths + [original_long_len], state.policy):
            continue

        if bool(getattr(state.policy, "phase1_enable_cohort_mode", False)):
            cohort_cap = max(
                1,
                int(original_long_len * float(getattr(state.policy, "phase1_short_cohort_long_fraction", 0.35))),
            )
            cohort_short_lengths = [int(v) for v in short_lengths if int(v) <= cohort_cap]
            if len(cohort_short_lengths) < int(getattr(state.policy, "phase1_cohort_min_count", 2) or 2):
                cohort_short_lengths = list(short_lengths)
            if not cohort_short_lengths:
                cohort_short_lengths = [int(min(short_lengths))]
            representative_short_len = max(
                1,
                int(round(sum(cohort_short_lengths) / max(1, len(cohort_short_lengths)))),
            )
        else:
            cohort_short_lengths = [int(min(short_lengths))]
            representative_short_len = int(cohort_short_lengths[0])

        candidate = _Phase1CohortStats(
            representative_short_len=representative_short_len,
            short_count=max(
                1,
                int(getattr(ingress_virtual, "short_count", len(cohort_short_lengths)) or len(cohort_short_lengths)),
            ),
            short_token_mass=max(
                representative_short_len,
                int(getattr(ingress_virtual, "short_token_mass", sum(cohort_short_lengths)) or sum(cohort_short_lengths)),
            ),
            short_lengths=[int(v) for v in cohort_short_lengths],
            long_len=max(1, int(remaining)),
            long_req_id=str(request_id),
            total_count=max(
                len(active_prompt_tokens),
                len(cohort_short_lengths) + 1,
                int(
                    getattr(
                        ingress_virtual,
                        "active_count",
                        len(active_prompt_tokens) or (len(cohort_short_lengths) + 1),
                    )
                    or (len(cohort_short_lengths) + 1)
                ),
            ),
        )
        score = (
            int(candidate.long_len),
            int(original_long_len),
            int(candidate.total_count),
        )
        if best_score is None or score > best_score:
            best = candidate
            best_score = score
    return best


def _phase1_group_snapshot_by_lora_cohort(
    snapshot: list[tuple[Any, int]],
) -> dict[str, list[tuple[Any, int]]]:
    grouped: dict[str, list[tuple[Any, int]]] = {}
    for seq_group, rem in snapshot:
        key = _phase1_lora_cohort_key(seq_group)
        if not key:
            continue
        grouped.setdefault(key, []).append((seq_group, int(rem)))
    return grouped


def _phase1_candidate_chunk_for_snapshot(
    *,
    state: _PatchState,
    snapshot: list[tuple[Any, int]],
    max_wait_us: float,
    queue_len: int,
    scheduler_cfg: Any,
    original_budget: Any,
    original_threshold: Any,
) -> Optional[tuple[_Phase1CohortStats, int]]:
    lengths = [int(rem) for _, rem in snapshot]
    if not _need_wave_slice(lengths, state.policy):
        return None

    cohort = _phase1_live_cohort_from_snapshot(
        snapshot,
        state.policy,
        request_id_getter=_safe_request_id,
    )
    if cohort is None or not cohort.long_req_id:
        return None

    long_seq_group = _phase1_find_seq_group_by_request_id(
        snapshot,
        cohort.long_req_id,
        request_id_getter=_safe_request_id,
    )
    if long_seq_group is None:
        return None

    short_len = int(cohort.representative_short_len)
    long_len = int(cohort.long_len)
    if long_len <= max(1, short_len):
        return None

    adjusted_queue_len = _phase1_adjusted_queue_len(cohort, queue_len, state.policy)
    baseline_chunk = _phase1_baseline_chunk_proxy(
        long_len=long_len,
        original_budget=original_budget,
        original_threshold=original_threshold,
        scheduler_cfg=scheduler_cfg,
        policy=state.policy,
    )
    best_chunk = state.slicer.choose_dynamic_chunk(
        short_len=short_len,
        long_len=long_len,
        scheduler=state.brain,
        t_wait_us=max_wait_us,
        queue_length=adjusted_queue_len,
        baseline_chunk=baseline_chunk,
    )
    explicit_chunk = _phase1_explicit_chunk_from_plan(
        state=state,
        cohort=cohort,
        snapshot=snapshot,
        t_wait_us=max_wait_us,
        queue_length=adjusted_queue_len,
        baseline_chunk=baseline_chunk,
        total_tokens_getter=_safe_total_tokens,
        request_id_getter=_safe_request_id,
    )
    if explicit_chunk is not None:
        best_chunk = int(explicit_chunk[0])

    if bool(state.policy.phase1_ingress_direct_authoritative):
        total_len = max(1, int(_safe_total_tokens(long_seq_group) or long_len))
        done_offset = max(0, int(total_len) - int(long_len))
        direct_cap_chunk = _phase1_direct_chunk_candidate(
            state=state,
            cohort=cohort,
            total_len=int(total_len),
            done_offset=int(done_offset),
            remaining_len=int(long_len),
            baseline_chunk=baseline_chunk,
        )
        if direct_cap_chunk is not None:
            best_chunk = min(int(best_chunk), int(direct_cap_chunk))

    if _phase1_lora_cohort_key(long_seq_group):
        # Under LoRA, the global ingress floor can be too coarse for the
        # per-adapter short/long pair. Clamp against the cohort target itself
        # so the short request can actually detach from its paired long prefill.
        lora_cohort_target = max(
            int(short_len),
            min(int(long_len) - 1, int(_phase1_cohort_target_len(cohort, state.policy))),
        )
        best_chunk = min(int(best_chunk), int(lora_cohort_target))

    best_chunk = max(int(short_len), min(int(best_chunk), int(long_len) - 1))
    if best_chunk >= int(long_len):
        return None
    return cohort, int(best_chunk)


def _phase1_collect_secondary_lora_caps(
    *,
    state: _PatchState,
    snapshot: list[tuple[Any, int]],
    primary_request_id: Optional[str],
    max_wait_us: float,
    queue_len: int,
    scheduler_cfg: Any,
    original_budget: Any,
    original_threshold: Any,
) -> dict[str, int]:
    if not snapshot:
        return {}

    caps: dict[str, int] = {}
    grouped = _phase1_group_snapshot_by_lora_cohort(snapshot)
    for cohort_snapshot in grouped.values():
        candidate = _phase1_candidate_chunk_for_snapshot(
            state=state,
            snapshot=cohort_snapshot,
            max_wait_us=max_wait_us,
            queue_len=queue_len,
            scheduler_cfg=scheduler_cfg,
            original_budget=original_budget,
            original_threshold=original_threshold,
        )
        if candidate is None:
            continue
        cohort, best_chunk = candidate
        req_id = str(cohort.long_req_id or "")
        if not req_id or req_id == str(primary_request_id or ""):
            continue
        caps[req_id] = min(int(best_chunk), int(caps.get(req_id, best_chunk)))
    return caps


def _phase1_apply_sequence_len_shadow(
    *,
    state: _PatchState,
    seq_group: Any,
    target_chunk: int,
) -> bool:
    if target_chunk <= 0 or seq_group is None:
        return False
    shadowed = False
    try:
        seqs = list(seq_group.get_seqs())
    except Exception:
        seqs = []
    for seq in seqs:
        try:
            if bool(seq.is_finished()):
                continue
        except Exception:
            pass
        data = getattr(seq, "data", None)
        if data is None:
            continue
        try:
            computed = int(data.get_num_computed_tokens())
            total = int(data.get_len())
        except Exception:
            continue
        shadow_len = max(computed + 1, min(total, computed + int(target_chunk)))
        state.phase1_shadow_seq_lens[id(data)] = int(shadow_len)
        shadowed = True
    return shadowed


def _phase1_rewrite_schedule_tuple(
    *,
    state: _PatchState,
    scheduler_obj: Any,
    seq_group_metadata_list: list[Any],
    scheduler_outputs: Any,
) -> tuple[list[Any], Any, bool]:
    del state, scheduler_obj
    scheduled = getattr(scheduler_outputs, "scheduled_seq_groups", None)
    if not isinstance(scheduled, list) or not isinstance(seq_group_metadata_list, list):
        return seq_group_metadata_list, scheduler_outputs, False

    chunk_by_request: dict[str, int] = {}
    for group in scheduled:
        seq_group = getattr(group, "seq_group", None)
        if seq_group is None:
            continue
        request_id = _safe_request_id(seq_group)
        if not request_id:
            continue
        chunk = int(getattr(group, "token_chunk_size", 0) or 0)
        if chunk <= 0:
            continue
        prev = chunk_by_request.get(request_id)
        if prev is None or chunk < prev:
            chunk_by_request[request_id] = chunk

    rewritten = False
    rewrite_count = 0
    for meta in seq_group_metadata_list:
        try:
            request_id = str(getattr(meta, "request_id", "") or "")
            if not request_id:
                continue
            target_chunk = chunk_by_request.get(request_id)
            if target_chunk is None or target_chunk <= 0:
                continue
            old_chunk = int(getattr(meta, "token_chunk_size", 0) or 0)
            if old_chunk <= 0 or target_chunk >= old_chunk:
                continue
            meta.token_chunk_size = int(target_chunk)
            try:
                if bool(getattr(meta, "is_prompt", False)):
                    setattr(meta, "do_sample", False)
            except Exception:
                pass
            rewritten = True
            rewrite_count += 1
        except Exception:
            continue

    if rewritten:
        logger.info(
            "[Wave-Slice][P1-public-schedule] aligned_metadata=%d",
            int(rewrite_count),
        )
    return seq_group_metadata_list, scheduler_outputs, rewritten


def _extract_phase2_lora_ranks(
    model_input: Any,
    runner_self: Optional[Any] = None,
) -> list[int]:
    ranks: list[int] = []

    if _is_v1_scheduler_output(model_input):
        try:
            num_sched = dict(getattr(model_input, "num_scheduled_tokens", {}) or {})
        except Exception:
            num_sched = {}
        req_states = getattr(runner_self, "requests", {}) if runner_self is not None else {}
        for rid_raw, tok_raw in num_sched.items():
            try:
                if int(tok_raw) <= 0:
                    continue
            except Exception:
                continue
            rid = str(rid_raw)
            st = req_states.get(rid) if isinstance(req_states, dict) else None
            rank = _infer_lora_rank(getattr(st, "lora_request", None) if st is not None else None)
            if rank > 0:
                ranks.append(rank)
        return sorted(ranks)

    try:
        reqs = list(getattr(model_input, "lora_requests", None) or [])
    except Exception:
        reqs = []
    for req in reqs:
        rank = _infer_lora_rank(req)
        if rank > 0:
            ranks.append(rank)
    return sorted(ranks)


def _phase12_joint_phase1_floor(
    *,
    state: _PatchState,
    snapshot: Any,
    policy: WaveSlicePolicy,
) -> Optional[int]:
    if not (policy.enable_phase1_scheduler and policy.enable_phase2_modelrunner):
        return None
    if not bool(policy.phase12_joint_coordination):
        return None
    if not (
        bool(policy.phase2_enable_execution_escape)
        or bool(policy.phase2_enable_scheduler_cashout)
    ):
        return None
    if hasattr(snapshot, "running") and hasattr(snapshot, "waiting"):
        seq_groups = list(snapshot.running) + list(snapshot.waiting)
    else:
        seq_groups = [seq_group for seq_group, _ in list(snapshot or [])]
    prefill_lens, lora_ranks = _phase12_collect_prefill_lora_state(
        seq_groups,
        rank_infer=_infer_lora_rank,
        remaining_getter=_safe_prefill_uncomputed_tokens,
    )
    if len(prefill_lens) < max(2, int(policy.phase2_min_prefill_count)):
        return None
    selective_ok, _ratio, pressure_ratio, lora_rank_hetero = _phase2_selective_gate(
        prefill_lens=prefill_lens,
        lora_ranks=lora_ranks,
        policy=policy,
        strict_mode=False,
    )
    if not selective_ok:
        return None
    if not lora_rank_hetero and pressure_ratio < float(policy.phase2_min_pressure_ratio):
        return None
    return max(1, int(policy.phase12_joint_min_chunk))


def _phase12_joint_phase2_ready(
    *,
    state: _PatchState,
    policy: WaveSlicePolicy,
    prefill_lens: list[int],
    num_decode_tokens: int,
    lora_ranks: list[int],
    req_infos: Optional[list[_ScheduledReqInfo]],
    strict_mode: bool,
) -> tuple[bool, str]:
    if not (policy.enable_phase1_scheduler and policy.enable_phase2_modelrunner):
        return True, "joint_disabled"
    if not bool(policy.phase12_joint_coordination):
        return True, "joint_coordination_off"
    cap_live = bool(state.phase1_virtual_token_caps)
    recent_ttl = int(getattr(state, "phase12_recent_phase1_apply_ttl", 0) or 0)
    requires_recent_phase1 = bool(policy.phase12_phase2_requires_recent_phase1)

    gate_mode = str(getattr(policy, "phase12_phase2_gate_mode", "hard") or "hard").strip().lower()
    if gate_mode == "hard":
        if cap_live:
            return True, "joint_recent_phase1_cap_live"
        if recent_ttl > 0:
            return True, "joint_recent_phase1_ttl"
        if not requires_recent_phase1:
            return True, "joint_recent_phase1_not_required"
        return False, "joint_waiting_for_phase1"

    pos_prefills = [int(v) for v in prefill_lens if int(v) > 0]
    if not pos_prefills:
        return False, "joint_soft_no_prefill"

    selective_ok, ratio, pressure_ratio, lora_rank_hetero = _phase2_selective_gate(
        prefill_lens=pos_prefills,
        lora_ranks=lora_ranks,
        policy=policy,
        strict_mode=strict_mode,
    )
    max_len = max(pos_prefills)
    soft_ratio = max(
        float(policy.phase2_min_hetero_ratio) * max(1.0, float(policy.phase12_phase2_soft_ratio_scale)),
        2.5 if not strict_mode else 3.0,
    )
    soft_pressure = max(
        float(policy.phase2_min_pressure_ratio) * max(1.0, float(policy.phase12_phase2_soft_pressure_scale)),
        2.0 if not strict_mode else 2.5,
    )
    soft_long = max(
        int(policy.phase2_min_long_prefill),
        int(policy.phase12_phase2_soft_min_long_prefill),
    )
    mixed_ok = (
        not strict_mode
        and bool(policy.phase12_phase2_soft_allow_mixed_decode)
        and num_decode_tokens > 0
        and _phase2_mixed_escape_ok(
            prefill_lens=pos_prefills,
            num_decode_tokens=num_decode_tokens,
            ratio=ratio,
            pressure_ratio=pressure_ratio,
            lora_rank_hetero=lora_rank_hetero,
            policy=policy,
            strict_mode=strict_mode,
        )
    )
    strong_prefill = (
        selective_ok
        and ratio >= soft_ratio
        and pressure_ratio >= soft_pressure
        and max_len >= soft_long
    )
    strong_rank_signal = (
        selective_ok
        and lora_rank_hetero
        and pressure_ratio >= float(policy.phase2_min_pressure_ratio)
        and max_len >= soft_long
    )
    recent_strength = float(getattr(state, "phase12_recent_phase1_strength", 0.0) or 0.0)
    if cap_live:
        recent_strength = max(recent_strength, 1.0)
    beneficiary_signal = _phase12_beneficiary_signal(
        state=state,
        policy=policy,
        req_infos=req_infos or [],
    )
    beneficiary_quality = min(
        1.0,
        float(beneficiary_signal.beneficiary_prefill_count)
        / float(max(1, int(policy.phase12_phase2_min_beneficiary_prefills))),
    )
    beneficiary_quality = max(
        beneficiary_quality,
        float(beneficiary_signal.beneficiary_cashout_quality),
    )
    beneficiary_quality = max(
        beneficiary_quality,
        float(beneficiary_signal.beneficiary_selected_quality),
    )
    beneficiary_quality_floor = max(
        0.0,
        min(1.0, float(getattr(policy, "phase12_phase2_beneficiary_quality_floor", 0.60) or 0.60)),
    )
    require_beneficiary_signal = bool(
        getattr(policy, "phase12_phase2_require_beneficiary_signal", True)
    )
    strong_prefill_beneficiary_floor = max(
        beneficiary_quality_floor,
        min(
            1.0,
            float(
                getattr(policy, "phase12_phase2_beneficiary_strong_prefill_quality_floor", 0.72) or 0.72
            ),
        ),
    )
    sparse_cooldown = max(
        0,
        int(getattr(policy, "phase12_phase2_sparse_cashout_cooldown", 2) or 0),
    )
    sparse_exception_quality = max(
        strong_prefill_beneficiary_floor,
        min(
            1.0,
            float(
                getattr(policy, "phase12_phase2_sparse_cashout_exception_quality", 0.90) or 0.90
            ),
        ),
    )
    phase2_cooldown_live = int(getattr(state, "phase12_recent_phase2_cashout_cooldown", 0) or 0)
    recent_chunk = max(0, int(getattr(state, "phase12_recent_phase1_chunk", 0) or 0))
    current_min_prefill = min(pos_prefills) if pos_prefills else 0
    chunk_match_scale = max(
        1.0,
        float(getattr(policy, "phase12_phase2_soft_recent_chunk_match_scale", 1.5) or 1.5),
    )
    recent_chunk_match = (
        recent_chunk > 0
        and current_min_prefill > 0
        and current_min_prefill
        <= max(recent_chunk + 64, int(float(recent_chunk) * chunk_match_scale))
    )
    expected_prefill_upper = max(recent_chunk + 64, int(float(recent_chunk) * chunk_match_scale))
    if recent_chunk > 0 and current_min_prefill > 0 and expected_prefill_upper > recent_chunk:
        if current_min_prefill <= recent_chunk:
            chunk_match_quality = 1.0
        elif current_min_prefill >= expected_prefill_upper:
            chunk_match_quality = 0.0
        else:
            chunk_match_quality = 1.0 - (
                float(current_min_prefill - recent_chunk)
                / float(max(1, expected_prefill_upper - recent_chunk))
            )
    else:
        chunk_match_quality = 0.0
    strong_recent = recent_strength >= max(
        0.0,
        float(getattr(policy, "phase12_phase2_soft_recent_strength_floor", 0.08) or 0.08),
    )
    recent_floor = max(
        1e-6,
        float(getattr(policy, "phase12_phase2_soft_recent_strength_floor", 0.08) or 0.08),
    )
    ttl_target = max(1, int(getattr(policy, "phase12_phase2_recent_ttl", 1) or 1))
    ttl_quality = min(1.0, max(0.0, float(recent_ttl)) / float(ttl_target))
    recent_quality = min(1.0, max(max(0.0, recent_strength) / recent_floor, ttl_quality))
    pressure_quality = min(1.0, max(0.0, pressure_ratio) / max(1e-6, soft_pressure))
    ratio_quality = min(1.0, max(0.0, ratio) / max(1e-6, soft_ratio))
    window_score = (
        float(getattr(policy, "phase12_phase2_soft_window_recent_weight", 0.40) or 0.40) * recent_quality
        + float(getattr(policy, "phase12_phase2_soft_window_chunk_weight", 0.25) or 0.25) * chunk_match_quality
        + float(getattr(policy, "phase12_phase2_soft_window_pressure_weight", 0.20) or 0.20) * pressure_quality
        + float(getattr(policy, "phase12_phase2_soft_window_ratio_weight", 0.10) or 0.10) * ratio_quality
        + float(getattr(policy, "phase12_phase2_beneficiary_weight", 0.35) or 0.35)
        * max(
            beneficiary_quality,
            float(beneficiary_signal.beneficiary_fraction),
            float(beneficiary_signal.beneficiary_wait_quality),
            float(beneficiary_signal.beneficiary_selected_quality),
        )
        + (
            float(getattr(policy, "phase12_phase2_soft_window_decode_bonus", 0.10) or 0.10)
            if mixed_ok
            else 0.0
        )
        + (0.10 if cap_live else 0.0)
        + (0.05 if lora_rank_hetero and selective_ok else 0.0)
    )
    if require_beneficiary_signal:
        beneficiary_deficit = max(0.0, beneficiary_quality_floor - float(beneficiary_signal.beneficiary_selected_quality))
        if beneficiary_deficit > 0.0 and strong_prefill and not mixed_ok:
            window_score -= min(0.25, beneficiary_deficit * 0.35)
    window_threshold = max(
        0.1,
        float(getattr(policy, "phase12_phase2_soft_window_score_threshold", 0.95) or 0.95),
    )
    if (
        window_score >= window_threshold
        and selective_ok
        and (pressure_ratio >= float(policy.phase2_min_pressure_ratio) or mixed_ok)
    ):
        return True, "joint_soft_window_quality"
    require_cashout_signal = bool(getattr(policy, "phase12_phase2_soft_require_cashout_signal", True))
    if (
        require_beneficiary_signal
        and (cap_live or recent_ttl > 0 or strong_recent)
        and selective_ok
        and beneficiary_signal.beneficiary_prefill_count
        < max(1, int(policy.phase12_phase2_min_beneficiary_prefills))
        and not mixed_ok
    ):
        return False, "joint_soft_no_beneficiary"
    if (
        require_beneficiary_signal
        and strong_prefill
        and (
            beneficiary_signal.beneficiary_selected_ids == []
            or float(beneficiary_signal.beneficiary_selected_quality) < strong_prefill_beneficiary_floor
        )
        and not mixed_ok
    ):
        return False, "joint_soft_low_beneficiary_quality"
    if (
        sparse_cooldown > 0
        and phase2_cooldown_live > 0
        and strong_prefill
        and float(beneficiary_signal.beneficiary_selected_quality) < sparse_exception_quality
        and not mixed_ok
    ):
        return False, "joint_soft_sparse_cashout_cooldown"
    if strong_prefill and require_cashout_signal and window_score < window_threshold and not mixed_ok:
        return False, "joint_soft_low_window_quality"
    if not requires_recent_phase1 and selective_ok and window_score >= max(0.35, window_threshold * 0.55):
        return True, "joint_soft_window_quality_no_recent_req"
    if strong_prefill and (
        not require_beneficiary_signal
        or float(beneficiary_signal.beneficiary_selected_quality) >= strong_prefill_beneficiary_floor
        or mixed_ok
    ):
        return True, "joint_soft_strong_prefill"
    if strong_rank_signal and (window_score >= max(0.5, window_threshold * 0.8) or strong_recent):
        return True, "joint_soft_rank_pressure"
    if mixed_ok and selective_ok and window_score >= max(0.45, window_threshold * 0.7):
        return True, "joint_soft_mixed_escape"
    return False, "joint_soft_waiting_for_phase1"


def _phase12_tick_recent_phase1(state: _PatchState) -> None:
    recent_ttl = int(getattr(state, "phase12_recent_phase1_apply_ttl", 0) or 0)
    if recent_ttl > 0:
        state.phase12_recent_phase1_apply_ttl = max(0, recent_ttl - 1)
        state.phase12_recent_phase1_strength = max(
            0.0,
            float(getattr(state, "phase12_recent_phase1_strength", 0.0) or 0.0) * 0.7,
        )
    else:
        state.phase12_recent_phase1_strength = 0.0
        state.phase12_recent_phase1_chunk = 0


def _phase12_tick_recent_phase2(state: _PatchState) -> None:
    cooldown = int(getattr(state, "phase12_recent_phase2_cashout_cooldown", 0) or 0)
    if cooldown > 0:
        state.phase12_recent_phase2_cashout_cooldown = max(0, cooldown - 1)
    lane_ttl = int(getattr(state, "phase2_escape_lane_ttl", 0) or 0)
    if lane_ttl > 0:
        state.phase2_escape_lane_ttl = max(0, lane_ttl - 1)
    if int(getattr(state, "phase2_escape_lane_ttl", 0) or 0) <= 0:
        state.phase2_escape_active_ids.clear()
        state.phase2_escape_deferred_ids.clear()


def _phase12_activate_escape_lane(
    state: _PatchState,
    *,
    beneficiary_ids: Iterable[str],
    deferred_ids: Iterable[str],
    lane_ttl: Optional[int] = None,
) -> None:
    active = {str(rid) for rid in beneficiary_ids if str(rid)}
    deferred = {str(rid) for rid in deferred_ids if str(rid)}
    state.phase2_escape_active_ids = active
    state.phase2_escape_deferred_ids = deferred
    base_ttl = (
        int(lane_ttl)
        if lane_ttl is not None
        else int(getattr(state.policy, "phase12_phase2_escape_lane_ttl", 2) or 2)
    )
    state.phase2_escape_lane_ttl = max(1, base_ttl)
    state.metrics.record_escape_lane_activation(
        active_ids=active,
        deferred_ids=deferred,
        lane_ttl=int(state.phase2_escape_lane_ttl),
    )


def _phase12_activate_execution_escape(
    state: _PatchState,
    *,
    active_ids: Iterable[str],
    deferred_ids: Iterable[str],
) -> bool:
    active = [str(rid) for rid in active_ids if str(rid)]
    deferred = [str(rid) for rid in deferred_ids if str(rid)]
    if not active or not deferred:
        return False
    _phase12_activate_escape_lane(
        state,
        beneficiary_ids=active,
        deferred_ids=deferred,
        lane_ttl=int(getattr(state.policy, "phase2_execution_escape_lane_ttl", 1) or 1),
    )
    return True


def _phase12_clear_escape_lane(
    state: _PatchState,
    *,
    request_ids: Optional[Iterable[str]] = None,
) -> None:
    if request_ids is None:
        state.phase2_escape_active_ids.clear()
        state.phase2_escape_deferred_ids.clear()
        state.phase2_escape_lane_ttl = 0
        return
    for rid in request_ids:
        srid = str(rid)
        if not srid:
            continue
        state.phase2_escape_active_ids.discard(srid)
        state.phase2_escape_deferred_ids.discard(srid)
    if not state.phase2_escape_active_ids:
        state.phase2_escape_deferred_ids.clear()
        state.phase2_escape_lane_ttl = 0


def _phase12_scheduler_cashout_rewrite(
    *,
    state: _PatchState,
    scheduler_outputs: Any,
) -> tuple[Any, bool]:
    WaveSliceMetrics._emit_cross_process_event("phase2_sched_post_enter", {})
    policy = state.policy
    if not (policy.enable_phase2_modelrunner and bool(policy.phase2_enable_scheduler_cashout)):
        return scheduler_outputs, False
    scheduled = getattr(scheduler_outputs, "scheduled_seq_groups", None)
    if not isinstance(scheduled, list) or len(scheduled) < 2:
        return scheduler_outputs, False

    snapshot: list[tuple[Any, int]] = []
    seq_groups: list[Any] = []
    for group in scheduled:
        seq_group = getattr(group, "seq_group", None)
        if seq_group is None:
            continue
        seq_groups.append(seq_group)
        try:
            is_prefill = bool(seq_group.is_prefill())
        except Exception:
            is_prefill = False
        if not is_prefill:
            continue
        remaining = _safe_prefill_uncomputed_tokens(seq_group) or 0
        if remaining > 1:
            snapshot.append((seq_group, int(remaining)))
    if len(snapshot) < 2:
        state.metrics.record_phase2_decision(False, "scheduler_cashout_no_prefill")
        return scheduler_outputs, False

    prefill_lens, lora_ranks = _phase12_collect_prefill_lora_state(
        seq_groups,
        rank_infer=_infer_lora_rank,
        remaining_getter=_safe_prefill_uncomputed_tokens,
    )
    req_infos = _phase12_collect_snapshot_req_infos(
        snapshot,
        state=state,
        policy=policy,
        request_id_getter=_safe_request_id,
        expected_chunk_getter=_phase12_expected_chunk_tokens_adapter,
        rank_infer=_infer_lora_rank,
    )
    phase12_ready, phase12_reason = _phase12_joint_phase2_ready(
        state=state,
        policy=policy,
        prefill_lens=prefill_lens,
        num_decode_tokens=0,
        lora_ranks=lora_ranks,
        req_infos=req_infos,
        strict_mode=_is_phase2_strict(policy),
    )
    if not phase12_ready:
        state.metrics.record_phase2_decision(False, f"{phase12_reason}_sched")
        return scheduler_outputs, False

    beneficiary_signal = _phase12_beneficiary_signal(
        state=state,
        policy=policy,
        req_infos=req_infos,
    )
    removable_prefills = 0
    for group in scheduled:
        seq_group = getattr(group, "seq_group", None)
        if seq_group is None:
            continue
        request_id = _safe_request_id(seq_group)
        try:
            is_prefill = bool(seq_group.is_prefill())
        except Exception:
            is_prefill = False
        if not is_prefill:
            continue
        remaining = _safe_prefill_uncomputed_tokens(seq_group) or 0
        if remaining > 1 and (request_id is None or str(request_id) not in set(str(rid) for rid in beneficiary_signal.beneficiary_selected_ids)):
            removable_prefills += 1
    grade = _phase12_scheduler_cashout_grade(
        policy=policy,
        selected_ids=[str(rid) for rid in beneficiary_signal.beneficiary_selected_ids],
        selected_quality=float(beneficiary_signal.beneficiary_selected_quality),
        removable_count=removable_prefills,
        value_signal=_phase12_scheduler_cashout_value_signal(
            req_infos=req_infos,
            beneficiary_signal=beneficiary_signal,
        ),
    )
    if not grade or not grade["allowed"]:
        state.metrics.record_phase2_decision(
            False,
            "scheduler_cashout_low_quality",
            selected_quality=float(beneficiary_signal.beneficiary_selected_quality),
            quality_floor=float((grade or {}).get("hard_floor", 0.0)),
            quality_soft_floor=float((grade or {}).get("soft_floor", 0.0)),
            selected_count=int(len(beneficiary_signal.beneficiary_selected_ids or [])),
            strength=float((grade or {}).get("strength", 0.0)),
            value_score=float((grade or {}).get("value_score", 0.0)),
            net_value=float((grade or {}).get("net_value", 0.0)),
            gain_score=float((grade or {}).get("gain_score", 0.0)),
            cost_score=float((grade or {}).get("cost_score", 0.0)),
            wait_gap=float((grade or {}).get("wait_gap", 0.0)),
        )
        return scheduler_outputs, False

    beneficiary_ids = set(str(rid) for rid in beneficiary_signal.beneficiary_selected_ids[: int(grade["selected_cap"])])
    keep_groups: list[Any] = []
    removed_groups = 0
    removed_tokens = 0
    kept_prefills = 0
    remove_cap = int(grade["remove_cap"])
    for group in scheduled:
        seq_group = getattr(group, "seq_group", None)
        if seq_group is None:
            keep_groups.append(group)
            continue
        request_id = _safe_request_id(seq_group)
        try:
            is_prefill = bool(seq_group.is_prefill())
        except Exception:
            is_prefill = False
        if not is_prefill:
            keep_groups.append(group)
            continue
        remaining = _safe_prefill_uncomputed_tokens(seq_group) or 0
        is_beneficiary = request_id is not None and str(request_id) in beneficiary_ids
        if is_beneficiary:
            keep_groups.append(group)
            kept_prefills += 1
            continue
        if remaining > 1 and removed_groups < remove_cap:
            removed_groups += 1
            removed_tokens += max(0, int(getattr(group, "token_chunk_size", 0) or 0))
            continue
        keep_groups.append(group)

    min_removed = max(
        1,
        int(getattr(policy, "phase12_phase2_scheduler_cashout_min_removed_prefills", 1) or 1),
    )
    if removed_groups < min_removed or not beneficiary_ids:
        state.metrics.record_phase2_decision(False, "scheduler_cashout_not_enough_removed")
        return scheduler_outputs, False

    scheduler_outputs.scheduled_seq_groups = keep_groups
    try:
        scheduler_outputs.num_batched_tokens = max(
            0,
            int(getattr(scheduler_outputs, "num_batched_tokens", 0)) - int(removed_tokens),
        )
    except Exception:
        pass
    try:
        scheduler_outputs.num_prefill_groups = max(0, int(kept_prefills))
    except Exception:
        pass

    state.metrics.record_phase2_decision(
        True,
        "scheduler_cashout_beneficiary",
        selected_quality=float(beneficiary_signal.beneficiary_selected_quality),
        quality_floor=float(grade["hard_floor"]),
        quality_soft_floor=float(grade["soft_floor"]),
        selected_count=int(len(beneficiary_ids)),
        strength=float(grade["strength"]),
        remove_cap=int(remove_cap),
        value_score=float(grade.get("value_score", 0.0)),
        net_value=float(grade.get("net_value", 0.0)),
        gain_score=float(grade.get("gain_score", 0.0)),
        cost_score=float(grade.get("cost_score", 0.0)),
        wait_gap=float(grade.get("wait_gap", 0.0)),
    )
    state.phase12_recent_phase2_cashout_cooldown = _phase12_scheduler_cashout_cooldown_for_grade(
        policy=policy,
        grade=grade,
    )
    logger.info(
        "[Wave-Slice][P2-scheduler-cashout] beneficiaries=%d removed_prefills=%d removed_tokens=%d quality=%.3f",
        len(beneficiary_ids),
        removed_groups,
        removed_tokens,
        float(beneficiary_signal.beneficiary_selected_quality),
    )
    return scheduler_outputs, True


def _phase12_apply_scheduler_cashout_to_queues(
    *,
    state: _PatchState,
    running: Any,
    waiting: Any,
) -> tuple[Any, Any, list[Any], list[Any], bool]:
    WaveSliceMetrics._emit_cross_process_event("phase2_sched_pre_enter", {})
    policy = state.policy
    if not (policy.enable_phase2_modelrunner and bool(policy.phase2_enable_scheduler_cashout)):
        return running, waiting, [], [], False
    cooldown_live = int(getattr(state, "phase12_recent_phase2_cashout_cooldown", 0) or 0)
    if cooldown_live > 0:
        state.metrics.record_phase2_decision(False, "scheduler_cashout_cooldown")
        return running, waiting, [], [], False

    snapshot, _ = _collect_live_snapshot(waiting, running)
    if len(snapshot) < 2:
        state.metrics.record_phase2_decision(False, "scheduler_cashout_no_prefill")
        return running, waiting, [], [], False

    req_infos = _phase12_collect_snapshot_req_infos(
        snapshot,
        state=state,
        policy=policy,
        request_id_getter=_safe_request_id,
        expected_chunk_getter=_phase12_expected_chunk_tokens_adapter,
        rank_infer=_infer_lora_rank,
    )
    prefill_lens, lora_ranks = _phase12_collect_prefill_lora_state(
        [sg for sg, _ in snapshot],
        rank_infer=_infer_lora_rank,
        remaining_getter=_safe_prefill_uncomputed_tokens,
    )
    combined = list(waiting) + list(running)
    num_decode_tokens = sum(
        1
        for sg in combined
        if (_safe_prefill_uncomputed_tokens(sg) or 0) <= 0 and (_safe_remaining_tokens(sg) or 0) > 0
    )
    phase12_ready, phase12_reason = _phase12_joint_phase2_ready(
        state=state,
        policy=policy,
        prefill_lens=prefill_lens,
        num_decode_tokens=num_decode_tokens,
        lora_ranks=lora_ranks,
        req_infos=req_infos,
        strict_mode=_is_phase2_strict(policy),
    )
    if not phase12_ready:
        state.metrics.record_phase2_decision(False, f"{phase12_reason}_sched_pre")
        return running, waiting, [], [], False

    beneficiary_signal = _phase12_beneficiary_signal(
        state=state,
        policy=policy,
        req_infos=req_infos,
    )
    beneficiary_selected_ids = [str(rid) for rid in beneficiary_signal.beneficiary_selected_ids]
    req_info_map = {str(info.request_id): info for info in req_infos}
    removable_candidate_request_ids: list[str] = []
    removable_candidate_chunk_tokens: dict[str, int] = {}
    waiting_request_ids = {
        str(rid)
        for rid in (
            _safe_request_id(seq_group)
            for seq_group in list(waiting)
        )
        if rid
    }
    max_expected_chunk = max((int(info.expected_chunk_tokens) for info in req_infos if int(info.expected_chunk_tokens) > 0), default=0)
    candidate_pool = max(
        1,
        int(getattr(policy, "phase12_phase2_scheduler_cashout_candidate_pool", 3) or 3),
    )
    candidate_size_ceiling = max(
        0.0,
        min(
            1.0,
            float(
                getattr(policy, "phase12_phase2_scheduler_cashout_candidate_size_ceiling", 0.20)
                or 0.20
            ),
        ),
    )
    fragment_cap_tokens = max(
        1,
        int(getattr(policy, "phase12_phase2_scheduler_cashout_fragment_cap_tokens", 256) or 256),
    )
    fragment_recent_scale = max(
        0.25,
        float(getattr(policy, "phase12_phase2_scheduler_cashout_fragment_recent_scale", 0.75) or 0.75),
    )
    recent_chunk = max(0, int(getattr(state, "phase12_recent_phase1_chunk", 0) or 0))
    removable_prefills = sum(
        1
        for seq_group in list(waiting)
        if (_safe_prefill_uncomputed_tokens(seq_group) or 0) > 1
        and (
            (_safe_request_id(seq_group) is None)
            or (str(_safe_request_id(seq_group)) not in set(beneficiary_selected_ids))
        )
    )
    removable_candidates: list[tuple[float, float, str, int]] = []
    now_s = time.time()
    for seq_group in list(waiting):
        rid = _safe_request_id(seq_group)
        remaining = _safe_prefill_uncomputed_tokens(seq_group) or 0
        if remaining <= 1:
            continue
        if rid is not None and str(rid) in set(beneficiary_selected_ids):
            continue
        if not rid:
            continue
        info = req_info_map.get(str(rid))
        chunk_tokens = int(getattr(info, "expected_chunk_tokens", remaining) if info is not None else remaining)
        fragment_tokens = max(1, min(int(chunk_tokens), int(fragment_cap_tokens)))
        if recent_chunk > 0:
            fragment_tokens = max(
                1,
                min(fragment_tokens, int(max(1.0, float(recent_chunk) * fragment_recent_scale))),
            )
        size_quality = (
            min(1.0, float(max(0, fragment_tokens)) / float(max_expected_chunk))
            if max_expected_chunk > 0
            else 0.0
        )
        if size_quality > candidate_size_ceiling:
            continue
        arrival_s = getattr(getattr(seq_group, "metrics", None), "arrival_time", None)
        if arrival_s is None:
            arrival_s = getattr(seq_group, "arrival_time", None)
        wait_s = max(0.0, now_s - float(arrival_s)) if arrival_s is not None else 0.0
        removable_candidates.append((size_quality, wait_s, str(rid), int(fragment_tokens)))
    removable_candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    chosen_candidates = removable_candidates[:candidate_pool]
    removable_candidate_request_ids = [rid for _, _, rid, _ in chosen_candidates]
    removable_candidate_chunk_tokens = {rid: int(tokens) for _, _, rid, tokens in chosen_candidates}
    grade = _phase12_scheduler_cashout_grade(
        policy=policy,
        selected_ids=beneficiary_selected_ids,
        selected_quality=float(beneficiary_signal.beneficiary_selected_quality),
        removable_count=removable_prefills,
        value_signal=_phase12_scheduler_cashout_value_signal(
            req_infos=req_infos,
            beneficiary_signal=beneficiary_signal,
            removable_request_ids=waiting_request_ids,
            removable_candidate_request_ids=removable_candidate_request_ids,
            removable_candidate_chunk_tokens=removable_candidate_chunk_tokens,
        ),
    )
    if not grade or not grade["allowed"]:
        state.metrics.record_phase2_decision(
            False,
            "scheduler_cashout_low_quality",
            selected_quality=float(beneficiary_signal.beneficiary_selected_quality),
            quality_floor=float((grade or {}).get("hard_floor", 0.0)),
            quality_soft_floor=float((grade or {}).get("soft_floor", 0.0)),
            selected_count=int(len(beneficiary_signal.beneficiary_selected_ids or [])),
            strength=float((grade or {}).get("strength", 0.0)),
            value_score=float((grade or {}).get("value_score", 0.0)),
            net_value=float((grade or {}).get("net_value", 0.0)),
            gain_score=float((grade or {}).get("gain_score", 0.0)),
            cost_score=float((grade or {}).get("cost_score", 0.0)),
            wait_gap=float((grade or {}).get("wait_gap", 0.0)),
            candidate_wait_quality=float((grade or {}).get("candidate_wait_quality", 0.0)),
            candidate_size_quality=float((grade or {}).get("candidate_size_quality", 0.0)),
            candidate_shape_penalty=float((grade or {}).get("candidate_shape_penalty", 0.0)),
            small_candidate_bonus=float((grade or {}).get("small_candidate_bonus", 0.0)),
            medium_candidate_penalty=float((grade or {}).get("medium_candidate_penalty", 0.0)),
        )
        return running, waiting, [], [], False

    beneficiary_ids = set(beneficiary_selected_ids[: int(grade["selected_cap"])])
    target_remove_ids = set(removable_candidate_request_ids[: int(grade["remove_cap"])])
    hidden_running: list[Any] = []
    hidden_waiting: list[Any] = []
    hidden_request_ids: list[str] = []
    remove_cap = int(grade["remove_cap"])
    strength = float(grade["strength"])
    removed_prefills = 0

    def _filter_queue(queue_obj: Any, hidden_out: list[Any]) -> Any:
        nonlocal removed_prefills
        kept: list[Any] = []
        for seq_group in list(queue_obj):
            request_id = _safe_request_id(seq_group)
            remaining = _safe_prefill_uncomputed_tokens(seq_group) or 0
            if (
                remaining > 1
                and removed_prefills < remove_cap
                and request_id is not None
                and str(request_id) in target_remove_ids
                and str(request_id) not in beneficiary_ids
            ):
                hidden_out.append(seq_group)
                removed_prefills += 1
                if request_id:
                    hidden_request_ids.append(str(request_id))
            else:
                kept.append(seq_group)
        return _rebuild_queue_like(queue_obj, kept)

    new_running = running
    new_waiting = _filter_queue(waiting, hidden_waiting)
    min_removed = max(
        1,
        int(getattr(policy, "phase12_phase2_scheduler_cashout_min_removed_prefills", 1) or 1),
    )
    if removed_prefills < min_removed:
        state.metrics.record_phase2_decision(False, "scheduler_cashout_not_enough_removed")
        return running, waiting, [], [], False

    now_s = time.time()
    new_waiting = _phase12_priority_bubble_waiting_queue(
        new_waiting,
        beneficiary_signal=beneficiary_signal,
        beneficiary_ids=beneficiary_ids,
        strength=strength,
        brain=state.brain,
        now_s=now_s,
        remaining_getter=_safe_remaining_tokens,
        wait_getter=_safe_wait_us,
        request_id_getter=_safe_request_id,
        solo_us_estimator=_estimate_solo_us,
        queue_rebuilder=_rebuild_queue_like,
    )

    state.metrics.record_phase2_decision(
        True,
        "scheduler_cashout_beneficiary",
        selected_quality=float(beneficiary_signal.beneficiary_selected_quality),
        quality_floor=float(grade["hard_floor"]),
        quality_soft_floor=float(grade["soft_floor"]),
        selected_count=int(len(beneficiary_ids)),
        strength=float(grade["strength"]),
        remove_cap=int(remove_cap),
        value_score=float(grade.get("value_score", 0.0)),
        net_value=float(grade.get("net_value", 0.0)),
        gain_score=float(grade.get("gain_score", 0.0)),
        cost_score=float(grade.get("cost_score", 0.0)),
        wait_gap=float(grade.get("wait_gap", 0.0)),
        candidate_wait_quality=float(grade.get("candidate_wait_quality", 0.0)),
        candidate_size_quality=float(grade.get("candidate_size_quality", 0.0)),
        candidate_shape_penalty=float(grade.get("candidate_shape_penalty", 0.0)),
        small_candidate_bonus=float(grade.get("small_candidate_bonus", 0.0)),
        medium_candidate_penalty=float(grade.get("medium_candidate_penalty", 0.0)),
    )
    state.phase12_recent_phase2_cashout_cooldown = _phase12_scheduler_cashout_cooldown_for_grade(
        policy=policy,
        grade=grade,
    )
    _phase12_activate_escape_lane(
        state,
        beneficiary_ids=list(beneficiary_ids),
        deferred_ids=hidden_request_ids,
        lane_ttl=int(grade["lane_ttl"]),
    )
    logger.info(
        "[Wave-Slice][P2-scheduler-prehide] beneficiaries=%d hidden_running=%d hidden_waiting=%d quality=%.3f",
        len(beneficiary_ids),
        len(hidden_running),
        len(hidden_waiting),
        float(beneficiary_signal.beneficiary_selected_quality),
    )
    return new_running, new_waiting, hidden_running, hidden_waiting, True


def _is_v1_scheduler_output(model_input: Any) -> bool:
    return hasattr(model_input, "num_scheduled_tokens") and hasattr(
        model_input, "total_num_scheduled_tokens"
    )


def _record_phase2_v1_output_probe(
    state: _PatchState,
    *,
    context: str,
    model_input: Any,
) -> bool:
    is_v1 = _is_v1_scheduler_output(model_input)
    state.metrics.record_phase2_debug_counter(
        f"{str(context)}_v1_output_{'hits' if is_v1 else 'misses'}"
    )
    return is_v1


def _phase12_expected_chunk_tokens_adapter(
    seq_group: Any,
    state: Optional[_PatchState],
    remaining: int,
) -> int:
    return _phase12_expected_chunk_tokens(
        seq_group,
        state=state,
        remaining=remaining,
    )


def _v1_request_status_name(request: Any) -> str:
    status = getattr(request, "status", None)
    if status is None:
        return ""
    return str(getattr(status, "name", status))


def _reconcile_v1_waiting_running_status(
    scheduler_obj: Any,
    state: _PatchState,
    *,
    context: str,
) -> None:
    """Keep vLLM V1's waiting queue free of RUNNING requests.

    vLLM V1 treats the queue object and request.status as a state-machine
    contract: requests in `waiting` must be WAITING/PREEMPTED/WAITING_*.
    Our scheduler-side prehide/restore can briefly perturb queue ownership
    around a native schedule() call, so repair the queue before handing control
    back to vLLM.
    """
    if state.scheduler_method_name != "schedule":
        return
    if not _has_running_waiting_queues(scheduler_obj):
        return
    running = getattr(scheduler_obj, "running", None)
    waiting = getattr(scheduler_obj, "waiting", None)
    if running is None or waiting is None:
        return

    try:
        waiting_items = list(waiting)
    except Exception:
        return
    if not waiting_items:
        return

    try:
        running_items = list(running)
    except Exception:
        running_items = []
    running_ids = {
        str(rid)
        for rid in (_safe_request_id(req) for req in running_items)
        if rid is not None
    }

    keep_waiting: list[Any] = []
    move_to_running: list[Any] = []
    dropped_duplicates = 0
    for req in waiting_items:
        if _v1_request_status_name(req) != "RUNNING":
            keep_waiting.append(req)
            continue
        rid = _safe_request_id(req)
        rid_s = str(rid) if rid is not None else ""
        if rid_s and rid_s in running_ids:
            dropped_duplicates += 1
            continue
        move_to_running.append(req)
        if rid_s:
            running_ids.add(rid_s)

    if not move_to_running and not dropped_duplicates:
        return

    try:
        scheduler_obj.waiting = _rebuild_queue_like(waiting, keep_waiting)
    except Exception:
        logger.exception("[Wave-Slice] failed to rebuild V1 waiting queue during status reconciliation.")
        return
    if move_to_running:
        try:
            if hasattr(scheduler_obj.running, "extend"):
                scheduler_obj.running.extend(move_to_running)
            else:
                scheduler_obj.running = _rebuild_queue_like(
                    scheduler_obj.running,
                    list(scheduler_obj.running) + move_to_running,
                )
        except Exception:
            logger.exception("[Wave-Slice] failed to restore RUNNING requests to V1 running queue.")
            return
    state.metrics.record_phase2_debug_counter(f"v1_waiting_running_reconciled_{context}")
    logger.warning(
        "[Wave-Slice][V1-reconcile] context=%s moved_running=%d dropped_duplicates=%d",
        str(context),
        len(move_to_running),
        dropped_duplicates,
    )


def _build_scheduler_hook(state: _PatchState) -> Callable[..., Any]:
    original_schedule = state.original_schedule
    schedule_impl = (
        _build_v1_schedule_waiting_patch(state, original_schedule)
        if state.scheduler_method_name == "schedule"
        else original_schedule
    )

    @functools.wraps(original_schedule)
    def _wave_schedule_hook(self: Any, *args: Any, **kwargs: Any) -> Any:
        WaveSliceMetrics._emit_cross_process_event("schedule_hook_enter", {})

        def _emit_schedule_hook_early(reason: str, **payload: Any) -> None:
            WaveSliceMetrics._emit_cross_process_event(
                "schedule_hook_early_return",
                {"reason": str(reason), **payload},
            )

        def _maybe_apply_phase2_schedule_cashout(outputs: Any) -> Any:
            if (
                state.scheduler_method_name == "schedule"
                and bool(state.policy.phase2_enable_scheduler_cashout)
            ):
                try:
                    outputs, _ = _phase12_scheduler_cashout_rewrite(
                        state=state,
                        scheduler_outputs=outputs,
                    )
                except Exception:
                    logger.exception("[Wave-Slice] inline scheduler cashout rewrite failed.")
            return outputs

        def _run_schedule_with_optional_phase2_cashout() -> Any:
            restore_phase2_running = None
            restore_phase2_waiting = None
            live_running = getattr(self, "running", None)
            live_waiting = getattr(self, "waiting", None)
            if (
                _phase2_scheduler_cashout_enabled(state.policy)
                and _has_running_waiting_queues(self)
                and live_running is not None
                and live_waiting is not None
            ):
                try:
                    restore_phase2_running, restore_phase2_waiting = _capture_queue_pair(self)
                    self.running, self.waiting, _, _, _ = _phase12_apply_scheduler_cashout_to_queues(
                        state=state,
                        running=self.running,
                        waiting=self.waiting,
                    )
                except Exception:
                    logger.exception("[Wave-Slice] pre-schedule cashout wrapper failed.")
            try:
                _reconcile_v1_waiting_running_status(
                    self,
                    state,
                    context="pre_native",
                )
                outputs = schedule_impl(self, *args, **kwargs)
            finally:
                try:
                    _restore_queue_pair(self, restore_phase2_running, restore_phase2_waiting)
                except Exception:
                    pass
            return _maybe_apply_phase2_schedule_cashout(outputs)

        def _is_lora_mode_enabled(scheduler_obj: Any) -> bool:
            # Conservative detection across vLLM variants.
            direct = getattr(scheduler_obj, "lora_config", None)
            if direct is not None:
                return True
            for attr in ("enable_lora", "lora_enabled", "has_lora", "_enable_lora"):
                if bool(getattr(scheduler_obj, attr, False)):
                    return True
            cfg = getattr(scheduler_obj, "scheduler_config", None)
            if cfg is not None:
                for attr in ("enable_lora", "lora_enabled"):
                    if bool(getattr(cfg, attr, False)):
                        return True
            return False

        if not _has_running_waiting_queues(self):
            _emit_schedule_hook_early("missing_running_waiting")
            return _run_schedule_with_optional_phase2_cashout()

        running = self.running
        waiting = self.waiting
        if running is None or waiting is None:
            _emit_schedule_hook_early("null_running_waiting")
            return _run_schedule_with_optional_phase2_cashout()

        if not state.policy.enable_phase1_scheduler:
            state.metrics.record_scheduler_decision(False)
            _emit_schedule_hook_early("phase1_disabled")
            return _run_schedule_with_optional_phase2_cashout()

        lora_mode_enabled = _is_lora_mode_enabled(self)
        can_phase1_threshold = state.policy.enable_phase1_dynamic_threshold and (
            (not lora_mode_enabled)
            or state.policy.allow_phase1_with_lora
            or state.policy.allow_phase1_threshold_with_lora
        )
        can_phase1_budget = state.policy.enable_phase1_budget_guidance and (
            (not lora_mode_enabled)
            or state.policy.allow_phase1_with_lora
            or state.policy.allow_phase1_budget_with_lora
        )
        can_phase1_tick_hide = state.policy.enable_tick_hide and (
            (not lora_mode_enabled)
            or state.policy.allow_phase1_with_lora
            or state.policy.allow_phase1_tick_hide_with_lora
        )

        # Keep LoRA correctness first: by default only threshold rewriting is
        # enabled under LoRA, while queue hiding / budget shaping stay off.
        if lora_mode_enabled and not (can_phase1_threshold or can_phase1_budget or can_phase1_tick_hide):
            state.metrics.record_scheduler_decision(False)
            _emit_schedule_hook_early("lora_guard")
            return _run_schedule_with_optional_phase2_cashout()

        # Metrics-side request observation from scheduler internals.
        for sg in list(waiting) + list(running):
            req_id = _safe_request_id(sg)
            if not req_id:
                continue
            total_tokens = _safe_total_tokens(sg)
            solo_us = _estimate_solo_us(state.brain, total_tokens)
            is_short = (total_tokens is not None) and (total_tokens <= state.policy.metrics_short_request_tokens)
            state.metrics.observe_scheduler_request(
                req_id,
                total_tokens=total_tokens,
                solo_us=solo_us,
                is_short=is_short,
            )

        scheduler_cfg = getattr(self, "scheduler_config", None)
        original_budget = getattr(scheduler_cfg, "max_num_batched_tokens", None)
        original_threshold = getattr(scheduler_cfg, "long_prefill_token_threshold", None)
        _phase12_tick_recent_phase1(state)
        _phase12_tick_recent_phase2(state)

        try:
            if state.policy.enable_sjf_reorder:
                now_s = time.time()
                self.running = _reorder_queue(
                    running,
                    brain=state.brain,
                    now_s=now_s,
                    mode=state.policy.queue_reorder_mode,
                    aging_quantum_us=state.policy.queue_reorder_aging_quantum_us,
                )
                self.waiting = _reorder_queue(
                    waiting,
                    brain=state.brain,
                    now_s=now_s,
                    mode=state.policy.queue_reorder_mode,
                    aging_quantum_us=state.policy.queue_reorder_aging_quantum_us,
                )
                running = self.running
                waiting = self.waiting
            if bool(state.policy.phase1_ingress_direct_authoritative):
                state.phase1_virtual_token_caps = _phase1_prune_ingress_virtual_caps(state)
            else:
                state.phase1_virtual_token_caps.clear()
            ingress_eager_chunk: Optional[int] = None
            scheduler_chunk_raw: Optional[int] = None
            direct_chunk_raw: Optional[int] = None
            cohort_target_raw: Optional[int] = None

            snapshot, max_wait_us = _collect_live_snapshot(waiting, running)
            snapshot_lengths = [int(rem) for _, rem in snapshot]
            selection_snapshot = snapshot
            global_activity_cohort = (
                _phase1_build_global_activity_cohort(state, snapshot)
                if not lora_mode_enabled
                else None
            )
            ingress_candidate = _phase1_find_ingress_virtual_candidate(
                state.phase1_ingress_virtuals,
                snapshot=snapshot,
                request_id_getter=_safe_request_id,
            )
            if lora_mode_enabled:
                selection_snapshot = _phase1_filter_snapshot_for_lora_cohort(
                    snapshot,
                    preferred_request_id=(
                        str(ingress_candidate[0].long_req_id)
                        if ingress_candidate is not None and getattr(ingress_candidate[0], "long_req_id", None)
                        else None
                    ),
                )
            selection_lengths = [int(rem) for _, rem in selection_snapshot]
            cohort = None
            if ingress_candidate is not None:
                ingress_virtual, _ingress_seq_group, ingress_remaining = ingress_candidate
                live_cohort = _phase1_live_cohort_from_snapshot(
                    selection_snapshot,
                    state.policy,
                    request_id_getter=_safe_request_id,
                )
                if live_cohort is not None:
                    cohort = _Phase1CohortStats(
                        representative_short_len=max(1, int(live_cohort.representative_short_len)),
                        short_count=max(1, int(live_cohort.short_count)),
                        short_token_mass=max(1, int(live_cohort.short_token_mass)),
                        short_lengths=[int(v) for v in live_cohort.short_lengths] or [max(1, int(live_cohort.representative_short_len))],
                        long_len=max(1, int(ingress_remaining)),
                        long_req_id=str(ingress_virtual.long_req_id),
                        total_count=max(int(live_cohort.total_count), int(ingress_virtual.active_count), 2),
                    )
                else:
                    cohort = _Phase1CohortStats(
                        representative_short_len=int(ingress_virtual.representative_short_len),
                        short_count=max(1, int(ingress_virtual.short_count)),
                        short_token_mass=int(ingress_virtual.short_token_mass),
                        short_lengths=[int(v) for v in ingress_virtual.short_lengths] or [int(ingress_virtual.representative_short_len)],
                        long_len=max(1, int(ingress_remaining)),
                        long_req_id=str(ingress_virtual.long_req_id),
                        total_count=max(2, int(ingress_virtual.active_count)),
                    )
                state.metrics.record_phase1_probe(
                    reason="ingress_virtual_override",
                    short_len=int(cohort.representative_short_len),
                    long_len=int(cohort.long_len),
                    queue_len=len(selection_lengths),
                    wait_us=max_wait_us,
                    slice_eligible=False,
                )
            else:
                if not _need_wave_slice(snapshot_lengths, state.policy):
                    fallback_cohort = global_activity_cohort
                    if fallback_cohort is None and not lora_mode_enabled:
                        fallback_cohort = _phase1_build_ingress_fallback_cohort(
                            state,
                            snapshot,
                        )
                    if fallback_cohort is not None:
                        cohort = fallback_cohort
                        state.metrics.record_phase1_probe(
                            reason="ingress_virtual_override",
                            short_len=int(cohort.representative_short_len),
                            long_len=int(cohort.long_len),
                            queue_len=len(snapshot_lengths),
                            wait_us=max_wait_us,
                            slice_eligible=False,
                        )
                    else:
                        state.metrics.record_phase1_probe(
                            reason="no_need_wave_slice",
                            short_len=min(snapshot_lengths) if snapshot_lengths else None,
                            long_len=max(snapshot_lengths) if snapshot_lengths else None,
                            queue_len=len(snapshot_lengths),
                            wait_us=max_wait_us,
                            slice_eligible=False,
                        )
                        state.metrics.record_scheduler_decision(False)
                        state.phase1_sticky_req_id = None
                        state.phase1_sticky_chunk = None
                        state.phase1_sticky_ttl_left = 0
                        if not waiting and not running:
                            state.phase1_explicit_plans.clear()
                        _emit_schedule_hook_early(
                            "no_need_wave_slice",
                            snapshot_count=len(snapshot_lengths),
                            min_len=(min(snapshot_lengths) if snapshot_lengths else 0),
                            max_len=(max(snapshot_lengths) if snapshot_lengths else 0),
                            hetero_ratio=(
                                (float(max(snapshot_lengths)) / float(max(1, min(snapshot_lengths))))
                                if snapshot_lengths else 0.0
                            ),
                        )
                        return _run_schedule_with_optional_phase2_cashout()
                if cohort is None:
                    cohort = _phase1_live_cohort_from_snapshot(
                        selection_snapshot,
                        state.policy,
                        request_id_getter=_safe_request_id,
                    )
                if cohort is None and global_activity_cohort is not None:
                    cohort = global_activity_cohort
                if (
                    cohort is None
                    and lora_mode_enabled
                    and selection_snapshot is not snapshot
                ):
                    cohort = _phase1_live_cohort_from_snapshot(
                        snapshot,
                        state.policy,
                        request_id_getter=_safe_request_id,
                    )
                if cohort is None:
                    state.metrics.record_phase1_probe(
                        reason="no_cohort",
                        short_len=min(snapshot_lengths) if snapshot_lengths else None,
                        long_len=max(snapshot_lengths) if snapshot_lengths else None,
                        queue_len=len(snapshot_lengths),
                        wait_us=max_wait_us,
                        slice_eligible=False,
                    )
                    state.metrics.record_scheduler_decision(False)
                    _emit_schedule_hook_early("no_cohort")
                    return _run_schedule_with_optional_phase2_cashout()

            long_seq_group = _phase1_find_seq_group_by_request_id(
                selection_snapshot,
                cohort.long_req_id,
                request_id_getter=_safe_request_id,
            )
            if long_seq_group is None and selection_snapshot is not snapshot:
                long_seq_group = _phase1_find_seq_group_by_request_id(
                    snapshot,
                    cohort.long_req_id,
                    request_id_getter=_safe_request_id,
                )

            short_len = int(cohort.representative_short_len)
            long_len = int(cohort.long_len)
            queue_len = len(waiting) + len(running)
            runtime_base_policy: Optional[WaveSlicePolicy] = None
            runtime_adaptive_meta: dict[str, float] = {}
            if bool(getattr(state.policy, "phase1_runtime_adaptive_enabled", False)):
                try:
                    waiting_short_count_for_pressure = _phase1_waiting_short_count(
                        waiting,
                        short_threshold_tokens=int(state.policy.metrics_short_request_tokens),
                    )
                    raw_runtime_meta = _phase1_runtime_pressure_meta(
                        policy=state.policy,
                        cohort=cohort,
                        queue_len=int(queue_len),
                        waiting_short_count=int(waiting_short_count_for_pressure),
                        max_wait_us=float(max_wait_us),
                        virtual_cap_hit_rate=float(state.metrics.phase1_virtual_cap_hit_ratio()),
                        previous_wall_pressure=float(state.phase1_runtime_wall_pressure_ema),
                    )
                    alpha = max(
                        0.0,
                        min(1.0, float(getattr(state.policy, "phase1_runtime_ema_alpha", 0.35) or 0.35)),
                    )
                    state.phase1_runtime_wall_pressure_ema = (
                        (1.0 - alpha) * float(state.phase1_runtime_wall_pressure_ema)
                        + alpha * float(raw_runtime_meta.get("wall_pressure", 0.0))
                    )
                    state.phase1_runtime_pressure_ema = (
                        (1.0 - alpha) * float(state.phase1_runtime_pressure_ema)
                        + alpha * float(raw_runtime_meta.get("effective_pressure", 0.0))
                    )
                    raw_runtime_meta["wall_pressure"] = float(state.phase1_runtime_wall_pressure_ema)
                    raw_runtime_meta["effective_pressure"] = float(state.phase1_runtime_pressure_ema)
                    adapted_policy, runtime_adaptive_meta = _phase1_runtime_adapt_policy(
                        state.policy,
                        raw_runtime_meta,
                    )
                    if adapted_policy is not state.policy:
                        runtime_base_policy = state.policy
                        state.policy = adapted_policy
                        state.phase1_runtime_last_meta = dict(runtime_adaptive_meta)
                        state.metrics.record_phase1_runtime_adaptation(
                            queue_len=int(queue_len),
                            waiting_short_count=int(waiting_short_count_for_pressure),
                            effective_pressure=float(runtime_adaptive_meta.get("effective_pressure", 0.0)),
                            wall_pressure=float(runtime_adaptive_meta.get("wall_pressure", 0.0)),
                            short_urgency=float(runtime_adaptive_meta.get("short_urgency", 0.0)),
                            target_fraction=float(runtime_adaptive_meta.get("phase1_target_long_fraction", state.policy.phase1_target_long_fraction)),
                            target_chunk=int(runtime_adaptive_meta.get("phase1_ingress_target_chunk", state.policy.phase1_ingress_target_chunk)),
                        )
                except Exception:
                    logger.exception("[Wave-Slice] runtime pressure adaptation failed; continue with base policy.")
                    runtime_base_policy = None
                    runtime_adaptive_meta = {}
            adjusted_queue_len = _phase1_adjusted_queue_len(cohort, queue_len, state.policy)
            baseline_chunk = _phase1_baseline_chunk_proxy(
                long_len=long_len,
                original_budget=original_budget,
                original_threshold=original_threshold,
                scheduler_cfg=scheduler_cfg,
                policy=state.policy,
            )
            explicit_plan_kind = None
            if (
                ingress_candidate is not None
                and bool(state.policy.phase1_ingress_direct_authoritative)
                and cohort.long_req_id
            ):
                upper = max(short_len + 1, long_len - 1)
                cohort_target_raw = int(_phase1_cohort_target_len(cohort, state.policy))
                ingress_min = _phase1_effective_ingress_min_chunk(
                    state.policy,
                    target=int(cohort_target_raw),
                )
                eager_target = min(upper, int(cohort_target_raw))
                if upper >= ingress_min:
                    eager_target = max(eager_target, ingress_min)
                eager_chunk = _phase1_authoritative_chunk(
                    state.policy,
                    state.slicer,
                    target=int(eager_target),
                    short_len=int(short_len),
                    upper=int(upper),
                )
                eager_floor = _phase1_authoritative_short_floor(
                    state.policy,
                    short_len=int(short_len),
                    target=int(eager_target),
                )
                ingress_eager_chunk = max(eager_floor, min(int(eager_chunk), upper))
                direct_cap_chunk = _phase1_direct_chunk_candidate(
                    state=state,
                    cohort=cohort,
                    total_len=max(1, int(_safe_total_tokens(long_seq_group) or long_len)),
                    done_offset=max(
                        0,
                        int((_safe_total_tokens(long_seq_group) or long_len) - long_len),
                    ),
                    remaining_len=int(long_len),
                    baseline_chunk=baseline_chunk,
                )
                state.phase1_virtual_token_caps[str(cohort.long_req_id)] = int(
                    direct_cap_chunk
                    if direct_cap_chunk is not None
                    else ingress_eager_chunk
                )

            best_chunk = state.slicer.choose_dynamic_chunk(
                short_len=short_len,
                long_len=long_len,
                scheduler=state.brain,
                t_wait_us=max_wait_us,
                queue_length=adjusted_queue_len,
                baseline_chunk=baseline_chunk,
            )
            scheduler_chunk_raw = int(best_chunk)
            cohort_target_raw = int(_phase1_cohort_target_len(cohort, state.policy))
            explicit_chunk = _phase1_explicit_chunk_from_plan(
                state=state,
                cohort=cohort,
                snapshot=selection_snapshot,
                t_wait_us=max_wait_us,
                queue_length=adjusted_queue_len,
                baseline_chunk=baseline_chunk,
                total_tokens_getter=_safe_total_tokens,
                request_id_getter=_safe_request_id,
            )
            if explicit_chunk is not None:
                direct_chunk_raw = int(explicit_chunk[0])
                best_chunk, explicit_plan_kind = explicit_chunk
            elif ingress_eager_chunk is not None:
                capped_chunk = min(int(best_chunk), int(ingress_eager_chunk))
                if capped_chunk < int(best_chunk):
                    best_chunk = int(capped_chunk)
                    explicit_plan_kind = "ingress_authoritative_cap"
            state.metrics.record_phase1_proposal(
                scheduler_chunk=scheduler_chunk_raw,
                direct_chunk=direct_chunk_raw,
                cohort_target=cohort_target_raw,
                direct_won=bool(
                    direct_chunk_raw is not None
                    and scheduler_chunk_raw is not None
                    and int(best_chunk) == int(direct_chunk_raw)
                    and int(direct_chunk_raw) < int(scheduler_chunk_raw)
                ),
            )
            reused_sticky = False
            if state.policy.phase1_enable_sticky_chunk:
                best_chunk, reused_sticky = _phase1_apply_sticky_chunk(
                    state=state,
                    cohort=cohort,
                    chosen_chunk=best_chunk,
                    slicer=state.slicer,
                )
            joint_floor_chunk = _phase12_joint_phase1_floor(
                state=state,
                snapshot=snapshot,
                policy=state.policy,
            )
            if joint_floor_chunk is not None:
                best_chunk = max(int(best_chunk), int(joint_floor_chunk))
            forced_chunk = False
            if explicit_plan_kind in {"direct_authoritative", "ingress_authoritative_eager"}:
                best_chunk = max(
                    int(best_chunk),
                    _phase1_effective_ingress_min_chunk(
                        state.policy,
                        target=int(best_chunk),
                    ),
                )
            else:
                best_chunk, forced_chunk = _maybe_force_phase1_chunk(
                    cohort=cohort,
                    queue_len=queue_len,
                    chosen_chunk=best_chunk,
                    slicer=state.slicer,
                    policy=state.policy,
                )
            state.metrics.record_phase1_probe(
                reason="candidate",
                short_len=short_len,
                long_len=long_len,
                baseline_chunk=baseline_chunk,
                best_chunk=best_chunk,
                queue_len=queue_len,
                wait_us=max_wait_us,
                slice_eligible=bool(best_chunk < long_len),
            )
            if best_chunk >= long_len:
                state.metrics.record_phase1_probe(
                    reason="best_chunk_ge_long",
                    short_len=short_len,
                    long_len=long_len,
                    baseline_chunk=baseline_chunk,
                    best_chunk=best_chunk,
                    queue_len=queue_len,
                    wait_us=max_wait_us,
                    slice_eligible=False,
                )
                state.metrics.record_scheduler_decision(False)
                if runtime_base_policy is not None:
                    state.policy = runtime_base_policy
                _phase1_update_sticky_chunk(
                    state=state,
                    cohort=cohort,
                    chosen_chunk=best_chunk,
                    applied=False,
                )
                _emit_schedule_hook_early("best_chunk_ge_long")
                return _run_schedule_with_optional_phase2_cashout()
        except Exception as exc:
            logger.exception("Wave-Slice Phase I pre-check failed; fallback to original scheduler.")
            if "runtime_base_policy" in locals() and runtime_base_policy is not None:
                state.policy = runtime_base_policy
            state.metrics.record_phase1_probe(
                reason=f"precheck_exception:{type(exc).__name__}",
            )
            state.metrics.record_scheduler_decision(False)
            _emit_schedule_hook_early(
                f"precheck_exception:{type(exc).__name__}",
                detail=str(exc)[:240],
                snapshot_count=len(snapshot) if "snapshot" in locals() else 0,
                min_len=(min(lengths) if "lengths" in locals() and lengths else 0),
                max_len=(max(lengths) if "lengths" in locals() and lengths else 0),
            )
            return _run_schedule_with_optional_phase2_cashout()

        hidden_long_tasks: list[Any] = []
        hidden_long_waiting_tasks: list[Any] = []
        hidden_phase2_running: list[Any] = []
        hidden_phase2_waiting: list[Any] = []
        state.metrics.record_scheduler_decision(True)
        short_token_mass = _phase1_effective_short_token_mass(
            cohort.short_lengths,
            short_len=short_len,
            best_chunk=best_chunk,
            policy=state.policy,
        )
        phase1_tick_hide_keep_ceiling = _phase1_tick_hide_keep_ceiling(
            best_chunk=best_chunk,
            cohort=cohort,
        )
        waiting_short_count = sum(
            1
            for sg in waiting
            if 0 < (_safe_prefill_uncomputed_tokens(sg) or 0) <= phase1_tick_hide_keep_ceiling
        )
        state.metrics.record_phase1_probe(
            reason="apply",
            short_len=short_len,
            long_len=long_len,
            baseline_chunk=baseline_chunk,
            best_chunk=best_chunk,
            queue_len=queue_len,
            wait_us=max_wait_us,
            slice_eligible=True,
        )
        if state.policy.phase1_enable_sticky_chunk:
            _phase1_update_sticky_chunk(
                state=state,
                cohort=cohort,
                chosen_chunk=best_chunk,
                applied=True,
            )
        state.phase12_recent_phase1_apply_ttl = max(1, int(state.policy.phase12_phase2_recent_ttl))
        state.phase12_last_phase1_req_id = str(cohort.long_req_id) if cohort.long_req_id else None
        state.phase12_recent_phase1_chunk = int(max(1, best_chunk))
        if long_len > 0:
            state.phase12_recent_phase1_strength = max(
                float(getattr(state, "phase12_recent_phase1_strength", 0.0) or 0.0),
                float(max(0, long_len - best_chunk)) / float(max(1, long_len)),
            )
        state.metrics.record_phase1_choice(
            chosen_chunk=best_chunk,
            baseline_chunk=baseline_chunk,
            explicit_plan=explicit_plan_kind is not None,
        )

        try:
            waiting_cap_cleanup_requests: list[Any] = []
            waiting_cap_restore_specs: list[tuple[Any, str, Any]] = []
            if bool(state.policy.phase1_ingress_direct_authoritative):
                try:
                    (
                        waiting_cap_cleanup_requests,
                        waiting_cap_restore_specs,
                    ) = _install_v1_waiting_cap_runtime_hooks(state, self)
                except Exception:
                    logger.exception("[Wave-Slice] failed to install v1 waiting-cap runtime hooks.")
            state.phase1_shadow_seq_lens.clear()
            secondary_lora_caps: dict[str, int] = {}
            if bool(state.policy.phase1_ingress_direct_authoritative) and lora_mode_enabled:
                try:
                    secondary_lora_caps = _phase1_collect_secondary_lora_caps(
                        state=state,
                        snapshot=snapshot,
                        primary_request_id=cohort.long_req_id,
                        max_wait_us=max_wait_us,
                        queue_len=queue_len,
                        scheduler_cfg=scheduler_cfg,
                        original_budget=original_budget,
                        original_threshold=original_threshold,
                    )
                except Exception:
                    logger.exception("[Wave-Slice] failed to collect secondary LoRA Phase-I caps.")
            if cohort.long_req_id and best_chunk < long_len:
                total_tokens_for_target = max(1, int(_safe_total_tokens(long_seq_group) or long_len))
                done_tokens_for_target = max(0, int(total_tokens_for_target - long_len))
                state.phase1_virtual_token_caps[str(cohort.long_req_id)] = int(best_chunk)
                state.phase1_public_skip_rewrite_requests.add(str(cohort.long_req_id))
                state.metrics.record_phase1_virtual_cap_probe(target_set=True)
                state.metrics.record_phase1_step_trace(
                    request_id=str(cohort.long_req_id),
                    event="phase1_target_set",
                    is_prefill=True,
                    num_computed_tokens=done_tokens_for_target,
                    uncached=int(long_len),
                    cached=0,
                    target_chunk=int(best_chunk),
                )
                logger.info(
                    "[Wave-Slice][P1-virtual-target] req=%s target_chunk=%d long_len=%d baseline_chunk=%s",
                    str(cohort.long_req_id),
                    int(best_chunk),
                    int(long_len),
                    str(baseline_chunk) if baseline_chunk is not None else "none",
                )
            for secondary_req_id, secondary_chunk in secondary_lora_caps.items():
                if int(secondary_chunk) <= 0:
                    continue
                existing_cap = state.phase1_virtual_token_caps.get(str(secondary_req_id))
                if existing_cap is not None and int(existing_cap) <= int(secondary_chunk):
                    continue
                secondary_seq_group = _phase1_find_seq_group_by_request_id(
                    snapshot,
                    str(secondary_req_id),
                    request_id_getter=_safe_request_id,
                )
                secondary_total = int(_safe_total_tokens(secondary_seq_group) or 0) if secondary_seq_group is not None else 0
                secondary_remaining = int(_safe_remaining_tokens(secondary_seq_group) or secondary_total)
                secondary_done = max(0, int(secondary_total) - int(secondary_remaining))
                state.phase1_virtual_token_caps[str(secondary_req_id)] = int(secondary_chunk)
                state.phase1_public_skip_rewrite_requests.add(str(secondary_req_id))
                state.metrics.record_phase1_virtual_cap_probe(target_set=True)
                state.metrics.record_phase1_step_trace(
                    request_id=str(secondary_req_id),
                    event="phase1_secondary_target_set",
                    is_prefill=True,
                    num_computed_tokens=int(secondary_done),
                    uncached=max(1, int(secondary_remaining)),
                    cached=0,
                    target_chunk=int(secondary_chunk),
                )
                logger.info(
                    "[Wave-Slice][P1-secondary-target] req=%s target_chunk=%d",
                    str(secondary_req_id),
                    int(secondary_chunk),
                )
            use_seq_len_shadow = not bool(state.policy.phase1_ingress_direct_authoritative)
            if use_seq_len_shadow and long_seq_group is not None and best_chunk < long_len:
                _phase1_apply_sequence_len_shadow(
                    state=state,
                    seq_group=long_seq_group,
                    target_chunk=best_chunk,
                )

            if can_phase1_tick_hide and waiting_short_count > 0:
                force_joint_lora_tick_hide = _phase12_should_force_lora_tick_hide(
                    state=state,
                    lora_mode_enabled=lora_mode_enabled,
                    waiting_short_count=waiting_short_count,
                )
                first_wait_len = _safe_remaining_tokens(waiting[0]) or 0
                if force_joint_lora_tick_hide or (
                    first_wait_len > 0 and first_wait_len <= phase1_tick_hide_keep_ceiling
                ):
                    if force_joint_lora_tick_hide:
                        state.metrics.record_phase1_step_trace(
                            request_id=str(cohort.long_req_id or ""),
                            event="phase12_joint_lora_tick_hide_force",
                            is_prefill=True,
                            num_computed_tokens=0,
                            uncached=int(long_len),
                            cached=0,
                            target_chunk=int(phase1_tick_hide_keep_ceiling),
                        )
                    new_waiting_items: list[Any] = []
                    for seq_group in waiting:
                        remaining = _safe_remaining_tokens(seq_group) or 0
                        if remaining > phase1_tick_hide_keep_ceiling:
                            hidden_long_waiting_tasks.append(seq_group)
                        else:
                            new_waiting_items.append(seq_group)
                    if hidden_long_waiting_tasks:
                        self.waiting = _rebuild_queue_like(waiting, new_waiting_items)
                        waiting = self.waiting
                    new_running_items: list[Any] = []
                    for seq_group in running:
                        remaining = _safe_remaining_tokens(seq_group) or 0
                        if remaining > phase1_tick_hide_keep_ceiling:
                            hidden_long_tasks.append(seq_group)
                        else:
                            new_running_items.append(seq_group)
                    self.running = _rebuild_queue_like(running, new_running_items)

            new_threshold = None
            if can_phase1_threshold and scheduler_cfg is not None:
                new_threshold = _compute_long_prefill_threshold(
                    best_chunk,
                    original_threshold,
                    self,
                )
                if new_threshold is not None:
                    scheduler_cfg.long_prefill_token_threshold = new_threshold

            new_budget = None
            if can_phase1_budget and scheduler_cfg is not None:
                if explicit_plan_kind is not None:
                    new_budget = _compute_explicit_plan_budget(
                        best_chunk=best_chunk,
                        short_len=short_len,
                        short_token_mass=short_token_mass,
                        policy=state.policy,
                        original_budget=original_budget,
                        baseline_chunk=baseline_chunk,
                    )
                else:
                    new_budget = _compute_budget(
                        best_chunk,
                        short_len,
                        long_len,
                        short_token_mass,
                        queue_len,
                        state.policy,
                        original_budget,
                        baseline_chunk=baseline_chunk,
                    )
                if new_budget is not None:
                    scheduler_cfg.max_num_batched_tokens = new_budget

            logger.info(
                "[Wave-Slice][P1] model=%s long=%d short=%d baseline_chunk=%s chunk=%d forced=%s explicit_plan=%s queued=%d short_mass=%d hidden=%d threshold=%s budget=%s",
                state.model_name,
                long_len,
                short_len,
                str(baseline_chunk) if baseline_chunk is not None else "none",
                best_chunk,
                f"{forced_chunk}|sticky={reused_sticky}|joint_floor={joint_floor_chunk}",
                str(explicit_plan_kind) if explicit_plan_kind is not None else "off",
                queue_len,
                short_token_mass,
                len(hidden_long_tasks) + len(hidden_long_waiting_tasks),
                str(new_threshold) if new_threshold is not None else "unchanged",
                str(new_budget) if new_budget is not None else "unchanged",
            )
            if runtime_adaptive_meta:
                logger.info(
                    "[Wave-Slice][P1-runtime-adapt] pressure=%.3f wall=%.3f urgency=%.3f target_fraction=%.3f target_chunk=%d phase2_ratio=%.2f phase2_pressure=%.2f",
                    float(runtime_adaptive_meta.get("effective_pressure", 0.0)),
                    float(runtime_adaptive_meta.get("wall_pressure", 0.0)),
                    float(runtime_adaptive_meta.get("short_urgency", 0.0)),
                    float(runtime_adaptive_meta.get("phase1_target_long_fraction", 0.0)),
                    int(runtime_adaptive_meta.get("phase1_ingress_target_chunk", 0.0)),
                    float(runtime_adaptive_meta.get("phase2_min_hetero_ratio", 0.0)),
                    float(runtime_adaptive_meta.get("phase2_min_pressure_ratio", 0.0)),
                )
            self.running, self.waiting, hidden_phase2_running, hidden_phase2_waiting, _ = _phase12_apply_scheduler_cashout_to_queues(
                state=state,
                running=self.running,
                waiting=self.waiting,
            )
            _reconcile_v1_waiting_running_status(
                self,
                state,
                context="phase12_pre_native",
            )
            outputs = schedule_impl(self, *args, **kwargs)
            outputs = _maybe_apply_phase2_schedule_cashout(outputs)
            outputs, rewritten, rewritten_groups, old_chunk_sum, new_chunk_sum, token_delta_sum = _phase1_rewrite_scheduler_outputs(
                outputs=outputs,
                request_id=cohort.long_req_id,
                target_chunk=best_chunk,
            )
            state.metrics.record_phase1_rewrite(
                rewritten_groups=rewritten_groups,
                old_chunk_sum=old_chunk_sum,
                new_chunk_sum=new_chunk_sum,
                token_delta_sum=token_delta_sum,
            )
            if rewritten:
                logger.info(
                    "[Wave-Slice][P1-rewrite] req=%s chunk=%d explicit_plan=%s",
                    str(cohort.long_req_id),
                    int(best_chunk),
                    str(explicit_plan_kind) if explicit_plan_kind is not None else "off",
                )
            return outputs
        finally:
            try:
                _restore_v1_waiting_cap_runtime_hooks(
                    waiting_cap_cleanup_requests if "waiting_cap_cleanup_requests" in locals() else [],
                    waiting_cap_restore_specs if "waiting_cap_restore_specs" in locals() else [],
                )
            except Exception:
                pass
            state.phase1_shadow_seq_lens.clear()
            state.phase1_virtual_token_caps.clear()
            if "runtime_base_policy" in locals() and runtime_base_policy is not None:
                state.policy = runtime_base_policy
            if scheduler_cfg is not None and isinstance(original_budget, int) and original_budget > 0:
                try:
                    scheduler_cfg.max_num_batched_tokens = original_budget
                except Exception:
                    pass
            if scheduler_cfg is not None and isinstance(original_threshold, int) and original_threshold >= 0:
                try:
                    scheduler_cfg.long_prefill_token_threshold = original_threshold
                except Exception:
                    pass
            if hidden_long_tasks and hasattr(self, "running"):
                try:
                    self.running.extend(hidden_long_tasks)
                except Exception:
                    pass
            if hidden_long_waiting_tasks and hasattr(self, "waiting"):
                try:
                    self.waiting = _restore_hidden_queue_items(
                        self.waiting,
                        hidden_long_waiting_tasks,
                        queue_rebuilder=_rebuild_queue_like,
                    )
                except Exception:
                    pass
            if hidden_phase2_running and hasattr(self, "running"):
                try:
                    self.running = _restore_hidden_queue_items(
                        self.running,
                        hidden_phase2_running,
                        queue_rebuilder=_rebuild_queue_like,
                    )
                except Exception:
                    pass
            if hidden_phase2_waiting and hasattr(self, "waiting"):
                try:
                    self.waiting = _restore_hidden_queue_items(
                        self.waiting,
                        hidden_phase2_waiting,
                        queue_rebuilder=_rebuild_queue_like,
                    )
                except Exception:
                    pass

    _wave_schedule_hook.__wave_slice_hook__ = True  # type: ignore[attr-defined]
    return _wave_schedule_hook


def _build_public_schedule_hook(state: _PatchState) -> Callable[..., Any]:
    original_public_schedule = state.original_public_schedule
    if original_public_schedule is None:
        raise RuntimeError("public schedule hook requested without original_public_schedule")
    schedule_globals = getattr(original_public_schedule, "__globals__", {})
    SequenceGroupMetadata = schedule_globals.get("SequenceGroupMetadata")
    SequenceGroupMetadataDelta = schedule_globals.get("SequenceGroupMetadataDelta")
    SequenceStatus = schedule_globals.get("SequenceStatus")
    if SequenceGroupMetadata is None or SequenceGroupMetadataDelta is None or SequenceStatus is None:
        raise RuntimeError("public schedule hook missing vLLM scheduler globals")

    @functools.wraps(original_public_schedule)
    def _wave_public_schedule_hook(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            scheduler_start_time = time.perf_counter()
            hidden_phase2_running: list[Any] = []
            hidden_phase2_waiting: list[Any] = []
            restore_phase2_running = None
            restore_phase2_waiting = None
            if _phase2_scheduler_cashout_enabled(state.policy) and _has_running_waiting_queues(self):
                try:
                    restore_phase2_running, restore_phase2_waiting = _capture_queue_pair(self)
                    self.running, self.waiting, hidden_phase2_running, hidden_phase2_waiting, _ = (
                        _phase12_apply_scheduler_cashout_to_queues(
                            state=state,
                            running=self.running,
                            waiting=self.waiting,
                        )
                    )
                except Exception:
                    logger.exception("[Wave-Slice] public pre-schedule cashout failed.")
                    hidden_phase2_running = []
                    hidden_phase2_waiting = []
            try:
                scheduler_outputs = getattr(self, state.scheduler_method_name)(*args, **kwargs)
            finally:
                try:
                    _restore_queue_pair(self, restore_phase2_running, restore_phase2_waiting)
                except Exception:
                    pass
            now = time.time()

            if state.policy.enable_phase1_scheduler:
                scheduler_outputs, _ = _phase1_force_public_schedule_rewrite(
                    state=state,
                    scheduler_obj=self,
                    scheduler_outputs=scheduler_outputs,
                )
            if bool(state.policy.phase2_enable_scheduler_cashout):
                scheduler_outputs, _ = _phase12_scheduler_cashout_rewrite(
                    state=state,
                    scheduler_outputs=scheduler_outputs,
                )

            allow_async_output_proc = self.use_async_output_proc
            seq_group_metadata_list: list[Any] = []
            for scheduled_seq_group in scheduler_outputs.scheduled_seq_groups:
                seq_group = scheduled_seq_group.seq_group
                seq_group_metadata = _build_public_seq_group_metadata(
                    scheduler_obj=self,
                    scheduled_seq_group=scheduled_seq_group,
                    scheduler_outputs=scheduler_outputs,
                    now=now,
                    state=state,
                    sequence_status_cls=SequenceStatus,
                    sequence_group_metadata_cls=SequenceGroupMetadata,
                    sequence_group_metadata_delta_cls=SequenceGroupMetadataDelta,
                    request_id_getter=_safe_request_id,
                )
                seq_group_metadata_list.append(seq_group_metadata)

                if allow_async_output_proc:
                    allow_async_output_proc = self._allow_async_output_proc(seq_group)

            for scheduled_seq_group in scheduler_outputs.scheduled_seq_groups:
                self.block_manager.mark_blocks_as_computed(
                    scheduled_seq_group.seq_group,
                    scheduled_seq_group.token_chunk_size,
                )

            self._seq_group_metadata_cache[self.next_cache_id].reset()

            scheduler_time = time.perf_counter() - scheduler_start_time
            for seq_group in self.running:
                if seq_group is not None and seq_group.metrics is not None:
                    if seq_group.metrics.scheduler_time is not None:
                        seq_group.metrics.scheduler_time += scheduler_time
                    else:
                        seq_group.metrics.scheduler_time = scheduler_time

            self.cache_id = self.next_cache_id
            return (seq_group_metadata_list, scheduler_outputs, allow_async_output_proc)
        except Exception:
            logger.exception("[Wave-Slice] public schedule rewrite failed; fallback to original tuple.")
            return original_public_schedule(self, *args, **kwargs)
        finally:
            state.phase1_public_skip_rewrite_requests.clear()

    _wave_public_schedule_hook.__wave_slice_public_hook__ = True  # type: ignore[attr-defined]
    return _wave_public_schedule_hook


def _build_model_runner_hook(state: _PatchState) -> Callable[..., Any]:
    original_execute = state.original_execute_model
    if original_execute is None:
        raise RuntimeError("internal error: original execute_model is missing")

    @functools.wraps(original_execute)
    def _wave_execute_model_hook(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not state.policy.enable_phase2_modelrunner:
            return original_execute(self, *args, **kwargs)
        if _phase2_modelrunner_passthrough(state.policy):
            return original_execute(self, *args, **kwargs)

        runtime_base_policy: Optional[WaveSlicePolicy] = None
        if (
            bool(getattr(state.policy, "phase2_runtime_adaptive_enabled", False))
            and bool(getattr(state.policy, "phase1_runtime_adaptive_enabled", False))
            and getattr(state, "phase1_runtime_last_meta", None)
        ):
            try:
                adapted_policy, _ = _phase1_runtime_adapt_policy(
                    state.policy,
                    dict(state.phase1_runtime_last_meta or {}),
                )
                if adapted_policy is not state.policy:
                    runtime_base_policy = state.policy
                    state.policy = adapted_policy
            except Exception:
                logger.exception("[Wave-Slice] modelrunner runtime gate adaptation failed; continue with base policy.")

        try:
            model_input = args[0] if len(args) > 0 else kwargs.get("model_input")

            if bool(getattr(state.policy, "phase2_enable_execution_escape", False)):
                try:
                    split_ids = None
                    model_input_is_v1 = _record_phase2_v1_output_probe(
                        state,
                        context="execution_escape",
                        model_input=model_input,
                    )
                    if model_input_is_v1:
                        split_ids = _v1_execution_escape_req_ids(
                            model_input,
                            state.policy,
                            runner_self=self,
                            state=state,
                            req_info_collector=lambda model_input, runner_self, state, policy: _phase12_collect_scheduled_req_infos(
                                model_input,
                                runner_self=runner_self,
                                state=state,
                                policy=policy,
                                rank_infer=_infer_lora_rank,
                            ),
                            beneficiary_signal_builder=lambda state, policy, req_infos: _phase12_beneficiary_signal(
                                state=state,
                                policy=policy,
                                req_infos=req_infos,
                            ),
                        )
                    if split_ids is not None:
                        active_ids, deferred_ids = split_ids
                        activated = _phase12_activate_execution_escape(
                            state,
                            active_ids=active_ids,
                            deferred_ids=deferred_ids,
                        )
                        state.metrics.record_phase2_decision(
                            activated,
                            "execution_escape_activate" if activated else "execution_escape_noop",
                            selected_count=int(len(active_ids or [])),
                        )
                    else:
                        # V0 model_input does not expose the V1 scheduler output shape
                        # needed by the execution-escape lane. Fall back to the legacy
                        # Phase-II decision path below instead of short-circuiting the
                        # whole Phase-II block as a permanent no-candidate.
                        state.metrics.record_phase2_debug_counter("execution_escape_fallback_steps")
                        state.metrics.record_phase2_decision(False, "execution_escape_v0_fallback")
                except Exception:
                    logger.exception("Wave-Slice execution escape activation failed; falling back to original execute.")
                    exc_type, exc, _tb = sys.exc_info()
                    state.metrics.record_phase2_decision(
                        False,
                        "execution_escape_exception",
                        exception_type=str(getattr(exc_type, "__name__", "UnknownError")),
                        exception_message=str(exc)[:240] if exc is not None else "",
                    )
                    return original_execute(self, *args, **kwargs)
                if model_input_is_v1:
                    state.metrics.record_phase2_debug_counter("execution_escape_original_execute_returns")
                    return original_execute(self, *args, **kwargs)

            # Non-current legacy stream rebinding was removed. The current Phase II
            # path is execution escape / scheduler-side promotion only.
            state.metrics.record_phase2_decision(False, "phase2_no_current_modelrunner_path")
            return original_execute(self, *args, **kwargs)
        finally:
            if runtime_base_policy is not None:
                state.policy = runtime_base_policy

    _wave_execute_model_hook.__wave_slice_phase2_hook__ = True  # type: ignore[attr-defined]
    return _wave_execute_model_hook


def _build_add_request_hook(state: _PatchState) -> Callable[..., Any]:
    return _build_add_request_hook_impl(
        state,
        estimate_prompt_tokens=_estimate_prompt_tokens,
        estimate_solo_us=_estimate_solo_us,
        lookup_engine_prompt_tokens=_lookup_engine_prompt_tokens,
        phase1_maybe_seed_ingress_virtual=_phase1_maybe_seed_ingress_virtual,
    )


def _build_add_processed_request_hook(state: _PatchState) -> Callable[..., Any]:
    return _build_add_processed_request_hook_impl(
        state,
        estimate_solo_us=_estimate_solo_us,
        phase1_maybe_seed_ingress_virtual=_phase1_maybe_seed_ingress_virtual,
    )


def _build_v1_process_inputs_hook(state: _PatchState) -> Callable[..., Any]:
    return _build_v1_process_inputs_hook_impl(
        state,
        estimate_solo_us=_estimate_solo_us,
        phase1_maybe_seed_ingress_virtual=_phase1_maybe_seed_ingress_virtual,
    )


def _build_v1_engine_core_add_request_hook(state: _PatchState) -> Callable[..., Any]:
    return _build_v1_engine_core_add_request_hook_impl(
        state,
        estimate_solo_us=_estimate_solo_us,
        phase1_maybe_seed_ingress_virtual=_phase1_maybe_seed_ingress_virtual,
    )


def _build_step_hook(state: _PatchState) -> Callable[..., Any]:
    return _build_step_hook_impl(
        state,
        maybe_install_v1_runtime_lifecycle_hooks=_maybe_install_v1_runtime_lifecycle_hooks,
        phase12_clear_escape_lane=_phase12_clear_escape_lane,
    )


def _maybe_install_v1_runtime_lifecycle_hooks(state: _PatchState) -> None:
    return _maybe_install_v1_runtime_lifecycle_hooks_impl(
        state,
        patch_lock=_PATCH_LOCK,
        load_v1_output_processor_cls=_load_v1_output_processor_cls,
        build_output_processor_add_request_hook=_build_output_processor_add_request_hook,
        build_output_processor_process_outputs_hook=lambda st: _build_output_processor_process_outputs_hook(
            st,
            phase12_clear_escape_lane=_phase12_clear_escape_lane,
        ),
        build_v1_scheduler_update_from_output_hook=lambda st: _build_v1_scheduler_update_from_output_hook(
            st,
            phase12_clear_escape_lane=_phase12_clear_escape_lane,
        ),
        build_v1_scheduler_finish_requests_hook=lambda st: _build_v1_scheduler_finish_requests_hook(
            st,
            phase12_clear_escape_lane=_phase12_clear_escape_lane,
        ),
    )


def _build_output_processor_add_request_hook(state: _PatchState) -> Callable[..., Any]:
    return _build_output_processor_add_request_hook_impl(state)


def _build_output_processor_process_outputs_hook(state: _PatchState) -> Callable[..., Any]:
    return _build_output_processor_process_outputs_hook_impl(
        state,
        phase12_clear_escape_lane=_phase12_clear_escape_lane,
    )


def _build_v1_scheduler_update_from_output_hook(state: _PatchState) -> Callable[..., Any]:
    return _build_v1_scheduler_update_from_output_hook_impl(
        state,
        phase12_clear_escape_lane=_phase12_clear_escape_lane,
    )


def _build_v1_scheduler_finish_requests_hook(state: _PatchState) -> Callable[..., Any]:
    return _build_v1_scheduler_finish_requests_hook_impl(
        state,
        phase12_clear_escape_lane=_phase12_clear_escape_lane,
    )


def _build_lora_logits_compat_hook(state: _PatchState) -> Callable[..., Any]:
    original = state.original_lora_get_logits
    if original is None:
        raise RuntimeError("internal error: original _get_logits is missing")

    glb = getattr(original, "__globals__", {})
    _apply_lora = glb.get("_apply_lora", None)
    tensor_model_parallel_gather = glb.get("tensor_model_parallel_gather", None)
    torch_mod = glb.get("torch", None)

    @functools.wraps(original)
    def _wave_logits_compat(self: Any, hidden_states: Any, embedding: Any, embedding_bias: Any = None):
        try:
            return original(self, hidden_states, embedding, embedding_bias)
        except RuntimeError as exc:
            msg = str(exc)
            # Only patch known vLLM 0.4.x LoRA shape mismatch patterns.
            if (
                "expanded size of the tensor" not in msg
                and "CHECK_EQ(indicies.size(0), x.size(0))" not in msg
            ):
                raise

            if torch_mod is None or tensor_model_parallel_gather is None or _apply_lora is None:
                raise

            # Safe fallback path mirrors vLLM logic with shape guards.
            logits = torch_mod.matmul(hidden_states, embedding.t())
            if embedding_bias is not None:
                logits += embedding_bias
            logits = tensor_model_parallel_gather(logits)
            if logits is None:
                return None

            lora_logits = torch_mod.empty(
                self.embeddings_tensors.shape[0] + 1,
                self.embeddings_tensors.shape[1],
                hidden_states.shape[0],
                dtype=self.embeddings_tensors.dtype,
                device=self.embeddings_tensors.device,
            )
            torch_mod.matmul(self.embeddings_tensors, hidden_states.T, out=lora_logits[:-1])
            lora_logits[-1] = float("-inf")
            lora_logits = lora_logits.mT
            lora_logits = (
                lora_logits.reshape(
                    lora_logits.shape[0] * lora_logits.shape[1],
                    lora_logits.shape[2],
                )
                .index_select(0, self.indices_padded[: self.indices_len[2]])
                .nan_to_num_(nan=float("-inf"), posinf=float("inf"), neginf=float("-inf"))
            )

            row_n = min(int(logits.shape[0]), int(lora_logits.shape[0]))
            col_n = min(
                int(logits.shape[1] - self.base_layer.org_vocab_size),
                int(lora_logits.shape[1]),
            )
            if row_n > 0 and col_n > 0:
                start = self.base_layer.org_vocab_size
                logits[:row_n, start : start + col_n] = lora_logits[:row_n, :col_n]

            active_indices = self.indices[: self.indices_len[1]]
            if hidden_states.shape[0] == 0:
                active_indices = active_indices[:0]
            elif active_indices.shape[0] > hidden_states.shape[0]:
                active_indices = active_indices[: hidden_states.shape[0]]

            if hidden_states.shape[0] > 0 and active_indices.shape[0] > 0:
                _apply_lora(
                    hidden_states,
                    self.lora_a_stacked,
                    self.lora_b_stacked,
                    active_indices,
                    logits,
                )

            logits = logits[:, : self.base_layer.vocab_size]
            logger.warning("[Wave-Slice] applied LoRA logits compat fallback for vLLM runtime mismatch.")
            return logits

    _wave_logits_compat.__wave_slice_lora_compat_hook__ = True  # type: ignore[attr-defined]
    return _wave_logits_compat
def _maybe_auto_inject_from_env() -> None:
    if os.environ.get(_AUTO_ENV_ENABLED, "").strip() != "1":
        return
    with _PATCH_LOCK:
        if _PATCH_STATE is not None:
            return
    model_name = os.environ.get(_AUTO_ENV_MODEL, "").strip()
    if not model_name:
        return
    gamma_raw = os.environ.get(_AUTO_ENV_GAMMA, "2.0").strip()
    policy_raw = os.environ.get(_AUTO_ENV_POLICY, "").strip()
    try:
        gamma = float(gamma_raw)
    except Exception:
        gamma = 2.0
    try:
        policy_payload = json.loads(policy_raw) if policy_raw else {}
        policy = WaveSlicePolicy(**policy_payload)
    except Exception:
        logger.exception("[Wave-Slice] failed to decode auto-inject policy; using defaults.")
        policy = WaveSlicePolicy()
    try:
        inject_wave_slice(model_name=model_name, gamma=gamma, policy=policy, force=False)
        logger.info("[Wave-Slice] auto-injected from child-process env for model=%s", model_name)
    except Exception:
        logger.exception("[Wave-Slice] child-process auto-inject failed.")


def waveslice_vllm_general_plugin() -> None:
    """General-plugin entrypoint executed inside vLLM child processes."""
    _maybe_auto_inject_from_env()


def inject_wave_slice(
    model_name: str,
    *,
    gamma: float = 2.0,
    policy: Optional[WaveSlicePolicy] = None,
    force: bool = False,
) -> None:
    """Install Wave-Slice hooks into vLLM runtime."""
    global _PATCH_STATE

    scheduler_cls, scheduler_method_name = _load_scheduler_target()
    chosen_policy = policy or WaveSlicePolicy()

    with _PATCH_LOCK:
        if _PATCH_STATE is not None and not force:
            logger.info("[Wave-Slice] already injected for model=%s", _PATCH_STATE.model_name)
            return
        if _PATCH_STATE is not None and force:
            uninject_wave_slice()

        current_schedule = getattr(scheduler_cls, scheduler_method_name, None)
        if current_schedule is None or not callable(current_schedule):
            raise RuntimeError(f"Scheduler.{scheduler_method_name} is missing or not callable.")

        objective_mode = str(chosen_policy.scheduler_objective_mode).strip().lower()
        if objective_mode not in {"fair_escape", "pure_gain"}:
            logger.warning(
                "[Wave-Slice] unknown scheduler_objective_mode=%s, fallback to fair_escape",
                chosen_policy.scheduler_objective_mode,
            )
            objective_mode = "fair_escape"

        state = _PatchState(
            scheduler_cls=scheduler_cls,
            scheduler_method_name=scheduler_method_name,
            original_schedule=current_schedule,
            brain=WaveScheduler(model_name=model_name, gamma=gamma, objective_mode=objective_mode),
            policy=chosen_policy,
            model_name=model_name,
            metrics=WaveSliceMetrics(short_threshold_tokens=chosen_policy.metrics_short_request_tokens),
        )

        try:
            sequence_data_cls = _load_sequence_data_cls()
            original_get_len = getattr(sequence_data_cls, "get_len", None)
            if callable(original_get_len):
                state.sequence_data_cls = sequence_data_cls
                state.original_sequence_data_get_len = original_get_len
                sequence_data_cls.get_len = _build_sequence_data_get_len_hook(state)
        except Exception:
            logger.exception("[Wave-Slice] failed to install SequenceData.get_len hook.")

        try:
            v1_request_cls = _load_v1_request_cls()
            num_tokens_prop = getattr(v1_request_cls, "num_tokens", None)
            num_tokens_with_spec_prop = getattr(v1_request_cls, "num_tokens_with_spec", None)
            if isinstance(num_tokens_prop, property) and isinstance(num_tokens_with_spec_prop, property):
                state.v1_request_cls = v1_request_cls
                state.original_v1_request_num_tokens = num_tokens_prop
                state.original_v1_request_num_tokens_with_spec = num_tokens_with_spec_prop
                v1_request_cls.num_tokens = _build_v1_request_num_tokens_hook(state)
                v1_request_cls.num_tokens_with_spec = _build_v1_request_num_tokens_with_spec_hook(state)
        except Exception:
            logger.exception("[Wave-Slice] failed to install v1 Request token-cap hooks.")

        setattr(scheduler_cls, scheduler_method_name, _build_scheduler_hook(state))
        scheduler_add_request = getattr(scheduler_cls, "add_request", None)
        if scheduler_method_name == "schedule" and callable(scheduler_add_request):
            state.original_scheduler_add_request = scheduler_add_request
            setattr(scheduler_cls, "add_request", _build_v1_scheduler_add_request_hook(state))
            scheduler_update_after_schedule = getattr(scheduler_cls, "_update_after_schedule", None)
            if callable(scheduler_update_after_schedule):
                state.original_scheduler_update_after_schedule = scheduler_update_after_schedule
                setattr(
                    scheduler_cls,
                    "_update_after_schedule",
                    _build_v1_scheduler_update_after_schedule_hook(state),
                )
        if scheduler_method_name != "schedule":
            public_schedule = getattr(scheduler_cls, "schedule", None)
            if callable(public_schedule):
                state.original_public_schedule = public_schedule
                setattr(scheduler_cls, "schedule", _build_public_schedule_hook(state))
        scheduler_helper = getattr(scheduler_cls, "_get_num_new_uncached_and_cached_tokens", None)
        if callable(scheduler_helper):
            state.original_get_new_uncached_and_cached_tokens = scheduler_helper
            setattr(
                scheduler_cls,
                "_get_num_new_uncached_and_cached_tokens",
                _build_get_num_new_uncached_and_cached_tokens_hook(state),
            )

        # Phase II: ModelRunner hook (optional)
        if chosen_policy.enable_phase2_modelrunner:
            try:
                model_runner_cls = _load_model_runner_cls()
                original_execute = getattr(model_runner_cls, "execute_model", None)
                if original_execute and callable(original_execute):
                    state.model_runner_cls = model_runner_cls
                    state.original_execute_model = original_execute
                    model_runner_cls.execute_model = _build_model_runner_hook(state)
                else:
                    logger.warning("[Wave-Slice] skip Phase II: ModelRunner.execute_model not callable.")
            except Exception:
                logger.exception("[Wave-Slice] Phase II injection failed; continue with Phase I.")

        # Metrics hooks: LLMEngine.add_request + step (optional)
        if chosen_policy.enable_metrics_hook:
            try:
                llm_engine_cls = _load_llm_engine_cls()
                original_add_request = getattr(llm_engine_cls, "add_request", None)
                original_add_processed_request = getattr(llm_engine_cls, "_add_processed_request", None)
                original_step = getattr(llm_engine_cls, "step", None)
                if callable(original_add_request) and callable(original_step):
                    state.llm_engine_cls = llm_engine_cls
                    state.original_add_request = original_add_request
                    if callable(original_add_processed_request):
                        state.original_add_processed_request = original_add_processed_request
                        llm_engine_cls._add_processed_request = _build_add_processed_request_hook(state)
                    state.original_step = original_step
                    llm_engine_cls.add_request = _build_add_request_hook(state)
                    llm_engine_cls.step = _build_step_hook(state)
                else:
                    logger.warning("[Wave-Slice] skip metrics hooks: add_request/step missing.")
            except Exception:
                logger.exception("[Wave-Slice] metrics hook injection failed; continue without metrics hooks.")
            try:
                v1_processor_cls = _load_v1_processor_cls()
                original_process_inputs = getattr(v1_processor_cls, "process_inputs", None)
                if callable(original_process_inputs):
                    state.v1_processor_cls = v1_processor_cls
                    state.original_v1_processor_process_inputs = original_process_inputs
                    v1_processor_cls.process_inputs = _build_v1_process_inputs_hook(state)
            except Exception:
                logger.exception("[Wave-Slice] failed to install v1 Processor.process_inputs hook.")
            try:
                v1_engine_core_cls = _load_v1_engine_core_cls()
                original_engine_core_add_request = getattr(v1_engine_core_cls, "add_request", None)
                if callable(original_engine_core_add_request):
                    state.v1_engine_core_cls = v1_engine_core_cls
                    state.original_v1_engine_core_add_request = original_engine_core_add_request
                    v1_engine_core_cls.add_request = _build_v1_engine_core_add_request_hook(state)
            except Exception:
                logger.exception("[Wave-Slice] failed to install v1 EngineCore.add_request hook.")

        # v1 lifecycle hooks are installed lazily from the first engine.step call.
        # Eagerly importing/patching v1 engine/output modules before the runtime is
        # fully initialized can interfere with CUDA/fork startup in v1.

        # Compatibility hook for known LoRA shape mismatches in some vLLM versions.
        if chosen_policy.enable_vllm_lora_compat_patch:
            try:
                lora_logits_cls = _load_logits_processor_lora_cls()
                original_get_logits = getattr(lora_logits_cls, "_get_logits", None)
                if callable(original_get_logits):
                    state.logits_processor_lora_cls = lora_logits_cls
                    state.original_lora_get_logits = original_get_logits
                    lora_logits_cls._get_logits = _build_lora_logits_compat_hook(state)
                else:
                    logger.warning("[Wave-Slice] skip LoRA compat patch: _get_logits missing.")
            except Exception:
                logger.exception("[Wave-Slice] failed to install LoRA compat patch.")

        _PATCH_STATE = state
        _publish_autoinject_env(model_name, gamma, chosen_policy)
        logger.info(
            "[Wave-Slice] injected model=%s scheduler=%s.%s phase2=%s dispatch=%s metrics=%s",
            model_name,
            scheduler_cls.__module__,
            scheduler_method_name,
            str(chosen_policy.enable_phase2_modelrunner),
            chosen_policy.phase2_dispatch_mode,
            str(chosen_policy.enable_metrics_hook),
        )


def uninject_wave_slice() -> None:
    """Restore all patched vLLM runtime methods."""
    global _PATCH_STATE

    with _PATCH_LOCK:
        if _PATCH_STATE is None:
            _clear_autoinject_env()
            return
        state = _PATCH_STATE

        try:
            setattr(state.scheduler_cls, state.scheduler_method_name, state.original_schedule)
        except Exception:
            logger.exception(
                "[Wave-Slice] failed to restore Scheduler.%s",
                state.scheduler_method_name,
            )
        if state.original_scheduler_add_request is not None:
            try:
                setattr(state.scheduler_cls, "add_request", state.original_scheduler_add_request)
            except Exception:
                logger.exception("[Wave-Slice] failed to restore Scheduler.add_request")
        if state.original_scheduler_update_after_schedule is not None:
            try:
                setattr(state.scheduler_cls, "_update_after_schedule", state.original_scheduler_update_after_schedule)
            except Exception:
                logger.exception("[Wave-Slice] failed to restore Scheduler._update_after_schedule")
        if state.original_public_schedule is not None:
            try:
                setattr(state.scheduler_cls, "schedule", state.original_public_schedule)
            except Exception:
                logger.exception("[Wave-Slice] failed to restore Scheduler.schedule")
        if state.original_get_new_uncached_and_cached_tokens is not None:
            try:
                setattr(
                    state.scheduler_cls,
                    "_get_num_new_uncached_and_cached_tokens",
                    state.original_get_new_uncached_and_cached_tokens,
                )
            except Exception:
                logger.exception(
                    "[Wave-Slice] failed to restore Scheduler._get_num_new_uncached_and_cached_tokens"
                )

        if state.model_runner_cls is not None and state.original_execute_model is not None:
            try:
                state.model_runner_cls.execute_model = state.original_execute_model
            except Exception:
                logger.exception("[Wave-Slice] failed to restore ModelRunner.execute_model")

        if state.sequence_data_cls is not None and state.original_sequence_data_get_len is not None:
            try:
                state.sequence_data_cls.get_len = state.original_sequence_data_get_len
            except Exception:
                logger.exception("[Wave-Slice] failed to restore SequenceData.get_len")

        if state.v1_request_cls is not None:
            if state.original_v1_request_num_tokens is not None:
                try:
                    state.v1_request_cls.num_tokens = state.original_v1_request_num_tokens
                except Exception:
                    logger.exception("[Wave-Slice] failed to restore Request.num_tokens")
            if state.original_v1_request_num_tokens_with_spec is not None:
                try:
                    state.v1_request_cls.num_tokens_with_spec = state.original_v1_request_num_tokens_with_spec
                except Exception:
                    logger.exception("[Wave-Slice] failed to restore Request.num_tokens_with_spec")

        if state.llm_engine_cls is not None:
            if state.original_add_request is not None:
                try:
                    state.llm_engine_cls.add_request = state.original_add_request
                except Exception:
                    logger.exception("[Wave-Slice] failed to restore LLMEngine.add_request")
            if state.original_add_processed_request is not None:
                try:
                    state.llm_engine_cls._add_processed_request = state.original_add_processed_request
                except Exception:
                    logger.exception("[Wave-Slice] failed to restore LLMEngine._add_processed_request")
            if state.original_step is not None:
                try:
                    state.llm_engine_cls.step = state.original_step
                except Exception:
                    logger.exception("[Wave-Slice] failed to restore LLMEngine.step")

        if state.v1_processor_cls is not None and state.original_v1_processor_process_inputs is not None:
            try:
                state.v1_processor_cls.process_inputs = state.original_v1_processor_process_inputs
            except Exception:
                logger.exception("[Wave-Slice] failed to restore v1 Processor.process_inputs")

        if state.v1_engine_core_cls is not None and state.original_v1_engine_core_add_request is not None:
            try:
                state.v1_engine_core_cls.add_request = state.original_v1_engine_core_add_request
            except Exception:
                logger.exception("[Wave-Slice] failed to restore v1 EngineCore.add_request")

        if state.v1_output_processor_cls is not None:
            if state.original_output_processor_add_request is not None:
                try:
                    state.v1_output_processor_cls.add_request = state.original_output_processor_add_request
                except Exception:
                    logger.exception("[Wave-Slice] failed to restore OutputProcessor.add_request")
            if state.original_output_processor_process_outputs is not None:
                try:
                    state.v1_output_processor_cls.process_outputs = state.original_output_processor_process_outputs
                except Exception:
                    logger.exception("[Wave-Slice] failed to restore OutputProcessor.process_outputs")

        if state.original_scheduler_update_from_output is not None:
            try:
                setattr(state.scheduler_cls, "update_from_output", state.original_scheduler_update_from_output)
            except Exception:
                logger.exception("[Wave-Slice] failed to restore Scheduler.update_from_output")

        if state.original_scheduler_finish_requests is not None:
            try:
                setattr(state.scheduler_cls, "finish_requests", state.original_scheduler_finish_requests)
            except Exception:
                logger.exception("[Wave-Slice] failed to restore Scheduler.finish_requests")

        if state.logits_processor_lora_cls is not None and state.original_lora_get_logits is not None:
            try:
                state.logits_processor_lora_cls._get_logits = state.original_lora_get_logits
            except Exception:
                logger.exception("[Wave-Slice] failed to restore LogitsProcessorWithLoRA._get_logits")

        _PATCH_STATE = None
        _clear_autoinject_env()
        logger.info("[Wave-Slice] un-injected from vLLM runtime.")


def is_wave_slice_injected() -> bool:
    with _PATCH_LOCK:
        return _PATCH_STATE is not None


def get_wave_slice_metrics(*, reset: bool = False) -> dict[str, Any]:
    with _PATCH_LOCK:
        state = _PATCH_STATE
    if state is None:
        return {}
    report = _merge_cross_process_metrics(state.metrics.summary())
    if reset:
        state.metrics.reset()
    return report


def reset_wave_slice_metrics() -> None:
    _reset_cross_process_metrics_file()
    with _PATCH_LOCK:
        state = _PATCH_STATE
    if state is not None:
        state.metrics.reset()


@contextlib.contextmanager
def wave_slice_session(
    model_name: str,
    *,
    gamma: float = 2.0,
    policy: Optional[WaveSlicePolicy] = None,
    force: bool = False,
):
    inject_wave_slice(model_name=model_name, gamma=gamma, policy=policy, force=force)
    try:
        yield
    finally:
        uninject_wave_slice()


_maybe_auto_inject_from_env()
