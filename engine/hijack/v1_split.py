from __future__ import annotations

from typing import Any, Callable, Optional

from engine.hijack.runtime_state import WaveSlicePolicy
from engine.hijack.types import _PatchState, _Phase12BeneficiarySignal, _ScheduledReqInfo

ReqInfoCollector = Callable[[Any, Optional[Any], Optional[_PatchState], WaveSlicePolicy], list[_ScheduledReqInfo]]
BeneficiarySignalBuilder = Callable[[_PatchState, WaveSlicePolicy, list[_ScheduledReqInfo]], _Phase12BeneficiarySignal]


def v1_partition_req_ids_diagnostic(
    model_input: Any,
    policy: WaveSlicePolicy,
    *,
    runner_self: Optional[Any] = None,
    state: Optional[_PatchState] = None,
    req_info_collector: ReqInfoCollector,
    beneficiary_signal_builder: BeneficiarySignalBuilder,
    include_non_prefill_in_long: bool = False,
) -> tuple[Optional[tuple[list[str], list[str]]], str, dict[str, int]]:
    debug: dict[str, int] = {}
    try:
        num_sched = dict(getattr(model_input, "num_scheduled_tokens", {}) or {})
    except Exception:
        return None, "num_sched_exception", debug
    if not num_sched:
        return None, "empty_num_sched", debug
    ordered_req_ids = [str(rid) for rid in num_sched.keys()]
    prefill_num_sched = {
        str(rid): int(tok)
        for rid, tok in num_sched.items()
        if int(tok) > 1
    }
    non_prefill_ids = [
        str(rid)
        for rid, tok in num_sched.items()
        if int(tok) <= 1
    ]
    debug["source_num_sched_len"] = len(ordered_req_ids)
    debug["prefill_req_count"] = len(prefill_num_sched)
    debug["decode_req_count"] = len(non_prefill_ids)
    if not prefill_num_sched:
        return None, "no_prefill_tokens", debug

    req_infos = req_info_collector(model_input, runner_self, state, policy)
    debug["req_info_count"] = len(req_infos)
    if state is not None and bool(getattr(policy, "phase12_phase2_require_beneficiary_signal", True)):
        beneficiary_signal = beneficiary_signal_builder(state, policy, req_infos)
        min_beneficiaries = max(1, int(policy.phase12_phase2_min_beneficiary_prefills))
        debug["beneficiary_long_anchor_present"] = int(
            beneficiary_signal.long_anchor_id is not None
        )
        debug["beneficiary_selected_count"] = len(
            beneficiary_signal.beneficiary_selected_ids or []
        )
        debug["beneficiary_min_required"] = min_beneficiaries
        if (
            beneficiary_signal.long_anchor_id is not None
            and len(beneficiary_signal.beneficiary_selected_ids) >= min_beneficiaries
        ):
            beneficiary_ids = set(beneficiary_signal.beneficiary_selected_ids)
            short_ids_set = {
                str(rid)
                for rid in prefill_num_sched.keys()
                if str(rid) in beneficiary_ids
            }
            long_ids_set = {
                str(rid)
                for rid in prefill_num_sched.keys()
                if str(rid) not in beneficiary_ids
            }
            if include_non_prefill_in_long:
                long_ids_set.update(non_prefill_ids)
            short_ids = [rid for rid in ordered_req_ids if rid in short_ids_set]
            long_ids = [rid for rid in ordered_req_ids if rid in long_ids_set]
            debug["beneficiary_short_count"] = len(short_ids)
            debug["beneficiary_long_count"] = len(long_ids)
            if short_ids and long_ids:
                return (short_ids, long_ids), "beneficiary_partition", debug
            return None, "beneficiary_partition_empty", debug

    req_info_map = {info.request_id: info for info in req_infos}
    prefill_items: list[tuple[str, int]] = []
    for rid_raw, tok_raw in prefill_num_sched.items():
        rid = str(rid_raw)
        tok = int(tok_raw)
        info = req_info_map.get(rid)
        approx_rem = info.remaining_tokens if info is not None else tok
        rank = info.lora_rank if info is not None else 1
        score = max(tok, approx_rem) * (rank if policy.phase2_lora_rank_aware else 1)
        if score > 1:
            prefill_items.append((rid, score))
    debug["prefill_items_count"] = len(prefill_items)
    if len(prefill_items) < 2:
        return None, "prefill_items_lt2", debug

    prefill_sorted = sorted(prefill_items, key=lambda x: x[1])
    min_tok = prefill_sorted[0][1]
    max_tok = prefill_sorted[-1][1]
    debug["score_min"] = int(min_tok)
    debug["score_max"] = int(max_tok)
    if min_tok <= 0:
        return None, "non_positive_min_tok", debug
    ratio = float(max_tok) / float(min_tok)
    threshold = max(2.0, float(policy.phase2_min_hetero_ratio))
    debug["ratio_x1000"] = int(ratio * 1000.0)
    debug["ratio_threshold_x1000"] = int(threshold * 1000.0)
    if ratio < threshold:
        return None, "ratio_below_threshold", debug

    pivot = max(min_tok * 2, int((min_tok * max_tok) ** 0.5))
    debug["pivot_score"] = int(pivot)
    short_prefill = [rid for rid, tok in prefill_sorted if tok <= pivot]
    long_prefill = [rid for rid, tok in prefill_sorted if tok > pivot]
    if not short_prefill or not long_prefill:
        half = max(1, len(prefill_sorted) // 2)
        short_prefill = [rid for rid, _ in prefill_sorted[:half]]
        long_prefill = [rid for rid, _ in prefill_sorted[half:]]
        debug["fallback_half_split"] = 1
        if not short_prefill or not long_prefill:
            return None, "half_split_empty", debug

    short_ids_set = set(short_prefill)
    long_ids_set = set(long_prefill)
    if include_non_prefill_in_long:
        long_ids_set.update(non_prefill_ids)
    short_ids = [rid for rid in ordered_req_ids if rid in short_ids_set]
    long_ids = [rid for rid in ordered_req_ids if rid in long_ids_set]
    debug["final_short_count"] = len(short_ids)
    debug["final_long_count"] = len(long_ids)
    if not short_ids or not long_ids:
        return None, "final_partition_empty", debug
    return (short_ids, long_ids), "heuristic_partition", debug


def v1_partition_req_ids(
    model_input: Any,
    policy: WaveSlicePolicy,
    *,
    runner_self: Optional[Any] = None,
    state: Optional[_PatchState] = None,
    req_info_collector: ReqInfoCollector,
    beneficiary_signal_builder: BeneficiarySignalBuilder,
    include_non_prefill_in_long: bool = False,
) -> Optional[tuple[list[str], list[str]]]:
    split_ids, _reason, _debug = v1_partition_req_ids_diagnostic(
        model_input,
        policy,
        runner_self=runner_self,
        state=state,
        req_info_collector=req_info_collector,
        beneficiary_signal_builder=beneficiary_signal_builder,
        include_non_prefill_in_long=include_non_prefill_in_long,
    )
    return split_ids


def v1_execution_escape_req_ids(
    model_input: Any,
    policy: WaveSlicePolicy,
    *,
    runner_self: Optional[Any] = None,
    state: Optional[_PatchState] = None,
    req_info_collector: ReqInfoCollector,
    beneficiary_signal_builder: BeneficiarySignalBuilder,
) -> Optional[tuple[list[str], list[str]]]:
    def _fallback_partition() -> Optional[tuple[list[str], list[str]]]:
        return v1_partition_req_ids(
            model_input,
            policy,
            runner_self=runner_self,
            state=state,
            req_info_collector=req_info_collector,
            beneficiary_signal_builder=beneficiary_signal_builder,
            include_non_prefill_in_long=True,
        )

    try:
        num_sched = dict(getattr(model_input, "num_scheduled_tokens", {}) or {})
    except Exception:
        return None
    if not num_sched or state is None:
        return None
    req_infos = req_info_collector(model_input, runner_self, state, policy)
    beneficiary_signal = beneficiary_signal_builder(state, policy, req_infos)
    mode = str(getattr(policy, "phase2_execution_escape_mode", "bounded_spillover") or "bounded_spillover").strip().lower()
    if mode == "broad_partition":
        return _fallback_partition()
    selected_ids = [str(rid) for rid in (beneficiary_signal.beneficiary_selected_ids or []) if str(rid)]
    if not selected_ids:
        return _fallback_partition()
    selected_set = set(selected_ids)
    req_info_map = {info.request_id: info for info in req_infos}
    active_ids = [str(rid) for rid in num_sched.keys() if str(rid) in selected_set]

    if mode == "beneficiary_only":
        deferred_ids = [str(rid) for rid in num_sched.keys() if str(rid) not in set(active_ids)]
        if not active_ids or not deferred_ids:
            return _fallback_partition()
        return active_ids, deferred_ids

    spillover_cap = max(
        0,
        int(getattr(policy, "phase2_execution_escape_spillover_cap", 2) or 0),
    )
    max_active = max(
        max(1, len(active_ids)),
        int(getattr(policy, "phase2_execution_escape_max_active", 4) or 4),
    )
    remaining_slots = max(0, max_active - len(active_ids))
    spillover_slots = min(spillover_cap, remaining_slots)
    if spillover_slots > 0:
        beneficiary_score_map = dict(beneficiary_signal.beneficiary_score_map or {})
        spillover_candidates: list[tuple[float, int, int, str]] = []
        for rid_raw in num_sched.keys():
            rid = str(rid_raw)
            if rid in selected_set:
                continue
            info = req_info_map.get(rid)
            if info is None:
                continue
            if int(info.remaining_tokens) <= 1:
                continue
            beneficiary_like = float(beneficiary_score_map.get(rid, 0.0) or 0.0)
            short_bonus = 1 if bool(info.is_short) else 0
            chunk_cost = max(0, int(info.expected_chunk_tokens))
            spillover_candidates.append(
                (
                    -beneficiary_like,
                    -short_bonus,
                    chunk_cost,
                    rid,
                )
            )
        spillover_candidates.sort()
        spillover_ids = [rid for *_rest, rid in spillover_candidates[:spillover_slots]]
        if spillover_ids:
            active_ids = [str(rid) for rid in num_sched.keys() if str(rid) in (selected_set | set(spillover_ids))]
    deferred_ids = [str(rid) for rid in num_sched.keys() if str(rid) not in set(active_ids)]
    if not active_ids or not deferred_ids:
        return _fallback_partition()
    return active_ids, deferred_ids


def v1_filter_cached_reqs(cached: Any, selected_req_ids: set[str]) -> Any:
    req_ids = [str(r) for r in list(getattr(cached, "req_ids", []) or [])]
    indices = [i for i, rid in enumerate(req_ids) if rid in selected_req_ids]
    cls = type(cached)

    def _pick_list_attr(attr: str, default: Any) -> list[Any]:
        vals = list(getattr(cached, attr, []) or [])
        return [vals[i] if i < len(vals) else default for i in indices]

    return cls(
        req_ids=[req_ids[i] for i in indices],
        resumed_from_preemption=_pick_list_attr("resumed_from_preemption", False),
        new_token_ids=_pick_list_attr("new_token_ids", []),
        new_block_ids=_pick_list_attr("new_block_ids", ()),
        num_computed_tokens=_pick_list_attr("num_computed_tokens", 0),
    )


def v1_build_subset_scheduler_output(
    model_input: Any,
    selected_req_ids: list[str],
    *,
    carry_finished: bool,
    carry_kv_metadata: bool,
) -> Any:
    selected_set = set(str(r) for r in selected_req_ids)
    output_cls = type(model_input)
    scheduled_new = [r for r in list(getattr(model_input, "scheduled_new_reqs", []) or []) if str(getattr(r, "req_id", "")) in selected_set]
    cached = v1_filter_cached_reqs(getattr(model_input, "scheduled_cached_reqs"), selected_set)

    num_sched = {
        str(rid): int(tok)
        for rid, tok in dict(getattr(model_input, "num_scheduled_tokens", {}) or {}).items()
        if str(rid) in selected_set
    }
    spec_tokens = {
        str(rid): list(tokens)
        for rid, tokens in dict(getattr(model_input, "scheduled_spec_decode_tokens", {}) or {}).items()
        if str(rid) in selected_set
    }
    encoder_inputs = {
        str(rid): list(inputs)
        for rid, inputs in dict(getattr(model_input, "scheduled_encoder_inputs", {}) or {}).items()
        if str(rid) in selected_set
    }
    structured_ids = {
        str(rid): int(idx)
        for rid, idx in dict(getattr(model_input, "structured_output_request_ids", {}) or {}).items()
        if str(rid) in selected_set
    }
    finished_ids = set(getattr(model_input, "finished_req_ids", set()) or set()) if carry_finished else set()
    free_encoder_ids = [
        (str(req_id), int(inp_idx))
        for req_id, inp_idx in list(getattr(model_input, "free_encoder_input_ids", []) or [])
        if (not carry_finished) or (str(req_id) in selected_set)
    ] if carry_finished else []
    num_common_prefix_blocks = [
        0 for _ in list(getattr(model_input, "num_common_prefix_blocks", []) or [])
    ]

    return output_cls(
        scheduled_new_reqs=scheduled_new,
        scheduled_cached_reqs=cached,
        num_scheduled_tokens=num_sched,
        total_num_scheduled_tokens=int(sum(num_sched.values())),
        scheduled_spec_decode_tokens=spec_tokens,
        scheduled_encoder_inputs=encoder_inputs,
        num_common_prefix_blocks=num_common_prefix_blocks,
        finished_req_ids=finished_ids,
        free_encoder_input_ids=free_encoder_ids,
        structured_output_request_ids=structured_ids,
        grammar_bitmask=None,
        kv_connector_metadata=getattr(model_input, "kv_connector_metadata", None) if carry_kv_metadata else None,
    )
