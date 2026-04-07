"""Wave-Slice runtime hijacker for vLLM.

This module implements:
- Phase I: scheduler-side fairness-aware chunk/budget hijacking.
- Phase II: optional ModelRunner-level stream rebinding hijack.
- Runtime metrics hooks: TTFT / P99 / slowdown accounting.

All changes are applied via monkey patching and can be fully reverted.
"""

from __future__ import annotations

import collections
import contextlib
import functools
import importlib
import json
import logging
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Deque, Iterable, Optional

from engine.base_slicer import SlicePlan, WaveBaseSlicer
from engine.runtime_bootstrap import bootstrap_vllm_runtime
from scheduler.wave_scheduler import WaveScheduler

logger = logging.getLogger("WaveSlice")
logger.addHandler(logging.NullHandler())

_AUTO_ENV_ENABLED = "WAVESLICE_AUTOINJECT_ENABLED"
_AUTO_ENV_MODEL = "WAVESLICE_AUTOINJECT_MODEL_NAME"
_AUTO_ENV_GAMMA = "WAVESLICE_AUTOINJECT_GAMMA"
_AUTO_ENV_POLICY = "WAVESLICE_AUTOINJECT_POLICY_JSON"
_AUTO_ENV_PREV_PYTHONPATH = "WAVESLICE_AUTOINJECT_PREV_PYTHONPATH"
_AUTO_ENV_PREV_VLLM_PLUGINS = "WAVESLICE_AUTOINJECT_PREV_VLLM_PLUGINS"


@dataclass(frozen=True)
class WaveSlicePolicy:
    """Runtime knobs for Phase I/II behavior and metrics."""

    # Phase I (scheduler)
    enable_phase1_scheduler: bool = True
    min_hetero_ratio: float = 3.0
    min_long_seq: int = 384
    short_escape_multiplier: int = 12
    max_budget_cap: int = 8192
    enable_sjf_reorder: bool = True
    queue_reorder_mode: str = "sjf"  # sjf | hrrn | aging
    queue_reorder_aging_quantum_us: float = 20_000.0
    enable_tick_hide: bool = False
    allow_phase1_with_lora: bool = False
    allow_phase1_threshold_with_lora: bool = True
    allow_phase1_budget_with_lora: bool = False
    allow_phase1_tick_hide_with_lora: bool = False
    enable_phase1_dynamic_threshold: bool = True
    enable_phase1_budget_guidance: bool = True
    enable_phase1_baseline_relative: bool = True
    enable_phase1_explicit_plan: bool = True
    enable_phase1_direct_explicit_override: bool = True
    phase1_ingress_direct_authoritative: bool = True
    scheduler_objective_mode: str = "fair_escape"  # fair_escape | pure_gain
    phase1_force_extreme_ratio: float = 6.0
    phase1_force_queue_len: int = 1
    phase1_force_min_chunk: int = 128
    phase1_ingress_exact_chunk: bool = True
    phase1_ingress_target_chunk: int = 384
    phase1_ingress_min_chunk: int = 256
    phase1_ingress_max_chunk: int = 512
    phase1_target_short_mul: float = 4.0
    phase1_target_long_fraction: float = 0.33
    phase1_budget_short_mass_factor: float = 1.75
    phase1_budget_bonus_tokens: int = 256
    phase1_budget_queue_bonus: int = 64
    phase1_explicit_budget_cap_tokens: int = 512
    phase1_enable_cohort_mode: bool = False
    phase1_enable_sticky_chunk: bool = False
    phase1_short_cohort_long_fraction: float = 0.4
    phase1_cohort_min_count: int = 2
    phase1_cohort_queue_bonus: int = 2
    phase1_cohort_mass_queue_factor: float = 0.5
    phase1_cohort_target_mass_factor: float = 1.0
    phase1_sticky_ttl: int = 4
    phase1_sticky_reuse_ratio: float = 0.85

    # Phase II (ModelRunner)
    enable_phase2_modelrunner: bool = False
    phase2_enable_mixed_prefill_decode: bool = True
    phase2_min_prefill_count: int = 1
    phase2_min_hetero_ratio: float = 2.0
    phase2_min_long_prefill: int = 256
    phase2_host_sync_after_dispatch: bool = False
    phase2_consistency_mode: str = "balanced"  # balanced | strict
    phase2_dispatch_mode: str = "synchronized"  # synchronized | async_experimental
    phase2_max_inflight_events: int = 2
    phase2_enable_v1_true_unbind: bool = False
    phase2_lora_rank_aware: bool = True
    phase2_min_lora_count: int = 2
    phase2_min_rank_ratio: float = 1.5
    phase2_min_rank_gap: int = 4
    phase2_min_pressure_ratio: float = 2.0
    phase2_selective_only: bool = True
    phase2_extreme_hetero_ratio: float = 3.0
    phase2_extreme_long_prefill: int = 512
    phase2_extreme_pressure_ratio: float = 3.0
    phase2_require_rank_hetero: bool = False
    phase12_joint_coordination: bool = True
    phase12_joint_min_chunk: int = 512
    phase12_phase2_requires_recent_phase1: bool = True
    phase12_phase2_recent_ttl: int = 4
    phase12_phase2_gate_mode: str = "soft"  # hard | soft
    phase12_phase2_soft_ratio_scale: float = 1.15
    phase12_phase2_soft_pressure_scale: float = 1.10
    phase12_phase2_soft_min_long_prefill: int = 512
    phase12_phase2_soft_allow_mixed_decode: bool = True
    phase12_phase2_soft_recent_strength_floor: float = 0.08
    phase12_phase2_soft_require_cashout_signal: bool = True
    phase12_phase2_soft_recent_chunk_match_scale: float = 1.5
    phase12_phase2_soft_window_score_threshold: float = 0.95
    phase12_phase2_soft_window_recent_weight: float = 0.40
    phase12_phase2_soft_window_chunk_weight: float = 0.25
    phase12_phase2_soft_window_pressure_weight: float = 0.20
    phase12_phase2_soft_window_ratio_weight: float = 0.10
    phase12_phase2_soft_window_decode_bonus: float = 0.10
    phase12_phase2_beneficiary_weight: float = 0.35
    phase12_phase2_beneficiary_prefill_scale: float = 1.5
    phase12_phase2_min_beneficiary_prefills: int = 1
    phase12_phase2_require_beneficiary_signal: bool = True
    phase12_phase2_beneficiary_score_threshold: float = 0.55
    phase12_phase2_beneficiary_wait_weight: float = 0.40
    phase12_phase2_beneficiary_size_weight: float = 0.60
    phase12_phase2_beneficiary_quality_floor: float = 0.60
    phase12_phase2_beneficiary_strong_prefill_quality_floor: float = 0.72
    phase12_phase2_beneficiary_max_selected: int = 2
    phase12_phase2_sparse_cashout_cooldown: int = 2
    phase12_phase2_sparse_cashout_exception_quality: float = 0.90
    enable_vllm_lora_compat_patch: bool = True

    # Metrics
    enable_metrics_hook: bool = True
    metrics_short_request_tokens: int = 256


@dataclass
class _RequestMetric:
    request_id: str
    arrival_s: Optional[float] = None
    first_token_s: Optional[float] = None
    finish_s: Optional[float] = None
    input_tokens: Optional[int] = None
    solo_us: Optional[float] = None
    is_short: Optional[bool] = None
    finished: bool = False
    generated_tokens: int = 0


class WaveSliceMetrics:
    """Thread-safe in-process metrics registry."""

    def __init__(self, short_threshold_tokens: int = 256):
        self._short_threshold_tokens = short_threshold_tokens
        self._lock = threading.RLock()
        self._requests: dict[str, _RequestMetric] = {}
        self._phase2_total = 0
        self._phase2_applied = 0
        self._phase2_v1_unbind_applied = 0
        self._phase2_reason_counter: dict[str, int] = {}
        self._sched_total = 0
        self._sched_applied = 0
        self._phase1_baseline_chunk_sum = 0.0
        self._phase1_baseline_chunk_count = 0
        self._phase1_chosen_chunk_sum = 0.0
        self._phase1_chosen_chunk_count = 0
        self._phase1_slice_ratio_sum = 0.0
        self._phase1_slice_ratio_count = 0
        self._phase1_explicit_total = 0
        self._phase1_rewrite_applied = 0
        self._phase1_rewrite_old_chunk_sum = 0.0
        self._phase1_rewrite_new_chunk_sum = 0.0
        self._phase1_rewrite_group_count = 0
        self._phase1_rewrite_token_delta_sum = 0.0
        self._phase1_virtual_cap_total = 0
        self._phase1_virtual_cap_applied = 0
        self._phase1_virtual_cap_old_sum = 0.0
        self._phase1_virtual_cap_new_sum = 0.0
        self._phase1_virtual_cap_target_set = 0
        self._phase1_virtual_cap_helper_calls = 0
        self._phase1_virtual_cap_prefill_calls = 0
        self._phase1_virtual_cap_target_hits = 0
        self._phase1_probe_total = 0
        self._phase1_probe_slice_eligible = 0
        self._phase1_probe_best_lt_long = 0
        self._phase1_probe_short_sum = 0.0
        self._phase1_probe_long_sum = 0.0
        self._phase1_probe_baseline_sum = 0.0
        self._phase1_probe_baseline_count = 0
        self._phase1_probe_best_sum = 0.0
        self._phase1_probe_best_count = 0
        self._phase1_probe_queue_sum = 0.0
        self._phase1_probe_wait_us_sum = 0.0
        self._phase1_probe_reason_counter: dict[str, int] = {}
        self._phase1_scheduler_prop_sum = 0.0
        self._phase1_scheduler_prop_count = 0
        self._phase1_direct_prop_sum = 0.0
        self._phase1_direct_prop_count = 0
        self._phase1_cohort_target_sum = 0.0
        self._phase1_cohort_target_count = 0
        self._phase1_direct_wins = 0
        self._phase1_request_traces: dict[str, list[dict[str, Any]]] = {}
        self._phase1_last_is_prompt: dict[str, bool] = {}

    @staticmethod
    def _percentile(values: list[float], p: float) -> Optional[float]:
        if not values:
            return None
        if p <= 0:
            return min(values)
        if p >= 100:
            return max(values)
        ordered = sorted(values)
        k = (len(ordered) - 1) * (p / 100.0)
        lo = int(k)
        hi = min(lo + 1, len(ordered) - 1)
        if lo == hi:
            return ordered[lo]
        frac = k - lo
        return ordered[lo] + (ordered[hi] - ordered[lo]) * frac

    def reset(self) -> None:
        with self._lock:
            self._requests.clear()
            self._phase2_total = 0
            self._phase2_applied = 0
            self._phase2_v1_unbind_applied = 0
            self._phase2_reason_counter.clear()
            self._sched_total = 0
            self._sched_applied = 0
            self._phase1_baseline_chunk_sum = 0.0
            self._phase1_baseline_chunk_count = 0
            self._phase1_chosen_chunk_sum = 0.0
            self._phase1_chosen_chunk_count = 0
            self._phase1_slice_ratio_sum = 0.0
            self._phase1_slice_ratio_count = 0
            self._phase1_explicit_total = 0
            self._phase1_rewrite_applied = 0
            self._phase1_rewrite_old_chunk_sum = 0.0
            self._phase1_rewrite_new_chunk_sum = 0.0
            self._phase1_rewrite_group_count = 0
            self._phase1_rewrite_token_delta_sum = 0.0
            self._phase1_virtual_cap_total = 0
            self._phase1_virtual_cap_applied = 0
            self._phase1_virtual_cap_old_sum = 0.0
            self._phase1_virtual_cap_new_sum = 0.0
            self._phase1_request_traces = {}
            self._phase1_last_is_prompt = {}
            self._phase1_virtual_cap_target_set = 0
            self._phase1_virtual_cap_helper_calls = 0
            self._phase1_virtual_cap_prefill_calls = 0
            self._phase1_virtual_cap_target_hits = 0
            self._phase1_probe_total = 0
            self._phase1_probe_slice_eligible = 0
            self._phase1_probe_best_lt_long = 0
            self._phase1_probe_short_sum = 0.0
            self._phase1_probe_long_sum = 0.0
            self._phase1_probe_baseline_sum = 0.0
            self._phase1_probe_baseline_count = 0
            self._phase1_probe_best_sum = 0.0
            self._phase1_probe_best_count = 0
            self._phase1_probe_queue_sum = 0.0
            self._phase1_probe_wait_us_sum = 0.0
            self._phase1_probe_reason_counter = {}
            self._phase1_scheduler_prop_sum = 0.0
            self._phase1_scheduler_prop_count = 0
            self._phase1_direct_prop_sum = 0.0
            self._phase1_direct_prop_count = 0
            self._phase1_cohort_target_sum = 0.0
            self._phase1_cohort_target_count = 0
            self._phase1_direct_wins = 0

    def register_request(
        self,
        request_id: str,
        *,
        arrival_s: Optional[float] = None,
        input_tokens: Optional[int] = None,
        solo_us: Optional[float] = None,
        is_short: Optional[bool] = None,
    ) -> None:
        with self._lock:
            metric = self._requests.get(request_id)
            if metric is None:
                metric = _RequestMetric(request_id=request_id)
                self._requests[request_id] = metric
            if arrival_s is not None and metric.arrival_s is None:
                metric.arrival_s = arrival_s
            if input_tokens is not None:
                metric.input_tokens = input_tokens
            if solo_us is not None:
                metric.solo_us = solo_us
            if is_short is not None:
                metric.is_short = is_short
            if metric.is_short is None and metric.input_tokens is not None:
                metric.is_short = metric.input_tokens <= self._short_threshold_tokens

    def observe_scheduler_request(
        self,
        request_id: str,
        *,
        total_tokens: Optional[int] = None,
        solo_us: Optional[float] = None,
        is_short: Optional[bool] = None,
    ) -> None:
        self.register_request(
            request_id=request_id,
            input_tokens=total_tokens,
            solo_us=solo_us,
            is_short=is_short,
        )

    def snapshot_requests(
        self,
        request_ids: Optional[Iterable[str]] = None,
    ) -> dict[str, dict[str, Any]]:
        with self._lock:
            if request_ids is None:
                items = list(self._requests.items())
            else:
                wanted = {str(rid) for rid in request_ids}
                items = [
                    (rid, rec)
                    for rid, rec in self._requests.items()
                    if str(rid) in wanted
                ]
        return {
            str(rid): {
                "arrival_s": rec.arrival_s,
                "input_tokens": rec.input_tokens,
                "solo_us": rec.solo_us,
                "is_short": rec.is_short,
                "generated_tokens": rec.generated_tokens,
                "finished": rec.finished,
            }
            for rid, rec in items
        }

    def record_scheduler_decision(self, applied: bool) -> None:
        with self._lock:
            self._sched_total += 1
            if applied:
                self._sched_applied += 1

    def record_phase1_choice(
        self,
        *,
        chosen_chunk: Optional[int],
        baseline_chunk: Optional[int],
        explicit_plan: bool,
    ) -> None:
        with self._lock:
            if baseline_chunk is not None and baseline_chunk > 0:
                self._phase1_baseline_chunk_sum += float(baseline_chunk)
                self._phase1_baseline_chunk_count += 1
            if chosen_chunk is not None and chosen_chunk > 0:
                self._phase1_chosen_chunk_sum += float(chosen_chunk)
                self._phase1_chosen_chunk_count += 1
            if (
                chosen_chunk is not None
                and chosen_chunk > 0
                and baseline_chunk is not None
                and baseline_chunk > 0
            ):
                self._phase1_slice_ratio_sum += float(chosen_chunk) / float(baseline_chunk)
                self._phase1_slice_ratio_count += 1
            if explicit_plan:
                self._phase1_explicit_total += 1

    def record_phase1_rewrite(
        self,
        *,
        rewritten_groups: int,
        old_chunk_sum: int,
        new_chunk_sum: int,
        token_delta_sum: int,
    ) -> None:
        with self._lock:
            if rewritten_groups <= 0:
                return
            self._phase1_rewrite_applied += 1
            self._phase1_rewrite_group_count += int(rewritten_groups)
            self._phase1_rewrite_old_chunk_sum += float(max(0, old_chunk_sum))
            self._phase1_rewrite_new_chunk_sum += float(max(0, new_chunk_sum))
            self._phase1_rewrite_token_delta_sum += float(max(0, token_delta_sum))

    def record_phase1_virtual_cap(
        self,
        *,
        old_total_tokens: int,
        new_total_tokens: int,
        applied: bool,
    ) -> None:
        with self._lock:
            self._phase1_virtual_cap_total += 1
            if not applied:
                return
            self._phase1_virtual_cap_applied += 1
            self._phase1_virtual_cap_old_sum += float(max(0, old_total_tokens))
            self._phase1_virtual_cap_new_sum += float(max(0, new_total_tokens))

    def record_phase1_virtual_cap_probe(
        self,
        *,
        target_set: bool = False,
        helper_called: bool = False,
        prefill_call: bool = False,
        target_hit: bool = False,
    ) -> None:
        with self._lock:
            if target_set:
                self._phase1_virtual_cap_target_set += 1
            if helper_called:
                self._phase1_virtual_cap_helper_calls += 1
            if prefill_call:
                self._phase1_virtual_cap_prefill_calls += 1
            if target_hit:
                self._phase1_virtual_cap_target_hits += 1

    @staticmethod
    def _trace_request_key(request_id: str) -> bool:
        rid = str(request_id or "")
        if not rid:
            return False
        return rid == "long_b" or rid.endswith(":long_b")

    def record_phase1_step_trace(
        self,
        *,
        request_id: str,
        event: str,
        is_prefill: Optional[bool] = None,
        token_chunk_size: Optional[int] = None,
        num_computed_tokens: Optional[int] = None,
        uncached: Optional[int] = None,
        cached: Optional[int] = None,
        target_chunk: Optional[int] = None,
    ) -> None:
        if not self._trace_request_key(request_id):
            return
        with self._lock:
            traces = self._phase1_request_traces.setdefault(str(request_id), [])
            rec: dict[str, Any] = {"event": str(event)}
            if is_prefill is not None:
                rec["is_prefill"] = bool(is_prefill)
            if token_chunk_size is not None:
                rec["token_chunk_size"] = int(token_chunk_size)
            if num_computed_tokens is not None:
                rec["num_computed_tokens"] = int(num_computed_tokens)
            if uncached is not None:
                rec["uncached"] = int(uncached)
            if cached is not None:
                rec["cached"] = int(cached)
            if target_chunk is not None:
                rec["target_chunk"] = int(target_chunk)
            if is_prefill is not None:
                prev = self._phase1_last_is_prompt.get(str(request_id))
                if prev is not None and bool(prev) and not bool(is_prefill):
                    rec["prefill_to_decode"] = True
                self._phase1_last_is_prompt[str(request_id)] = bool(is_prefill)
            traces.append(rec)
            if len(traces) > 2048:
                del traces[: len(traces) - 2048]

    def record_phase1_probe(
        self,
        *,
        reason: str,
        short_len: Optional[int] = None,
        long_len: Optional[int] = None,
        baseline_chunk: Optional[int] = None,
        best_chunk: Optional[int] = None,
        queue_len: Optional[int] = None,
        wait_us: Optional[float] = None,
        slice_eligible: bool = False,
    ) -> None:
        with self._lock:
            self._phase1_probe_total += 1
            if slice_eligible:
                self._phase1_probe_slice_eligible += 1
            if (
                best_chunk is not None
                and long_len is not None
                and int(best_chunk) > 0
                and int(long_len) > 0
                and int(best_chunk) < int(long_len)
            ):
                self._phase1_probe_best_lt_long += 1
            if short_len is not None and int(short_len) > 0:
                self._phase1_probe_short_sum += float(short_len)
            if long_len is not None and int(long_len) > 0:
                self._phase1_probe_long_sum += float(long_len)
            if baseline_chunk is not None and int(baseline_chunk) > 0:
                self._phase1_probe_baseline_sum += float(baseline_chunk)
                self._phase1_probe_baseline_count += 1
            if best_chunk is not None and int(best_chunk) > 0:
                self._phase1_probe_best_sum += float(best_chunk)
                self._phase1_probe_best_count += 1
            if queue_len is not None and int(queue_len) >= 0:
                self._phase1_probe_queue_sum += float(queue_len)
            if wait_us is not None and float(wait_us) >= 0.0:
                self._phase1_probe_wait_us_sum += float(wait_us)
            self._phase1_probe_reason_counter[reason] = self._phase1_probe_reason_counter.get(reason, 0) + 1

    def record_phase1_proposal(
        self,
        *,
        scheduler_chunk: Optional[int] = None,
        direct_chunk: Optional[int] = None,
        cohort_target: Optional[int] = None,
        direct_won: bool = False,
    ) -> None:
        with self._lock:
            if scheduler_chunk is not None and int(scheduler_chunk) > 0:
                self._phase1_scheduler_prop_sum += float(scheduler_chunk)
                self._phase1_scheduler_prop_count += 1
            if direct_chunk is not None and int(direct_chunk) > 0:
                self._phase1_direct_prop_sum += float(direct_chunk)
                self._phase1_direct_prop_count += 1
            if cohort_target is not None and int(cohort_target) > 0:
                self._phase1_cohort_target_sum += float(cohort_target)
                self._phase1_cohort_target_count += 1
            if direct_won:
                self._phase1_direct_wins += 1

    def record_phase2_decision(self, applied: bool, reason: str) -> None:
        with self._lock:
            self._phase2_total += 1
            if applied:
                self._phase2_applied += 1
            self._phase2_reason_counter[reason] = self._phase2_reason_counter.get(reason, 0) + 1

    def record_phase2_v1_unbind(self) -> None:
        with self._lock:
            self._phase2_v1_unbind_applied += 1

    def observe_engine_outputs(self, outputs: Any, now_s: Optional[float] = None) -> None:
        now = now_s if now_s is not None else time.perf_counter()
        for out in outputs or []:
            request_id = str(getattr(out, "request_id", ""))
            if not request_id:
                continue
            with self._lock:
                metric = self._requests.get(request_id)
                if metric is None:
                    metric = _RequestMetric(request_id=request_id, arrival_s=None)
                    self._requests[request_id] = metric

                token_count = 0
                try:
                    payload = out.outputs[0]
                    token_count = len(getattr(payload, "token_ids", []) or [])
                except Exception:
                    token_count = 0

                if token_count > 0 and metric.first_token_s is None and metric.arrival_s is not None:
                    metric.first_token_s = now
                metric.generated_tokens = max(metric.generated_tokens, token_count)

                if bool(getattr(out, "finished", False)):
                    metric.finished = True
                    if metric.arrival_s is not None:
                        metric.finish_s = now

    def summary(self) -> dict[str, Any]:
        with self._lock:
            records = list(self._requests.values())

        ttft_ms_all: list[float] = []
        ttft_ms_short: list[float] = []
        slowdown_all: list[float] = []
        slowdown_short: list[float] = []

        for rec in records:
            if rec.arrival_s is not None and rec.first_token_s is not None:
                ttft = (rec.first_token_s - rec.arrival_s) * 1000.0
                ttft_ms_all.append(ttft)
                if rec.is_short:
                    ttft_ms_short.append(ttft)

            if rec.arrival_s is not None and rec.finish_s is not None and rec.solo_us and rec.solo_us > 0:
                slowdown = ((rec.finish_s - rec.arrival_s) * 1e6) / rec.solo_us
                slowdown_all.append(slowdown)
                if rec.is_short:
                    slowdown_short.append(slowdown)

        def _stat(values: list[float]) -> dict[str, Optional[float]]:
            return {
                "count": float(len(values)),
                "p50": self._percentile(values, 50.0),
                "p95": self._percentile(values, 95.0),
                "p99": self._percentile(values, 99.0),
            }

        with self._lock:
            phase2_total = self._phase2_total
            phase2_applied = self._phase2_applied
            phase2_v1_unbind_applied = self._phase2_v1_unbind_applied
            phase2_reasons = dict(self._phase2_reason_counter)
            sched_total = self._sched_total
            sched_applied = self._sched_applied
            phase1_baseline_chunk_sum = self._phase1_baseline_chunk_sum
            phase1_baseline_chunk_count = self._phase1_baseline_chunk_count
            phase1_chosen_chunk_sum = self._phase1_chosen_chunk_sum
            phase1_chosen_chunk_count = self._phase1_chosen_chunk_count
            phase1_slice_ratio_sum = self._phase1_slice_ratio_sum
            phase1_slice_ratio_count = self._phase1_slice_ratio_count
            phase1_explicit_total = self._phase1_explicit_total
            phase1_rewrite_applied = self._phase1_rewrite_applied
            phase1_rewrite_old_chunk_sum = self._phase1_rewrite_old_chunk_sum
            phase1_rewrite_new_chunk_sum = self._phase1_rewrite_new_chunk_sum
            phase1_rewrite_group_count = self._phase1_rewrite_group_count
            phase1_rewrite_token_delta_sum = self._phase1_rewrite_token_delta_sum
            phase1_virtual_cap_total = self._phase1_virtual_cap_total
            phase1_virtual_cap_applied = self._phase1_virtual_cap_applied
            phase1_virtual_cap_old_sum = self._phase1_virtual_cap_old_sum
            phase1_virtual_cap_new_sum = self._phase1_virtual_cap_new_sum
            phase1_virtual_cap_target_set = self._phase1_virtual_cap_target_set
            phase1_virtual_cap_helper_calls = self._phase1_virtual_cap_helper_calls
            phase1_virtual_cap_prefill_calls = self._phase1_virtual_cap_prefill_calls
            phase1_virtual_cap_target_hits = self._phase1_virtual_cap_target_hits
            phase1_request_traces = {
                rid: list(rows) for rid, rows in self._phase1_request_traces.items()
            }
            phase1_probe_total = self._phase1_probe_total
            phase1_probe_slice_eligible = self._phase1_probe_slice_eligible
            phase1_probe_best_lt_long = self._phase1_probe_best_lt_long
            phase1_probe_short_sum = self._phase1_probe_short_sum
            phase1_probe_long_sum = self._phase1_probe_long_sum
            phase1_probe_baseline_sum = self._phase1_probe_baseline_sum
            phase1_probe_baseline_count = self._phase1_probe_baseline_count
            phase1_probe_best_sum = self._phase1_probe_best_sum
            phase1_probe_best_count = self._phase1_probe_best_count
            phase1_probe_queue_sum = self._phase1_probe_queue_sum
            phase1_probe_wait_us_sum = self._phase1_probe_wait_us_sum
            phase1_probe_reason_counter = dict(self._phase1_probe_reason_counter)
            phase1_scheduler_prop_sum = self._phase1_scheduler_prop_sum
            phase1_scheduler_prop_count = self._phase1_scheduler_prop_count
            phase1_direct_prop_sum = self._phase1_direct_prop_sum
            phase1_direct_prop_count = self._phase1_direct_prop_count
            phase1_cohort_target_sum = self._phase1_cohort_target_sum
            phase1_cohort_target_count = self._phase1_cohort_target_count
            phase1_direct_wins = self._phase1_direct_wins
            req_total = len(self._requests)
            req_finished = sum(1 for r in self._requests.values() if r.finished)

        return {
            "requests": {"total": req_total, "finished": req_finished},
            "scheduler": {
                "attempts": sched_total,
                "applied": sched_applied,
                "apply_ratio": (sched_applied / sched_total) if sched_total else 0.0,
                "baseline_chunk_avg": (
                    phase1_baseline_chunk_sum / phase1_baseline_chunk_count
                    if phase1_baseline_chunk_count else None
                ),
                "chosen_chunk_avg": (
                    phase1_chosen_chunk_sum / phase1_chosen_chunk_count
                    if phase1_chosen_chunk_count else None
                ),
                "chosen_vs_baseline_ratio_avg": (
                    phase1_slice_ratio_sum / phase1_slice_ratio_count
                    if phase1_slice_ratio_count else None
                ),
                "explicit_plan_ratio": (
                    phase1_explicit_total / sched_total if sched_total else 0.0
                ),
                "rewrite_applied": phase1_rewrite_applied,
                "rewrite_apply_ratio": (
                    phase1_rewrite_applied / sched_total if sched_total else 0.0
                ),
                "rewrite_group_count": phase1_rewrite_group_count,
                "rewrite_old_chunk_avg": (
                    phase1_rewrite_old_chunk_sum / phase1_rewrite_group_count
                    if phase1_rewrite_group_count else None
                ),
                "rewrite_new_chunk_avg": (
                    phase1_rewrite_new_chunk_sum / phase1_rewrite_group_count
                    if phase1_rewrite_group_count else None
                ),
                "rewrite_token_delta_avg": (
                    phase1_rewrite_token_delta_sum / phase1_rewrite_group_count
                    if phase1_rewrite_group_count else None
                ),
                "virtual_cap_apply_ratio": (
                    phase1_virtual_cap_applied / phase1_virtual_cap_total
                    if phase1_virtual_cap_total else 0.0
                ),
                "virtual_cap_old_avg": (
                    phase1_virtual_cap_old_sum / phase1_virtual_cap_applied
                    if phase1_virtual_cap_applied else None
                ),
                "virtual_cap_new_avg": (
                    phase1_virtual_cap_new_sum / phase1_virtual_cap_applied
                    if phase1_virtual_cap_applied else None
                ),
                "virtual_cap_target_set": float(phase1_virtual_cap_target_set),
                "virtual_cap_helper_calls": float(phase1_virtual_cap_helper_calls),
                "virtual_cap_prefill_calls": float(phase1_virtual_cap_prefill_calls),
                "virtual_cap_target_hits": float(phase1_virtual_cap_target_hits),
                "request_traces": phase1_request_traces,
                "probe_total": float(phase1_probe_total),
                "probe_slice_eligible_ratio": (
                    phase1_probe_slice_eligible / phase1_probe_total if phase1_probe_total else 0.0
                ),
                "probe_best_lt_long_ratio": (
                    phase1_probe_best_lt_long / phase1_probe_total if phase1_probe_total else 0.0
                ),
                "probe_short_avg": (
                    phase1_probe_short_sum / phase1_probe_total if phase1_probe_total else None
                ),
                "probe_long_avg": (
                    phase1_probe_long_sum / phase1_probe_total if phase1_probe_total else None
                ),
                "probe_baseline_avg": (
                    phase1_probe_baseline_sum / phase1_probe_baseline_count
                    if phase1_probe_baseline_count else None
                ),
                "probe_best_avg": (
                    phase1_probe_best_sum / phase1_probe_best_count
                    if phase1_probe_best_count else None
                ),
                "probe_queue_avg": (
                    phase1_probe_queue_sum / phase1_probe_total if phase1_probe_total else None
                ),
                "probe_wait_us_avg": (
                    phase1_probe_wait_us_sum / phase1_probe_total if phase1_probe_total else None
                ),
                "proposal_scheduler_avg": (
                    phase1_scheduler_prop_sum / phase1_scheduler_prop_count
                    if phase1_scheduler_prop_count else None
                ),
                "proposal_direct_avg": (
                    phase1_direct_prop_sum / phase1_direct_prop_count
                    if phase1_direct_prop_count else None
                ),
                "proposal_cohort_target_avg": (
                    phase1_cohort_target_sum / phase1_cohort_target_count
                    if phase1_cohort_target_count else None
                ),
                "proposal_direct_win_ratio": (
                    phase1_direct_wins / sched_total if sched_total else 0.0
                ),
                "probe_reasons": phase1_probe_reason_counter,
            },
            "phase2": {
                "attempts": phase2_total,
                "applied": phase2_applied,
                "apply_ratio": (phase2_applied / phase2_total) if phase2_total else 0.0,
                "v1_true_unbind_applied": phase2_v1_unbind_applied,
                "v1_true_unbind_ratio": (phase2_v1_unbind_applied / phase2_total) if phase2_total else 0.0,
                "reasons": phase2_reasons,
            },
            "ttft_ms_all": _stat(ttft_ms_all),
            "ttft_ms_short": _stat(ttft_ms_short),
            "slowdown_all": _stat(slowdown_all),
            "slowdown_short": _stat(slowdown_short),
        }


@dataclass
class _PatchState:
    scheduler_cls: type
    scheduler_method_name: str
    original_schedule: Callable[..., Any]
    brain: WaveScheduler
    policy: WaveSlicePolicy
    model_name: str
    original_public_schedule: Optional[Callable[..., Any]] = None
    metrics: WaveSliceMetrics = field(default_factory=WaveSliceMetrics)
    slicer: WaveBaseSlicer = field(default_factory=WaveBaseSlicer)
    model_runner_cls: Optional[type] = None
    original_execute_model: Optional[Callable[..., Any]] = None
    llm_engine_cls: Optional[type] = None
    original_add_request: Optional[Callable[..., Any]] = None
    original_step: Optional[Callable[..., Any]] = None
    logits_processor_lora_cls: Optional[type] = None
    original_lora_get_logits: Optional[Callable[..., Any]] = None
    sequence_data_cls: Optional[type] = None
    original_sequence_data_get_len: Optional[Callable[..., Any]] = None
    original_get_new_uncached_and_cached_tokens: Optional[Callable[..., Any]] = None
    phase1_sticky_req_id: Optional[str] = None
    phase1_sticky_chunk: Optional[int] = None
    phase1_sticky_ttl_left: int = 0
    phase1_explicit_plans: dict[str, list[SlicePlan]] = field(default_factory=dict)
    phase1_shadow_seq_lens: dict[int, int] = field(default_factory=dict)
    phase1_virtual_token_caps: dict[str, int] = field(default_factory=dict)
    phase1_active_prompt_tokens: dict[str, int] = field(default_factory=dict)
    phase1_ingress_virtuals: dict[str, _Phase1IngressVirtualSlice] = field(default_factory=dict)
    phase1_public_skip_rewrite_requests: set[str] = field(default_factory=set)
    phase12_recent_phase1_apply_ttl: int = 0
    phase12_last_phase1_req_id: Optional[str] = None
    phase12_recent_phase1_strength: float = 0.0
    phase12_recent_phase1_chunk: int = 0
    phase12_recent_phase2_cashout_cooldown: int = 0


@dataclass
class _Phase2Decision:
    apply: bool
    reason: str
    prefill_lens: list[int]
    num_prefills: int
    num_decode_tokens: int
    lora_ranks: list[int] = field(default_factory=list)


@dataclass
class _RunnerStreamState:
    device: Any
    fast_stream: Any
    inflight_events: Deque[Any] = field(default_factory=collections.deque)


@dataclass(frozen=True)
class _ScheduledReqInfo:
    request_id: str
    scheduled_tokens: int
    remaining_tokens: int
    input_tokens: Optional[int]
    arrival_s: Optional[float]
    is_short: bool
    lora_rank: int


@dataclass(frozen=True)
class _Phase12BeneficiarySignal:
    long_anchor_id: Optional[str]
    beneficiary_prefill_ids: list[str]
    beneficiary_prefill_count: int
    beneficiary_fraction: float
    beneficiary_wait_quality: float
    beneficiary_size_quality: float
    beneficiary_cashout_quality: float
    beneficiary_selected_quality: float
    beneficiary_selected_ids: list[str]


@dataclass(frozen=True)
class _Phase1CohortStats:
    representative_short_len: int
    short_count: int
    short_token_mass: int
    short_lengths: list[int]
    long_len: int
    long_req_id: Optional[str]
    total_count: int


@dataclass(frozen=True)
class _Phase1IngressVirtualSlice:
    long_req_id: str
    representative_short_len: int
    short_count: int
    short_token_mass: int
    short_lengths: list[int]
    original_long_len: int
    active_count: int


_PATCH_LOCK = threading.RLock()
_PATCH_STATE: Optional[_PatchState] = None
_LORA_RANK_CACHE: dict[str, int] = {}
_LORA_RANK_RE = re.compile(r"rank[_-]?(\d+)", re.IGNORECASE)


def _is_phase2_strict(policy: WaveSlicePolicy) -> bool:
    return str(policy.phase2_consistency_mode).strip().lower() == "strict"


def _is_phase2_async_experimental(policy: WaveSlicePolicy) -> bool:
    return str(policy.phase2_dispatch_mode).strip().lower() == "async_experimental"


def _load_scheduler_target() -> tuple[type, str]:
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


def _load_model_runner_cls() -> type:
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


def _load_llm_engine_cls() -> type:
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


def _load_sequence_data_cls() -> type:
    bootstrap_vllm_runtime()
    mod = importlib.import_module("vllm.sequence")
    cls = getattr(mod, "SequenceData", None)
    if cls is None:
        raise RuntimeError("vLLM SequenceData class not found.")
    return cls


def _load_logits_processor_lora_cls() -> type:
    bootstrap_vllm_runtime()
    mod = importlib.import_module("vllm.lora.layers")
    cls = getattr(mod, "LogitsProcessorWithLoRA", None)
    if cls is None:
        raise RuntimeError("vLLM LogitsProcessorWithLoRA class not found.")
    return cls


def _safe_first_seq(seq_group: Any) -> Optional[Any]:
    try:
        return next(iter(seq_group.get_seqs()), None)
    except Exception:
        return None


def _safe_total_tokens(seq_group: Any) -> Optional[int]:
    seq = _safe_first_seq(seq_group)
    if seq is None:
        # vLLM v1 Request path
        for attr in ("num_tokens_with_spec", "num_prompt_tokens"):
            val = getattr(seq_group, attr, None)
            if val is not None:
                try:
                    return int(val)
                except Exception:
                    pass
        return None
    try:
        return int(seq.get_len())
    except Exception:
        return None


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
                        state,
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


def _safe_remaining_tokens(seq_group: Any) -> Optional[int]:
    seq = _safe_first_seq(seq_group)
    if seq is None:
        # vLLM v1 Request path
        try:
            total = int(getattr(seq_group, "num_tokens_with_spec"))
            done = int(getattr(seq_group, "num_computed_tokens"))
            return max(0, total - done)
        except Exception:
            return None
    try:
        total = int(seq.get_len())
        done = int(seq.data.get_num_computed_tokens())
    except Exception:
        return None
    return max(0, total - done)


def _safe_prefill_uncomputed_tokens(seq_group: Any) -> Optional[int]:
    seq = _safe_first_seq(seq_group)
    if seq is None:
        try:
            is_prefill = bool(getattr(seq_group, "is_prefill")())
        except Exception:
            is_prefill = False
        if not is_prefill:
            return 0
        try:
            total = int(getattr(seq_group, "num_tokens_with_spec"))
            done = int(getattr(seq_group, "num_computed_tokens"))
            return max(0, total - done)
        except Exception:
            return None
    try:
        if not bool(seq.is_prefill()):
            return 0
    except Exception:
        return 0
    try:
        return max(0, int(seq.get_num_uncomputed_tokens()))
    except Exception:
        pass
    try:
        total = int(seq.get_len())
        done = int(seq.data.get_num_computed_tokens())
        return max(0, total - done)
    except Exception:
        return None


def _safe_request_id(seq_group: Any) -> Optional[str]:
    for attr in ("request_id", "req_id", "id"):
        val = getattr(seq_group, attr, None)
        if val is not None:
            return str(val)
    seq = _safe_first_seq(seq_group)
    if seq is None:
        return None
    for attr in ("request_id", "req_id", "id"):
        val = getattr(seq, attr, None)
        if val is not None:
            return str(val)
    return None


def _safe_wait_us(seq_group: Any, now_s: float) -> float:
    try:
        arrival_s = float(seq_group.metrics.arrival_time)
    except Exception:
        try:
            arrival_s = float(getattr(seq_group, "arrival_time"))
        except Exception:
            return 0.0
    if arrival_s <= 0:
        return 0.0
    return max(0.0, (now_s - arrival_s) * 1e6)


def _queue_reorder_key(
    seq_group: Any,
    *,
    brain: WaveScheduler,
    now_s: float,
    mode: str,
    aging_quantum_us: float,
) -> Any:
    remaining = max(1, int(_safe_remaining_tokens(seq_group) or 1))
    service_us = _estimate_solo_us(brain, remaining) or float(remaining)
    wait_us = _safe_wait_us(seq_group, now_s)
    mode = str(mode or "sjf").strip().lower()

    if mode == "hrrn":
        # Highest Response Ratio Next: larger score gets higher priority.
        response_ratio = (wait_us + service_us) / max(1.0, service_us)
        return (-response_ratio, service_us, remaining)
    if mode == "aging":
        # Aging-SJF: old requests gradually shrink their effective service time.
        quantum = max(1.0, float(aging_quantum_us))
        aged_service = service_us / (1.0 + (wait_us / quantum))
        return (aged_service, service_us, remaining)
    return (service_us, remaining)


def _reorder_queue(
    queue_like: Iterable[Any],
    *,
    brain: WaveScheduler,
    now_s: float,
    mode: str,
    aging_quantum_us: float,
) -> Any:
    queue = list(queue_like)
    queue.sort(
        key=lambda sg: _queue_reorder_key(
            sg,
            brain=brain,
            now_s=now_s,
            mode=mode,
            aging_quantum_us=aging_quantum_us,
        )
    )
    if isinstance(queue_like, collections.deque):
        return collections.deque(queue)
    if isinstance(queue_like, list):
        return queue
    return collections.deque(queue)


def _collect_live_lengths(waiting: Iterable[Any], running: Iterable[Any]) -> tuple[list[int], float]:
    lengths: list[int] = []
    max_wait_us = 0.0
    now_s = time.time()
    for sg in waiting:
        remaining = _safe_prefill_uncomputed_tokens(sg)
        if remaining and remaining > 0:
            lengths.append(remaining)
            max_wait_us = max(max_wait_us, _safe_wait_us(sg, now_s))
    for sg in running:
        remaining = _safe_prefill_uncomputed_tokens(sg)
        if remaining and remaining > 0:
            lengths.append(remaining)
    return lengths, max_wait_us


def _collect_live_snapshot(waiting: Iterable[Any], running: Iterable[Any]) -> tuple[list[tuple[Any, int]], float]:
    snapshot: list[tuple[Any, int]] = []
    max_wait_us = 0.0
    now_s = time.time()
    for sg in list(waiting) + list(running):
        remaining = _safe_prefill_uncomputed_tokens(sg)
        if remaining and remaining > 0:
            rem = int(remaining)
            snapshot.append((sg, rem))
            max_wait_us = max(max_wait_us, _safe_wait_us(sg, now_s))
    return snapshot, max_wait_us


def _phase1_build_cohort(snapshot: list[tuple[Any, int]], policy: WaveSlicePolicy) -> Optional[_Phase1CohortStats]:
    positive = sorted(int(rem) for _, rem in snapshot if int(rem) > 0)
    if len(positive) < 2:
        return None
    long_len = positive[-1]
    cohort_cap = max(1, int(long_len * float(policy.phase1_short_cohort_long_fraction)))
    short_lengths = [int(v) for v in positive[:-1] if int(v) <= cohort_cap]
    if len(short_lengths) < int(policy.phase1_cohort_min_count):
        short_lengths = positive[:-1]
    if not short_lengths:
        short_lengths = [positive[0]]
    short_count = len(short_lengths)
    short_token_mass = sum(short_lengths)
    representative = max(1, int(round(short_token_mass / max(1, short_count))))
    long_req_id = None
    for seq_group, rem in snapshot:
        if int(rem) == int(long_len):
            long_req_id = _safe_request_id(seq_group)
            break
    return _Phase1CohortStats(
        representative_short_len=representative,
        short_count=short_count,
        short_token_mass=short_token_mass,
        short_lengths=list(short_lengths),
        long_len=int(long_len),
        long_req_id=long_req_id,
        total_count=len(positive),
    )


def _phase1_basic_cohort(snapshot: list[tuple[Any, int]]) -> Optional[_Phase1CohortStats]:
    positive = sorted(int(rem) for _, rem in snapshot if int(rem) > 0)
    if len(positive) < 2:
        return None
    short_len = int(positive[0])
    long_len = int(positive[-1])
    long_req_id = None
    for seq_group, rem in snapshot:
        if int(rem) == long_len:
            long_req_id = _safe_request_id(seq_group)
            break
    return _Phase1CohortStats(
        representative_short_len=short_len,
        short_count=1,
        short_token_mass=short_len,
        short_lengths=[short_len],
        long_len=long_len,
        long_req_id=long_req_id,
        total_count=len(positive),
    )


def _phase1_live_cohort_from_snapshot(
    snapshot: list[tuple[Any, int]],
    policy: WaveSlicePolicy,
) -> Optional[_Phase1CohortStats]:
    if policy.phase1_enable_cohort_mode:
        return _phase1_build_cohort(snapshot, policy)
    return _phase1_basic_cohort(snapshot)


def _phase1_maybe_seed_ingress_virtual(
    state: _PatchState,
    *,
    request_id: str,
    input_tokens: Optional[int],
) -> None:
    if input_tokens is None or int(input_tokens) <= 0:
        return
    state.phase1_active_prompt_tokens[str(request_id)] = int(input_tokens)
    positive_items = [
        (rid, int(tok))
        for rid, tok in state.phase1_active_prompt_tokens.items()
        if int(tok) > 0
    ]
    if len(positive_items) < 2:
        return
    positive = sorted(tok for _, tok in positive_items)
    if not _need_wave_slice(positive, state.policy):
        return
    long_len = positive[-1]
    short_len = positive[0]
    long_req_id = None
    for rid, tok in positive_items:
        if int(tok) == int(long_len):
            long_req_id = str(rid)
            break
    if long_req_id is None:
        return
    state.phase1_ingress_virtuals[long_req_id] = _Phase1IngressVirtualSlice(
        long_req_id=long_req_id,
        representative_short_len=int(short_len),
        short_count=max(1, len(positive) - 1),
        short_token_mass=int(sum(positive[:-1])) if len(positive) > 1 else int(short_len),
        short_lengths=[int(v) for v in positive[:-1]] if len(positive) > 1 else [int(short_len)],
        original_long_len=int(long_len),
        active_count=len(positive_items),
    )
    if bool(state.policy.phase1_ingress_direct_authoritative):
        seed_cohort = _Phase1CohortStats(
            representative_short_len=int(short_len),
            short_count=max(1, len(positive) - 1),
            short_token_mass=int(sum(positive[:-1])) if len(positive) > 1 else int(short_len),
            short_lengths=[int(v) for v in positive[:-1]] if len(positive) > 1 else [int(short_len)],
            long_len=int(long_len),
            long_req_id=str(long_req_id),
            total_count=len(positive_items),
        )
        ingress_target = int(_phase1_cohort_target_len(seed_cohort, state.policy))
        ingress_cap = _phase1_authoritative_chunk(
            state,
            target=int(ingress_target),
            short_len=int(short_len),
            upper=max(1, int(long_len) - 1),
        )
        state.phase1_virtual_token_caps[long_req_id] = max(1, int(ingress_cap))
    logger.info(
        "[Wave-Slice][P1-ingress-seed] long_req=%s short=%d long=%d active=%d",
        long_req_id,
        int(short_len),
        int(long_len),
        len(positive_items),
    )


def _phase1_authoritative_chunk(
    state: _PatchState,
    *,
    target: int,
    short_len: int = 0,
    upper: Optional[int] = None,
) -> int:
    ingress_min = max(1, int(state.policy.phase1_ingress_min_chunk))
    ingress_max = max(ingress_min, int(state.policy.phase1_ingress_max_chunk))
    target = max(ingress_min, min(int(target), ingress_max))
    floor = _phase1_authoritative_short_floor(
        state,
        short_len=int(short_len),
        target=int(target),
    )
    if upper is None:
        upper = max(floor + 1, target)
    upper = max(floor + 1, int(upper))
    target = min(target, upper)
    if bool(state.policy.phase1_ingress_exact_chunk):
        return max(floor, min(int(target), upper))
    chunk = int(state.slicer._conservative_map_down(int(target)))
    return max(floor, min(chunk, upper))


def _phase1_authoritative_short_floor(
    state: _PatchState,
    *,
    short_len: int,
    target: int,
) -> int:
    base_floor = max(1, int(short_len))
    if bool(state.policy.phase1_ingress_exact_chunk):
        return max(1, min(base_floor, int(target)))
    return base_floor


def _phase1_find_ingress_virtual_candidate(
    state: _PatchState,
    *,
    snapshot: list[tuple[Any, int]],
) -> Optional[tuple[_Phase1IngressVirtualSlice, Any, int]]:
    if not state.phase1_ingress_virtuals:
        return None
    for seq_group, remaining in snapshot:
        req_id = _safe_request_id(seq_group)
        if not req_id:
            continue
        candidate = state.phase1_ingress_virtuals.get(str(req_id))
        if candidate is None:
            continue
        rem = int(remaining)
        if rem <= 0:
            continue
        return candidate, seq_group, rem
    return None


def _need_wave_slice(lengths: list[int], policy: WaveSlicePolicy) -> bool:
    if len(lengths) < 2:
        return False
    s_min = min(lengths)
    s_max = max(lengths)
    if s_min <= 0:
        return False
    if (
        s_max >= int(max(1, policy.phase1_force_min_chunk))
        and s_max >= int(s_min * max(1.0, policy.phase1_force_extreme_ratio))
    ):
        return True
    if s_max <= policy.min_long_seq:
        return False
    return s_max >= int(s_min * policy.min_hetero_ratio)


def _compute_budget(
    best_chunk: int,
    short_len: int,
    long_len: int,
    short_token_mass: int,
    queue_len: int,
    policy: WaveSlicePolicy,
    original_budget: Any,
    baseline_chunk: Optional[int] = None,
) -> Optional[int]:
    _ = long_len
    if not isinstance(original_budget, int) or original_budget <= 0:
        return None
    escape_allowance = short_len * policy.short_escape_multiplier
    mass_allowance = int(
        max(0.0, float(short_token_mass))
        * max(0.0, float(policy.phase1_budget_short_mass_factor))
    )
    queue_allowance = int(max(0, queue_len)) * int(
        max(0, policy.phase1_budget_queue_bonus)
    )
    max_inflation = 1024
    total_inflation = min(
        max_inflation, escape_allowance + mass_allowance + queue_allowance
    )

    candidate = best_chunk + total_inflation + int(
        max(0, policy.phase1_budget_bonus_tokens)
    )
    candidate = max(best_chunk, candidate)
    if baseline_chunk is not None and int(baseline_chunk) > 0:
        baseline_chunk = max(1, int(baseline_chunk))
        baseline_ceiling = max(
            best_chunk,
            baseline_chunk
            + total_inflation
            + int(max(0, policy.phase1_budget_bonus_tokens)),
        )
        candidate = min(candidate, baseline_ceiling)
    candidate = min(candidate, policy.max_budget_cap)
    return max(1, candidate)


def _compute_explicit_plan_budget(
    *,
    best_chunk: int,
    short_len: int,
    short_token_mass: int,
    policy: WaveSlicePolicy,
    original_budget: Any,
    baseline_chunk: Optional[int],
) -> Optional[int]:
    if not isinstance(original_budget, int) or original_budget <= 0:
        return None
    explicit_inflation = min(
        int(max(0, policy.phase1_explicit_budget_cap_tokens)),
        max(
            short_len,
            int(
                max(0.0, float(short_token_mass))
                * max(0.0, float(policy.phase1_budget_short_mass_factor))
            ),
        ),
    )
    candidate = max(1, int(best_chunk) + explicit_inflation)
    if baseline_chunk is not None and int(baseline_chunk) > 0:
        candidate = min(candidate, int(baseline_chunk))
    candidate = min(candidate, int(original_budget), int(policy.max_budget_cap))
    return max(int(best_chunk), candidate)


def _phase1_baseline_chunk_proxy(
    *,
    long_len: int,
    original_budget: Any,
    original_threshold: Any,
    scheduler_cfg: Any,
    policy: WaveSlicePolicy,
) -> Optional[int]:
    if not bool(policy.enable_phase1_baseline_relative):
        return None
    chunked_enabled = True
    try:
        chunked_enabled = bool(getattr(scheduler_cfg, "enable_chunked_prefill", True))
    except Exception:
        chunked_enabled = True
    if not chunked_enabled:
        return None

    candidates = [max(1, int(long_len))]
    if isinstance(original_budget, int) and original_budget > 0:
        candidates.append(int(original_budget))
    if isinstance(original_threshold, int) and original_threshold > 0:
        candidates.append(int(original_threshold))
    baseline_chunk = min(candidates)
    if baseline_chunk >= int(long_len):
        return None
    return max(1, baseline_chunk)


def _phase1_adjusted_queue_len(
    cohort: _Phase1CohortStats,
    queue_len: int,
    policy: WaveSlicePolicy,
) -> int:
    extra = max(0, int(cohort.short_count) - 1) * int(max(0, policy.phase1_cohort_queue_bonus))
    mass_units = float(cohort.short_token_mass) / float(max(1, cohort.representative_short_len))
    extra += int(max(0.0, mass_units - 1.0) * max(0.0, float(policy.phase1_cohort_mass_queue_factor)))
    return max(1, int(queue_len) + extra)


def _phase1_cohort_target_len(
    cohort: _Phase1CohortStats,
    policy: WaveSlicePolicy,
) -> int:
    mean_short = max(1, int(round(cohort.short_token_mass / max(1, cohort.short_count))))
    target_by_short = int(max(policy.phase1_force_min_chunk, mean_short * float(policy.phase1_target_short_mul)))
    target_by_mass = int(max(policy.phase1_force_min_chunk, mean_short * float(policy.phase1_cohort_target_mass_factor)))
    target_by_fraction = int(max(policy.phase1_force_min_chunk, cohort.long_len * float(policy.phase1_target_long_fraction)))
    return max(1, min(target_by_fraction, max(target_by_short, target_by_mass), cohort.long_len - 1))


def _phase1_effective_short_token_mass(
    lengths: list[int],
    *,
    short_len: int,
    best_chunk: int,
    policy: WaveSlicePolicy,
) -> int:
    limit = max(int(best_chunk), int(short_len * max(1.0, policy.phase1_target_short_mul)))
    mass = 0
    for val in lengths:
        iv = int(val)
        if iv <= 0:
            continue
        if iv <= limit:
            mass += iv
    return max(short_len, mass)


def _maybe_force_phase1_chunk(
    *,
    cohort: _Phase1CohortStats,
    queue_len: int,
    chosen_chunk: int,
    slicer: WaveBaseSlicer,
    policy: WaveSlicePolicy,
) -> tuple[int, bool]:
    short_len = max(1, int(cohort.representative_short_len))
    long_len = max(short_len + 1, int(cohort.long_len))
    chosen_chunk = max(short_len, min(int(chosen_chunk), long_len))
    ratio = float(long_len) / float(max(1, short_len))
    should_force = (
        ratio >= float(policy.phase1_force_extreme_ratio)
        and queue_len >= int(policy.phase1_force_queue_len)
        and long_len >= int(policy.phase1_force_min_chunk)
    )
    if not should_force:
        return chosen_chunk, False

    cohort_target = _phase1_cohort_target_len(cohort, policy)
    forced_cap = max(short_len + 1, min(int(cohort_target), long_len - 1))
    forced_chunk = slicer._conservative_map_down(forced_cap)
    forced_chunk = max(short_len, min(int(forced_chunk), long_len - 1))
    if forced_chunk < chosen_chunk or chosen_chunk >= long_len:
        return forced_chunk, True
    return chosen_chunk, False


def _phase1_apply_sticky_chunk(
    *,
    state: _PatchState,
    cohort: _Phase1CohortStats,
    chosen_chunk: int,
    slicer: WaveBaseSlicer,
) -> tuple[int, bool]:
    sticky_req = state.phase1_sticky_req_id
    sticky_chunk = state.phase1_sticky_chunk
    ttl_left = state.phase1_sticky_ttl_left
    if (
        sticky_req
        and sticky_chunk
        and ttl_left > 0
        and cohort.long_req_id
        and sticky_req == cohort.long_req_id
        and cohort.long_len >= int(sticky_chunk * max(1.0, float(state.policy.phase1_sticky_reuse_ratio)))
    ):
        reused = slicer._conservative_map_down(min(int(sticky_chunk), cohort.long_len - 1))
        reused = max(cohort.representative_short_len, min(int(reused), cohort.long_len - 1))
        state.phase1_sticky_ttl_left = max(0, ttl_left - 1)
        return min(chosen_chunk, reused), True

    state.phase1_sticky_req_id = None
    state.phase1_sticky_chunk = None
    state.phase1_sticky_ttl_left = 0
    return chosen_chunk, False


def _phase1_update_sticky_chunk(
    *,
    state: _PatchState,
    cohort: _Phase1CohortStats,
    chosen_chunk: int,
    applied: bool,
) -> None:
    if applied and cohort.long_req_id:
        state.phase1_sticky_req_id = cohort.long_req_id
        state.phase1_sticky_chunk = int(chosen_chunk)
        state.phase1_sticky_ttl_left = max(0, int(state.policy.phase1_sticky_ttl))
        return
    if state.phase1_sticky_ttl_left > 0:
        state.phase1_sticky_ttl_left = max(0, state.phase1_sticky_ttl_left - 1)
    if state.phase1_sticky_ttl_left == 0:
        state.phase1_sticky_req_id = None
        state.phase1_sticky_chunk = None


def _phase1_find_seq_group_by_request_id(
    snapshot: list[tuple[Any, int]],
    request_id: Optional[str],
) -> Optional[Any]:
    if not request_id:
        return None
    for seq_group, _ in snapshot:
        if _safe_request_id(seq_group) == request_id:
            return seq_group
    return None


def _phase1_prune_explicit_plans(
    plans: list[SlicePlan],
    current_offset: int,
) -> list[SlicePlan]:
    kept = [
        plan
        for plan in plans
        if int(plan.long_offset + plan.chunk_len) > int(current_offset)
    ]
    return kept


def _phase1_build_direct_explicit_plans(
    *,
    state: _PatchState,
    cohort: _Phase1CohortStats,
    total_len: int,
    done_offset: int,
    remaining_len: int,
    baseline_chunk: Optional[int],
) -> list[SlicePlan]:
    if not bool(state.policy.enable_phase1_direct_explicit_override):
        return []
    short_len = max(1, int(cohort.representative_short_len))
    long_len = max(short_len + 1, int(remaining_len))
    direct_target = _phase1_cohort_target_len(
        _Phase1CohortStats(
            representative_short_len=short_len,
            short_count=cohort.short_count,
            short_token_mass=cohort.short_token_mass,
            short_lengths=list(cohort.short_lengths),
            long_len=long_len,
            long_req_id=cohort.long_req_id,
            total_count=cohort.total_count,
        ),
        state.policy,
    )
    upper = long_len - 1
    if baseline_chunk is not None and int(baseline_chunk) > 0:
        upper = min(upper, int(baseline_chunk) - 1)
    if upper <= short_len:
        return []
    ingress_min = max(1, int(state.policy.phase1_ingress_min_chunk))
    ingress_max = max(ingress_min, int(state.policy.phase1_ingress_max_chunk))
    if upper >= ingress_min:
        direct_target = max(int(direct_target), ingress_min)
    direct_target = min(int(direct_target), upper)
    direct_chunk = _phase1_authoritative_chunk(
        state,
        target=int(direct_target),
        short_len=int(short_len),
        upper=int(upper),
    )
    direct_floor = _phase1_authoritative_short_floor(
        state,
        short_len=int(short_len),
        target=int(direct_target),
    )
    direct_chunk = max(direct_floor, min(int(direct_chunk), upper))
    if direct_chunk >= long_len:
        return []
    return [
        state.slicer.make_plan(
            short_len=short_len,
            long_total_len=int(total_len),
            chunk_len=int(direct_chunk),
            long_offset=int(chunk_offset),
        )
        for chunk_offset, _ in state.slicer.iter_long_chunks(
            long_total_len=int(total_len),
            chunk_len=int(direct_chunk),
            start_offset=int(done_offset),
        )
    ]


def _phase1_direct_chunk_candidate(
    *,
    state: _PatchState,
    cohort: _Phase1CohortStats,
    total_len: int,
    done_offset: int,
    remaining_len: int,
    baseline_chunk: Optional[int],
) -> Optional[int]:
    plans = _phase1_build_direct_explicit_plans(
        state=state,
        cohort=cohort,
        total_len=int(total_len),
        done_offset=int(done_offset),
        remaining_len=int(remaining_len),
        baseline_chunk=baseline_chunk,
    )
    if not plans:
        return None
    return int(plans[0].chunk_len)


def _phase1_explicit_chunk_from_plan(
    *,
    state: _PatchState,
    cohort: _Phase1CohortStats,
    snapshot: list[tuple[Any, int]],
    t_wait_us: float,
    queue_length: int,
    baseline_chunk: Optional[int],
) -> Optional[tuple[int, str]]:
    if not bool(state.policy.enable_phase1_explicit_plan):
        return None
    request_id = cohort.long_req_id
    if not request_id:
        return None
    seq_group = _phase1_find_seq_group_by_request_id(snapshot, request_id)
    if seq_group is None:
        state.phase1_explicit_plans.pop(request_id, None)
        return None

    total_len = _safe_total_tokens(seq_group)
    remaining_len = _safe_remaining_tokens(seq_group)
    if total_len is None or remaining_len is None:
        state.phase1_explicit_plans.pop(request_id, None)
        return None
    done_offset = max(0, int(total_len) - int(remaining_len))
    if remaining_len <= 0:
        state.phase1_explicit_plans.pop(request_id, None)
        return None

    existing_plans = list(state.phase1_explicit_plans.get(request_id, []))
    plans = _phase1_prune_explicit_plans(
        existing_plans,
        done_offset,
    )
    plan_kind = "reuse" if plans else "new"
    if not plans:
        plans = state.slicer.build_long_prefill_plan(
            short_len=int(cohort.representative_short_len),
            long_total_len=int(total_len),
            scheduler=state.brain,
            t_wait_us=float(max(0.0, t_wait_us)),
            queue_length=int(max(0, queue_length)),
            start_offset=int(done_offset),
            baseline_chunk=baseline_chunk,
        )
    direct_plans = _phase1_build_direct_explicit_plans(
        state=state,
        cohort=cohort,
        total_len=int(total_len),
        done_offset=int(done_offset),
        remaining_len=int(remaining_len),
        baseline_chunk=baseline_chunk,
    )
    if direct_plans:
        if bool(state.policy.phase1_ingress_direct_authoritative) and request_id in state.phase1_ingress_virtuals:
            plans = direct_plans
            plan_kind = "direct_authoritative"
        else:
            first_direct = direct_plans[0]
            first_plan = plans[0] if plans else None
            current_chunk = int(first_plan.chunk_len) if first_plan is not None else int(remaining_len)
            direct_chunk = int(first_direct.chunk_len)
            ingress_min = max(1, int(state.policy.phase1_ingress_min_chunk))
            if direct_chunk < current_chunk or (
                current_chunk < ingress_min <= direct_chunk
            ):
                plans = direct_plans
                plan_kind = "direct"
    if not plans:
        state.phase1_explicit_plans.pop(request_id, None)
        return None

    state.phase1_explicit_plans[request_id] = plans
    active = plans[0]
    chunk_len = max(1, min(int(active.chunk_len), int(remaining_len)))
    return chunk_len, plan_kind


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
        cohort = _phase1_build_cohort(snapshot, state.policy)
    else:
        cohort = _phase1_basic_cohort(snapshot)
    if cohort is None or not cohort.long_req_id:
        return scheduler_outputs, False

    long_seq_group = _phase1_find_seq_group_by_request_id(snapshot, cohort.long_req_id)
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
        logger.info(
            "[Wave-Slice][P1-public-force] req=%s baseline_chunk=%s chunk=%d explicit_plan=%s groups=%d",
            str(cohort.long_req_id),
            str(baseline_chunk) if baseline_chunk is not None else "none",
            int(best_chunk),
            str(explicit_plan),
            int(rewritten_groups),
        )
    return scheduler_outputs, rewritten


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


def _compute_long_prefill_threshold(
    best_chunk: int,
    original_threshold: Any,
    scheduler_obj: Any,
) -> Optional[int]:
    if best_chunk <= 0:
        return None
    max_model_len = None
    try:
        scheduler_cfg = getattr(scheduler_obj, "scheduler_config", None)
        max_model_len = getattr(scheduler_cfg, "max_model_len", None)
    except Exception:
        max_model_len = None
    threshold = int(best_chunk)
    if isinstance(max_model_len, int) and max_model_len > 0:
        threshold = min(threshold, max_model_len)
    if isinstance(original_threshold, int) and original_threshold > 0:
        threshold = min(threshold, max(original_threshold, 1))
    return max(1, threshold)


def _estimate_prompt_tokens(
    prompt_or_ids: Any,
    *,
    engine_self: Any = None,
    lora_request: Any = None,
) -> Optional[int]:
    if prompt_or_ids is None:
        return None
    if isinstance(prompt_or_ids, dict):
        prompt_token_ids = prompt_or_ids.get("prompt_token_ids")
        if isinstance(prompt_token_ids, (list, tuple)):
            return len(prompt_token_ids)
        prompt_text = prompt_or_ids.get("prompt")
        if isinstance(prompt_text, str):
            prompt_or_ids = prompt_text
    if isinstance(prompt_or_ids, str):
        if engine_self is not None:
            try:
                tokenizer = engine_self.get_tokenizer(lora_request=lora_request)
            except Exception:
                tokenizer = None
            if tokenizer is not None:
                try:
                    encoded = tokenizer.encode(prompt_or_ids, add_special_tokens=False)
                    if isinstance(encoded, (list, tuple)):
                        return len(encoded)
                except TypeError:
                    try:
                        encoded = tokenizer.encode(prompt_or_ids)
                        if isinstance(encoded, (list, tuple)):
                            return len(encoded)
                    except Exception:
                        pass
                except Exception:
                    pass
        return max(1, len(prompt_or_ids.split()))
    if isinstance(prompt_or_ids, (list, tuple)):
        return len(prompt_or_ids)
    return None


def _estimate_solo_us(brain: WaveScheduler, input_tokens: Optional[int]) -> Optional[float]:
    if input_tokens is None or input_tokens <= 0:
        return None
    try:
        bucket = brain._conservative_map_up(input_tokens)
        return float(brain.t_solo_dict.get(bucket, 0.0)) or None
    except Exception:
        return None


def _safe_lora_path(lora_request: Any) -> Optional[str]:
    if lora_request is None:
        return None
    for attr in ("lora_path", "path", "lora_local_path", "local_path"):
        try:
            val = getattr(lora_request, attr, None)
        except Exception:
            val = None
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _extract_rank_from_text(text: str) -> int:
    if not text:
        return 0
    match = _LORA_RANK_RE.search(text)
    if match is None:
        return 0
    try:
        return max(0, int(match.group(1)))
    except Exception:
        return 0


def _infer_lora_rank(lora_request: Any) -> int:
    if lora_request is None:
        return 0

    path = _safe_lora_path(lora_request)
    if path:
        cached = _LORA_RANK_CACHE.get(path)
        if cached is not None:
            return cached

        rank = 0
        cfg_path = os.path.join(path, "adapter_config.json")
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                rank = int(payload.get("r") or payload.get("rank") or 0)
            except Exception:
                rank = 0

        if rank <= 0:
            rank = _extract_rank_from_text(path)
        if rank <= 0:
            rank = _extract_rank_from_text(str(getattr(lora_request, "lora_name", "") or ""))

        rank = max(0, int(rank))
        _LORA_RANK_CACHE[path] = rank
        return rank

    return _extract_rank_from_text(str(getattr(lora_request, "lora_name", "") or ""))


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


def _phase2_has_lora_heterogeneity(ranks: list[int], policy: WaveSlicePolicy) -> bool:
    if not policy.phase2_lora_rank_aware:
        return False
    pos = [int(r) for r in ranks if int(r) > 0]
    if len(pos) < max(2, int(policy.phase2_min_lora_count)):
        return False
    min_rank = max(1, min(pos))
    max_rank = max(pos)
    rank_ratio = float(max_rank) / float(min_rank)
    rank_gap = int(max_rank) - int(min_rank)
    return rank_ratio >= float(policy.phase2_min_rank_ratio) and rank_gap >= int(policy.phase2_min_rank_gap)


def _phase2_rank_ratio(lora_ranks: list[int]) -> float:
    pos = [int(r) for r in lora_ranks if int(r) > 0]
    if len(pos) < 2:
        return 1.0
    min_rank = max(1, min(pos))
    max_rank = max(pos)
    return float(max_rank) / float(min_rank)


def _phase2_pressure_ratio(prefill_lens: list[int], lora_ranks: list[int]) -> float:
    if not prefill_lens:
        return 1.0
    min_len = max(1, min(int(v) for v in prefill_lens if int(v) > 0))
    max_len = max(int(v) for v in prefill_lens if int(v) > 0)
    length_ratio = float(max_len) / float(min_len)
    pos_ranks = [int(r) for r in lora_ranks if int(r) > 0]
    if not pos_ranks:
        return length_ratio
    min_rank = max(1, min(pos_ranks))
    max_rank = max(pos_ranks)
    rank_ratio = float(max_rank) / float(min_rank)
    return length_ratio * rank_ratio


def _phase2_selective_gate(
    *,
    prefill_lens: list[int],
    lora_ranks: list[int],
    policy: WaveSlicePolicy,
    strict_mode: bool,
) -> tuple[bool, float, float, bool]:
    pos_prefills = [int(v) for v in prefill_lens if int(v) > 0]
    if not pos_prefills:
        return False, 1.0, 1.0, False
    min_len = max(1, min(pos_prefills))
    max_len = max(pos_prefills)
    length_ratio = float(max_len) / float(min_len)
    pressure_ratio = _phase2_pressure_ratio(pos_prefills, lora_ranks)
    rank_ratio = _phase2_rank_ratio(lora_ranks)
    ratio = max(length_ratio, rank_ratio)
    lora_rank_hetero = _phase2_has_lora_heterogeneity(lora_ranks, policy)
    need_rank_hetero = bool(policy.phase2_require_rank_hetero)
    if need_rank_hetero and not lora_rank_hetero:
        return False, ratio, pressure_ratio, lora_rank_hetero

    # Balanced mode should honor the normal Phase-II thresholds. The
    # "extreme_*" knobs are for optional escalation paths, not the default
    # selective gate; otherwise balanced mode is silently promoted to a much
    # stricter policy and almost never applies.
    min_ratio = max(
        float(policy.phase2_min_hetero_ratio),
        3.0 if strict_mode else 0.0,
    )
    min_long_prefill = max(
        int(policy.phase2_min_long_prefill),
        512 if strict_mode else 0,
    )
    min_pressure = float(policy.phase2_min_pressure_ratio)
    selective = ratio >= min_ratio and max_len >= min_long_prefill and pressure_ratio >= min_pressure
    return selective, ratio, pressure_ratio, lora_rank_hetero


def _phase2_mixed_escape_ok(
    *,
    prefill_lens: list[int],
    num_decode_tokens: int,
    ratio: float,
    pressure_ratio: float,
    lora_rank_hetero: bool,
    policy: WaveSlicePolicy,
    strict_mode: bool,
) -> bool:
    if strict_mode or num_decode_tokens <= 0:
        return False
    pos_prefills = [int(v) for v in prefill_lens if int(v) > 0]
    if not pos_prefills:
        return False
    max_len = max(pos_prefills)
    if max_len < int(policy.phase2_min_long_prefill):
        return False
    soft_ratio = max(1.25, float(policy.phase2_min_hetero_ratio) * 0.75)
    soft_pressure = max(1.5, float(policy.phase2_min_pressure_ratio))
    return (
        ratio >= soft_ratio
        and (pressure_ratio >= soft_pressure or lora_rank_hetero)
    )


def _phase12_collect_prefill_lora_state(seq_groups: list[Any]) -> tuple[list[int], list[int]]:
    prefill_lens: list[int] = []
    lora_ranks: list[int] = []
    for seq_group in seq_groups:
        remaining = _safe_prefill_uncomputed_tokens(seq_group) or 0
        if remaining <= 0:
            continue
        prefill_lens.append(int(remaining))
        rank = max(1, _infer_lora_rank(getattr(seq_group, "lora_request", None)) or 1)
        lora_ranks.append(int(rank))
    return prefill_lens, lora_ranks


def _phase12_collect_scheduled_req_infos(
    model_input: Any,
    *,
    runner_self: Optional[Any],
    state: Optional[_PatchState],
    policy: WaveSlicePolicy,
) -> list[_ScheduledReqInfo]:
    try:
        num_sched = dict(getattr(model_input, "num_scheduled_tokens", {}) or {})
    except Exception:
        return []
    if not num_sched:
        return []

    req_states = getattr(runner_self, "requests", {}) if runner_self is not None else {}
    metric_snapshot = (
        state.metrics.snapshot_requests(num_sched.keys())
        if state is not None
        else {}
    )
    infos: list[_ScheduledReqInfo] = []
    for rid_raw, tok_raw in num_sched.items():
        rid = str(rid_raw)
        scheduled = max(0, int(tok_raw))
        remaining = scheduled
        rank = 1
        st = req_states.get(rid) if isinstance(req_states, dict) else None
        if st is not None:
            try:
                remaining = max(0, int(st.num_tokens) - int(st.num_computed_tokens))
            except Exception:
                remaining = scheduled
            rank = max(1, _infer_lora_rank(getattr(st, "lora_request", None)) or 1)
        metric = metric_snapshot.get(rid, {})
        input_tokens = metric.get("input_tokens")
        arrival_s = metric.get("arrival_s")
        metric_is_short = metric.get("is_short")
        is_short = bool(metric_is_short)
        if metric_is_short is None and input_tokens is not None:
            is_short = int(input_tokens) <= int(policy.metrics_short_request_tokens)
        infos.append(
            _ScheduledReqInfo(
                request_id=rid,
                scheduled_tokens=scheduled,
                remaining_tokens=max(scheduled, remaining),
                input_tokens=int(input_tokens) if input_tokens is not None else None,
                arrival_s=float(arrival_s) if arrival_s is not None else None,
                is_short=is_short,
                lora_rank=rank,
            )
        )
    return infos


def _phase12_beneficiary_signal(
    *,
    state: _PatchState,
    policy: WaveSlicePolicy,
    req_infos: list[_ScheduledReqInfo],
) -> _Phase12BeneficiarySignal:
    if not req_infos:
        return _Phase12BeneficiarySignal(None, [], 0, 0.0, 0.0, 0.0, 0.0, 0.0, [])
    recent_ttl = int(getattr(state, "phase12_recent_phase1_apply_ttl", 0) or 0)
    recent_chunk = max(0, int(getattr(state, "phase12_recent_phase1_chunk", 0) or 0))
    if recent_ttl <= 0 or recent_chunk <= 0:
        return _Phase12BeneficiarySignal(None, [], 0, 0.0, 0.0, 0.0, 0.0, 0.0, [])

    anchor_id = str(getattr(state, "phase12_last_phase1_req_id", "") or "")
    req_map = {info.request_id: info for info in req_infos}
    anchor = req_map.get(anchor_id)
    if anchor is None:
        prefill_infos = [info for info in req_infos if info.remaining_tokens > 1]
        if not prefill_infos:
            return _Phase12BeneficiarySignal(None, [], 0, 0.0, 0.0, 0.0, 0.0, 0.0, [])
        anchor = max(prefill_infos, key=lambda info: (info.remaining_tokens, info.scheduled_tokens))
        anchor_id = anchor.request_id

    anchor_arrival = anchor.arrival_s
    now_s = time.perf_counter()
    beneficiary_upper = max(
        recent_chunk + 64,
        int(float(recent_chunk) * max(1.0, float(policy.phase12_phase2_beneficiary_prefill_scale))),
    )
    beneficiary_ids: list[str] = []
    beneficiary_scores: dict[str, float] = {}
    candidate_pool = 0
    candidate_waits: list[float] = []
    candidate_sizes: list[int] = []
    for info in req_infos:
        if info.request_id == anchor_id:
            continue
        if info.remaining_tokens <= 1:
            continue
        candidate_pool += 1
        size_ok = info.remaining_tokens <= beneficiary_upper
        arrival_ok = (
            anchor_arrival is None
            or info.arrival_s is None
            or float(info.arrival_s) >= float(anchor_arrival)
        )
        shortish_ok = info.is_short or (
            info.input_tokens is not None and int(info.input_tokens) <= beneficiary_upper
        )
        if size_ok and arrival_ok and shortish_ok:
            beneficiary_ids.append(info.request_id)
            wait_s = max(0.0, now_s - float(info.arrival_s)) if info.arrival_s is not None else 0.0
            candidate_waits.append(wait_s)
            candidate_sizes.append(int(info.remaining_tokens))
    beneficiary_fraction = (
        float(len(beneficiary_ids)) / float(candidate_pool)
        if candidate_pool > 0
        else 0.0
    )
    if beneficiary_ids:
        max_wait = max(candidate_waits) if candidate_waits else 0.0
        selected_ids: list[str] = []
        wait_scores: list[float] = []
        size_scores: list[float] = []
        score_threshold = max(
            0.0,
            min(1.0, float(getattr(policy, "phase12_phase2_beneficiary_score_threshold", 0.55) or 0.55)),
        )
        wait_weight = max(0.0, float(getattr(policy, "phase12_phase2_beneficiary_wait_weight", 0.40) or 0.40))
        size_weight = max(0.0, float(getattr(policy, "phase12_phase2_beneficiary_size_weight", 0.60) or 0.60))
        norm = max(1e-6, wait_weight + size_weight)
        for rid in beneficiary_ids:
            info = req_map.get(rid)
            if info is None:
                continue
            wait_s = max(0.0, now_s - float(info.arrival_s)) if info.arrival_s is not None else 0.0
            wait_quality = min(1.0, wait_s / max(1e-6, max_wait)) if max_wait > 0.0 else 0.0
            if info.remaining_tokens <= recent_chunk:
                size_quality = 1.0
            elif info.remaining_tokens >= beneficiary_upper:
                size_quality = 0.0
            else:
                size_quality = 1.0 - (
                    float(info.remaining_tokens - recent_chunk)
                    / float(max(1, beneficiary_upper - recent_chunk))
                )
            score = ((wait_weight * wait_quality) + (size_weight * size_quality)) / norm
            beneficiary_scores[rid] = score
            wait_scores.append(wait_quality)
            size_scores.append(size_quality)
            if score >= score_threshold:
                selected_ids.append(rid)
        selected_ids.sort(key=lambda rid: beneficiary_scores.get(rid, 0.0), reverse=True)
        max_selected = max(0, int(getattr(policy, "phase12_phase2_beneficiary_max_selected", 2) or 0))
        if max_selected > 0:
            selected_ids = selected_ids[:max_selected]
        beneficiary_wait_quality = sum(wait_scores) / float(len(wait_scores)) if wait_scores else 0.0
        beneficiary_size_quality = sum(size_scores) / float(len(size_scores)) if size_scores else 0.0
        beneficiary_cashout_quality = (
            sum(beneficiary_scores.get(rid, 0.0) for rid in beneficiary_scores)
            / float(len(beneficiary_scores))
            if beneficiary_scores
            else 0.0
        )
        beneficiary_selected_quality = (
            sum(beneficiary_scores.get(rid, 0.0) for rid in selected_ids)
            / float(len(selected_ids))
            if selected_ids
            else 0.0
        )
    else:
        selected_ids = []
        beneficiary_wait_quality = 0.0
        beneficiary_size_quality = 0.0
        beneficiary_cashout_quality = 0.0
        beneficiary_selected_quality = 0.0
    return _Phase12BeneficiarySignal(
        long_anchor_id=anchor_id,
        beneficiary_prefill_ids=beneficiary_ids,
        beneficiary_prefill_count=len(beneficiary_ids),
        beneficiary_fraction=beneficiary_fraction,
        beneficiary_wait_quality=beneficiary_wait_quality,
        beneficiary_size_quality=beneficiary_size_quality,
        beneficiary_cashout_quality=beneficiary_cashout_quality,
        beneficiary_selected_quality=beneficiary_selected_quality,
        beneficiary_selected_ids=selected_ids,
    )


def _phase12_joint_phase1_floor(
    *,
    state: _PatchState,
    snapshot: _SchedulerSnapshot,
    policy: WaveSlicePolicy,
) -> Optional[int]:
    if not (policy.enable_phase1_scheduler and policy.enable_phase2_modelrunner):
        return None
    if not bool(policy.phase12_joint_coordination):
        return None
    seq_groups = list(snapshot.running) + list(snapshot.waiting)
    prefill_lens, lora_ranks = _phase12_collect_prefill_lora_state(seq_groups)
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


def _extract_prefill_lens(attn_meta: Any) -> list[int]:
    lens_tensor = getattr(attn_meta, "prompt_lens_tensor", None)
    if lens_tensor is None:
        lens_tensor = getattr(attn_meta, "seq_lens_tensor", None)
    if lens_tensor is None:
        return []
    try:
        raw = lens_tensor.tolist()
        return [int(x) for x in raw if int(x) > 0]
    except Exception:
        return []


def _phase2_decide(
    model_input: Any,
    policy: WaveSlicePolicy,
    *,
    runner_self: Optional[Any] = None,
) -> _Phase2Decision:
    with _PATCH_LOCK:
        state = _PATCH_STATE
    lora_ranks = _extract_phase2_lora_ranks(model_input, runner_self=runner_self)
    req_infos = _phase12_collect_scheduled_req_infos(
        model_input,
        runner_self=runner_self,
        state=state,
        policy=policy,
    )

    # vLLM v1 path: GPUModelRunner.execute_model(scheduler_output, ...)
    if hasattr(model_input, "num_scheduled_tokens") and hasattr(model_input, "total_num_scheduled_tokens"):
        strict_mode = _is_phase2_strict(policy)
        try:
            vals = [int(v) for v in getattr(model_input, "num_scheduled_tokens", {}).values()]
        except Exception:
            vals = []
        prefill_lens = [v for v in vals if v > 1]
        num_prefills = len(prefill_lens)
        num_decode_tokens = sum(1 for v in vals if v <= 1)
        selective_ok, ratio, pressure_ratio, lora_rank_hetero = _phase2_selective_gate(
            prefill_lens=prefill_lens,
            lora_ranks=lora_ranks,
            policy=policy,
            strict_mode=strict_mode,
        )
        if state is not None:
            phase12_ready, phase12_reason = _phase12_joint_phase2_ready(
                state=state,
                policy=policy,
                prefill_lens=prefill_lens,
                num_decode_tokens=num_decode_tokens,
                lora_ranks=lora_ranks,
                req_infos=req_infos,
                strict_mode=strict_mode,
            )
        else:
            phase12_ready, phase12_reason = True, "joint_state_absent"
        if not phase12_ready:
            return _Phase2Decision(
                False,
                f"{phase12_reason}_v1",
                prefill_lens,
                num_prefills,
                num_decode_tokens,
                lora_ranks,
            )

        if strict_mode and num_decode_tokens > 0:
            return _Phase2Decision(
                False,
                "strict_no_mixed_prefill_decode",
                prefill_lens,
                num_prefills,
                num_decode_tokens,
                lora_ranks,
            )

        if (
            not strict_mode
            and policy.phase2_enable_mixed_prefill_decode
            and num_prefills > 0
            and num_decode_tokens > 0
        ):
            mixed_escape_ok = _phase2_mixed_escape_ok(
                prefill_lens=prefill_lens,
                num_decode_tokens=num_decode_tokens,
                ratio=ratio,
                pressure_ratio=pressure_ratio,
                lora_rank_hetero=lora_rank_hetero,
                policy=policy,
                strict_mode=strict_mode,
            )
            if bool(policy.phase2_selective_only) and not selective_ok:
                if mixed_escape_ok:
                    return _Phase2Decision(
                        True,
                        "selective_mixed_prefill_decode_soft_v1",
                        prefill_lens,
                        num_prefills,
                        num_decode_tokens,
                        lora_ranks,
                    )
                return _Phase2Decision(
                    False,
                    "mixed_prefill_decode_not_extreme_v1",
                    prefill_lens,
                    num_prefills,
                    num_decode_tokens,
                    lora_ranks,
                )
            return _Phase2Decision(
                True,
                "selective_mixed_prefill_decode_v1",
                prefill_lens,
                num_prefills,
                num_decode_tokens,
                lora_ranks,
            )

        if not strict_mode and num_prefills >= max(1, policy.phase2_min_prefill_count):
            if bool(policy.phase2_selective_only) and not selective_ok:
                return _Phase2Decision(
                    False,
                    "prefill_batch_not_extreme_v1",
                    prefill_lens,
                    num_prefills,
                    num_decode_tokens,
                    lora_ranks,
                )
            return _Phase2Decision(
                True,
                "selective_lora_extreme_prefill_v1",
                prefill_lens,
                num_prefills,
                num_decode_tokens,
                lora_ranks,
            )

        if len(prefill_lens) < max(2 if strict_mode else 1, policy.phase2_min_prefill_count):
            return _Phase2Decision(
                False,
                "insufficient_prefill_batch",
                prefill_lens,
                num_prefills,
                num_decode_tokens,
                lora_ranks,
            )

        min_len = max(1, min(prefill_lens))
        max_len = max(prefill_lens)
        ratio = float(max_len) / float(min_len)
        pressure_ratio = _phase2_pressure_ratio(prefill_lens, lora_ranks)
        min_hetero_ratio = max(policy.phase2_min_hetero_ratio, 3.0 if strict_mode else 0.0)
        min_long_prefill = max(policy.phase2_min_long_prefill, 512 if strict_mode else 0)
        if ratio < min_hetero_ratio:
            if lora_rank_hetero and pressure_ratio >= float(policy.phase2_min_pressure_ratio) and selective_ok:
                return _Phase2Decision(
                    True,
                    "strict_selective_lora_rank_pressure_prefill_v1"
                    if strict_mode
                    else "selective_lora_rank_pressure_prefill_v1",
                    prefill_lens,
                    num_prefills,
                    num_decode_tokens,
                    lora_ranks,
                )
            return _Phase2Decision(
                False,
                "strict_hetero_ratio_too_low" if strict_mode else "hetero_ratio_too_low",
                prefill_lens,
                num_prefills,
                num_decode_tokens,
                lora_ranks,
            )
        if max_len < min_long_prefill:
            return _Phase2Decision(
                False,
                "strict_long_prefill_too_short" if strict_mode else "long_prefill_too_short",
                prefill_lens,
                num_prefills,
                num_decode_tokens,
                lora_ranks,
            )
        if bool(policy.phase2_selective_only) and not selective_ok:
            return _Phase2Decision(
                False,
                "strict_prefill_not_extreme_v1" if strict_mode else "prefill_not_extreme_v1",
                prefill_lens,
                num_prefills,
                num_decode_tokens,
                lora_ranks,
            )
        return _Phase2Decision(
            True,
            "strict_selective_lora_extreme_prefill_v1" if strict_mode else "selective_lora_extreme_prefill_v1",
            prefill_lens,
            num_prefills,
            num_decode_tokens,
            lora_ranks,
        )

    attn_meta = getattr(model_input, "attn_metadata", None)
    if attn_meta is None:
        return _Phase2Decision(False, "no_attn_meta", [], 0, 0, lora_ranks)

    num_prefills = int(getattr(attn_meta, "num_prefills", 0) or 0)
    num_decode_tokens = int(getattr(attn_meta, "num_decode_tokens", 0) or 0)
    prefill_lens = _extract_prefill_lens(attn_meta)
    strict_mode = _is_phase2_strict(policy)
    selective_ok, ratio, pressure_ratio, lora_rank_hetero = _phase2_selective_gate(
        prefill_lens=prefill_lens,
        lora_ranks=lora_ranks,
        policy=policy,
        strict_mode=strict_mode,
    )
    if state is not None:
        phase12_ready, phase12_reason = _phase12_joint_phase2_ready(
            state=state,
            policy=policy,
            prefill_lens=prefill_lens,
            num_decode_tokens=num_decode_tokens,
            lora_ranks=lora_ranks,
            req_infos=req_infos,
            strict_mode=strict_mode,
        )
    else:
        phase12_ready, phase12_reason = True, "joint_state_absent"
    if not phase12_ready:
        return _Phase2Decision(
            False,
            phase12_reason,
            prefill_lens,
            num_prefills,
            num_decode_tokens,
            lora_ranks,
        )

    if strict_mode and num_decode_tokens > 0:
        return _Phase2Decision(
            False,
            "strict_no_mixed_prefill_decode",
            prefill_lens,
            num_prefills,
            num_decode_tokens,
            lora_ranks,
        )

    if (
        not strict_mode
        and
        policy.phase2_enable_mixed_prefill_decode
        and num_prefills > 0
        and num_decode_tokens > 0
    ):
        mixed_escape_ok = _phase2_mixed_escape_ok(
            prefill_lens=prefill_lens,
            num_decode_tokens=num_decode_tokens,
            ratio=ratio,
            pressure_ratio=pressure_ratio,
            lora_rank_hetero=lora_rank_hetero,
            policy=policy,
            strict_mode=strict_mode,
        )
        if bool(policy.phase2_selective_only) and not selective_ok:
            if mixed_escape_ok:
                return _Phase2Decision(
                    True,
                    "selective_mixed_prefill_decode_soft",
                    prefill_lens,
                    num_prefills,
                    num_decode_tokens,
                    lora_ranks,
                )
            return _Phase2Decision(
                False,
                "mixed_prefill_decode_not_extreme",
                prefill_lens,
                num_prefills,
                num_decode_tokens,
                lora_ranks,
            )
        return _Phase2Decision(
            True,
            "selective_mixed_prefill_decode",
            prefill_lens,
            num_prefills,
            num_decode_tokens,
            lora_ranks,
        )

    min_prefill_count = policy.phase2_min_prefill_count
    if strict_mode:
        min_prefill_count = max(min_prefill_count, 2)

    if len(prefill_lens) < min_prefill_count:
        return _Phase2Decision(
            False,
            "insufficient_prefill_batch",
            prefill_lens,
            num_prefills,
            num_decode_tokens,
            lora_ranks,
        )

    min_len = max(1, min(prefill_lens))
    max_len = max(prefill_lens)
    min_hetero_ratio = policy.phase2_min_hetero_ratio
    min_long_prefill = policy.phase2_min_long_prefill
    if strict_mode:
        min_hetero_ratio = max(min_hetero_ratio, 3.0)
        min_long_prefill = max(min_long_prefill, 512)

    if ratio < min_hetero_ratio:
        if lora_rank_hetero and pressure_ratio >= float(policy.phase2_min_pressure_ratio) and selective_ok:
            return _Phase2Decision(
                True,
                "strict_selective_lora_rank_pressure_prefill"
                if strict_mode
                else "selective_lora_rank_pressure_prefill",
                prefill_lens,
                num_prefills,
                num_decode_tokens,
                lora_ranks,
            )
        return _Phase2Decision(
            False,
            "strict_hetero_ratio_too_low" if strict_mode else "hetero_ratio_too_low",
            prefill_lens,
            num_prefills,
            num_decode_tokens,
            lora_ranks,
        )
    if max_len < min_long_prefill:
        return _Phase2Decision(
            False,
            "strict_long_prefill_too_short" if strict_mode else "long_prefill_too_short",
            prefill_lens,
            num_prefills,
            num_decode_tokens,
            lora_ranks,
        )
    if bool(policy.phase2_selective_only) and not selective_ok:
        return _Phase2Decision(
            False,
            "strict_prefill_not_extreme" if strict_mode else "prefill_not_extreme",
            prefill_lens,
            num_prefills,
            num_decode_tokens,
            lora_ranks,
        )
    return _Phase2Decision(
        True,
        "strict_selective_lora_extreme_prefill" if strict_mode else "selective_lora_extreme_prefill",
        prefill_lens,
        num_prefills,
        num_decode_tokens,
        lora_ranks,
    )


def _is_v1_scheduler_output(model_input: Any) -> bool:
    return hasattr(model_input, "num_scheduled_tokens") and hasattr(
        model_input, "total_num_scheduled_tokens"
    )


def _safe_v1_pipeline_size(model_runner: Any) -> int:
    try:
        pc = getattr(model_runner, "parallel_config", None)
        if pc is None:
            return 1
        return int(getattr(pc, "pipeline_parallel_size", 1) or 1)
    except Exception:
        return 1


def _v1_partition_req_ids(
    model_input: Any,
    policy: WaveSlicePolicy,
    *,
    runner_self: Optional[Any] = None,
    state: Optional[_PatchState] = None,
) -> Optional[tuple[list[str], list[str]]]:
    try:
        num_sched = dict(getattr(model_input, "num_scheduled_tokens", {}) or {})
    except Exception:
        return None
    if not num_sched:
        return None

    req_infos = _phase12_collect_scheduled_req_infos(
        model_input,
        runner_self=runner_self,
        state=state,
        policy=policy,
    )
    if state is not None and bool(getattr(policy, "phase12_phase2_require_beneficiary_signal", True)):
        beneficiary_signal = _phase12_beneficiary_signal(
            state=state,
            policy=policy,
            req_infos=req_infos,
        )
        min_beneficiaries = max(1, int(policy.phase12_phase2_min_beneficiary_prefills))
        if (
            beneficiary_signal.long_anchor_id is not None
            and len(beneficiary_signal.beneficiary_selected_ids) >= min_beneficiaries
        ):
            beneficiary_ids = set(beneficiary_signal.beneficiary_selected_ids)
            short_ids = [
                str(rid)
                for rid in num_sched.keys()
                if str(rid) in beneficiary_ids
            ]
            long_ids = [
                str(rid)
                for rid in num_sched.keys()
                if str(rid) not in beneficiary_ids
            ]
            if short_ids and long_ids:
                return short_ids, long_ids

    req_info_map = {info.request_id: info for info in req_infos}
    prefill_items: list[tuple[str, int]] = []
    for rid_raw, tok_raw in num_sched.items():
        rid = str(rid_raw)
        tok = int(tok_raw)
        info = req_info_map.get(rid)
        approx_rem = info.remaining_tokens if info is not None else tok
        rank = info.lora_rank if info is not None else 1
        score = max(tok, approx_rem) * (rank if policy.phase2_lora_rank_aware else 1)
        if score > 1:
            prefill_items.append((rid, score))
    if len(prefill_items) < 2:
        return None

    prefill_sorted = sorted(prefill_items, key=lambda x: x[1])
    min_tok = prefill_sorted[0][1]
    max_tok = prefill_sorted[-1][1]
    if min_tok <= 0:
        return None
    ratio = float(max_tok) / float(min_tok)
    if ratio < max(2.0, float(policy.phase2_min_hetero_ratio)):
        return None

    pivot = max(min_tok * 2, int((min_tok * max_tok) ** 0.5))
    short_prefill = [rid for rid, tok in prefill_sorted if tok <= pivot]
    long_prefill = [rid for rid, tok in prefill_sorted if tok > pivot]
    if not short_prefill or not long_prefill:
        half = max(1, len(prefill_sorted) // 2)
        short_prefill = [rid for rid, _ in prefill_sorted[:half]]
        long_prefill = [rid for rid, _ in prefill_sorted[half:]]
        if not short_prefill or not long_prefill:
            return None

    short_ids_set = set(short_prefill)
    long_ids_set = set(long_prefill)
    short_ids = [str(rid) for rid in num_sched.keys() if str(rid) in short_ids_set]
    long_ids = [str(rid) for rid in num_sched.keys() if str(rid) in long_ids_set]
    if not short_ids or not long_ids:
        return None
    return short_ids, long_ids


def _v1_filter_cached_reqs(cached: Any, selected_req_ids: set[str]) -> Any:
    req_ids = [str(r) for r in list(getattr(cached, "req_ids", []) or [])]
    indices = [i for i, rid in enumerate(req_ids) if rid in selected_req_ids]
    cls = type(cached)
    return cls(
        req_ids=[req_ids[i] for i in indices],
        resumed_from_preemption=[getattr(cached, "resumed_from_preemption", [])[i] for i in indices],
        new_token_ids=[getattr(cached, "new_token_ids", [])[i] for i in indices],
        new_block_ids=[getattr(cached, "new_block_ids", [])[i] for i in indices],
        num_computed_tokens=[getattr(cached, "num_computed_tokens", [])[i] for i in indices],
    )


def _v1_build_subset_scheduler_output(
    model_input: Any,
    selected_req_ids: list[str],
    *,
    carry_finished: bool,
    carry_kv_metadata: bool,
) -> Any:
    selected_set = set(str(r) for r in selected_req_ids)
    output_cls = type(model_input)
    scheduled_new = [r for r in list(getattr(model_input, "scheduled_new_reqs", []) or []) if str(getattr(r, "req_id", "")) in selected_set]
    cached = _v1_filter_cached_reqs(getattr(model_input, "scheduled_cached_reqs"), selected_set)

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


def _merge_kv_connector_outputs(a: Any, b: Any) -> Any:
    if a is None:
        return b
    if b is None:
        return a
    cls = type(a)
    finished_sending = set(getattr(a, "finished_sending", set()) or set()) | set(
        getattr(b, "finished_sending", set()) or set()
    )
    finished_recving = set(getattr(a, "finished_recving", set()) or set()) | set(
        getattr(b, "finished_recving", set()) or set()
    )
    return cls(
        finished_sending=finished_sending or None,
        finished_recving=finished_recving or None,
    )


def _merge_v1_runner_outputs(original_order: list[str], out_a: Any, out_b: Any) -> Any:
    if not hasattr(out_a, "req_ids") or not hasattr(out_b, "req_ids"):
        raise TypeError("v1 split merge requires ModelRunnerOutput-like outputs.")

    a_ids = [str(x) for x in list(getattr(out_a, "req_ids", []) or [])]
    b_ids = [str(x) for x in list(getattr(out_b, "req_ids", []) or [])]
    a_map = {rid: i for i, rid in enumerate(a_ids)}
    b_map = {rid: i for i, rid in enumerate(b_ids)}
    all_ids = set(a_ids) | set(b_ids)
    req_ids = [rid for rid in original_order if rid in all_ids]
    req_id_to_index = {rid: i for i, rid in enumerate(req_ids)}

    def _pick_list_attr(obj: Any, attr: str, idx: int, default: Any) -> Any:
        vals = getattr(obj, attr, None)
        if vals is None:
            return default
        try:
            return vals[idx]
        except Exception:
            return default

    sampled_token_ids: list[list[int]] = []
    pooler_output: list[Optional[Any]] = []
    spec_token_ids_needed = (getattr(out_a, "spec_token_ids", None) is not None) or (
        getattr(out_b, "spec_token_ids", None) is not None
    )
    spec_token_ids: Optional[list[list[int]]] = [] if spec_token_ids_needed else None

    for rid in req_ids:
        if rid in a_map:
            idx = a_map[rid]
            sampled = list(_pick_list_attr(out_a, "sampled_token_ids", idx, []))
            pool_val = _pick_list_attr(out_a, "pooler_output", idx, None)
            spec_val = (
                list(_pick_list_attr(out_a, "spec_token_ids", idx, []))
                if spec_token_ids_needed
                else []
            )
        else:
            idx = b_map[rid]
            sampled = list(_pick_list_attr(out_b, "sampled_token_ids", idx, []))
            pool_val = _pick_list_attr(out_b, "pooler_output", idx, None)
            spec_val = (
                list(_pick_list_attr(out_b, "spec_token_ids", idx, []))
                if spec_token_ids_needed
                else []
            )
        sampled_token_ids.append(sampled)
        pooler_output.append(pool_val)
        if spec_token_ids is not None:
            spec_token_ids.append(spec_val)

    logprobs = None
    if getattr(out_a, "logprobs", None) is not None or getattr(out_b, "logprobs", None) is not None:
        lp_cls = type(getattr(out_a, "logprobs", None) or getattr(out_b, "logprobs", None))
        a_lp = getattr(out_a, "logprobs", None)
        b_lp = getattr(out_b, "logprobs", None)
        logprob_token_ids = []
        logprobs_vals = []
        sampled_token_ranks = []
        for rid in req_ids:
            if rid in a_map and a_lp is not None:
                i = a_map[rid]
                logprob_token_ids.append(list(a_lp.logprob_token_ids[i]))
                logprobs_vals.append(list(a_lp.logprobs[i]))
                sampled_token_ranks.append(int(a_lp.sampled_token_ranks[i]))
            elif rid in b_map and b_lp is not None:
                i = b_map[rid]
                logprob_token_ids.append(list(b_lp.logprob_token_ids[i]))
                logprobs_vals.append(list(b_lp.logprobs[i]))
                sampled_token_ranks.append(int(b_lp.sampled_token_ranks[i]))
            else:
                logprob_token_ids.append([])
                logprobs_vals.append([])
                sampled_token_ranks.append(0)
        logprobs = lp_cls(logprob_token_ids, logprobs_vals, sampled_token_ranks)

    prompt_logprobs_dict = {}
    prompt_logprobs_dict.update(getattr(out_a, "prompt_logprobs_dict", {}) or {})
    prompt_logprobs_dict.update(getattr(out_b, "prompt_logprobs_dict", {}) or {})

    num_nans_in_logits = {}
    num_nans_in_logits.update(getattr(out_a, "num_nans_in_logits", {}) or {})
    num_nans_in_logits.update(getattr(out_b, "num_nans_in_logits", {}) or {})

    out_cls = type(out_a)
    return out_cls(
        req_ids=req_ids,
        req_id_to_index=req_id_to_index,
        sampled_token_ids=sampled_token_ids,
        spec_token_ids=spec_token_ids,
        logprobs=logprobs,
        prompt_logprobs_dict=prompt_logprobs_dict,
        pooler_output=pooler_output,
        kv_connector_output=_merge_kv_connector_outputs(
            getattr(out_a, "kv_connector_output", None),
            getattr(out_b, "kv_connector_output", None),
        ),
        num_nans_in_logits=num_nans_in_logits or None,
    )


def _call_original_execute_with_model_input(
    original_execute: Callable[..., Any],
    runner_self: Any,
    model_input: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    if len(args) > 0:
        new_args = (model_input,) + tuple(args[1:])
    else:
        new_args = (model_input,)
    new_kwargs = dict(kwargs)
    if "model_input" in new_kwargs:
        new_kwargs.pop("model_input", None)
    if "scheduler_output" in new_kwargs:
        new_kwargs.pop("scheduler_output", None)
    return original_execute(runner_self, *new_args, **new_kwargs)


def _try_v1_true_unbind_execute(
    *,
    runner_self: Any,
    state: _PatchState,
    original_execute: Callable[..., Any],
    model_input: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    torch_mod: Any,
    stream_state: _RunnerStreamState,
) -> Optional[Any]:
    if not state.policy.phase2_enable_v1_true_unbind:
        return None
    if not _is_v1_scheduler_output(model_input):
        return None
    if _safe_v1_pipeline_size(runner_self) > 1:
        return None
    if getattr(model_input, "grammar_bitmask", None) is not None:
        return None
    if dict(getattr(model_input, "structured_output_request_ids", {}) or {}):
        return None
    if dict(getattr(model_input, "scheduled_spec_decode_tokens", {}) or {}):
        return None
    if getattr(model_input, "kv_connector_metadata", None) is not None:
        return None
    if getattr(getattr(runner_self, "input_batch", None), "pooling_params", None):
        return None
    # Keep the v1 replay path focused on heterogeneous prefills only.
    try:
        if any(int(tok) <= 1 for tok in dict(getattr(model_input, "num_scheduled_tokens", {}) or {}).values()):
            return None
    except Exception:
        return None

    split_ids = _v1_partition_req_ids(
        model_input,
        state.policy,
        runner_self=runner_self,
        state=state,
    )
    if split_ids is None:
        return None
    short_ids, long_ids = split_ids
    if not short_ids or not long_ids:
        return None

    short_sched = _v1_build_subset_scheduler_output(
        model_input,
        short_ids,
        carry_finished=True,
        carry_kv_metadata=False,
    )
    long_sched = _v1_build_subset_scheduler_output(
        model_input,
        long_ids,
        carry_finished=False,
        carry_kv_metadata=False,
    )
    if getattr(short_sched, "total_num_scheduled_tokens", 0) <= 0 or getattr(long_sched, "total_num_scheduled_tokens", 0) <= 0:
        return None

    strict_mode = _is_phase2_strict(state.policy)
    if strict_mode:
        torch_mod.cuda.synchronize(device=stream_state.device)

    main_stream = torch_mod.cuda.current_stream(device=stream_state.device)
    stream_state.fast_stream.wait_stream(main_stream)
    with torch_mod.cuda.stream(stream_state.fast_stream):
        out_short = _call_original_execute_with_model_input(
            original_execute, runner_self, short_sched, args, kwargs
        )
        evt_short = torch_mod.cuda.Event(enable_timing=False)
        evt_short.record(stream_state.fast_stream)
    main_stream.wait_event(evt_short)

    out_long = _call_original_execute_with_model_input(
        original_execute, runner_self, long_sched, args, kwargs
    )

    merged = _merge_v1_runner_outputs(
        original_order=[str(rid) for rid in dict(getattr(model_input, "num_scheduled_tokens", {}) or {}).keys()],
        out_a=out_short,
        out_b=out_long,
    )

    if strict_mode:
        torch_mod.cuda.synchronize(device=stream_state.device)

    logger.info(
        "[Wave-Slice][P2-v1-unbind] short=%d long=%d short_tokens=%d long_tokens=%d",
        len(short_ids),
        len(long_ids),
        int(getattr(short_sched, "total_num_scheduled_tokens", 0)),
        int(getattr(long_sched, "total_num_scheduled_tokens", 0)),
    )
    state.phase12_recent_phase2_cashout_cooldown = max(
        0,
        int(getattr(state.policy, "phase12_phase2_sparse_cashout_cooldown", 2) or 0),
    )
    return merged


def _build_scheduler_hook(state: _PatchState) -> Callable[..., Any]:
    original_schedule = state.original_schedule

    @functools.wraps(original_schedule)
    def _wave_schedule_hook(self: Any, *args: Any, **kwargs: Any) -> Any:
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

        if not hasattr(self, "running") or not hasattr(self, "waiting"):
            return original_schedule(self, *args, **kwargs)

        running = self.running
        waiting = self.waiting
        if running is None or waiting is None:
            return original_schedule(self, *args, **kwargs)

        if not state.policy.enable_phase1_scheduler:
            state.metrics.record_scheduler_decision(False)
            return original_schedule(self, *args, **kwargs)

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
            return original_schedule(self, *args, **kwargs)

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
                state.phase1_virtual_token_caps = {
                    str(req_id): int(chunk)
                    for req_id, chunk in state.phase1_virtual_token_caps.items()
                    if str(req_id) in state.phase1_ingress_virtuals and int(chunk) > 0
                }
            else:
                state.phase1_virtual_token_caps.clear()
            ingress_eager_chunk: Optional[int] = None
            scheduler_chunk_raw: Optional[int] = None
            direct_chunk_raw: Optional[int] = None
            cohort_target_raw: Optional[int] = None

            snapshot, max_wait_us = _collect_live_snapshot(waiting, running)
            ingress_candidate = _phase1_find_ingress_virtual_candidate(
                state,
                snapshot=snapshot,
            )
            lengths = [int(rem) for _, rem in snapshot]
            cohort = None
            if ingress_candidate is not None:
                ingress_virtual, _ingress_seq_group, ingress_remaining = ingress_candidate
                live_cohort = _phase1_live_cohort_from_snapshot(snapshot, state.policy)
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
                    queue_len=len(lengths),
                    wait_us=max_wait_us,
                    slice_eligible=False,
                )
            else:
                if not _need_wave_slice(lengths, state.policy):
                    state.metrics.record_phase1_probe(
                        reason="no_need_wave_slice",
                        short_len=min(lengths) if lengths else None,
                        long_len=max(lengths) if lengths else None,
                        queue_len=len(lengths),
                        wait_us=max_wait_us,
                        slice_eligible=False,
                    )
                    state.metrics.record_scheduler_decision(False)
                    state.phase1_sticky_req_id = None
                    state.phase1_sticky_chunk = None
                    state.phase1_sticky_ttl_left = 0
                    if not waiting and not running:
                        state.phase1_explicit_plans.clear()
                    return original_schedule(self, *args, **kwargs)

                cohort = _phase1_live_cohort_from_snapshot(snapshot, state.policy)
                if cohort is None:
                    state.metrics.record_phase1_probe(
                        reason="no_cohort",
                        short_len=min(lengths) if lengths else None,
                        long_len=max(lengths) if lengths else None,
                        queue_len=len(lengths),
                        wait_us=max_wait_us,
                        slice_eligible=False,
                    )
                    state.metrics.record_scheduler_decision(False)
                    return original_schedule(self, *args, **kwargs)

            long_seq_group = _phase1_find_seq_group_by_request_id(snapshot, cohort.long_req_id)

            short_len = int(cohort.representative_short_len)
            long_len = int(cohort.long_len)
            queue_len = len(waiting) + len(running)
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
                ingress_min = max(1, int(state.policy.phase1_ingress_min_chunk))
                cohort_target_raw = int(_phase1_cohort_target_len(cohort, state.policy))
                eager_target = min(upper, int(cohort_target_raw))
                if upper >= ingress_min:
                    eager_target = max(eager_target, ingress_min)
                eager_chunk = _phase1_authoritative_chunk(
                    state,
                    target=int(eager_target),
                    short_len=int(short_len),
                    upper=int(upper),
                )
                eager_floor = _phase1_authoritative_short_floor(
                    state,
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
                snapshot=snapshot,
                t_wait_us=max_wait_us,
                queue_length=adjusted_queue_len,
                baseline_chunk=baseline_chunk,
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
                    max(1, int(state.policy.phase1_ingress_min_chunk)),
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
                _phase1_update_sticky_chunk(
                    state=state,
                    cohort=cohort,
                    chosen_chunk=best_chunk,
                    applied=False,
                )
                return original_schedule(self, *args, **kwargs)
        except Exception as exc:
            logger.exception("Wave-Slice Phase I pre-check failed; fallback to original scheduler.")
            state.metrics.record_phase1_probe(
                reason=f"precheck_exception:{type(exc).__name__}",
            )
            state.metrics.record_scheduler_decision(False)
            return original_schedule(self, *args, **kwargs)

        hidden_long_tasks: list[Any] = []
        state.metrics.record_scheduler_decision(True)
        short_token_mass = _phase1_effective_short_token_mass(
            cohort.short_lengths,
            short_len=short_len,
            best_chunk=best_chunk,
            policy=state.policy,
        )
        waiting_short_count = sum(
            1
            for sg in waiting
            if 0 < (_safe_prefill_uncomputed_tokens(sg) or 0) <= max(best_chunk, short_len)
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
            state.phase1_shadow_seq_lens.clear()
            if cohort.long_req_id and best_chunk < long_len:
                state.phase1_virtual_token_caps[str(cohort.long_req_id)] = int(best_chunk)
                state.phase1_public_skip_rewrite_requests.add(str(cohort.long_req_id))
                state.metrics.record_phase1_virtual_cap_probe(target_set=True)
                logger.info(
                    "[Wave-Slice][P1-virtual-target] req=%s target_chunk=%d long_len=%d baseline_chunk=%s",
                    str(cohort.long_req_id),
                    int(best_chunk),
                    int(long_len),
                    str(baseline_chunk) if baseline_chunk is not None else "none",
                )
            use_seq_len_shadow = not bool(state.policy.phase1_ingress_direct_authoritative)
            if use_seq_len_shadow and long_seq_group is not None and best_chunk < long_len:
                _phase1_apply_sequence_len_shadow(
                    state=state,
                    seq_group=long_seq_group,
                    target_chunk=best_chunk,
                )

            if can_phase1_tick_hide and waiting_short_count > 0:
                first_wait_len = _safe_remaining_tokens(waiting[0]) or 0
                if first_wait_len > 0 and first_wait_len <= max(best_chunk, short_len):
                    new_running = [] if isinstance(running, list) else collections.deque()
                    for seq_group in running:
                        remaining = _safe_remaining_tokens(seq_group) or 0
                        if remaining > best_chunk:
                            hidden_long_tasks.append(seq_group)
                        else:
                            new_running.append(seq_group)
                    self.running = new_running

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
                len(hidden_long_tasks),
                str(new_threshold) if new_threshold is not None else "unchanged",
                str(new_budget) if new_budget is not None else "unchanged",
            )
            outputs = original_schedule(self, *args, **kwargs)
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
            state.phase1_shadow_seq_lens.clear()
            state.phase1_virtual_token_caps.clear()
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
            scheduler_outputs = getattr(self, state.scheduler_method_name)(*args, **kwargs)
            now = time.time()

            if state.policy.enable_phase1_scheduler:
                scheduler_outputs, _ = _phase1_force_public_schedule_rewrite(
                    state=state,
                    scheduler_obj=self,
                    scheduler_outputs=scheduler_outputs,
                )

            if not self.cache_config.enable_prefix_caching:
                common_computed_block_nums = []

            allow_async_output_proc = self.use_async_output_proc
            seq_group_metadata_list: list[Any] = []
            for i, scheduled_seq_group in enumerate(scheduler_outputs.scheduled_seq_groups):
                seq_group = scheduled_seq_group.seq_group
                token_chunk_size = scheduled_seq_group.token_chunk_size
                seq_group.maybe_set_first_scheduled_time(now)
                trace_request_id = _safe_request_id(seq_group)
                trace_is_prompt = None
                trace_num_computed_tokens = None
                try:
                    trace_is_prompt = bool(seq_group.is_prefill())
                except Exception:
                    trace_is_prompt = None
                try:
                    seqs_for_trace = seq_group.get_seqs()
                    if seqs_for_trace:
                        trace_num_computed_tokens = int(
                            seqs_for_trace[0].data.get_num_computed_tokens()
                        )
                except Exception:
                    trace_num_computed_tokens = None
                if trace_request_id:
                    state.metrics.record_phase1_step_trace(
                        request_id=str(trace_request_id),
                        event="public_schedule_group",
                        is_prefill=trace_is_prompt,
                        token_chunk_size=int(token_chunk_size),
                        num_computed_tokens=trace_num_computed_tokens,
                    )

                seq_group_metadata = self._seq_group_metadata_cache[self.cache_id].get_object()
                seq_group_metadata.seq_data.clear()
                seq_group_metadata.block_tables.clear()

                seq_data: dict[int, Any] = {}
                block_tables: dict[int, list[int]] = {}

                if seq_group.is_encoder_decoder():
                    encoder_seq = seq_group.get_encoder_seq()
                    assert encoder_seq is not None
                    encoder_seq_data = encoder_seq.data
                    cross_block_table = self.block_manager.get_cross_block_table(seq_group)
                else:
                    encoder_seq_data = None
                    cross_block_table = None

                for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                    seq_id = seq.seq_id
                    seq_data[seq_id] = seq.data
                    block_tables[seq_id] = self.block_manager.get_block_table(seq)
                    self.block_manager.access_all_blocks_in_seq(seq, now)

                if self.cache_config.enable_prefix_caching:
                    common_computed_block_nums = self.block_manager.get_common_computed_block_ids(
                        seq_group.get_seqs(status=SequenceStatus.RUNNING)
                    )

                do_sample = True
                is_prompt = seq_group.is_prefill()
                is_first_prefill = False
                if is_prompt:
                    seqs = seq_group.get_seqs()
                    assert len(seqs) == 1
                    num_computed_tokens = seqs[0].data.get_num_computed_tokens()
                    is_first_prefill = num_computed_tokens == 0
                    if token_chunk_size + num_computed_tokens < seqs[0].data.get_len():
                        do_sample = False

                if is_first_prefill or not self.scheduler_config.send_delta_data:
                    seq_group_metadata = SequenceGroupMetadata(
                        request_id=seq_group.request_id,
                        is_prompt=is_prompt,
                        seq_data=seq_data,
                        sampling_params=seq_group.sampling_params,
                        block_tables=block_tables,
                        do_sample=do_sample,
                        pooling_params=seq_group.pooling_params,
                        token_chunk_size=token_chunk_size,
                        lora_request=seq_group.lora_request,
                        computed_block_nums=common_computed_block_nums,
                        encoder_seq_data=encoder_seq_data,
                        cross_block_table=cross_block_table,
                        state=seq_group.state,
                        token_type_ids=seq_group.token_type_ids,
                        multi_modal_data=(
                            seq_group.multi_modal_data
                            if scheduler_outputs.num_prefill_groups > 0 else None
                        ),
                        multi_modal_placeholders=(
                            seq_group.multi_modal_placeholders
                            if scheduler_outputs.num_prefill_groups > 0 else None
                        ),
                    )
                else:
                    seq_data_delta = {}
                    for seq_id, data in seq_data.items():
                        seq_data_delta[seq_id] = data.get_delta_and_reset()
                    seq_group_metadata = SequenceGroupMetadataDelta(
                        seq_data_delta,
                        seq_group.request_id,
                        block_tables,
                        is_prompt,
                        do_sample=do_sample,
                        token_chunk_size=token_chunk_size,
                        computed_block_nums=common_computed_block_nums,
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


def _safe_import_torch() -> Optional[Any]:
    try:
        import torch
        return torch
    except Exception:
        return None


def _infer_runner_device(model_runner: Any, torch_mod: Any) -> Optional[Any]:
    device = getattr(model_runner, "device", None)
    if device is None:
        model = getattr(model_runner, "model", None)
        device = getattr(model, "device", None)
    if device == "":
        try:
            device = f"cuda:{int(torch_mod.cuda.current_device())}"
        except Exception:
            device = "cuda"
    if device is None:
        return None
    try:
        return torch_mod.device(device)
    except Exception:
        return None


def _get_or_create_runner_stream_state(
    model_runner: Any,
    torch_mod: Any,
    policy: WaveSlicePolicy,
) -> Optional[_RunnerStreamState]:
    stream_state = getattr(model_runner, "_wave_slice_stream_state", None)
    if isinstance(stream_state, _RunnerStreamState):
        return stream_state

    device = _infer_runner_device(model_runner, torch_mod)
    if device is None or device.type != "cuda":
        return None
    try:
        priority = 0 if _is_phase2_strict(policy) else -1
        fast_stream = torch_mod.cuda.Stream(device=device, priority=priority)
    except Exception:
        return None
    stream_state = _RunnerStreamState(device=device, fast_stream=fast_stream)
    setattr(model_runner, "_wave_slice_stream_state", stream_state)
    return stream_state


def _build_model_runner_hook(state: _PatchState) -> Callable[..., Any]:
    original_execute = state.original_execute_model
    if original_execute is None:
        raise RuntimeError("internal error: original execute_model is missing")

    @functools.wraps(original_execute)
    def _wave_execute_model_hook(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not state.policy.enable_phase2_modelrunner:
            return original_execute(self, *args, **kwargs)

        model_input = args[0] if len(args) > 0 else kwargs.get("model_input")
        decision = _phase2_decide(model_input, state.policy, runner_self=self)
        state.metrics.record_phase2_decision(decision.apply, decision.reason)
        if not decision.apply:
            return original_execute(self, *args, **kwargs)

        torch_mod = _safe_import_torch()
        if torch_mod is None or not torch_mod.cuda.is_available():
            return original_execute(self, *args, **kwargs)

        strict_mode = _is_phase2_strict(state.policy)
        stream_state = _get_or_create_runner_stream_state(self, torch_mod, state.policy)
        if stream_state is None:
            return original_execute(self, *args, **kwargs)

        try:
            v1_unbind_output = _try_v1_true_unbind_execute(
                runner_self=self,
                state=state,
                original_execute=original_execute,
                model_input=model_input,
                args=args,
                kwargs=kwargs,
                torch_mod=torch_mod,
                stream_state=stream_state,
            )
            if v1_unbind_output is not None:
                state.metrics.record_phase2_v1_unbind()
                return v1_unbind_output

            if strict_mode:
                # Enforce a conservative boundary to reduce numerical drift.
                torch_mod.cuda.synchronize(device=stream_state.device)
            main_stream = torch_mod.cuda.current_stream(device=stream_state.device)
            stream_state.fast_stream.wait_stream(main_stream)
            with torch_mod.cuda.stream(stream_state.fast_stream):
                output = _call_original_execute_with_model_input(
                    original_execute,
                    self,
                    model_input,
                    args,
                    kwargs,
                )
                done_evt = torch_mod.cuda.Event(enable_timing=False)
                done_evt.record(stream_state.fast_stream)

            async_mode = (not strict_mode) and _is_phase2_async_experimental(state.policy)
            if async_mode:
                stream_state.inflight_events.append(done_evt)
                max_inflight = max(1, int(state.policy.phase2_max_inflight_events))
                while len(stream_state.inflight_events) > max_inflight:
                    evt = stream_state.inflight_events.popleft()
                    evt.synchronize()
            else:
                main_stream.wait_event(done_evt)
                if strict_mode or state.policy.phase2_host_sync_after_dispatch:
                    done_evt.synchronize()
            if strict_mode:
                torch_mod.cuda.synchronize(device=stream_state.device)

            logger.info(
                "[Wave-Slice][P2] mode=%s dispatch=%s reason=%s prefills=%d decodes=%d lens=%s lora_ranks=%s",
                state.policy.phase2_consistency_mode,
                state.policy.phase2_dispatch_mode,
                decision.reason,
                decision.num_prefills,
                decision.num_decode_tokens,
                decision.prefill_lens,
                decision.lora_ranks,
            )
            return output
        except Exception:
            logger.exception("Wave-Slice Phase II stream dispatch failed; fallback to original execute_model.")
            return original_execute(self, *args, **kwargs)

    _wave_execute_model_hook.__wave_slice_phase2_hook__ = True  # type: ignore[attr-defined]
    return _wave_execute_model_hook


def _build_add_request_hook(state: _PatchState) -> Callable[..., Any]:
    original_add_request = state.original_add_request
    if original_add_request is None:
        raise RuntimeError("internal error: original add_request is missing")

    @functools.wraps(original_add_request)
    def _wave_add_request_hook(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_add_request(self, *args, **kwargs)
        try:
            request_id = None
            if len(args) >= 1:
                request_id = str(args[0])
            elif "request_id" in kwargs:
                request_id = str(kwargs["request_id"])
            if not request_id:
                return result

            prompt_obj = args[1] if len(args) >= 2 else kwargs.get("prompt")
            if prompt_obj is None:
                prompt_obj = kwargs.get("prompt_token_ids")
            lora_request = kwargs.get("lora_request")
            if lora_request is None and len(args) >= 4:
                lora_request = args[3]
            input_tokens = _estimate_prompt_tokens(
                prompt_obj,
                engine_self=self,
                lora_request=lora_request,
            )
            solo_us = _estimate_solo_us(state.brain, input_tokens)
            is_short = (input_tokens is not None) and (input_tokens <= state.policy.metrics_short_request_tokens)
            state.metrics.register_request(
                request_id,
                arrival_s=time.perf_counter(),
                input_tokens=input_tokens,
                solo_us=solo_us,
                is_short=is_short,
            )
            _phase1_maybe_seed_ingress_virtual(
                state,
                request_id=request_id,
                input_tokens=input_tokens,
            )
        except Exception:
            logger.exception("Wave-Slice metrics add_request hook failed.")
        return result

    _wave_add_request_hook.__wave_slice_metrics_hook__ = True  # type: ignore[attr-defined]
    return _wave_add_request_hook


def _build_step_hook(state: _PatchState) -> Callable[..., Any]:
    original_step = state.original_step
    if original_step is None:
        raise RuntimeError("internal error: original step is missing")

    @functools.wraps(original_step)
    def _wave_step_hook(self: Any, *args: Any, **kwargs: Any) -> Any:
        outputs = original_step(self, *args, **kwargs)
        try:
            for out in outputs or []:
                try:
                    if bool(getattr(out, "finished", False)):
                        req_id = str(getattr(out, "request_id", ""))
                        if req_id:
                            state.phase1_explicit_plans.pop(req_id, None)
                            state.phase1_active_prompt_tokens.pop(req_id, None)
                            state.phase1_ingress_virtuals.pop(req_id, None)
                            state.phase1_virtual_token_caps.pop(req_id, None)
                except Exception:
                    continue
            state.metrics.observe_engine_outputs(outputs, now_s=time.perf_counter())
        except Exception:
            logger.exception("Wave-Slice metrics step hook failed.")
        return outputs

    _wave_step_hook.__wave_slice_metrics_hook__ = True  # type: ignore[attr-defined]
    return _wave_step_hook


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


def _publish_autoinject_env(model_name: str, gamma: float, policy: WaveSlicePolicy) -> None:
    try:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        prev_pythonpath = os.environ.get("PYTHONPATH")
        prev_vllm_plugins = os.environ.get("VLLM_PLUGINS")
        os.environ[_AUTO_ENV_ENABLED] = "1"
        os.environ[_AUTO_ENV_MODEL] = str(model_name)
        os.environ[_AUTO_ENV_GAMMA] = str(float(gamma))
        os.environ[_AUTO_ENV_POLICY] = json.dumps(asdict(policy), sort_keys=True)
        os.environ[_AUTO_ENV_PREV_PYTHONPATH] = prev_pythonpath if prev_pythonpath is not None else ""
        os.environ[_AUTO_ENV_PREV_VLLM_PLUGINS] = prev_vllm_plugins if prev_vllm_plugins is not None else ""

        py_entries = [p for p in (prev_pythonpath or "").split(os.pathsep) if p]
        if repo_root not in py_entries:
            py_entries.insert(0, repo_root)
        os.environ["PYTHONPATH"] = os.pathsep.join(py_entries)

        plugin_entries = [p.strip() for p in (prev_vllm_plugins or "").split(",") if p.strip()]
        if "waveslice_autoinject" not in plugin_entries:
            plugin_entries.append("waveslice_autoinject")
        os.environ["VLLM_PLUGINS"] = ",".join(plugin_entries)
    except Exception:
        logger.exception("[Wave-Slice] failed to publish child auto-inject env.")


def _clear_autoinject_env() -> None:
    prev_pythonpath = os.environ.pop(_AUTO_ENV_PREV_PYTHONPATH, None)
    prev_vllm_plugins = os.environ.pop(_AUTO_ENV_PREV_VLLM_PLUGINS, None)
    if prev_pythonpath is not None:
        if prev_pythonpath:
            os.environ["PYTHONPATH"] = prev_pythonpath
        else:
            os.environ.pop("PYTHONPATH", None)
    if prev_vllm_plugins is not None:
        if prev_vllm_plugins:
            os.environ["VLLM_PLUGINS"] = prev_vllm_plugins
        else:
            os.environ.pop("VLLM_PLUGINS", None)
    for key in (_AUTO_ENV_ENABLED, _AUTO_ENV_MODEL, _AUTO_ENV_GAMMA, _AUTO_ENV_POLICY):
        os.environ.pop(key, None)


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

        setattr(scheduler_cls, scheduler_method_name, _build_scheduler_hook(state))
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
                original_step = getattr(llm_engine_cls, "step", None)
                if callable(original_add_request) and callable(original_step):
                    state.llm_engine_cls = llm_engine_cls
                    state.original_add_request = original_add_request
                    state.original_step = original_step
                    llm_engine_cls.add_request = _build_add_request_hook(state)
                    llm_engine_cls.step = _build_step_hook(state)
                else:
                    logger.warning("[Wave-Slice] skip metrics hooks: add_request/step missing.")
            except Exception:
                logger.exception("[Wave-Slice] metrics hook injection failed; continue without metrics hooks.")

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

        if state.llm_engine_cls is not None:
            if state.original_add_request is not None:
                try:
                    state.llm_engine_cls.add_request = state.original_add_request
                except Exception:
                    logger.exception("[Wave-Slice] failed to restore LLMEngine.add_request")
            if state.original_step is not None:
                try:
                    state.llm_engine_cls.step = state.original_step
                except Exception:
                    logger.exception("[Wave-Slice] failed to restore LLMEngine.step")

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
    report = state.metrics.summary()
    if reset:
        state.metrics.reset()
    return report


def reset_wave_slice_metrics() -> None:
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
