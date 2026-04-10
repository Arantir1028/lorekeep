from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable, Optional

AUTO_ENV_METRICS_FILE = "WAVESLICE_AUTOINJECT_METRICS_FILE"


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
        self._phase2_debug_counter: dict[str, int] = {}
        self._phase2_true_unbind_gate_reasons: dict[str, int] = {}
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
        self._escape_lane_activations = 0
        self._escape_lane_active_sum = 0.0
        self._escape_lane_deferred_sum = 0.0
        self._escape_lane_ttl_sum = 0.0
        self._escape_lane_seen_events = 0
        self._escape_lane_seen_active_hits = 0
        self._escape_lane_finished_events = 0
        self._escape_lane_finished_active_hits = 0
        self._escape_lane_last_active_ids: list[str] = []
        self._escape_lane_last_deferred_ids: list[str] = []

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

    @staticmethod
    def _emit_cross_process_event(kind: str, payload: dict[str, Any]) -> None:
        path = os.environ.get(AUTO_ENV_METRICS_FILE, "").strip()
        if not path:
            return
        try:
            record = {
                "kind": str(kind),
                "pid": int(os.getpid()),
                "payload": payload,
            }
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                fh.write("\n")
        except Exception:
            pass

    def reset(self) -> None:
        with self._lock:
            self._requests.clear()
            self._phase2_total = 0
            self._phase2_applied = 0
            self._phase2_v1_unbind_applied = 0
            self._phase2_reason_counter.clear()
            self._phase2_debug_counter.clear()
            self._phase2_true_unbind_gate_reasons.clear()
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
            self._escape_lane_activations = 0
            self._escape_lane_active_sum = 0.0
            self._escape_lane_deferred_sum = 0.0
            self._escape_lane_ttl_sum = 0.0
            self._escape_lane_seen_events = 0
            self._escape_lane_seen_active_hits = 0
            self._escape_lane_finished_events = 0
            self._escape_lane_finished_active_hits = 0
            self._escape_lane_last_active_ids = []
            self._escape_lane_last_deferred_ids = []

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
        self._emit_cross_process_event(
            "scheduler_decision",
            {"applied": bool(applied)},
        )

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
        self._emit_cross_process_event(
            "phase1_choice",
            {
                "chosen_chunk": int(chosen_chunk) if chosen_chunk is not None else None,
                "baseline_chunk": int(baseline_chunk) if baseline_chunk is not None else None,
                "explicit_plan": bool(explicit_plan),
            },
        )

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
        self._emit_cross_process_event(
            "phase1_rewrite",
            {
                "rewritten_groups": int(rewritten_groups),
                "old_chunk_sum": int(max(0, old_chunk_sum)),
                "new_chunk_sum": int(max(0, new_chunk_sum)),
                "token_delta_sum": int(max(0, token_delta_sum)),
            },
        )

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
                pass
            else:
                self._phase1_virtual_cap_applied += 1
                self._phase1_virtual_cap_old_sum += float(max(0, old_total_tokens))
                self._phase1_virtual_cap_new_sum += float(max(0, new_total_tokens))
        self._emit_cross_process_event(
            "phase1_virtual_cap",
            {
                "old_total_tokens": int(max(0, old_total_tokens)),
                "new_total_tokens": int(max(0, new_total_tokens)),
                "applied": bool(applied),
            },
        )

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
        self._emit_cross_process_event(
            "phase1_virtual_cap_probe",
            {
                "target_set": bool(target_set),
                "helper_called": bool(helper_called),
                "prefill_call": bool(prefill_call),
                "target_hit": bool(target_hit),
            },
        )

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
        if not self._trace_request_key(request_id) and target_chunk is None:
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
        self._emit_cross_process_event(
            "phase1_step_trace",
            {
                "request_id": str(request_id),
                "event": str(event),
                "is_prefill": (bool(is_prefill) if is_prefill is not None else None),
                "token_chunk_size": (int(token_chunk_size) if token_chunk_size is not None else None),
                "num_computed_tokens": (
                    int(num_computed_tokens) if num_computed_tokens is not None else None
                ),
                "uncached": (int(uncached) if uncached is not None else None),
                "cached": (int(cached) if cached is not None else None),
                "target_chunk": (int(target_chunk) if target_chunk is not None else None),
            },
        )

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
        self._emit_cross_process_event(
            "phase1_probe",
            {
                "reason": str(reason),
                "short_len": (int(short_len) if short_len is not None else None),
                "long_len": (int(long_len) if long_len is not None else None),
                "baseline_chunk": (int(baseline_chunk) if baseline_chunk is not None else None),
                "best_chunk": (int(best_chunk) if best_chunk is not None else None),
                "queue_len": (int(queue_len) if queue_len is not None else None),
                "wait_us": (float(wait_us) if wait_us is not None else None),
                "slice_eligible": bool(slice_eligible),
            },
        )

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
        self._emit_cross_process_event(
            "phase1_proposal",
            {
                "scheduler_chunk": (
                    int(scheduler_chunk) if scheduler_chunk is not None else None
                ),
                "direct_chunk": (
                    int(direct_chunk) if direct_chunk is not None else None
                ),
                "cohort_target": (
                    int(cohort_target) if cohort_target is not None else None
                ),
                "direct_won": bool(direct_won),
            },
        )

    def record_phase2_decision(
        self,
        applied: bool,
        reason: str,
        **payload: Any,
    ) -> None:
        with self._lock:
            self._phase2_total += 1
            if applied:
                self._phase2_applied += 1
            self._phase2_reason_counter[reason] = self._phase2_reason_counter.get(reason, 0) + 1
        self._emit_cross_process_event(
            "phase2_decision",
            {
                "applied": bool(applied),
                "reason": str(reason),
                **payload,
            },
        )

    def record_phase2_v1_unbind(self) -> None:
        with self._lock:
            self._phase2_v1_unbind_applied += 1
        self._emit_cross_process_event("phase2_v1_unbind", {})

    def record_phase2_debug_counter(self, name: str, amount: int = 1) -> None:
        counter_name = str(name or "").strip()
        delta = int(amount)
        if not counter_name or delta == 0:
            return
        with self._lock:
            self._phase2_debug_counter[counter_name] = (
                self._phase2_debug_counter.get(counter_name, 0) + delta
            )
        self._emit_cross_process_event(
            "phase2_debug_counter",
            {"name": counter_name, "amount": delta},
        )

    def record_phase2_true_unbind_gate(self, reason: str) -> None:
        gate_reason = str(reason or "").strip()
        if not gate_reason:
            return
        with self._lock:
            self._phase2_true_unbind_gate_reasons[gate_reason] = (
                self._phase2_true_unbind_gate_reasons.get(gate_reason, 0) + 1
            )
        self._emit_cross_process_event(
            "phase2_true_unbind_gate",
            {"reason": gate_reason},
        )

    def record_escape_lane_activation(
        self,
        *,
        active_ids: Iterable[str],
        deferred_ids: Iterable[str],
        lane_ttl: int,
    ) -> None:
        active = [str(rid) for rid in active_ids if str(rid)]
        deferred = [str(rid) for rid in deferred_ids if str(rid)]
        with self._lock:
            self._escape_lane_activations += 1
            self._escape_lane_active_sum += float(len(active))
            self._escape_lane_deferred_sum += float(len(deferred))
            self._escape_lane_ttl_sum += float(max(0, int(lane_ttl)))
            self._escape_lane_last_active_ids = active[:16]
            self._escape_lane_last_deferred_ids = deferred[:16]
        self._emit_cross_process_event(
            "escape_lane_activation",
            {
                "active_count": int(len(active)),
                "deferred_count": int(len(deferred)),
                "lane_ttl": int(max(0, int(lane_ttl))),
                "active_ids": active[:16],
                "deferred_ids": deferred[:16],
            },
        )

    def record_escape_lane_observation(
        self,
        *,
        active_ids: Iterable[str],
        seen_request_ids: Iterable[str] = (),
        finished_request_ids: Iterable[str] = (),
    ) -> None:
        active = {str(rid) for rid in active_ids if str(rid)}
        seen = [str(rid) for rid in seen_request_ids if str(rid)]
        finished = [str(rid) for rid in finished_request_ids if str(rid)]
        if active:
            with self._lock:
                self._escape_lane_seen_events += 1
                self._escape_lane_seen_active_hits += sum(1 for rid in seen if rid in active)
                if finished:
                    self._escape_lane_finished_events += 1
                    self._escape_lane_finished_active_hits += sum(1 for rid in finished if rid in active)
        self._emit_cross_process_event(
            "escape_lane_observation",
            {
                "active_count": int(len(active)),
                "active_ids": sorted(active)[:16],
                "seen_count": int(len(seen)),
                "seen_active_hits": int(sum(1 for rid in seen if rid in active)),
                "finished_count": int(len(finished)),
                "finished_active_hits": int(sum(1 for rid in finished if rid in active)),
                "seen_ids": seen[:16],
                "finished_ids": finished[:16],
            },
        )

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
            phase2_debug_counter = dict(self._phase2_debug_counter)
            phase2_true_unbind_gate_reasons = dict(self._phase2_true_unbind_gate_reasons)
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
            escape_lane_activations = self._escape_lane_activations
            escape_lane_active_sum = self._escape_lane_active_sum
            escape_lane_deferred_sum = self._escape_lane_deferred_sum
            escape_lane_ttl_sum = self._escape_lane_ttl_sum
            escape_lane_seen_events = self._escape_lane_seen_events
            escape_lane_seen_active_hits = self._escape_lane_seen_active_hits
            escape_lane_finished_events = self._escape_lane_finished_events
            escape_lane_finished_active_hits = self._escape_lane_finished_active_hits
            escape_lane_last_active_ids = list(self._escape_lane_last_active_ids)
            escape_lane_last_deferred_ids = list(self._escape_lane_last_deferred_ids)
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
                "escape_lane": {
                    "activations": float(escape_lane_activations),
                    "active_count_avg": (
                        escape_lane_active_sum / escape_lane_activations if escape_lane_activations else None
                    ),
                    "deferred_count_avg": (
                        escape_lane_deferred_sum / escape_lane_activations if escape_lane_activations else None
                    ),
                    "ttl_avg": (
                        escape_lane_ttl_sum / escape_lane_activations if escape_lane_activations else None
                    ),
                    "seen_events": float(escape_lane_seen_events),
                    "seen_active_hits": float(escape_lane_seen_active_hits),
                    "seen_active_hits_per_event": (
                        escape_lane_seen_active_hits / escape_lane_seen_events if escape_lane_seen_events else None
                    ),
                    "finished_events": float(escape_lane_finished_events),
                    "finished_active_hits": float(escape_lane_finished_active_hits),
                    "finished_active_hits_per_event": (
                        escape_lane_finished_active_hits / escape_lane_finished_events
                        if escape_lane_finished_events else None
                    ),
                    "last_active_ids": escape_lane_last_active_ids,
                    "last_deferred_ids": escape_lane_last_deferred_ids,
                },
                "debug": {
                    "counters": phase2_debug_counter,
                    "true_unbind_gate_reasons": phase2_true_unbind_gate_reasons,
                },
            },
            "ttft_ms_all": _stat(ttft_ms_all),
            "ttft_ms_short": _stat(ttft_ms_short),
            "slowdown_all": _stat(slowdown_all),
            "slowdown_short": _stat(slowdown_short),
        }
