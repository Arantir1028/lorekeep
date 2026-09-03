"""Runtime metrics collected by the WaveSlice vLLM integration."""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

RUNTIME_METRICS_FILE_ENV = "WAVESLICE_METRICS_FILE"


@dataclass
class _RequestMetric:
    arrival_s: float | None = None
    first_token_s: float | None = None
    finish_s: float | None = None
    input_tokens: int | None = None
    solo_us: float | None = None
    is_short: bool | None = None
    finished: bool = False
    generated_tokens: int = 0


class WaveSliceMetrics:
    """Thread-safe request and scheduler metrics used by the runtime hooks."""

    def __init__(self, short_threshold_tokens: int = 256):
        self._short_threshold_tokens = short_threshold_tokens
        self._lock = threading.RLock()
        self._requests: dict[str, _RequestMetric] = {}
        self._clear_aggregate_state()

    def _clear_aggregate_state(self) -> None:
        self._n: dict[str, float | int] = {}
        self._phase2_reason_counter: dict[str, int] = {}
        self._priority_lane_last_active_ids: list[str] = []
        self._priority_lane_last_deferred_ids: list[str] = []

    def _add(self, **values: float) -> None:
        for name, value in values.items():
            self._n[name] = self._n.get(name, 0) + value

    def _get(self, name: str) -> float:
        return self._n.get(name, 0)

    @staticmethod
    def _percentile(values: list[float], p: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        k = (len(ordered) - 1) * max(0.0, min(100.0, p)) / 100.0
        lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)

    @staticmethod
    def _emit_cross_process_event(payload: dict[str, Any]) -> None:
        path = os.environ.get(RUNTIME_METRICS_FILE_ENV, "").strip()
        if not path:
            return
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {"pid": os.getpid(), "payload": payload}, ensure_ascii=False, sort_keys=True
                    )
                    + "\n"
                )
        except OSError:
            return

    def _record(
        self,
        values: dict[str, float | int],
        *,
        reason: str | None = None,
        active_ids: list[str] | None = None,
        deferred_ids: list[str] | None = None,
    ) -> None:
        with self._lock:
            self._add(**values)
            if reason:
                self._phase2_reason_counter[reason] = self._phase2_reason_counter.get(reason, 0) + 1
            if active_ids is not None:
                self._priority_lane_last_active_ids = active_ids
                self._priority_lane_last_deferred_ids = deferred_ids or []
        payload: dict[str, Any] = {"values": values}
        if reason:
            payload["reason"] = reason
        if active_ids is not None:
            payload.update(last_active_ids=active_ids, last_deferred_ids=deferred_ids or [])
        self._emit_cross_process_event(payload)

    def reset(self) -> None:
        with self._lock:
            self._requests.clear()
            self._clear_aggregate_state()

    def register_request(
        self,
        request_id: str,
        *,
        arrival_s: float | None = None,
        input_tokens: int | None = None,
        solo_us: float | None = None,
        is_short: bool | None = None,
    ) -> None:
        with self._lock:
            rec = self._requests.setdefault(request_id, _RequestMetric())
            if arrival_s is not None and rec.arrival_s is None:
                rec.arrival_s = arrival_s
            if input_tokens is not None:
                rec.input_tokens = input_tokens
            if solo_us is not None:
                rec.solo_us = solo_us
            if is_short is not None:
                rec.is_short = is_short
            elif rec.is_short is None and rec.input_tokens is not None:
                rec.is_short = rec.input_tokens <= self._short_threshold_tokens

    def observe_scheduler_request(
        self,
        request_id: str,
        *,
        total_tokens: int | None = None,
        solo_us: float | None = None,
        is_short: bool | None = None,
    ) -> None:
        self.register_request(
            request_id, input_tokens=total_tokens, solo_us=solo_us, is_short=is_short
        )

    def snapshot_requests(
        self, request_ids: Iterable[str] | None = None
    ) -> dict[str, dict[str, Any]]:
        wanted = None if request_ids is None else {str(rid) for rid in request_ids}
        with self._lock:
            items = [
                (rid, rec)
                for rid, rec in self._requests.items()
                if wanted is None or str(rid) in wanted
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
        self._record({"sched_total": 1, "sched_applied": int(applied)})

    def record_phase1_choice(
        self, *, chosen_chunk: int | None, baseline_chunk: int | None, explicit_plan: bool
    ) -> None:
        values: dict[str, float | int] = {"explicit_total": int(explicit_plan)}
        if baseline_chunk and baseline_chunk > 0:
            values.update(baseline_sum=baseline_chunk, baseline_count=1)
        if chosen_chunk and chosen_chunk > 0:
            values.update(chosen_sum=chosen_chunk, chosen_count=1)
        if chosen_chunk and chosen_chunk > 0 and baseline_chunk and baseline_chunk > 0:
            values.update(slice_ratio_sum=chosen_chunk / baseline_chunk, slice_ratio_count=1)
        self._record(values)

    def record_phase1_rewrite(
        self, *, rewritten_groups: int, old_chunk_sum: int, new_chunk_sum: int, token_delta_sum: int
    ) -> None:
        if rewritten_groups <= 0:
            return
        self._record(
            {
                "rewrite_applied": 1,
                "rewrite_groups": rewritten_groups,
                "rewrite_old": max(0, int(old_chunk_sum)),
                "rewrite_new": max(0, int(new_chunk_sum)),
                "rewrite_delta": max(0, int(token_delta_sum)),
            }
        )

    def record_phase1_virtual_cap(
        self, *, old_total_tokens: int, new_total_tokens: int, applied: bool
    ) -> None:
        values = {"virtual_total": 1, "virtual_applied": int(applied)}
        if applied:
            values.update(
                virtual_old=max(0, int(old_total_tokens)), virtual_new=max(0, int(new_total_tokens))
            )
        self._record(values)

    def record_phase1_virtual_cap_probe(
        self,
        *,
        target_set: bool = False,
        helper_called: bool = False,
        prefill_call: bool = False,
        target_hit: bool = False,
    ) -> None:
        self._record(
            {
                "virtual_target_set": int(target_set),
                "virtual_helper": int(helper_called),
                "virtual_prefill": int(prefill_call),
                "virtual_hits": int(target_hit),
            }
        )

    def record_phase1_probe(
        self,
        *,
        short_len: int | None = None,
        long_len: int | None = None,
        baseline_chunk: int | None = None,
        best_chunk: int | None = None,
        queue_len: int | None = None,
        wait_us: float | None = None,
        slice_eligible: bool = False,
    ) -> None:
        aggregate: dict[str, float | int] = {
            "probe_total": 1,
            "probe_eligible": int(slice_eligible),
            "probe_best_lt_long": int(bool(best_chunk and long_len and 0 < best_chunk < long_len)),
        }
        for source, total, count in (
            (short_len, "probe_short", None),
            (long_len, "probe_long", None),
            (baseline_chunk, "probe_baseline", "probe_baseline_count"),
            (best_chunk, "probe_best", "probe_best_count"),
            (queue_len, "probe_queue", None),
            (wait_us, "probe_wait", None),
        ):
            if (
                source is not None
                and float(source) >= 0
                and (source > 0 or total in {"probe_queue", "probe_wait"})
            ):
                aggregate[total] = float(source)
                if count:
                    aggregate[count] = 1
        self._record(aggregate)

    def phase1_virtual_cap_hit_ratio(self) -> float:
        with self._lock:
            denominator = max(self._get("virtual_prefill"), self._get("virtual_target_set"))
            return min(1.0, self._get("virtual_hits") / denominator) if denominator else 0.0

    def record_phase1_runtime_adaptation(
        self,
        *,
        queue_len: int,
        waiting_short_count: int,
        effective_pressure: float,
        wall_pressure: float,
        short_urgency: float,
        target_fraction: float,
        target_chunk: int,
    ) -> None:
        self._record(
            {
                "runtime_total": 1,
                "runtime_queue": max(0, int(queue_len)),
                "runtime_waiting_short": max(0, int(waiting_short_count)),
                "runtime_pressure": effective_pressure,
                "runtime_wall": wall_pressure,
                "runtime_urgency": short_urgency,
                "runtime_fraction": target_fraction,
                "runtime_chunk": max(1, int(target_chunk)),
            }
        )

    def record_phase2_decision(self, applied: bool, reason: str) -> None:
        self._record({"phase2_total": 1, "phase2_applied": int(applied)}, reason=reason)

    def record_priority_lane_activation(
        self, *, active_ids: Iterable[str], deferred_ids: Iterable[str], lane_ttl: int
    ) -> None:
        active = [str(rid) for rid in active_ids if str(rid)]
        deferred = [str(rid) for rid in deferred_ids if str(rid)]
        self._record(
            {
                "priority_activations": 1,
                "priority_active": len(active),
                "priority_deferred": len(deferred),
                "priority_ttl": max(0, int(lane_ttl)),
            },
            active_ids=active[:16],
            deferred_ids=deferred[:16],
        )

    def record_priority_lane_observation(
        self,
        *,
        active_ids: Iterable[str],
        seen_request_ids: Iterable[str] = (),
        finished_request_ids: Iterable[str] = (),
    ) -> None:
        active = {str(rid) for rid in active_ids if str(rid)}
        seen, finished = (
            [str(rid) for rid in values if str(rid)]
            for values in (seen_request_ids, finished_request_ids)
        )
        seen_hits = sum(rid in active for rid in seen)
        finished_hits = sum(rid in active for rid in finished)
        if active:
            self._record(
                {
                    "priority_seen_events": 1,
                    "priority_seen_hits": seen_hits,
                    "priority_finished_events": int(bool(finished)),
                    "priority_finished_hits": finished_hits,
                }
            )

    def observe_engine_outputs(self, outputs: Any, now_s: float | None = None) -> None:
        now = time.perf_counter() if now_s is None else now_s
        for out in outputs or []:
            request_id = str(getattr(out, "request_id", ""))
            if not request_id:
                continue
            with self._lock:
                rec = self._requests.setdefault(request_id, _RequestMetric())
                payloads = getattr(out, "outputs", ()) or ()
                tokens = len(getattr(payloads[0], "token_ids", ()) or ()) if payloads else 0
                if tokens and rec.first_token_s is None and rec.arrival_s is not None:
                    rec.first_token_s = now
                rec.generated_tokens = max(rec.generated_tokens, tokens)
                if getattr(out, "finished", False):
                    rec.finished = True
                    rec.finish_s = now if rec.arrival_s is not None else rec.finish_s

    def summary(self, foreign: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            records, n = list(self._requests.values()), dict(self._n)
            reasons = dict(self._phase2_reason_counter)
            last_active, last_deferred = (
                list(self._priority_lane_last_active_ids),
                list(self._priority_lane_last_deferred_ids),
            )
        if foreign:
            for name, value in foreign["values"].items():
                n[name] = n.get(name, 0) + value
            for reason, count in foreign["reasons"].items():
                reasons[reason] = reasons.get(reason, 0) + count
            if foreign["last_active_ids"] or foreign["last_deferred_ids"]:
                last_active, last_deferred = (
                    foreign["last_active_ids"],
                    foreign["last_deferred_ids"],
                )

        def get(key: str) -> float:
            return n.get(key, 0.0)

        def avg(total: str, count: str) -> float | None:
            return get(total) / get(count) if get(count) else None

        def ratio(total: str, count: str) -> float:
            return get(total) / get(count) if get(count) else 0.0

        samples = {
            name: []
            for name in (
                "ttft_ms_all",
                "ttft_ms_short",
                "ttft_ms_long",
                "slowdown_all",
                "slowdown_short",
                "slowdown_long",
            )
        }
        for rec in records:
            side = "short" if rec.is_short else "long"
            if rec.arrival_s is not None and rec.first_token_s is not None:
                value = (rec.first_token_s - rec.arrival_s) * 1000.0
                samples["ttft_ms_all"].append(value)
                samples[f"ttft_ms_{side}"].append(value)
            if (
                rec.arrival_s is not None
                and rec.finish_s is not None
                and rec.solo_us
                and rec.solo_us > 0
            ):
                value = (rec.finish_s - rec.arrival_s) * 1e6 / rec.solo_us
                samples["slowdown_all"].append(value)
                samples[f"slowdown_{side}"].append(value)

        def stat(values: list[float]) -> dict[str, float | None]:
            return {
                "count": float(len(values)),
                **{f"p{p}": self._percentile(values, p) for p in (50, 95, 99)},
            }

        scheduler = {
            "attempts": int(get("sched_total")),
            "applied": int(get("sched_applied")),
            "apply_ratio": ratio("sched_applied", "sched_total"),
            "baseline_chunk_avg": avg("baseline_sum", "baseline_count"),
            "chosen_chunk_avg": avg("chosen_sum", "chosen_count"),
            "chosen_vs_baseline_ratio_avg": avg("slice_ratio_sum", "slice_ratio_count"),
            "explicit_plan_ratio": ratio("explicit_total", "sched_total"),
            "rewrite_applied": int(get("rewrite_applied")),
            "rewrite_apply_ratio": ratio("rewrite_applied", "sched_total"),
            "rewrite_group_count": int(get("rewrite_groups")),
            "rewrite_old_chunk_avg": avg("rewrite_old", "rewrite_groups"),
            "rewrite_new_chunk_avg": avg("rewrite_new", "rewrite_groups"),
            "rewrite_token_delta_avg": avg("rewrite_delta", "rewrite_groups"),
            "virtual_cap_apply_ratio": ratio("virtual_applied", "virtual_total"),
            "virtual_cap_old_avg": avg("virtual_old", "virtual_applied"),
            "virtual_cap_new_avg": avg("virtual_new", "virtual_applied"),
            "virtual_cap_target_set": get("virtual_target_set"),
            "virtual_cap_helper_calls": get("virtual_helper"),
            "virtual_cap_prefill_calls": get("virtual_prefill"),
            "virtual_cap_target_hits": get("virtual_hits"),
            "probe_total": get("probe_total"),
            "probe_slice_eligible_ratio": ratio("probe_eligible", "probe_total"),
            "probe_best_lt_long_ratio": ratio("probe_best_lt_long", "probe_total"),
            "probe_short_avg": avg("probe_short", "probe_total"),
            "probe_long_avg": avg("probe_long", "probe_total"),
            "probe_baseline_avg": avg("probe_baseline", "probe_baseline_count"),
            "probe_best_avg": avg("probe_best", "probe_best_count"),
            "probe_queue_avg": avg("probe_queue", "probe_total"),
            "probe_wait_us_avg": avg("probe_wait", "probe_total"),
            "runtime_adaptive_total": get("runtime_total"),
            "runtime_effective_pressure_avg": avg("runtime_pressure", "runtime_total"),
            "runtime_wall_pressure_avg": avg("runtime_wall", "runtime_total"),
            "runtime_short_urgency_avg": avg("runtime_urgency", "runtime_total"),
            "runtime_target_fraction_avg": avg("runtime_fraction", "runtime_total"),
            "runtime_target_chunk_avg": avg("runtime_chunk", "runtime_total"),
            "runtime_queue_avg": avg("runtime_queue", "runtime_total"),
            "runtime_waiting_short_avg": avg("runtime_waiting_short", "runtime_total"),
        }
        activations = get("priority_activations")
        phase2 = {
            "attempts": int(get("phase2_total")),
            "applied": int(get("phase2_applied")),
            "apply_ratio": ratio("phase2_applied", "phase2_total"),
            "reasons": reasons,
            "priority_lane": {
                "activations": activations,
                "active_count_avg": avg("priority_active", "priority_activations"),
                "deferred_count_avg": avg("priority_deferred", "priority_activations"),
                "ttl_avg": avg("priority_ttl", "priority_activations"),
                "seen_events": get("priority_seen_events"),
                "seen_active_hits": get("priority_seen_hits"),
                "seen_active_hits_per_event": avg("priority_seen_hits", "priority_seen_events"),
                "finished_events": get("priority_finished_events"),
                "finished_active_hits": get("priority_finished_hits"),
                "finished_active_hits_per_event": avg(
                    "priority_finished_hits", "priority_finished_events"
                ),
                "last_active_ids": last_active,
                "last_deferred_ids": last_deferred,
            },
        }
        return {
            "requests": {"total": len(records), "finished": sum(rec.finished for rec in records)},
            "scheduler": scheduler,
            "phase2": phase2,
            **{name: stat(values) for name, values in samples.items()},
        }
