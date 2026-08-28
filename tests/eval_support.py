from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from typing import Any

from experiments.result_io import percentile, ratio


@dataclass(frozen=True)
class Req:
    req_id: str
    prompt: str
    is_short: bool
    lora_tag: str | None = None
    arrival_offset_s: float = 0.0


def load_reqs_json(path: str) -> list[Req]:
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"requests json must be a list: {path}")
    output = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"request item #{index} must be an object")
        prompt = str(row.get("prompt") or "")
        if not prompt.strip():
            raise ValueError(f"request item #{index} has empty prompt")
        output.append(
            Req(
                str(row.get("req_id") or row.get("id") or f"req_{index}"),
                prompt,
                bool(row.get("is_short")),
                row.get("lora_tag"),
                max(0.0, float(row.get("arrival_offset_s") or row.get("arrival_s") or 0)),
            )
        )
    return output


def stats(values: list[float | None]) -> dict[str, float | None]:
    data = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not data:
        return {
            key: 0.0 if key == "count" else None
            for key in ("count", "mean", "p50", "p95", "p99", "min", "max")
        }
    return {
        "count": float(len(data)),
        "mean": sum(data) / len(data),
        "p50": percentile(data, 50),
        "p95": percentile(data, 95),
        "p99": percentile(data, 99),
        "min": min(data),
        "max": max(data),
    }


def measure_input_tokens(
    model_path: str, reqs: list[Req], *, trust_remote_code: bool = False
) -> dict[str, int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    return {
        request.req_id: len(tokenizer.encode(request.prompt, add_special_tokens=False))
        for request in reqs
    }


def bool_arg_from_argv(flag: str, default: bool) -> bool:
    return False if f"--no-{flag}" in sys.argv else True if f"--{flag}" in sys.argv else default


def str_arg_from_argv(flag: str, default: str = "") -> str:
    try:
        index = sys.argv.index(f"--{flag}") + 1
        return str(sys.argv[index])
    except (ValueError, IndexError):
        return default


def text_match_rate(a: dict[str, str], b: dict[str, str]) -> float:
    keys = sorted(set(a) & set(b))
    return sum(a.get(key, "") == b.get(key, "") for key in keys) / len(keys) if keys else 0.0


def semantic_check(req_id: str, text: str) -> dict[str, Any]:
    value, low = str(text or "").strip(), str(text or "").strip().lower()
    if not value:
        return {"pass": False, "score": 0.0, "reason": "empty"}
    if req_id == "short_a":
        markers = (
            " je ",
            " j'",
            " les ",
            " des ",
            " une ",
            " un ",
            " et ",
            " apprentissage ",
            " automatique",
            " systemes",
            " systèmes",
            " efficaces",
            " pipelines",
        )
        good, leaked = any(marker in f" {low} " for marker in markers), "translate to french" in low
        return {
            "pass": good and not leaked,
            "score": float(good) + 0.5 * (not leaked),
            "reason": "french_like" if good and not leaked else "not_french_like",
        }
    if req_id == "short_b":
        hits = sum(
            marker in low for marker in ("110", "130", "150", "50+60", "60+70", "70+80", "pattern")
        )
        leaked = "continue the arithmetic pattern" in low
        return {
            "pass": hits >= 1 and not leaked,
            "score": float(hits) - 0.5 * leaked,
            "reason": "pattern_continuation"
            if hits >= 1 and not leaked
            else "weak_pattern_continuation",
        }
    if req_id == "long_b":
        topic = sum(
            marker in low
            for marker in (
                "artificial intelligence",
                "ai",
                "systems",
                "engineering",
                "deployment",
                "lora",
                "workload",
                "workloads",
                "serving",
                "heterogeneous",
            )
        )
        bad = sum(
            marker in low
            for marker in (
                "1. introduction",
                "2. related work",
                "3. methodology",
                "4. experiments",
                "5. conclusion",
                "6. references",
            )
        )
        passed = topic >= 2 and value.count(".") <= 2 and "\n" not in value and bad == 0
        return {
            "pass": passed,
            "score": float(topic - bad),
            "reason": "topic_summary" if passed else "off_topic_or_outline",
        }
    return {"pass": True, "score": 1.0, "reason": "non_empty"}


def semantic_pass_rate(texts: dict[str, str]) -> float:
    checks = [semantic_check(key, value)["pass"] for key, value in texts.items()]
    return sum(checks) / len(checks) if checks else 0.0


_P1_SCHEDULER_FIELDS = (
    "baseline_chunk_avg",
    "chosen_chunk_avg",
    "chosen_vs_baseline_ratio_avg",
    "explicit_plan_ratio",
    "rewrite_apply_ratio",
    "rewrite_old_chunk_avg",
    "rewrite_new_chunk_avg",
    "rewrite_token_delta_avg",
    "virtual_cap_apply_ratio",
    "virtual_cap_old_avg",
    "virtual_cap_new_avg",
    "virtual_cap_target_set",
    "virtual_cap_helper_calls",
    "virtual_cap_prefill_calls",
    "virtual_cap_target_hits",
    "probe_total",
    "probe_slice_eligible_ratio",
    "probe_best_lt_long_ratio",
    "probe_short_avg",
    "probe_long_avg",
    "probe_baseline_avg",
    "probe_best_avg",
    "probe_queue_avg",
    "probe_wait_us_avg",
    "runtime_adaptive_total",
    "runtime_effective_pressure_avg",
    "runtime_wall_pressure_avg",
    "runtime_short_urgency_avg",
    "runtime_target_fraction_avg",
    "runtime_target_chunk_avg",
    "runtime_queue_avg",
    "runtime_waiting_short_avg",
)
_P1_SUMMARY_FIELDS = (
    "ttft_improve_ratio",
    "round_wall_improve_ratio",
    "error_rate",
    "baseline_noise_error_rate",
    "incremental_error_rate",
    "base_semantic_pass_rate",
    "wave_semantic_pass_rate",
    "semantic_pass_delta",
    "scheduler_apply_ratio",
) + _P1_SCHEDULER_FIELDS
_P2_SUMMARY_FIELDS = (
    "ttft_improve_ratio",
    "slowdown_improve_ratio",
    "round_wall_improve_ratio",
    "wave_error_rate",
    "baseline_noise_error_rate",
    "incremental_error_rate",
    "phase2_apply_ratio",
    "phase2_priority_lane_activations",
    "phase2_priority_lane_seen_active_hits",
    "phase2_priority_lane_finished_active_hits",
)


def _summary(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[str, Any]:
    return {field: stats([row.get(field) for row in rows]) for field in fields}


def run_phase1_pair(
    *,
    base_rows: list[dict[str, Any]],
    base_repeat_rows: list[dict[str, Any]],
    wave_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = []
    for base, repeated, wave in zip(base_rows, base_repeat_rows, wave_rows, strict=False):
        match, noise = (
            text_match_rate(base["texts"], wave["texts"]),
            text_match_rate(base["texts"], repeated["texts"]),
        )
        base_sem, wave_sem = semantic_pass_rate(base["texts"]), semantic_pass_rate(wave["texts"])
        scheduler = (wave.get("hook_report") or {}).get("scheduler", {})
        row = {
            "base_ttft_short_p99_ms": base["ttft_short_p99_ms"],
            "wave_ttft_short_p99_ms": wave["ttft_short_p99_ms"],
            "ttft_improve_ratio": ratio(base["ttft_short_p99_ms"], wave["ttft_short_p99_ms"]),
            "base_round_wall_ms": base.get("round_wall_ms"),
            "wave_round_wall_ms": wave.get("round_wall_ms"),
            "round_wall_improve_ratio": ratio(base.get("round_wall_ms"), wave.get("round_wall_ms")),
            "text_match_rate": match,
            "error_rate": 1 - match,
            "baseline_noise_match_rate": noise,
            "baseline_noise_error_rate": 1 - noise,
            "incremental_error_rate": noise - match,
            "base_semantic_pass_rate": base_sem,
            "wave_semantic_pass_rate": wave_sem,
            "semantic_pass_delta": wave_sem - base_sem,
            "base_request_timings": base.get("request_timings"),
            "wave_request_timings": wave.get("request_timings"),
            "scheduler_apply_ratio": scheduler.get("apply_ratio"),
            "scheduler_applied": scheduler.get("applied"),
            "scheduler_attempts": scheduler.get("attempts"),
            **{field: scheduler.get(field) for field in _P1_SCHEDULER_FIELDS},
            "baseline_timed_out": base.get("timed_out"),
            "wave_timed_out": wave.get("timed_out"),
        }
        rows.append(row)
    return {"rows": rows, "summary": _summary(rows, _P1_SUMMARY_FIELDS)}


def _phase2_row(
    index: int, base: dict[str, Any], repeated: dict[str, Any], wave: dict[str, Any]
) -> dict[str, Any]:
    base_report, wave_report = base.get("hook_report") or {}, wave.get("hook_report") or {}
    base_slow, wave_slow = (
        (base_report.get("slowdown_short") or {}).get("p99"),
        (wave_report.get("slowdown_short") or {}).get("p99"),
    )
    phase2, lane = (
        wave_report.get("phase2") or {},
        (wave_report.get("phase2") or {}).get("priority_lane") or {},
    )
    match, noise = (
        text_match_rate(base["texts"], wave["texts"]),
        text_match_rate(base["texts"], repeated["texts"]),
    )
    return {
        "repeat_index": index,
        "base_ttft_short_p99_ms": base["ttft_short_p99_ms"],
        "wave_ttft_short_p99_ms": wave["ttft_short_p99_ms"],
        "ttft_improve_ratio": ratio(base["ttft_short_p99_ms"], wave["ttft_short_p99_ms"]),
        "base_slowdown_short_p99": base_slow,
        "wave_slowdown_short_p99": wave_slow,
        "slowdown_improve_ratio": ratio(base_slow, wave_slow),
        "base_round_wall_ms": base.get("round_wall_ms"),
        "wave_round_wall_ms": wave.get("round_wall_ms"),
        "round_wall_improve_ratio": ratio(base.get("round_wall_ms"), wave.get("round_wall_ms")),
        "baseline_noise_match_rate": noise,
        "baseline_noise_error_rate": 1 - noise,
        "wave_match_rate": match,
        "wave_error_rate": 1 - match,
        "incremental_error_rate": noise - match,
        "phase2_apply_ratio": phase2.get("apply_ratio"),
        "phase2_applied": phase2.get("applied"),
        "phase2_attempts": phase2.get("attempts"),
        "phase2_reasons": phase2.get("reasons"),
        "phase2_priority_lane_activations": lane.get("activations"),
        "phase2_priority_lane_active_count_avg": lane.get("active_count_avg"),
        "phase2_priority_lane_deferred_count_avg": lane.get("deferred_count_avg"),
        "phase2_priority_lane_seen_active_hits": lane.get("seen_active_hits"),
        "phase2_priority_lane_finished_active_hits": lane.get("finished_active_hits"),
        "baseline_timed_out": base.get("timed_out"),
        "wave_timed_out": wave.get("timed_out"),
        "base_request_timings": base.get("request_timings"),
        "wave_request_timings": wave.get("request_timings"),
    }


def run_phase2_block(
    *,
    base_rows: list[dict[str, Any]],
    base_repeat_rows: list[dict[str, Any]],
    wave_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = [
        _phase2_row(index, *values)
        for index, values in enumerate(zip(base_rows, base_repeat_rows, wave_rows, strict=False))
    ]
    return {"rows": rows, "summary": _summary(rows, _P2_SUMMARY_FIELDS)}


def raw_mode_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    return {
        "ttft_short_p99_ms": stats([row.get("ttft_short_p99_ms") for row in rows]),
        "round_wall_ms": stats([row.get("round_wall_ms") for row in rows]),
        "timeout_rate": stats([1.0 if row.get("timed_out") else 0.0 for row in rows]),
    }
