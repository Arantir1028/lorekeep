from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class Req:
    req_id: str
    prompt: str
    is_short: bool
    lora_tag: Optional[str] = None
    arrival_offset_s: float = 0.0


def load_reqs_json(path: str) -> list[Req]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"requests json must be a list: {path}")
    reqs: list[Req] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"request item #{i} must be an object")
        req_id = str(item.get("req_id") or item.get("id") or f"req_{i}")
        prompt = str(item.get("prompt") or "")
        if not prompt.strip():
            raise ValueError(f"request item #{i} has empty prompt")
        is_short = bool(item.get("is_short"))
        lora_tag = item.get("lora_tag")
        arrival_offset_s = float(
            item.get("arrival_offset_s")
            or item.get("arrival_s")
            or 0.0
        )
        reqs.append(
            Req(
                req_id=req_id,
                prompt=prompt,
                is_short=is_short,
                lora_tag=lora_tag,
                arrival_offset_s=max(0.0, arrival_offset_s),
            )
        )
    return reqs


def percentile(values: list[float], p: float) -> Optional[float]:
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


def stats(values: list[Optional[float]]) -> dict[str, Optional[float]]:
    data = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not data:
        return {
            "count": 0.0,
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
        }
    return {
        "count": float(len(data)),
        "mean": float(sum(data) / len(data)),
        "p50": percentile(data, 50.0),
        "p95": percentile(data, 95.0),
        "p99": percentile(data, 99.0),
        "min": min(data),
        "max": max(data),
    }


def ratio(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b <= 0:
        return None
    return a / b


def mk_requests(short_a_repeat: int, short_b_repeat: int, long_repeat: int) -> list[Req]:
    short_1 = (
        (
            "Background note: efficient ML serving systems benefit from stable pipelines, "
            "careful batching, and predictable latency. "
        )
        * short_a_repeat
        + "Task: Translate the quoted sentence into French and output only the translation. "
        + "Sentence: 'I love machine learning systems and efficient serving pipelines.'"
    )
    short_2 = (
        (
            "Observed examples: 10+20=30. 20+30=50. 30+40=70. 40+50=90. "
            "The first addend grows by 10 each step and the second addend also grows by 10. "
        )
        * short_b_repeat
        + "Task: Continue the pattern with the next three equations and then give one short rule sentence. "
        + "Answer without repeating the prompt."
    )
    long_1 = (
        (
            "Passage: artificial intelligence changes systems engineering and deployment, "
            "especially when the serving stack must handle heterogeneous LoRA workloads, "
            "long-tail prompt lengths, fairness constraints, and resource contention. "
        )
        * long_repeat
        + "Task: Write exactly one sentence that summarizes the passage above. "
        + "Do not use section headers, numbering, bullet points, or repeated outline phrases."
    )
    return [
        Req("short_a", short_1, True, "A"),
        Req("short_b", short_2, True, "B"),
        Req("long_b", long_1, False, "B"),
    ]


def mk_lora_requests(short_a_repeat: int, short_b_repeat: int, long_repeat: int) -> list[Req]:
    short_1 = (
        "Task: Translate the quoted sentence into French and output only the translation. "
        + "Sentence: 'I love machine learning.'"
    )
    mid_1 = (
        (
            "Observed examples: 10+20=30. 20+30=50. 30+40=70. 40+50=90. "
            "The first addend grows by 10 each step and the second addend also grows by 10. "
        )
        * short_b_repeat
        + "Task: Continue the pattern with the next three equations and then give one short rule sentence. "
        + "Answer without repeating the prompt."
    )
    long_1 = (
        "Task: Write exactly one sentence that summarizes the passage. "
        + ("Artificial intelligence is transforming industry, deployment, and systems engineering. " * max(1, int(long_repeat * 0.6)))
    )
    long_2 = (
        "Task: Write exactly one sentence that summarizes the passage. "
        + ("Artificial intelligence is transforming industry, deployment, and systems engineering. " * long_repeat)
    )
    return [
        Req("short_a", short_1, True, "A"),
        Req("mid_b", mid_1, True, "B"),
        Req("long_a", long_1, False, "A"),
        Req("long_b", long_2, False, "B"),
    ]


def measure_input_tokens(
    model_path: str,
    reqs: list[Req],
    *,
    trust_remote_code: bool = False,
) -> dict[str, int]:
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        return {
            req.req_id: len(tok.encode(req.prompt, add_special_tokens=False))
            for req in reqs
        }
    except Exception:
        return {
            req.req_id: max(1, len(req.prompt.split()))
            for req in reqs
        }


def fit_requests_to_context(
    *,
    model_path: str,
    short_a_repeat: int,
    short_b_repeat: int,
    long_repeat: int,
    max_prompt_tokens: int,
    trust_remote_code: bool = False,
) -> tuple[list[Req], dict[str, int], int]:
    cur_long_repeat = max(1, long_repeat)
    for _ in range(8):
        reqs = mk_requests(short_a_repeat, short_b_repeat, cur_long_repeat)
        tok_lens = measure_input_tokens(model_path, reqs, trust_remote_code=trust_remote_code)
        if max(tok_lens.values()) <= max_prompt_tokens:
            return reqs, tok_lens, cur_long_repeat
        long_len = tok_lens.get("long_b", max(tok_lens.values()))
        if long_len <= 0 or cur_long_repeat <= 1:
            break
        cur_long_repeat = max(1, int(cur_long_repeat * (max_prompt_tokens / long_len) * 0.96))
    reqs = mk_requests(short_a_repeat, short_b_repeat, cur_long_repeat)
    tok_lens = measure_input_tokens(model_path, reqs, trust_remote_code=trust_remote_code)
    return reqs, tok_lens, cur_long_repeat


def bool_arg_from_argv(flag: str, default: bool) -> bool:
    if f"--no-{flag}" in sys.argv:
        return False
    if f"--{flag}" in sys.argv:
        return True
    return default


def str_arg_from_argv(flag: str, default: str = "") -> str:
    try:
        idx = sys.argv.index(f"--{flag}")
    except ValueError:
        return default
    if idx + 1 >= len(sys.argv):
        return default
    return str(sys.argv[idx + 1])


def fit_lora_requests_to_context(
    *,
    model_path: str,
    short_a_repeat: int,
    short_b_repeat: int,
    long_repeat: int,
    max_prompt_tokens: int,
    trust_remote_code: bool = False,
) -> tuple[list[Req], dict[str, int], int]:
    cur_long_repeat = max(1, long_repeat)
    for _ in range(8):
        reqs = mk_lora_requests(short_a_repeat, short_b_repeat, cur_long_repeat)
        tok_lens = measure_input_tokens(model_path, reqs, trust_remote_code=trust_remote_code)
        if max(tok_lens.values()) <= max_prompt_tokens:
            return reqs, tok_lens, cur_long_repeat
        long_len = max(v for k, v in tok_lens.items() if "long" in k)
        if long_len <= 0 or cur_long_repeat <= 1:
            break
        cur_long_repeat = max(1, int(cur_long_repeat * (max_prompt_tokens / long_len) * 0.96))
    reqs = mk_lora_requests(short_a_repeat, short_b_repeat, cur_long_repeat)
    tok_lens = measure_input_tokens(model_path, reqs, trust_remote_code=trust_remote_code)
    return reqs, tok_lens, cur_long_repeat


def text_match_rate(a: dict[str, str], b: dict[str, str]) -> float:
    keys = sorted(set(a.keys()) & set(b.keys()))
    if not keys:
        return 0.0
    hit = sum(1 for k in keys if a.get(k, "") == b.get(k, ""))
    return float(hit) / float(len(keys))


def text_mismatch_details(
    a: dict[str, str],
    b: dict[str, str],
    *,
    max_preview: int = 160,
) -> list[dict[str, Any]]:
    keys = sorted(set(a.keys()) & set(b.keys()))
    details: list[dict[str, Any]] = []
    for k in keys:
        left = str(a.get(k, "") or "")
        right = str(b.get(k, "") or "")
        if left == right:
            continue
        common = 0
        for ch_l, ch_r in zip(left, right):
            if ch_l != ch_r:
                break
            common += 1
        details.append(
            {
                "req_id": k,
                "common_prefix_chars": int(common),
                "base_preview": left[:max_preview],
                "wave_preview": right[:max_preview],
            }
        )
    return details


def semantic_check(req_id: str, text: str) -> dict[str, Any]:
    raw = str(text or "")
    s = raw.strip()
    low = s.lower()
    result: dict[str, Any] = {
        "pass": False,
        "score": 0.0,
        "reason": "empty",
    }
    if not s:
        return result

    if req_id == "short_a":
        french_markers = [
            " je ", " j'", " les ", " des ", " une ", " un ", " et ",
            " apprentissage ", " automatique", " systemes", " systèmes",
            " efficaces", " efficaces", " pipelines",
        ]
        has_french = any(m in f" {low} " for m in french_markers)
        leaked_prompt = "translate to french" in low
        score = 0.0
        if has_french:
            score += 1.0
        if not leaked_prompt:
            score += 0.5
        result.update(
            {
                "pass": bool(has_french and not leaked_prompt),
                "score": score,
                "reason": "french_like" if has_french and not leaked_prompt else "not_french_like",
            }
        )
        return result

    if req_id == "short_b":
        expected_markers = ["110", "130", "150", "50+60", "60+70", "70+80", "pattern"]
        hits = sum(1 for m in expected_markers if m in low)
        leaked_prompt = "continue the arithmetic pattern" in low
        result.update(
            {
                "pass": bool(hits >= 1 and not leaked_prompt),
                "score": float(hits) - (0.5 if leaked_prompt else 0.0),
                "reason": "pattern_continuation" if hits >= 1 and not leaked_prompt else "weak_pattern_continuation",
            }
        )
        return result

    if req_id == "long_b":
        topic_markers = [
            "artificial intelligence", "ai", "systems", "engineering", "deployment",
            "lora", "workload", "workloads", "serving", "heterogeneous",
        ]
        bad_markers = [
            "1. introduction", "2. related work", "3. methodology",
            "4. experiments", "5. conclusion", "6. references",
        ]
        topic_hits = sum(1 for m in topic_markers if m in low)
        bad_hits = sum(1 for m in bad_markers if m in low)
        one_sentence_like = s.count(".") <= 2 and s.count("\n") == 0
        result.update(
            {
                "pass": bool(topic_hits >= 2 and one_sentence_like and bad_hits == 0),
                "score": float(topic_hits) - float(bad_hits),
                "reason": (
                    "topic_summary"
                    if topic_hits >= 2 and one_sentence_like and bad_hits == 0
                    else "off_topic_or_outline"
                ),
            }
        )
        return result

    result.update(
        {
            "pass": bool(s),
            "score": 1.0,
            "reason": "non_empty",
        }
    )
    return result


def semantic_summary(texts: dict[str, str]) -> dict[str, Any]:
    details = {req_id: semantic_check(req_id, txt) for req_id, txt in texts.items()}
    passes = [1.0 if item.get("pass") else 0.0 for item in details.values()]
    return {
        "pass_rate": float(sum(passes) / len(passes)) if passes else 0.0,
        "details": details,
    }


def run_phase1_pair(
    *,
    base_rows: list[dict[str, Any]],
    base_repeat_rows: list[dict[str, Any]],
    wave_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for base, base_repeat, p1 in zip(base_rows, base_repeat_rows, wave_rows):
        match = text_match_rate(base["texts"], p1["texts"])
        match_noise = text_match_rate(base["texts"], base_repeat["texts"])
        mismatch_details = text_mismatch_details(base["texts"], p1["texts"])
        base_sem = semantic_summary(base["texts"])
        wave_sem = semantic_summary(p1["texts"])
        p1_sched = (p1["hook_report"] or {}).get("scheduler", {})
        rows.append(
            {
                "base_ttft_short_p99_ms": base["ttft_short_p99_ms"],
                "wave_ttft_short_p99_ms": p1["ttft_short_p99_ms"],
                "ttft_improve_ratio": ratio(base["ttft_short_p99_ms"], p1["ttft_short_p99_ms"]),
                "base_round_wall_ms": base.get("round_wall_ms"),
                "wave_round_wall_ms": p1.get("round_wall_ms"),
                "round_wall_improve_ratio": ratio(base.get("round_wall_ms"), p1.get("round_wall_ms")),
                "text_match_rate": match,
                "mismatch_details": mismatch_details,
                "error_rate": 1.0 - match,
                "baseline_noise_match_rate": match_noise,
                "baseline_noise_error_rate": 1.0 - match_noise,
                "incremental_error_rate": (1.0 - match) - (1.0 - match_noise),
                "base_semantic_pass_rate": base_sem.get("pass_rate"),
                "wave_semantic_pass_rate": wave_sem.get("pass_rate"),
                "semantic_pass_delta": (wave_sem.get("pass_rate") or 0.0) - (base_sem.get("pass_rate") or 0.0),
                "base_semantic_details": base_sem.get("details"),
                "wave_semantic_details": wave_sem.get("details"),
                "base_texts": base.get("texts"),
                "wave_texts": p1.get("texts"),
                "scheduler_apply_ratio": p1_sched.get("apply_ratio"),
                "scheduler_applied": p1_sched.get("applied"),
                "scheduler_attempts": p1_sched.get("attempts"),
                "baseline_chunk_avg": p1_sched.get("baseline_chunk_avg"),
                "chosen_chunk_avg": p1_sched.get("chosen_chunk_avg"),
                "chosen_vs_baseline_ratio_avg": p1_sched.get("chosen_vs_baseline_ratio_avg"),
                "explicit_plan_ratio": p1_sched.get("explicit_plan_ratio"),
                "rewrite_apply_ratio": p1_sched.get("rewrite_apply_ratio"),
                "rewrite_old_chunk_avg": p1_sched.get("rewrite_old_chunk_avg"),
                "rewrite_new_chunk_avg": p1_sched.get("rewrite_new_chunk_avg"),
                "rewrite_token_delta_avg": p1_sched.get("rewrite_token_delta_avg"),
                "virtual_cap_apply_ratio": p1_sched.get("virtual_cap_apply_ratio"),
                "virtual_cap_old_avg": p1_sched.get("virtual_cap_old_avg"),
                "virtual_cap_new_avg": p1_sched.get("virtual_cap_new_avg"),
                "virtual_cap_target_set": p1_sched.get("virtual_cap_target_set"),
                "virtual_cap_helper_calls": p1_sched.get("virtual_cap_helper_calls"),
                "virtual_cap_prefill_calls": p1_sched.get("virtual_cap_prefill_calls"),
                "virtual_cap_target_hits": p1_sched.get("virtual_cap_target_hits"),
                "probe_total": p1_sched.get("probe_total"),
                "probe_slice_eligible_ratio": p1_sched.get("probe_slice_eligible_ratio"),
                "probe_best_lt_long_ratio": p1_sched.get("probe_best_lt_long_ratio"),
                "probe_short_avg": p1_sched.get("probe_short_avg"),
                "probe_long_avg": p1_sched.get("probe_long_avg"),
                "probe_baseline_avg": p1_sched.get("probe_baseline_avg"),
                "probe_best_avg": p1_sched.get("probe_best_avg"),
                "probe_queue_avg": p1_sched.get("probe_queue_avg"),
                "probe_wait_us_avg": p1_sched.get("probe_wait_us_avg"),
                "probe_reasons": p1_sched.get("probe_reasons"),
                "phase1_request_traces": p1_sched.get("request_traces"),
                "hook_report": p1.get("hook_report"),
                "baseline_timed_out": base.get("timed_out"),
                "wave_timed_out": p1.get("timed_out"),
            }
        )
    return {
        "rows": rows,
        "summary": {
            "ttft_improve_ratio": stats([r.get("ttft_improve_ratio") for r in rows]),
            "round_wall_improve_ratio": stats([r.get("round_wall_improve_ratio") for r in rows]),
            "error_rate": stats([r.get("error_rate") for r in rows]),
            "baseline_noise_error_rate": stats([r.get("baseline_noise_error_rate") for r in rows]),
            "incremental_error_rate": stats([r.get("incremental_error_rate") for r in rows]),
            "base_semantic_pass_rate": stats([r.get("base_semantic_pass_rate") for r in rows]),
            "wave_semantic_pass_rate": stats([r.get("wave_semantic_pass_rate") for r in rows]),
            "semantic_pass_delta": stats([r.get("semantic_pass_delta") for r in rows]),
            "scheduler_apply_ratio": stats([r.get("scheduler_apply_ratio") for r in rows]),
            "baseline_chunk_avg": stats([r.get("baseline_chunk_avg") for r in rows]),
            "chosen_chunk_avg": stats([r.get("chosen_chunk_avg") for r in rows]),
            "chosen_vs_baseline_ratio_avg": stats([r.get("chosen_vs_baseline_ratio_avg") for r in rows]),
            "explicit_plan_ratio": stats([r.get("explicit_plan_ratio") for r in rows]),
            "rewrite_apply_ratio": stats([r.get("rewrite_apply_ratio") for r in rows]),
            "rewrite_old_chunk_avg": stats([r.get("rewrite_old_chunk_avg") for r in rows]),
            "rewrite_new_chunk_avg": stats([r.get("rewrite_new_chunk_avg") for r in rows]),
            "rewrite_token_delta_avg": stats([r.get("rewrite_token_delta_avg") for r in rows]),
            "virtual_cap_apply_ratio": stats([r.get("virtual_cap_apply_ratio") for r in rows]),
            "virtual_cap_old_avg": stats([r.get("virtual_cap_old_avg") for r in rows]),
            "virtual_cap_new_avg": stats([r.get("virtual_cap_new_avg") for r in rows]),
            "virtual_cap_target_set": stats([r.get("virtual_cap_target_set") for r in rows]),
            "virtual_cap_helper_calls": stats([r.get("virtual_cap_helper_calls") for r in rows]),
            "virtual_cap_prefill_calls": stats([r.get("virtual_cap_prefill_calls") for r in rows]),
            "virtual_cap_target_hits": stats([r.get("virtual_cap_target_hits") for r in rows]),
            "probe_total": stats([r.get("probe_total") for r in rows]),
            "probe_slice_eligible_ratio": stats([r.get("probe_slice_eligible_ratio") for r in rows]),
            "probe_best_lt_long_ratio": stats([r.get("probe_best_lt_long_ratio") for r in rows]),
            "probe_short_avg": stats([r.get("probe_short_avg") for r in rows]),
            "probe_long_avg": stats([r.get("probe_long_avg") for r in rows]),
            "probe_baseline_avg": stats([r.get("probe_baseline_avg") for r in rows]),
            "probe_best_avg": stats([r.get("probe_best_avg") for r in rows]),
            "probe_queue_avg": stats([r.get("probe_queue_avg") for r in rows]),
            "probe_wait_us_avg": stats([r.get("probe_wait_us_avg") for r in rows]),
        },
    }


def run_phase2_block(
    *,
    base_rows: list[dict[str, Any]],
    base_repeat_rows: list[dict[str, Any]],
    wave_rows: list[dict[str, Any]],
    strict_rows: Optional[list[dict[str, Any]]],
    include_strict: bool,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    strict_eval_rows: list[dict[str, Any]] = []
    for idx, (lora_base, lora_base_repeat, lora_p2) in enumerate(zip(base_rows, base_repeat_rows, wave_rows)):
        base_report = lora_base["hook_report"] or {}
        wave_report = lora_p2["hook_report"] or {}
        base_slow = (base_report.get("slowdown_short") or {}).get("p99")
        wave_slow = (wave_report.get("slowdown_short") or {}).get("p99")

        match_wave = text_match_rate(lora_base["texts"], lora_p2["texts"])
        match_noise = text_match_rate(lora_base["texts"], lora_base_repeat["texts"])
        p2_info = wave_report.get("phase2", {})
        lane_info = p2_info.get("escape_lane", {}) if isinstance(p2_info, dict) else {}

        row: dict[str, Any] = {
            "repeat_index": idx,
            "base_ttft_short_p99_ms": lora_base["ttft_short_p99_ms"],
            "wave_ttft_short_p99_ms": lora_p2["ttft_short_p99_ms"],
            "ttft_improve_ratio": ratio(lora_base["ttft_short_p99_ms"], lora_p2["ttft_short_p99_ms"]),
            "base_slowdown_short_p99": base_slow,
            "wave_slowdown_short_p99": wave_slow,
            "slowdown_improve_ratio": ratio(base_slow, wave_slow),
            "base_round_wall_ms": lora_base.get("round_wall_ms"),
            "wave_round_wall_ms": lora_p2.get("round_wall_ms"),
            "round_wall_improve_ratio": ratio(lora_base.get("round_wall_ms"), lora_p2.get("round_wall_ms")),
            "baseline_noise_match_rate": match_noise,
            "baseline_noise_error_rate": 1.0 - match_noise,
            "wave_match_rate": match_wave,
            "wave_error_rate": 1.0 - match_wave,
            "incremental_error_rate": (1.0 - match_wave) - (1.0 - match_noise),
            "phase2_apply_ratio": p2_info.get("apply_ratio"),
            "phase2_applied": p2_info.get("applied"),
            "phase2_attempts": p2_info.get("attempts"),
            "phase2_escape_lane_activations": lane_info.get("activations"),
            "phase2_escape_lane_active_count_avg": lane_info.get("active_count_avg"),
            "phase2_escape_lane_deferred_count_avg": lane_info.get("deferred_count_avg"),
            "phase2_escape_lane_seen_active_hits": lane_info.get("seen_active_hits"),
            "phase2_escape_lane_finished_active_hits": lane_info.get("finished_active_hits"),
            "baseline_timed_out": lora_base.get("timed_out"),
            "wave_timed_out": lora_p2.get("timed_out"),
            "base_request_timings": lora_base.get("request_timings"),
            "wave_request_timings": lora_p2.get("request_timings"),
            "base_texts": lora_base.get("texts"),
            "wave_texts": lora_p2.get("texts"),
            "base_hook_report": base_report,
            "wave_hook_report": wave_report,
        }
        rows.append(row)

    if include_strict:
        strict_rows = strict_rows or []
        for idx, (lora_base, lora_base_repeat, lora_p2_strict) in enumerate(
            zip(base_rows, base_repeat_rows, strict_rows)
        ):
            strict_report = lora_p2_strict["hook_report"] or {}
            base_slow = ((lora_base["hook_report"] or {}).get("slowdown_short") or {}).get("p99")
            strict_slow = (strict_report.get("slowdown_short") or {}).get("p99")
            strict_info = strict_report.get("phase2", {})
            strict_match = text_match_rate(lora_base["texts"], lora_p2_strict["texts"])
            match_noise = text_match_rate(lora_base["texts"], lora_base_repeat["texts"])
            strict_eval_rows.append(
                {
                    "repeat_index": idx,
                    "strict_ttft_short_p99_ms": lora_p2_strict["ttft_short_p99_ms"],
                    "strict_ttft_improve_ratio": ratio(
                        lora_base["ttft_short_p99_ms"],
                        lora_p2_strict["ttft_short_p99_ms"],
                    ),
                    "strict_slowdown_short_p99": strict_slow,
                    "strict_slowdown_improve_ratio": ratio(base_slow, strict_slow),
                    "strict_round_wall_ms": lora_p2_strict.get("round_wall_ms"),
                    "strict_round_wall_improve_ratio": ratio(
                        lora_base.get("round_wall_ms"),
                        lora_p2_strict.get("round_wall_ms"),
                    ),
                    "strict_match_rate": strict_match,
                    "strict_error_rate": 1.0 - strict_match,
                    "strict_incremental_error_rate": (1.0 - strict_match) - (1.0 - match_noise),
                    "strict_phase2_apply_ratio": strict_info.get("apply_ratio"),
                    "strict_phase2_applied": strict_info.get("applied"),
                    "strict_phase2_attempts": strict_info.get("attempts"),
                }
            )
    return {
        "rows": rows,
        "summary": {
            "ttft_improve_ratio": stats([r.get("ttft_improve_ratio") for r in rows]),
            "slowdown_improve_ratio": stats([r.get("slowdown_improve_ratio") for r in rows]),
            "round_wall_improve_ratio": stats([r.get("round_wall_improve_ratio") for r in rows]),
            "wave_error_rate": stats([r.get("wave_error_rate") for r in rows]),
            "baseline_noise_error_rate": stats([r.get("baseline_noise_error_rate") for r in rows]),
            "incremental_error_rate": stats([r.get("incremental_error_rate") for r in rows]),
            "phase2_apply_ratio": stats([r.get("phase2_apply_ratio") for r in rows]),
            "phase2_escape_lane_activations": stats([r.get("phase2_escape_lane_activations") for r in rows]),
            "phase2_escape_lane_seen_active_hits": stats(
                [r.get("phase2_escape_lane_seen_active_hits") for r in rows]
            ),
            "phase2_escape_lane_finished_active_hits": stats(
                [r.get("phase2_escape_lane_finished_active_hits") for r in rows]
            ),
        },
        "strict_rows": strict_eval_rows,
        "strict_summary": {
            "ttft_improve_ratio": stats([r.get("strict_ttft_improve_ratio") for r in strict_eval_rows]),
            "slowdown_improve_ratio": stats([r.get("strict_slowdown_improve_ratio") for r in strict_eval_rows]),
            "round_wall_improve_ratio": stats([r.get("strict_round_wall_improve_ratio") for r in strict_eval_rows]),
            "error_rate": stats([r.get("strict_error_rate") for r in strict_eval_rows]),
            "incremental_error_rate": stats([r.get("strict_incremental_error_rate") for r in strict_eval_rows]),
            "apply_ratio": stats([r.get("strict_phase2_apply_ratio") for r in strict_eval_rows]),
        },
    }


def raw_mode_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Optional[float]]]:
    return {
        "ttft_short_p99_ms": stats([r.get("ttft_short_p99_ms") for r in rows]),
        "round_wall_ms": stats([r.get("round_wall_ms") for r in rows]),
        "timeout_rate": stats([1.0 if r.get("timed_out") else 0.0 for r in rows]),
    }
