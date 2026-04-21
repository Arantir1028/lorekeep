from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = REPO_ROOT / "tests"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from experiments.chapter2_prestudy import (
    _candidate_model_entries,
    _config_for_model,
    _ensure_dir,
    _load_json,
    _phase1_prompt_long,
    _phase1_prompt_short,
    _resolve_model,
    _resolve_out_root,
    _write_json,
)
from experiments.openworkload_models import resolve_model_entry
from tests.evaluate_waveslice_claims import _build_engine, _cleanup_engine


@dataclass(frozen=True)
class ReqSpec:
    req_id: str
    prompt: str
    arrival_offset_s: float
    max_new_tokens: int
    is_short: bool


def _p99(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    vals = sorted(float(v) for v in values)
    idx = max(0, min(len(vals) - 1, int(round(0.99 * (len(vals) - 1)))))
    return vals[idx]


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _run_round(
    engine: Any,
    *,
    reqs: list[ReqSpec],
    timeout_sec: int,
    run_tag: str,
) -> dict[str, Any]:
    from vllm.sampling_params import SamplingParams

    pending = sorted(reqs, key=lambda r: (r.arrival_offset_s, r.req_id))
    sampling = {
        r.req_id: SamplingParams(
            max_tokens=int(r.max_new_tokens),
            temperature=0.0,
            ignore_eos=True,
        )
        for r in pending
    }
    trackers: dict[str, dict[str, Any]] = {}
    next_idx = 0
    round_start = time.perf_counter()
    deadline = time.time() + timeout_sec

    while time.time() < deadline and (next_idx < len(pending) or engine.has_unfinished_requests()):
        now = time.perf_counter()
        elapsed = now - round_start
        while next_idx < len(pending) and pending[next_idx].arrival_offset_s <= elapsed:
            req = pending[next_idx]
            request_id = f"{run_tag}:{req.req_id}"
            engine.add_request(request_id, req.prompt, sampling[req.req_id])
            trackers[request_id] = {
                "orig_req_id": req.req_id,
                "arrival_s": now,
                "arrival_offset_s": req.arrival_offset_s,
                "first_s": None,
                "finish_s": None,
                "is_short": req.is_short,
                "max_new_tokens": req.max_new_tokens,
                "generated_tokens": 0,
            }
            next_idx += 1

        if not engine.has_unfinished_requests():
            if next_idx < len(pending):
                sleep_s = max(0.0, min(0.005, pending[next_idx].arrival_offset_s - (time.perf_counter() - round_start)))
                if sleep_s > 0:
                    time.sleep(sleep_s)
            continue

        outputs = engine.step()
        now = time.perf_counter()
        for out in outputs:
            tracker = trackers.get(out.request_id)
            if tracker is None:
                continue
            tok_count = 0
            try:
                tok_count = len(out.outputs[0].token_ids)
            except Exception:
                pass
            tracker["generated_tokens"] = max(int(tracker["generated_tokens"]), int(tok_count))
            if tok_count > 0 and tracker["first_s"] is None:
                tracker["first_s"] = now
            if out.finished:
                tracker["finish_s"] = now

    result: dict[str, Any] = {"request_timings": {}, "timed_out": False}
    for tracker in trackers.values():
        first = tracker["first_s"]
        finish = tracker["finish_s"]
        arrival = tracker["arrival_s"]
        result["request_timings"][tracker["orig_req_id"]] = {
            "arrival_offset_s": tracker["arrival_offset_s"],
            "is_short": tracker["is_short"],
            "max_new_tokens": tracker["max_new_tokens"],
            "generated_tokens": tracker["generated_tokens"],
            "first_latency_ms": None if first is None else (first - arrival) * 1000.0,
            "finish_latency_ms": None if finish is None else (finish - arrival) * 1000.0,
        }
        if finish is None:
            result["timed_out"] = True
    return result


def _build_candidate(name: str, *, long_prompt_repeat: int, short_count: int, short_start_s: float, short_gap_s: float, short_decode_tokens: int) -> dict[str, Any]:
    reqs = [
        ReqSpec(
            req_id="long_00",
            prompt=_phase1_prompt_long(long_prompt_repeat),
            arrival_offset_s=0.0,
            max_new_tokens=64,
            is_short=False,
        )
    ]
    for idx in range(short_count):
        reqs.append(
            ReqSpec(
                req_id=f"short_{idx:02d}",
                prompt=_phase1_prompt_short(idx, 2),
                arrival_offset_s=short_start_s + idx * short_gap_s,
                max_new_tokens=short_decode_tokens,
                is_short=True,
            )
        )
    return {
        "name": name,
        "long_prompt_repeat": long_prompt_repeat,
        "short_count": short_count,
        "short_start_s": short_start_s,
        "short_gap_s": short_gap_s,
        "short_decode_tokens": short_decode_tokens,
        "requests": reqs,
    }


def _evaluate_candidate(engine: Any, *, candidate: dict[str, Any], timeout_sec: int) -> dict[str, Any]:
    overlap = _run_round(engine, reqs=list(candidate["requests"]), timeout_sec=timeout_sec, run_tag=f"{candidate['name']}_overlap")
    refs: dict[str, float] = {}
    solo_rows: dict[str, Any] = {}
    for req in candidate["requests"]:
        if not req.is_short:
            continue
        solo = _run_round(
            engine,
            reqs=[ReqSpec(req.req_id, req.prompt, 0.0, req.max_new_tokens, True)],
            timeout_sec=timeout_sec,
            run_tag=f"{candidate['name']}_{req.req_id}_solo",
        )
        row = (solo.get("request_timings") or {}).get(req.req_id) or {}
        solo_rows[req.req_id] = row
        finish = row.get("finish_latency_ms")
        if finish is not None:
            refs[req.req_id] = float(finish)

    overlap_rows = overlap.get("request_timings") or {}
    short_ttft = [float(v["first_latency_ms"]) for v in overlap_rows.values() if v.get("is_short") and v.get("first_latency_ms") is not None]
    isolated_ttft = [float(v["first_latency_ms"]) for v in solo_rows.values() if v.get("first_latency_ms") is not None]
    slowdown = []
    for req_id, row in overlap_rows.items():
        if not row.get("is_short"):
            continue
        finish = row.get("finish_latency_ms")
        ref = refs.get(req_id)
        if finish is not None and ref is not None and ref > 0:
            slowdown.append(float(finish) / ref)

    overlap_ttft_p99 = _p99(short_ttft)
    isolated_ttft_p99 = _p99(isolated_ttft)
    slowdown_p99 = _p99(slowdown)
    ttft_ratio = None
    if overlap_ttft_p99 is not None and isolated_ttft_p99 is not None and isolated_ttft_p99 > 0:
        ttft_ratio = overlap_ttft_p99 / isolated_ttft_p99

    return {
        "candidate": {
            k: v for k, v in candidate.items() if k != "requests"
        },
        "strict_reference_by_req_ms": refs,
        "overlap": overlap,
        "solo": solo_rows,
        "metrics": {
            "strict_isolated_short_ttft_p99_ms": isolated_ttft_p99,
            "naive_overlap_short_ttft_p99_ms": overlap_ttft_p99,
            "short_slowdown_p99_vs_strict_isolated": slowdown_p99,
            "ttft_ratio_vs_strict": ttft_ratio,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick Obs1 search runner for Chapter 2.")
    parser.add_argument("--config", default="experiments/configs/chapter2_prestudy_v1.json")
    parser.add_argument("--out-root", default="")
    parser.add_argument("--model-key", default="gemma-2-9b-it")
    parser.add_argument("--timeout-sec", type=int, default=120)
    args = parser.parse_args()

    config = _load_json(Path(args.config))
    try:
        resolved_model = resolve_model_entry(args.model_key)
    except Exception:
        resolved_model = None
        for entry in _candidate_model_entries(config):
            if str((entry or {}).get("key") or "").strip() == args.model_key:
                resolved_model = resolve_model_entry(entry)
                break
        if resolved_model is None:
            raise

    config = _config_for_model(config, resolved_model)
    out_root = _resolve_out_root(config, args.out_root or None)
    exp_root = _ensure_dir(out_root / "E1_obs1_quick")
    model = _resolve_model(config)
    eval_cfg = dict(config.get("eval") or {})
    phase1_cfg = dict(config.get("phase1") or {})
    phase12_cfg = dict(config.get("phase12_soft_gate") or {})

    candidates = [
        _build_candidate("k4_dense", long_prompt_repeat=48, short_count=6, short_start_s=0.12, short_gap_s=0.02, short_decode_tokens=4),
        _build_candidate("k8_dense", long_prompt_repeat=48, short_count=6, short_start_s=0.12, short_gap_s=0.02, short_decode_tokens=8),
        _build_candidate("k8_earlier", long_prompt_repeat=56, short_count=6, short_start_s=0.08, short_gap_s=0.015, short_decode_tokens=8),
        _build_candidate("k16_dense", long_prompt_repeat=56, short_count=6, short_start_s=0.10, short_gap_s=0.02, short_decode_tokens=16),
    ]

    engine, _ = _build_engine(
        model_path=str(model["model_path"]),
        model_name=str(model["model_name"]),
        mode="baseline",
        enable_lora=False,
        max_num_batched_tokens=int(eval_cfg.get("max_num_batched_tokens", 1536)),
        max_model_len=int(model["max_model_len"]),
        gpu_memory_utilization=float(eval_cfg.get("gpu_memory_utilization", 0.60)),
        queue_reorder_mode=str(eval_cfg.get("queue_reorder_mode", "sjf")),
        queue_reorder_aging_quantum_us=float(eval_cfg.get("queue_reorder_aging_quantum_us", 20000)),
        phase1_objective_mode=str(phase1_cfg.get("objective_mode", "fair_escape")),
        phase1_gamma=float(phase1_cfg.get("gamma", 1.0)),
        phase1_ingress_target_chunk=int(phase1_cfg.get("ingress_target_chunk", 768)),
        phase1_ingress_direct_authoritative=True,
        phase1_ingress_exact_chunk=True,
        phase1_force_min_chunk=int(phase1_cfg.get("force_min_chunk", 128)),
        phase1_target_long_fraction=float(phase1_cfg.get("target_long_fraction", 0.33)),
        phase12_phase2_gate_mode=str(phase12_cfg.get("phase2_gate_mode", "soft")),
        phase12_phase2_soft_ratio_scale=float(phase12_cfg.get("phase2_soft_ratio_scale", 1.15)),
        phase12_phase2_soft_pressure_scale=float(phase12_cfg.get("phase2_soft_pressure_scale", 1.10)),
        phase12_phase2_soft_min_long_prefill=int(phase12_cfg.get("phase2_soft_min_long_prefill", 512)),
        phase12_phase2_soft_allow_mixed_decode=bool(phase12_cfg.get("phase2_soft_allow_mixed_decode", True)),
        phase12_phase2_soft_recent_strength_floor=float(phase12_cfg.get("phase2_soft_recent_strength_floor", 0.08)),
        phase12_phase2_soft_require_cashout_signal=bool(phase12_cfg.get("phase2_soft_require_cashout_signal", True)),
        phase12_phase2_soft_recent_chunk_match_scale=float(phase12_cfg.get("phase2_soft_recent_chunk_match_scale", 1.5)),
        phase12_phase2_soft_window_score_threshold=float(phase12_cfg.get("phase2_soft_window_score_threshold", 0.95)),
        phase12_phase2_soft_window_recent_weight=float(phase12_cfg.get("phase2_soft_window_recent_weight", 0.40)),
        phase12_phase2_soft_window_chunk_weight=float(phase12_cfg.get("phase2_soft_window_chunk_weight", 0.25)),
        phase12_phase2_soft_window_pressure_weight=float(phase12_cfg.get("phase2_soft_window_pressure_weight", 0.20)),
        phase12_phase2_soft_window_ratio_weight=float(phase12_cfg.get("phase2_soft_window_ratio_weight", 0.10)),
        phase12_phase2_soft_window_decode_bonus=float(phase12_cfg.get("phase2_soft_window_decode_bonus", 0.10)),
        phase12_phase2_scheduler_cashout_soft_floor=0.55,
        phase12_phase2_scheduler_cashout_quality_floor=0.78,
        phase12_phase2_scheduler_cashout_cooldown_ticks=2,
        phase12_phase2_require_beneficiary_signal=True,
        phase12_phase2_beneficiary_score_threshold=0.55,
        phase2_enable_mixed_prefill_decode=False,
        phase2_min_hetero_ratio=4.0,
        phase2_min_long_prefill=768,
        phase2_min_pressure_ratio=4.0,
        phase2_enable_scheduler_cashout=True,
        phase2_enable_execution_escape=True,
        phase2_enable_v1_true_unbind=False,
        phase2_execution_escape_mode="bounded_spillover",
        phase2_execution_escape_spillover_cap=3,
        phase2_execution_escape_max_active=5,
        max_num_partial_prefills=1,
        max_long_partial_prefills=1,
        enable_chunked_prefill=True,
        trust_remote_code=bool(model["trust_remote_code"]),
    )

    try:
        results = []
        for candidate in candidates:
            result = _evaluate_candidate(engine, candidate=candidate, timeout_sec=int(args.timeout_sec))
            results.append(result)

        def _score(item: dict[str, Any]) -> float:
            metrics = item.get("metrics") or {}
            slow = float(metrics.get("short_slowdown_p99_vs_strict_isolated") or 0.0)
            ttft_ratio = float(metrics.get("ttft_ratio_vs_strict") or 999.0)
            penalty = max(0.0, ttft_ratio - 1.20) * 4.0
            return slow - penalty

        best = max(results, key=_score)
        output = {
            "model": model,
            "results": results,
            "best": best,
        }
        _write_json(exp_root / "summary.json", output)
        print(json.dumps({"best_candidate": best["candidate"], "metrics": best["metrics"]}, indent=2))
    finally:
        _cleanup_engine(engine)


if __name__ == "__main__":
    main()
