from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.chapter2_prestudy import (
    _candidate_model_entries,
    _config_for_model,
    _ensure_dir,
    _load_json,
    _make_lora_requests,
    _make_phase1_requests,
    _mean,
    _p99,
    _resolve_model,
    _resolve_out_root,
    _run_eval_case,
    _write_json,
    _write_md,
)
from experiments.openworkload_models import resolve_model_entry


def _strict_ref_by_req(per_req_runs: dict[str, dict[str, Any]]) -> dict[str, float]:
    ref: dict[str, float] = {}
    for req_id, summary in per_req_runs.items():
        rows = list((summary.get("per_repeat") or {}).get("phase2") or [])
        finishes: list[float] = []
        for row in rows:
            timings = row.get("base_request_timings") or {}
            item = timings.get(req_id) or {}
            finish = item.get("finish_latency_ms")
            if finish is not None:
                finishes.append(float(finish))
        if finishes:
            ref[req_id] = float(sum(finishes) / len(finishes))
    return ref


def _maybe_reuse_eval(case_root: Path, case_name: str) -> dict[str, Any] | None:
    out_json = case_root / "results" / f"{case_name}_eval.json"
    if out_json.exists():
        return _load_json(out_json)
    return None


def _row_short_slowdown_p99(row: dict[str, Any], reference_by_req: dict[str, float]) -> float | None:
    timings = row.get("base_request_timings") or {}
    ratios: list[float] = []
    for req_id, item in timings.items():
        if not isinstance(item, dict) or not bool(item.get("is_short")):
            continue
        finish = item.get("finish_latency_ms")
        ref = reference_by_req.get(str(req_id))
        if finish is None or ref is None or ref <= 0:
            continue
        ratios.append(float(finish) / ref)
    return _p99(ratios) if ratios else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict isolated-reference variant for Chapter 2 E1.")
    parser.add_argument(
        "--config",
        default="experiments/configs/chapter2_prestudy_v1.json",
        help="Path to the Chapter 2 prestudy config.",
    )
    parser.add_argument(
        "--out-root",
        default="",
        help="Optional output root override.",
    )
    parser.add_argument(
        "--model-key",
        default="gemma-2-9b-it",
        help="Catalog model key to use for the strict E1 run.",
    )
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
    exp_root = _ensure_dir(out_root / "E1_strict_isolated_reference_fixed_decode")

    model = _resolve_model(config)
    eval_cfg = dict(config.get("eval") or {})
    phase1_cfg = dict(config.get("phase1") or {})
    phase12_cfg = dict(config.get("phase12_soft_gate") or {})
    phase2_cfg = dict(config.get("phase2") or {})
    micro_cfg = dict(config.get("microbenchmark") or {})

    shared_kwargs = {
        "short_count": int(micro_cfg.get("short_count", 6)),
        "long_count": int(micro_cfg.get("long_count", 1)),
        "short_prompt_repeat": int(micro_cfg.get("short_prompt_repeat", 2)),
        "long_prompt_repeat": int(micro_cfg.get("long_prompt_repeat", 40)),
        "short_start_s": float(micro_cfg.get("short_start_s", 0.20)),
        "short_gap_s": float(micro_cfg.get("short_gap_s", 0.05)),
        "long_gap_s": float(micro_cfg.get("long_gap_s", 0.0)),
        "sequential_delay_s": float(micro_cfg.get("sequential_delay_s", 6.0)),
    }

    overlap_lora = _make_lora_requests(pattern="long_first", mixed_adapters=True, **shared_kwargs)
    phase1_stub = _make_phase1_requests(
        pattern="long_first",
        short_count=shared_kwargs["short_count"],
        short_prompt_repeat=shared_kwargs["short_prompt_repeat"],
        long_prompt_repeat=shared_kwargs["long_prompt_repeat"],
        short_start_s=shared_kwargs["short_start_s"],
        short_gap_s=shared_kwargs["short_gap_s"],
        sequential_delay_s=shared_kwargs["sequential_delay_s"],
    )

    overlap = _maybe_reuse_eval(exp_root, "dynamic_overlap")
    if overlap is None:
        overlap = _run_eval_case(
            case_root=exp_root,
            case_name="dynamic_overlap",
            model=model,
            eval_cfg=eval_cfg,
            phase1_cfg=phase1_cfg,
            phase12_cfg=phase12_cfg,
            phase2_cfg=phase2_cfg,
            requests=phase1_stub,
            lora_requests=overlap_lora,
            include_phase12=False,
            skip_phase2=False,
            phase1_baseline_mode="chunked",
            ignore_eos=True,
        )

    per_short_runs: dict[str, dict[str, Any]] = {}
    for item in overlap_lora:
        if not bool(item.get("is_short")):
            continue
        req_id = str(item["req_id"])
        solo_item = dict(item)
        solo_item["arrival_offset_s"] = 0.0
        solo_phase1_item = {
            "req_id": req_id,
            "prompt": str(item.get("prompt") or ""),
            "is_short": True,
            "arrival_offset_s": 0.0,
        }
        case_name = f"solo_{req_id}"
        reused = _maybe_reuse_eval(exp_root, case_name)
        if reused is not None:
            per_short_runs[req_id] = reused
            continue
        per_short_runs[req_id] = _run_eval_case(
            case_root=exp_root,
            case_name=case_name,
            model=model,
            eval_cfg=eval_cfg,
            phase1_cfg=phase1_cfg,
            phase12_cfg=phase12_cfg,
            phase2_cfg=phase2_cfg,
            requests=[solo_phase1_item],
            lora_requests=[solo_item],
            include_phase12=False,
            skip_phase2=False,
            phase1_baseline_mode="chunked",
            ignore_eos=True,
        )

    ref_by_req = _strict_ref_by_req(per_short_runs)
    overlap_rows = list((overlap.get("per_repeat") or {}).get("phase2") or [])
    slowdown_rows = [_row_short_slowdown_p99(row, ref_by_req) for row in overlap_rows]
    slowdown_rows = [v for v in slowdown_rows if v is not None]

    ttft_rows = [float(row.get("base_ttft_short_p99_ms")) for row in overlap_rows if row.get("base_ttft_short_p99_ms") is not None]
    wall_rows = [float(row.get("base_round_wall_ms")) for row in overlap_rows if row.get("base_round_wall_ms") is not None]

    summary = {
        "model": model,
        "strict_reference_by_req_ms": ref_by_req,
        "naive_overlap": {
            "short_ttft_p99_ms": _mean(ttft_rows),
            "short_slowdown_p99_vs_strict_isolated": _mean(slowdown_rows),
            "round_wall_ms": _mean(wall_rows),
        },
    }
    _write_json(exp_root / "summary.json", summary)
    _write_md(
        exp_root / "summary.md",
        [
            "# E1 Strict Isolated Reference",
            "",
            f"- Model: {model['label']}",
            f"- Naive overlap short TTFT p99 (ms): {summary['naive_overlap']['short_ttft_p99_ms']}",
            f"- Naive overlap short slowdown p99 vs strict isolated: {summary['naive_overlap']['short_slowdown_p99_vs_strict_isolated']}",
            f"- Naive overlap round wall (ms): {summary['naive_overlap']['round_wall_ms']}",
            "- Decode budget: fixed via --ignore-eos",
        ],
    )


if __name__ == "__main__":
    main()
