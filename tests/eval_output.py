from __future__ import annotations

import json
import os
import time
from argparse import Namespace
from typing import Any, Optional

from eval_config import build_summary_config
from eval_support import raw_mode_summary


def build_summary(
    *,
    args: Namespace,
    short_a_repeat: int,
    short_b_repeat: int,
    tok_lens: dict[str, int],
    need_chunked_baseline: bool,
    need_no_chunk_baseline: bool,
    phase1_base_rounds: list[dict[str, Any]],
    phase1_no_chunk_rounds: list[dict[str, Any]],
    phase1: dict[str, Any],
    phase2: dict[str, Any],
    phase1_no_chunk_control: Optional[dict[str, Any]],
    phase1_lora: Optional[dict[str, Any]],
    phase12: Optional[dict[str, Any]],
) -> dict[str, Any]:
    summary = {
        "config": build_summary_config(
            args,
            short_a_repeat=short_a_repeat,
            short_b_repeat=short_b_repeat,
        ),
        "token_lengths": tok_lens,
        "phase1_baseline_chunked_raw": raw_mode_summary(phase1_base_rounds) if need_chunked_baseline else None,
        "phase1_baseline_no_chunk_raw": raw_mode_summary(phase1_no_chunk_rounds) if need_no_chunk_baseline else None,
        "phase1": phase1["summary"],
        "phase2": phase2["summary"],
        "per_repeat": {
            "phase1_baseline_chunked": phase1_base_rounds if need_chunked_baseline else None,
            "phase1_baseline_no_chunk": phase1_no_chunk_rounds if need_no_chunk_baseline else None,
            "phase1": phase1["rows"],
            "phase2": phase2["rows"],
        },
    }
    if phase1_no_chunk_control is not None:
        summary["phase1_chunked_vs_no_chunk"] = phase1_no_chunk_control["summary"]
        summary["per_repeat"]["phase1_chunked_vs_no_chunk"] = phase1_no_chunk_control["rows"]
    if args.include_strict:
        summary["phase2_strict"] = phase2["strict_summary"]
        summary["per_repeat"]["phase2_strict"] = phase2["strict_rows"]
    if phase1_lora is not None:
        summary["phase1_lora"] = phase1_lora["summary"]
        summary["per_repeat"]["phase1_lora"] = phase1_lora["rows"]
    if phase12 is not None:
        summary["phase12"] = phase12["summary"]
        summary["per_repeat"]["phase12"] = phase12["rows"]
    return summary


def print_summary(
    summary: dict[str, Any],
    *,
    include_strict: bool,
    include_phase1_lora_only: bool,
    include_phase12: bool,
) -> None:
    print("\n[Summary] Phase-I")
    if summary.get("phase1_baseline_no_chunk_raw") is not None:
        print(f"  no_chunk_ttft_short_p99_ms={summary['phase1_baseline_no_chunk_raw']['ttft_short_p99_ms']}")
        print(f"  no_chunk_round_wall_ms={summary['phase1_baseline_no_chunk_raw']['round_wall_ms']}")
    if summary.get("phase1_baseline_chunked_raw") is not None:
        print(f"  chunked_ttft_short_p99_ms={summary['phase1_baseline_chunked_raw']['ttft_short_p99_ms']}")
        print(f"  chunked_round_wall_ms={summary['phase1_baseline_chunked_raw']['round_wall_ms']}")
    if summary.get("phase1_chunked_vs_no_chunk") is not None:
        print(f"  chunked_vs_no_chunk_ttft_ratio={summary['phase1_chunked_vs_no_chunk']['ttft_improve_ratio']}")
        print(f"  chunked_vs_no_chunk_wall_ratio={summary['phase1_chunked_vs_no_chunk']['round_wall_improve_ratio']}")
    print(f"  ttft_improve_ratio={summary['phase1']['ttft_improve_ratio']}")
    print(f"  round_wall_improve_ratio={summary['phase1']['round_wall_improve_ratio']}")
    print(f"  error_rate={summary['phase1']['error_rate']}")
    print(f"  baseline_noise_error_rate={summary['phase1']['baseline_noise_error_rate']}")
    print(f"  incremental_error_rate={summary['phase1']['incremental_error_rate']}")
    print(f"  scheduler_apply_ratio={summary['phase1']['scheduler_apply_ratio']}")
    print(f"  baseline_chunk_avg={summary['phase1']['baseline_chunk_avg']}")
    print(f"  chosen_chunk_avg={summary['phase1']['chosen_chunk_avg']}")
    print(f"  chosen_vs_baseline_ratio_avg={summary['phase1']['chosen_vs_baseline_ratio_avg']}")
    print(f"  explicit_plan_ratio={summary['phase1']['explicit_plan_ratio']}")
    print(f"  rewrite_apply_ratio={summary['phase1']['rewrite_apply_ratio']}")
    print(f"  rewrite_old_chunk_avg={summary['phase1']['rewrite_old_chunk_avg']}")
    print(f"  rewrite_new_chunk_avg={summary['phase1']['rewrite_new_chunk_avg']}")
    print(f"  rewrite_token_delta_avg={summary['phase1']['rewrite_token_delta_avg']}")
    print(f"  virtual_cap_apply_ratio={summary['phase1']['virtual_cap_apply_ratio']}")
    print(f"  virtual_cap_old_avg={summary['phase1']['virtual_cap_old_avg']}")
    print(f"  virtual_cap_new_avg={summary['phase1']['virtual_cap_new_avg']}")
    print(f"  runtime_effective_pressure_avg={summary['phase1'].get('runtime_effective_pressure_avg')}")
    print(f"  runtime_target_fraction_avg={summary['phase1'].get('runtime_target_fraction_avg')}")
    print(f"  runtime_target_chunk_avg={summary['phase1'].get('runtime_target_chunk_avg')}")

    print("\n[Summary] Phase-II")
    print(f"  ttft_improve_ratio={summary['phase2']['ttft_improve_ratio']}")
    print(f"  slowdown_improve_ratio={summary['phase2']['slowdown_improve_ratio']}")
    print(f"  round_wall_improve_ratio={summary['phase2']['round_wall_improve_ratio']}")
    print(f"  wave_error_rate={summary['phase2']['wave_error_rate']}")
    print(f"  baseline_noise_error_rate={summary['phase2']['baseline_noise_error_rate']}")
    print(f"  incremental_error_rate={summary['phase2']['incremental_error_rate']}")
    print(f"  phase2_apply_ratio={summary['phase2']['phase2_apply_ratio']}")

    if include_phase1_lora_only and summary.get("phase1_lora") is not None:
        print("\n[Summary] Phase-I on LoRA")
        print(f"  ttft_improve_ratio={summary['phase1_lora']['ttft_improve_ratio']}")
        print(f"  slowdown_improve_ratio={summary['phase1_lora']['slowdown_improve_ratio']}")
        print(f"  round_wall_improve_ratio={summary['phase1_lora']['round_wall_improve_ratio']}")
        print(f"  wave_error_rate={summary['phase1_lora']['wave_error_rate']}")
        print(f"  baseline_noise_error_rate={summary['phase1_lora']['baseline_noise_error_rate']}")
        print(f"  incremental_error_rate={summary['phase1_lora']['incremental_error_rate']}")
        print(f"  phase2_apply_ratio={summary['phase1_lora']['phase2_apply_ratio']}")

    if include_strict:
        print("\n[Summary] Phase-II Strict")
        print(f"  ttft_improve_ratio={summary['phase2_strict']['ttft_improve_ratio']}")
        print(f"  slowdown_improve_ratio={summary['phase2_strict']['slowdown_improve_ratio']}")
        print(f"  round_wall_improve_ratio={summary['phase2_strict']['round_wall_improve_ratio']}")
        print(f"  error_rate={summary['phase2_strict']['error_rate']}")
        print(f"  incremental_error_rate={summary['phase2_strict']['incremental_error_rate']}")
        print(f"  apply_ratio={summary['phase2_strict']['apply_ratio']}")

    if include_phase12 and summary.get("phase12") is not None:
        print("\n[Summary] Phase-I + Phase-II")
        print(f"  ttft_improve_ratio={summary['phase12']['ttft_improve_ratio']}")
        print(f"  slowdown_improve_ratio={summary['phase12']['slowdown_improve_ratio']}")
        print(f"  round_wall_improve_ratio={summary['phase12']['round_wall_improve_ratio']}")
        print(f"  wave_error_rate={summary['phase12']['wave_error_rate']}")
        print(f"  baseline_noise_error_rate={summary['phase12']['baseline_noise_error_rate']}")
        print(f"  incremental_error_rate={summary['phase12']['incremental_error_rate']}")
        print(f"  phase2_apply_ratio={summary['phase12']['phase2_apply_ratio']}")


def write_summary_json(summary: dict[str, Any], out_json: Optional[str]) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = out_json or f"results/waveslice_repeated_eval_{ts}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return path
