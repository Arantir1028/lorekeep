from __future__ import annotations

import time
from argparse import Namespace
from pathlib import Path
from typing import Any

from eval_config import build_summary_config
from eval_support import raw_mode_summary

from experiments.result_io import write_json


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
    phase1_no_chunk_control: dict[str, Any] | None,
    phase12: dict[str, Any] | None,
) -> dict[str, Any]:
    summary = {
        "config": build_summary_config(
            args, short_a_repeat=short_a_repeat, short_b_repeat=short_b_repeat
        ),
        "token_lengths": tok_lens,
        "phase1_baseline_chunked_raw": raw_mode_summary(phase1_base_rounds)
        if need_chunked_baseline
        else None,
        "phase1_baseline_no_chunk_raw": raw_mode_summary(phase1_no_chunk_rounds)
        if need_no_chunk_baseline
        else None,
        "phase1": phase1["summary"],
        "phase2": phase2["summary"],
        "per_repeat": {
            "phase1_baseline_chunked": phase1_base_rounds if need_chunked_baseline else None,
            "phase1_baseline_no_chunk": phase1_no_chunk_rounds if need_no_chunk_baseline else None,
            "phase1": phase1["rows"],
            "phase2": phase2["rows"],
        },
    }
    optional = (
        (phase1_no_chunk_control, "phase1_chunked_vs_no_chunk"),
        (phase12, "phase12"),
    )
    for block, key in optional:
        if block is not None:
            summary[key] = block["summary"]
            summary["per_repeat"][key] = block["rows"]
    return summary


def print_summary(summary: dict[str, Any], *, include_phase12: bool) -> None:
    sections = (("phase1", True), ("phase2", True), ("phase12", include_phase12))
    for key, enabled in sections:
        if enabled and summary.get(key) is not None:
            print(f"[Summary] {key}={summary[key]}")


def write_summary_json(summary: dict[str, Any], out_json: str | None) -> str:
    path = out_json or f"results/waveslice_repeated_eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    write_json(Path(path), summary)
    return path
