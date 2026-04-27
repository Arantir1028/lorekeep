from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _case(summary: dict[str, Any], name: str) -> dict[str, Any]:
    for row in summary["rows"]:
        if row["case"] == name:
            return row
    raise KeyError(f"missing case {name!r}")


def _draw_boundary_figure(summary: dict[str, Any], out_pdf: Path) -> None:
    one_long = _case(summary, "one_long")
    two_long = _case(summary, "two_long_contenders")
    sweep = [_case(summary, f"two_long_chunk_{chunk}") for chunk in (512, 768, 1536)]

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 1, figsize=(3.55, 4.55), constrained_layout=True)

    contention_labels = ["1 long\nprefill", "2 long\nprefills"]
    contention_values = [
        float(one_long["short_ttft_p99_ms"]),
        float(two_long["short_ttft_p99_ms"]),
    ]
    axes[0].bar(
        range(len(contention_values)),
        contention_values,
        color=["#6B7280", "#2563EB"],
        width=0.56,
    )
    axes[0].set_title("(a) Boundaries under contention")
    axes[0].set_ylabel("Short p99 TTFT (ms)")
    axes[0].set_xticks(range(len(contention_labels)), contention_labels)
    axes[0].set_ylim(0, max(contention_values) * 1.22)
    for idx, value in enumerate(contention_values):
        axes[0].text(idx, value + max(contention_values) * 0.035, f"{value:.1f}", ha="center")
    axes[0].grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    axes[0].set_axisbelow(True)

    sweep_labels = ["512", "768", "1536"]
    sweep_values = [float(row["short_ttft_p99_ms"]) for row in sweep]
    axes[1].plot(
        range(len(sweep_values)),
        sweep_values,
        marker="o",
        markersize=6,
        linewidth=2.2,
        color="#C2410C",
    )
    axes[1].set_title("(b) Fixed chunk-size sweep")
    axes[1].set_ylabel("Short p99 TTFT (ms)")
    axes[1].set_xlabel("Chunk size (tokens)")
    axes[1].set_xticks(range(len(sweep_labels)), sweep_labels)
    axes[1].set_xlim(-0.35, len(sweep_values) - 0.65)
    ymin = min(sweep_values) - 40
    ymax = max(sweep_values) + 45
    axes[1].set_ylim(max(0, ymin), ymax)
    for idx, value in enumerate(sweep_values):
        x_offset = 0.10 if idx == 0 else (-0.10 if idx == len(sweep_values) - 1 else 0.0)
        axes[1].text(idx + x_offset, value + 10, f"{value:.1f}", ha="center")
    axes[1].grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    axes[1].set_axisbelow(True)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render paper-facing Chapter 2 boundary observation figures from baseline-only summaries."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "results/chapter2_observations_v1/"
            "20260427_sara_baseline_obs_long70/"
            "obs2_boundaries_do_not_guarantee_short_benefit/"
            "summary.json"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "results/chapter2_observations_v1/"
            "20260427_sara_baseline_obs_long70/"
            "paper_figures/chapter2_boundaries_not_benefits.pdf"
        ),
    )
    parser.add_argument("--paper-copy", type=Path, default=None)
    args = parser.parse_args()

    summary = _load_json(args.summary)
    _draw_boundary_figure(summary, args.out)
    if args.paper_copy is not None:
        args.paper_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.out, args.paper_copy)
    print(args.out)
    if args.paper_copy is not None:
        print(args.paper_copy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
