from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CUCUMIS_FIG_DIR = ROOT / "results" / "chapter5_exports" / "default"
DEFAULT_MAIN_RUN = ROOT / "results" / "openworkload_v1_local_realworld_lora8" / "20260420_chapter5_eq1_eq5_lora8"
DEFAULT_E5_SUMMARY = (
    ROOT
    / "results"
    / "chapter2_prestudy_v1"
    / "20260420_lora_nonllama_eq1_eq5"
    / "E5_lora_multitenancy_relevance"
    / "summary_all_models.json"
)
CUCUMIS_FIG_DIR = DEFAULT_CUCUMIS_FIG_DIR
MAIN_RUN = DEFAULT_MAIN_RUN
E5_SUMMARY = DEFAULT_E5_SUMMARY
EXCLUDED_MODEL_KEYS = {"gemma-2-9b-it"}
DENSITY_ORDER = ["low", "mid", "high", "peak"]
MODEL_LABELS = {
    "mistral-7b-v0.1": "Mistral",
    "mistral-7b-instruct-v0.2": "M-I",
    "zephyr-7b-beta": "Zephyr",
    "openchat-3.5-0106": "OpenChat",
    "gemma-7b-it": "Gemma",
    "baichuan2-7b-chat": "BC2",
    "qwen2.5-7b-instruct": "Qwen",
}


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _apply_paper_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.titlesize": 15,
        }
    )


def _annotate_bar_values(
    ax: plt.Axes,
    xs: list[int],
    means: list[float],
    errs: list[float],
    *,
    suffix: str,
) -> None:
    ymax = ax.get_ylim()[1]
    lift = ymax * 0.02
    for x, mean_val, err_val in zip(xs, means, errs):
        ax.text(
            x,
            mean_val + err_val + lift,
            f"{mean_val:.1f}{suffix}",
            ha="center",
            va="bottom",
            fontsize=11,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 0.3},
            zorder=5,
        )


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(float(v) for v in values)
    if not ordered:
        raise ValueError("percentile on empty sequence")
    if q <= 0:
        return ordered[0]
    if q >= 100:
        return ordered[-1]
    k = (len(ordered) - 1) * (q / 100.0)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return ordered[lo]
    frac = k - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def _all_ttft_p99_ms(request_timings: dict[str, object]) -> float:
    vals = []
    for item in request_timings.values():
        if isinstance(item, dict) and isinstance(item.get("first_latency_ms"), (int, float)):
            vals.append(float(item["first_latency_ms"]))
    if not vals:
        raise ValueError("missing first_latency_ms values")
    return _percentile(vals, 99.0)


def _mean_ci95(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("empty values")
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), 1.96 * stdev(values) / math.sqrt(len(values))


def _load_density_case_metrics() -> dict[str, dict[str, dict[str, list[float]]]]:
    suite_rows = _read_json(MAIN_RUN / "metadata" / "suite_results.json")
    assert isinstance(suite_rows, list)

    grouped: dict[str, dict[str, dict[str, list[float]]]] = {density: {} for density in DENSITY_ORDER}
    for row in suite_rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("status")) != "ok":
            continue
        model_key = str(row.get("model_key") or "")
        density = str(row.get("density") or "")
        if model_key in EXCLUDED_MODEL_KEYS or density not in grouped:
            continue
        model_label = MODEL_LABELS.get(model_key, model_key)

        result_path = MAIN_RUN / "raw" / density / f"{model_key}_dataset_eval.json"
        result = _read_json(result_path)
        assert isinstance(result, dict)
        phase12 = ((result.get("per_repeat") or {}).get("phase12") or [])
        base_short_vals = []
        wave_short_vals = []
        base_all_vals = []
        wave_all_vals = []
        for repeat in phase12:
            if not isinstance(repeat, dict):
                continue
            base_short = float(repeat["base_ttft_short_p99_ms"])
            wave_short = float(repeat["wave_ttft_short_p99_ms"])
            base_all = _all_ttft_p99_ms(repeat.get("base_request_timings") or {})
            wave_all = _all_ttft_p99_ms(repeat.get("wave_request_timings") or {})
            base_short_vals.append(base_short)
            wave_short_vals.append(wave_short)
            base_all_vals.append(base_all)
            wave_all_vals.append(wave_all)

        bucket = grouped[density].setdefault(
            model_label,
            {
                "base_short_ms": [],
                "wave_short_ms": [],
                "base_all_ms": [],
                "wave_all_ms": [],
            },
        )
        bucket["base_short_ms"].extend(base_short_vals)
        bucket["wave_short_ms"].extend(wave_short_vals)
        bucket["base_all_ms"].extend(base_all_vals)
        bucket["wave_all_ms"].extend(wave_all_vals)
    return grouped


def regenerate_density_sweep() -> Path:
    density_metrics = _load_density_case_metrics()
    groups = list(MODEL_LABELS.values()) + ["All"]

    _apply_paper_style()
    fig, axes = plt.subplots(2, 2, figsize=(16.8, 10.2), dpi=220, constrained_layout=True)
    method_specs = [("base", "vLLM", "#7F7F7F"), ("wave", "CUCUMIS", "#E45756")]
    short_bar_width = 0.16
    all_bar_width = 0.16
    xs = list(range(len(groups)))
    short_offset_base = -0.27
    short_offset_wave = -0.09
    all_offset_base = 0.09
    all_offset_wave = 0.27

    short_max = 0.0
    all_max = 0.0
    for density in DENSITY_ORDER:
        for bucket in density_metrics[density].values():
            for key in ("base_short_ms", "wave_short_ms"):
                values = bucket.get(key) or []
                if values:
                    m, ci = _mean_ci95([float(v) for v in values])
                    short_max = max(short_max, m + ci)
            for key in ("base_all_ms", "wave_all_ms"):
                values = bucket.get(key) or []
                if values:
                    m, ci = _mean_ci95([float(v) for v in values])
                    all_max = max(all_max, m + ci)

    legend_handles = []
    legend_labels = []

    flat_axes = list(axes.flat)
    for col_idx, density in enumerate(DENSITY_ORDER):
        ax_short = flat_axes[col_idx]
        ax_all = ax_short.twinx()

        short_specs = [
            ("base_short_ms", "Short / vLLM", "#7F7F7F", short_offset_base, short_bar_width, None),
            ("wave_short_ms", "Short / CUCUMIS", "#E45756", short_offset_wave, short_bar_width, None),
        ]
        all_specs = [
            ("base_all_ms", "All / vLLM", "#C7C7C7", all_offset_base, all_bar_width, "//"),
            ("wave_all_ms", "All / CUCUMIS", "#F2A2A0", all_offset_wave, all_bar_width, "//"),
        ]

        for metric_key, label, color, offset, width, hatch in short_specs:
            means = []
            errs = []
            for group in groups:
                if group == "All":
                    per_model_means = [
                        mean(bucket.get(metric_key) or [])
                        for bucket in density_metrics[density].values()
                        if bucket.get(metric_key)
                    ]
                    m, ci = _mean_ci95(per_model_means)
                else:
                    bucket = density_metrics[density].get(group) or {}
                    m, ci = _mean_ci95([float(v) for v in (bucket.get(metric_key) or [])])
                means.append(m)
                errs.append(ci)
            bars = ax_short.bar(
                [x + offset for x in xs],
                means,
                yerr=errs,
                width=width,
                color=color,
                edgecolor="#4A4A4A",
                linewidth=0.3,
                capsize=2.5,
                error_kw={"elinewidth": 0.9, "capthick": 0.9},
                label=label,
                zorder=3,
            )
            if col_idx == 0:
                legend_handles.append(bars[0])
                legend_labels.append(label)

        for metric_key, label, color, offset, width, hatch in all_specs:
            means = []
            errs = []
            for group in groups:
                if group == "All":
                    per_model_means = [
                        mean(bucket.get(metric_key) or [])
                        for bucket in density_metrics[density].values()
                        if bucket.get(metric_key)
                    ]
                    m, ci = _mean_ci95(per_model_means)
                else:
                    bucket = density_metrics[density].get(group) or {}
                    m, ci = _mean_ci95([float(v) for v in (bucket.get(metric_key) or [])])
                means.append(m)
                errs.append(ci)
            bars = ax_all.bar(
                [x + offset for x in xs],
                means,
                yerr=errs,
                width=width,
                color=color,
                edgecolor="#4A4A4A",
                linewidth=0.3,
                hatch=hatch,
                capsize=2.5,
                error_kw={"elinewidth": 0.9, "capthick": 0.9},
                label=label,
                zorder=2,
            )
            if col_idx == 0:
                legend_handles.append(bars[0])
                legend_labels.append(label)

        ax_short.set_title(density.capitalize())
        ax_short.set_xticks(xs, groups, rotation=18)
        ax_short.set_ylim(0, short_max * 1.15)
        ax_all.set_ylim(0, all_max * 1.15)
        ax_short.set_axisbelow(True)
        ax_short.axvline(len(groups) - 1.5, color="#A0AEC0", linestyle=":", linewidth=1.0, alpha=0.9)

        if col_idx == 0:
            ax_short.set_ylabel("Short-Request p99 TTFT (ms)")
            ax_all.set_ylabel("All-Request p99 TTFT (ms)")

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.08),
        frameon=True,
        facecolor="white",
        edgecolor="#D9D9D9",
        framealpha=0.95,
    )
    out_path = CUCUMIS_FIG_DIR / "density_sweep.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _short_label(model_key: str) -> str:
    mapping = {
        "mistral-7b-v0.1": "Mistral",
        "mistral-7b-instruct-v0.2": "Mistral-I",
        "zephyr-7b-beta": "Zephyr",
        "openchat-3.5-0106": "OpenChat",
        "gemma-7b-it": "Gemma",
        "baichuan2-7b-chat": "Baichuan2",
        "qwen2.5-7b-instruct": "Qwen2.5",
    }
    return mapping.get(model_key, model_key)


def regenerate_lora_latency_dispersion() -> Path:
    summary = _read_json(E5_SUMMARY)
    assert isinstance(summary, dict)
    model_entries = summary.get("models") or []
    per_model = summary.get("per_model") or {}

    ordered_keys = [
        str(item.get("key"))
        for item in model_entries
        if isinstance(item, dict) and str(item.get("key")) not in EXCLUDED_MODEL_KEYS
    ]

    homogeneous_ttft = []
    mixed_ttft = []
    homogeneous_ratio_dist = []
    mixed_ratio_dist = []
    labels = []
    for model_key in ordered_keys:
        entry = per_model.get(model_key) or {}
        homo = entry.get("lora_homogeneous") or {}
        mixed = entry.get("lora_mixed_adapters") or {}
        if not isinstance(homo, dict) or not isinstance(mixed, dict):
            continue
        homogeneous_ttft.append(float(homo["short_ttft_p99_ms"]))
        mixed_ttft.append(float(mixed["short_ttft_p99_ms"]))
        homogeneous_ratio_dist.extend(float(x) for x in (entry.get("lora_homogeneous_ratio_distribution") or []))
        mixed_ratio_dist.extend(float(x) for x in (entry.get("lora_mixed_ratio_distribution") or []))
        labels.append(_short_label(model_key))

    _apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 5.4), dpi=200, constrained_layout=True)
    xs = list(range(len(labels)))

    for i, (homo, mixed) in enumerate(zip(homogeneous_ttft, mixed_ttft)):
        axes[0].plot([i, i], [homo, mixed], color="#B0B7C3", linewidth=1.4, zorder=1)
    axes[0].scatter(xs, homogeneous_ttft, color="#2B6CB0", s=62, label="Homogeneous LoRA", zorder=3)
    axes[0].scatter(xs, mixed_ttft, color="#D97706", s=62, label="Mixed-adapter LoRA", zorder=3)
    axes[0].set_title("Short Requests: p99 TTFT by Model")
    axes[0].set_ylabel("TTFT p99 (ms)")
    axes[0].set_xticks(xs, labels, rotation=22, ha="center")
    axes[0].set_xlim(-0.35, len(labels) - 0.05)
    axes[0].legend(
        frameon=True,
        facecolor="white",
        edgecolor="#D9D9D9",
        framealpha=0.95,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.995),
        borderaxespad=0.35,
        labelspacing=0.35,
    )

    box = axes[1].boxplot(
        [homogeneous_ratio_dist, mixed_ratio_dist],
        labels=["Homogeneous", "Mixed"],
        patch_artist=True,
        widths=0.5,
        showfliers=False,
    )
    for patch, color in zip(box["boxes"], ["#2B6CB0", "#D97706"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axes[1].axhline(1.0, color="#4A5568", linestyle="--", linewidth=1.0)
    axes[1].set_title("Relative Short-Request Interference")
    axes[1].set_ylabel("Interference ratio")

    out_path = CUCUMIS_FIG_DIR / "lora_latency_dispersion.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate Chapter 5 partial figures from Chapter 5 and Chapter 2 outputs.")
    parser.add_argument("--main-run", default=str(DEFAULT_MAIN_RUN))
    parser.add_argument("--out-dir", default=str(DEFAULT_CUCUMIS_FIG_DIR))
    parser.add_argument("--e5-summary", default=str(DEFAULT_E5_SUMMARY))
    parser.add_argument("--skip-density-sweep", action="store_true")
    parser.add_argument("--skip-lora-latency-dispersion", action="store_true")
    args = parser.parse_args()

    global MAIN_RUN, CUCUMIS_FIG_DIR, E5_SUMMARY
    MAIN_RUN = Path(args.main_run).expanduser().resolve()
    CUCUMIS_FIG_DIR = Path(args.out_dir).expanduser().resolve()
    E5_SUMMARY = Path(args.e5_summary).expanduser().resolve()
    if not (MAIN_RUN / "metadata" / "suite_results.json").exists():
        raise FileNotFoundError(f"main suite metadata not found under: {MAIN_RUN}")
    CUCUMIS_FIG_DIR.mkdir(parents=True, exist_ok=True)
    if not args.skip_density_sweep:
        density_path = regenerate_density_sweep()
        print(f"updated {density_path}")
    if not args.skip_lora_latency_dispersion:
        if not E5_SUMMARY.exists():
            raise FileNotFoundError(f"E5 summary not found: {E5_SUMMARY}")
        lora_path = regenerate_lora_latency_dispersion()
        print(f"updated {lora_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
