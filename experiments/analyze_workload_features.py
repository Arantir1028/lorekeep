"""Analyze workload features against Wave-Slice gains.

This script builds a reusable workload feature table by joining:
- result CSV manifests under results/
- evaluation JSON summaries
- workload meta JSON
- request JSON / lora request JSON when present

It then emits:
- a flattened feature CSV
- a JSON summary with simple Spearman-style rank correlations
- a Markdown note describing the most promising workload family
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Any


RESULT_TARGETS = [
    "phase12_ttft_improve_mean",
    "phase12_wall_improve_mean",
    "phase12_slowdown_improve_mean",
    "phase12_apply_mean",
    "phase12_incremental_error_mean",
]


@dataclass
class Record:
    key: str
    row: dict[str, Any]


def _safe_float(v: Any) -> float | None:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except Exception:
        return None


def _coerce_scalar(v: Any) -> Any:
    if not isinstance(v, str):
        return v
    s = v.strip()
    if s == "":
        return v
    if s.lower() in {"true", "false"}:
        return s.lower() == "true"
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except Exception:
        return v


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _std(xs: list[float]) -> float | None:
    return statistics.pstdev(xs) if len(xs) >= 2 else 0.0 if xs else None


def _quantile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    idx = int(round((len(ys) - 1) * q))
    idx = max(0, min(idx, len(ys) - 1))
    return ys[idx]


def _cv(xs: list[float]) -> float | None:
    if not xs:
        return None
    mu = _mean(xs)
    if mu is None or mu == 0:
        return None
    sd = _std(xs)
    return None if sd is None else sd / mu


def _rankdata(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def _pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mx = _mean(x)
    my = _mean(y)
    if mx is None or my is None:
        return None
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denx = math.sqrt(sum((a - mx) ** 2 for a in x))
    deny = math.sqrt(sum((b - my) ** 2 for b in y))
    if denx == 0 or deny == 0:
        return None
    return num / (denx * deny)


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    return _pearson(_rankdata(x), _rankdata(y))


def _load_json(path: str) -> Any | None:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _infer_policy_variant(result_path: str) -> str:
    stem = os.path.splitext(os.path.basename(result_path))[0].lower()
    if stem.endswith("_hrrn"):
        return "hrrn"
    if stem.endswith("_aging"):
        return "aging"
    return "sjf"


def _infer_density_label(result_path: str, meta: dict[str, Any] | None) -> str | None:
    if meta:
        p1 = _safe_float(meta.get("phase1_arrival_rate"))
        p2 = _safe_float(meta.get("phase2_arrival_rate"))
        if p1 is not None and p1 == p2:
            return f"rate_{int(p1) if p1.is_integer() else p1}"
    stem = os.path.splitext(os.path.basename(result_path))[0]
    parts = stem.split("_")
    for token in reversed(parts):
        if token in {"low", "midlow", "mid", "midhigh", "highmid", "high"}:
            return token
    return None


def _glob_for(prefix: str) -> list[str]:
    return sorted(glob.glob(prefix))


def _infer_workload_paths(result_path: str, workload_root: str) -> tuple[str | None, str | None, str | None]:
    stem = os.path.splitext(os.path.basename(result_path))[0]
    parent = os.path.basename(os.path.dirname(result_path))
    parts = stem.split("_")
    if len(parts) < 3:
        return None, None, None
    model = "_".join(parts[:-2])
    density = parts[-2]
    suffix_hints = [hint for hint in ("beneficiary", "reprofile") if hint in parent]
    patterns = [
        os.path.join(workload_root, f"{model}_{density}_meta.json"),
    ]
    patterns.extend(
        os.path.join(workload_root, f"{model}_{density}_{hint}_meta.json")
        for hint in suffix_hints
    )
    matches: list[str] = []
    for pattern in patterns:
        matches.extend(_glob_for(pattern))
    if not matches:
        return None, None, None
    exact_matches = [m for m in set(matches) if os.path.basename(m) == f"{model}_{density}_meta.json"]
    if exact_matches:
        meta_path = sorted(exact_matches, key=lambda p: os.path.getmtime(p), reverse=True)[0]
    elif len(set(matches)) == 1:
        meta_path = list(set(matches))[0]
    else:
        hinted = []
        for hint in suffix_hints:
            hinted.extend([m for m in set(matches) if f"_{hint}_meta.json" in os.path.basename(m)])
        if len(hinted) == 1:
            meta_path = hinted[0]
        else:
            return None, None, None
    prefix = meta_path[: -len("_meta.json")]
    req_path = prefix + "_requests.json"
    lora_path = prefix + "_lora_requests.json"
    return meta_path, req_path if os.path.exists(req_path) else None, lora_path if os.path.exists(lora_path) else None


def _summarize_requests(reqs: list[dict[str, Any]], *, prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not reqs:
        return out
    toks = [float(r.get("tokens", 0.0)) for r in reqs]
    arrivals = [float(r.get("arrival_offset_s", 0.0)) for r in reqs]
    shorts = [r for r in reqs if bool(r.get("is_short"))]
    longs = [r for r in reqs if not bool(r.get("is_short"))]
    short_toks = [float(r.get("tokens", 0.0)) for r in shorts]
    long_toks = [float(r.get("tokens", 0.0)) for r in longs]
    out[f"{prefix}_count"] = len(reqs)
    out[f"{prefix}_short_count"] = len(shorts)
    out[f"{prefix}_long_count"] = len(longs)
    out[f"{prefix}_short_frac"] = len(shorts) / len(reqs)
    out[f"{prefix}_token_mean"] = _mean(toks)
    out[f"{prefix}_token_p50"] = _quantile(toks, 0.50)
    out[f"{prefix}_token_p90"] = _quantile(toks, 0.90)
    out[f"{prefix}_token_cv"] = _cv(toks)
    out[f"{prefix}_token_mass"] = sum(toks)
    out[f"{prefix}_short_token_mass_frac"] = (sum(short_toks) / sum(toks)) if toks and sum(toks) > 0 else None
    out[f"{prefix}_long_to_short_mean_ratio"] = (_mean(long_toks) / _mean(short_toks)) if short_toks and long_toks and _mean(short_toks) not in (None, 0) else None
    out[f"{prefix}_arrival_span_s"] = max(arrivals) - min(arrivals) if arrivals else 0.0
    ordered = sorted(reqs, key=lambda r: float(r.get("arrival_offset_s", 0.0)))
    inter = [
        float(ordered[i]["arrival_offset_s"]) - float(ordered[i - 1]["arrival_offset_s"])
        for i in range(1, len(ordered))
    ]
    out[f"{prefix}_interarrival_mean_s"] = _mean(inter)
    out[f"{prefix}_interarrival_cv"] = _cv(inter)
    out[f"{prefix}_interarrival_p90_s"] = _quantile(inter, 0.90)

    prior_long_counts: list[int] = []
    prior_long_seen = 0
    shorts_with_prior_long = 0
    for r in ordered:
        if bool(r.get("is_short")):
            prior_long_counts.append(prior_long_seen)
            if prior_long_seen > 0:
                shorts_with_prior_long += 1
        else:
            prior_long_seen += 1
    out[f"{prefix}_beneficiary_short_frac"] = (shorts_with_prior_long / len(shorts)) if shorts else None
    out[f"{prefix}_prior_longs_per_short_mean"] = _mean([float(x) for x in prior_long_counts]) if prior_long_counts else None

    early = ordered[: max(1, len(ordered) // 4)]
    out[f"{prefix}_early_long_frac"] = (sum(1 for r in early if not bool(r.get("is_short"))) / len(early)) if early else None

    class_seq = ["S" if bool(r.get("is_short")) else "L" for r in ordered]
    class_switches = sum(1 for a, b in zip(class_seq, class_seq[1:]) if a != b)
    out[f"{prefix}_class_switch_rate"] = (class_switches / max(1, len(class_seq) - 1)) if class_seq else None
    first_long_idx = next((i for i, cls in enumerate(class_seq) if cls == "L"), None)
    if first_long_idx is None:
        out[f"{prefix}_first_long_position_frac"] = None
        out[f"{prefix}_shorts_after_first_long_frac"] = 0.0
    else:
        out[f"{prefix}_first_long_position_frac"] = first_long_idx / max(1, len(class_seq) - 1)
        shorts_after = sum(1 for cls in class_seq[first_long_idx + 1 :] if cls == "S")
        out[f"{prefix}_shorts_after_first_long_frac"] = (shorts_after / len(shorts)) if shorts else None

    if any("lora_tag" in r for r in reqs):
        tags = [str(r.get("lora_tag", "")) for r in ordered if r.get("lora_tag") is not None]
        counts = Counter(tags)
        total = sum(counts.values())
        out[f"{prefix}_tag_count"] = len(counts)
        out[f"{prefix}_minority_tag_frac"] = (min(counts.values()) / total) if counts and total > 0 else None
        transitions = 0
        for a, b in zip(tags, tags[1:]):
            if a != b:
                transitions += 1
        out[f"{prefix}_tag_switch_rate"] = (transitions / max(1, len(tags) - 1)) if tags else None
    return out


def _flatten_result_summary(result: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    cfg = result.get("config") or {}
    out["model_name"] = cfg.get("model_name")
    out["model_path"] = cfg.get("model_path")
    out["phase2_dispatch_mode"] = cfg.get("phase2_dispatch_mode")
    out["max_model_len"] = cfg.get("max_model_len")
    out["max_num_batched_tokens"] = cfg.get("max_num_batched_tokens")
    phase12 = result.get("phase12") or {}
    out["phase12_ttft_improve_mean"] = _safe_float((phase12.get("ttft_improve_ratio") or {}).get("mean"))
    out["phase12_wall_improve_mean"] = _safe_float((phase12.get("round_wall_improve_ratio") or {}).get("mean"))
    out["phase12_slowdown_improve_mean"] = _safe_float((phase12.get("slowdown_improve_ratio") or {}).get("mean"))
    out["phase12_apply_mean"] = _safe_float((phase12.get("phase2_apply_ratio") or {}).get("mean"))
    out["phase12_incremental_error_mean"] = _safe_float((phase12.get("incremental_error_rate") or {}).get("mean"))
    return out


def _discover_records(results_root: str, workload_root: str) -> list[Record]:
    records: dict[str, dict[str, Any]] = {}

    for csv_path in sorted(glob.glob(os.path.join(results_root, "*.csv"))):
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    row = {k: _coerce_scalar(v) for k, v in row.items()}
                    result_json = row.get("result_json") or ""
                    key = result_json or f"csv::{csv_path}::{row.get('density_label','row')}"
                    merged = dict(records.get(key, {}))
                    merged.update(row)
                    merged.setdefault("source_csv", csv_path)
                    records[key] = merged
        except Exception:
            continue

    for result_json in sorted(glob.glob(os.path.join(results_root, "**", "*_eval*.json"), recursive=True)):
        if "/dataset_workloads" in result_json:
            continue
        key = result_json
        merged = dict(records.get(key, {}))
        merged["result_json"] = result_json
        merged.setdefault("status", "ok")
        records[key] = merged

    out: list[Record] = []
    for key, row in sorted(records.items()):
        result_json = row.get("result_json") or ""
        if row.get("status") not in (None, "", "ok"):
            continue
        result = _load_json(result_json) if result_json else None
        if result:
            row.update(_flatten_result_summary(result))
        meta_path = row.get("workload_meta_json") or None
        req_path = None
        lora_path = None
        if meta_path:
            prefix = str(meta_path)[: -len("_meta.json")] if str(meta_path).endswith("_meta.json") else None
            if prefix:
                candidate_req = prefix + "_requests.json"
                candidate_lora = prefix + "_lora_requests.json"
                req_path = candidate_req if os.path.exists(candidate_req) else None
                lora_path = candidate_lora if os.path.exists(candidate_lora) else None
        if not meta_path or not os.path.exists(str(meta_path)):
            inferred_meta, inferred_req, inferred_lora = _infer_workload_paths(result_json, workload_root)
            meta_path = meta_path if meta_path and os.path.exists(str(meta_path)) else inferred_meta
            req_path = req_path or inferred_req
            lora_path = lora_path or inferred_lora
        meta = _load_json(str(meta_path)) if meta_path else None
        reqs = _load_json(req_path) if req_path else None
        lora_reqs = _load_json(lora_path) if lora_path else None
        row["workload_meta_json"] = meta_path
        row["requests_json"] = req_path
        row["lora_requests_json"] = lora_path
        if result_json:
            row.setdefault("policy_variant", _infer_policy_variant(result_json))
            row.setdefault("density_label", _infer_density_label(result_json, meta))
        if meta:
            for k, v in meta.items():
                row[f"meta_{k}"] = v
        if isinstance(reqs, list):
            row.update(_summarize_requests(reqs, prefix="phase1req"))
        if isinstance(lora_reqs, list):
            row.update(_summarize_requests(lora_reqs, prefix="phase2req"))
        out.append(Record(key=key, row=row))
    return out


def _build_correlations(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    numeric_cols = sorted({k for row in rows for k, v in row.items() if isinstance(v, (int, float)) and k not in RESULT_TARGETS})
    report: dict[str, list[dict[str, Any]]] = {}
    for target in RESULT_TARGETS[:3]:
        pairs = []
        for col in numeric_cols:
            xs: list[float] = []
            ys: list[float] = []
            for row in rows:
                x = row.get(col)
                y = row.get(target)
                if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                    xs.append(float(x))
                    ys.append(float(y))
            corr = _spearman(xs, ys)
            if corr is None:
                continue
            pairs.append({"feature": col, "spearman": corr, "n": len(xs)})
        pairs.sort(key=lambda item: abs(item["spearman"]), reverse=True)
        report[target] = pairs[:12]
    return report


def _candidate_family(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        row for row in rows
        if isinstance(row.get("phase12_ttft_improve_mean"), (int, float))
        and isinstance(row.get("phase12_wall_improve_mean"), (int, float))
    ]
    for row in eligible:
        row["combined_gain"] = float(row["phase12_ttft_improve_mean"]) * float(row["phase12_wall_improve_mean"])
    best = sorted(eligible, key=lambda r: r.get("combined_gain", 0.0), reverse=True)[:3]
    positive = [
        row for row in eligible
        if float(row.get("phase12_ttft_improve_mean", 0.0)) > 1.0 and float(row.get("phase12_wall_improve_mean", 0.0)) > 1.0
    ]
    summary: dict[str, Any] = {
        "best_rows": [
            {
                "result_json": row.get("result_json"),
                "density_label": row.get("density_label"),
                "policy_variant": row.get("policy_variant"),
                "phase12_ttft_improve_mean": row.get("phase12_ttft_improve_mean"),
                "phase12_wall_improve_mean": row.get("phase12_wall_improve_mean"),
                "phase12_slowdown_improve_mean": row.get("phase12_slowdown_improve_mean"),
                "phase12_apply_mean": row.get("phase12_apply_mean"),
                "phase1_arrival_rate": row.get("meta_phase1_arrival_rate"),
                "phase1_request_count": row.get("phase1req_count") or row.get("meta_phase1_request_count"),
                "phase2_request_count": row.get("phase2req_count") or row.get("meta_phase2_request_count"),
                "phase1req_beneficiary_short_frac": row.get("phase1req_beneficiary_short_frac"),
                "phase2req_minority_tag_frac": row.get("phase2req_minority_tag_frac"),
            }
            for row in best
        ],
        "positive_count": len(positive),
        "total_count": len(eligible),
    }
    if positive:
        rates = [float(r["meta_phase1_arrival_rate"]) for r in positive if isinstance(r.get("meta_phase1_arrival_rate"), (int, float, str)) and _safe_float(r.get("meta_phase1_arrival_rate")) is not None]
        ben = [float(r["phase1req_beneficiary_short_frac"]) for r in positive if isinstance(r.get("phase1req_beneficiary_short_frac"), (int, float))]
        short_frac = [float(r["phase1req_short_frac"]) for r in positive if isinstance(r.get("phase1req_short_frac"), (int, float))]
        shorts_after = [float(r["phase1req_shorts_after_first_long_frac"]) for r in positive if isinstance(r.get("phase1req_shorts_after_first_long_frac"), (int, float))]
        if ben and max(ben) <= 0.0 and shorts_after and max(shorts_after) <= 0.0:
            description = (
                "The best rows currently cluster around middle arrival density, but they are not truly beneficiary-rich: "
                "short requests almost never arrive after the first long request. This suggests the current Poisson workload "
                "randomizes inter-arrival gaps without sufficiently mixing short/long request order, which likely suppresses "
                "the advantage that Phase-I is supposed to create."
            )
        else:
            description = (
                "Most promising rows cluster where arrival density is in a middle band, beneficiary-rich short requests "
                "exist behind long requests, and Phase-II applies without saturating the system."
            )
        summary["family_hypothesis"] = {
            "arrival_rate_range": [min(rates), max(rates)] if rates else None,
            "beneficiary_short_frac_range": [min(ben), max(ben)] if ben else None,
            "shorts_after_first_long_frac_range": [min(shorts_after), max(shorts_after)] if shorts_after else None,
            "phase1_short_frac_range": [min(short_frac), max(short_frac)] if short_frac else None,
            "description": description,
        }
    return summary


def _write_markdown(path: str, rows: list[dict[str, Any]], corr: dict[str, list[dict[str, Any]]], fam: dict[str, Any]) -> None:
    lines = ["# Workload Analysis", "", f"Rows analyzed: {len(rows)}", ""]
    lines.append("## Best Candidate Rows")
    lines.append("")
    for idx, row in enumerate(fam.get("best_rows", []), start=1):
        lines.append(f"{idx}. `{os.path.basename(str(row.get('result_json','')) )}`")
        lines.append(f"   - density: `{row.get('density_label')}`")
        lines.append(f"   - policy: `{row.get('policy_variant')}`")
        lines.append(f"   - ttft: `{row.get('phase12_ttft_improve_mean')}`")
        lines.append(f"   - wall: `{row.get('phase12_wall_improve_mean')}`")
        lines.append(f"   - slowdown: `{row.get('phase12_slowdown_improve_mean')}`")
        lines.append(f"   - phase2_apply: `{row.get('phase12_apply_mean')}`")
    lines.append("")
    lines.append("## Top Feature Correlations")
    lines.append("")
    for target, pairs in corr.items():
        lines.append(f"### `{target}`")
        for item in pairs[:6]:
            lines.append(f"- `{item['feature']}`: spearman={item['spearman']:.4f} (n={item['n']})")
        lines.append("")
    fam_h = fam.get("family_hypothesis") or {}
    if fam_h:
        lines.append("## Candidate Workload Family")
        lines.append("")
        lines.append(f"- arrival rate range: `{fam_h.get('arrival_rate_range')}`")
        lines.append(f"- beneficiary short fraction range: `{fam_h.get('beneficiary_short_frac_range')}`")
        lines.append(f"- phase1 short fraction range: `{fam_h.get('phase1_short_frac_range')}`")
        lines.append(f"- note: {fam_h.get('description')}")
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze workload features against Wave-Slice gains.")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--workload-root", default=os.path.join("results", "dataset_workloads_poisson"))
    parser.add_argument("--out-dir", default=os.path.join("results", "workload_analysis"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    records = _discover_records(args.results_root, args.workload_root)
    rows = [r.row for r in records]
    feature_csv = os.path.join(args.out_dir, "workload_feature_table.csv")
    with open(feature_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = sorted({k for row in rows for k in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    corr = _build_correlations(rows)
    fam = _candidate_family(rows)
    summary = {
        "row_count": len(rows),
        "correlations": corr,
        "candidate_family": fam,
    }
    summary_json = os.path.join(args.out_dir, "workload_analysis_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    summary_md = os.path.join(args.out_dir, "workload_analysis_summary.md")
    _write_markdown(summary_md, rows, corr, fam)

    print(f"[Saved] {feature_csv}")
    print(f"[Saved] {summary_json}")
    print(f"[Saved] {summary_md}")
    print(f"[Rows] {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
