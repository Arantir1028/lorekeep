from __future__ import annotations

import csv
import json
import math
from collections.abc import Callable
from pathlib import Path
from statistics import mean
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def ratio(numerator: Any, denominator: Any) -> float | None:
    num, den = safe_float(numerator), safe_float(denominator)
    return None if num is None or den is None or den <= 0 else num / den


def mean_values(values: list[Any]) -> float | None:
    numbers = [number for value in values if (number := safe_float(value)) is not None]
    return float(mean(numbers)) if numbers else None


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered, percent = sorted(map(float, values)), max(0, min(100, percent))
    position = (len(ordered) - 1) * (percent / 100.0)
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def timing_summary(
    request_timings: Any,
    *,
    include_fraction: bool = False,
    include_wall: bool = False,
    float_counts: bool = False,
) -> dict[str, Any]:
    rows = (
        [item for item in request_timings.values() if isinstance(item, dict)]
        if isinstance(request_timings, dict)
        else []
    )
    scopes = {
        "all": rows,
        "short": [item for item in rows if item.get("is_short")],
        "long": [item for item in rows if not item.get("is_short")],
    }
    count = float if float_counts else int
    output: dict[str, Any] = {
        "request_count": count(len(rows)),
        "short_request_count": count(len(scopes["short"])),
        "long_request_count": count(len(scopes["long"])),
    }
    if include_fraction:
        output["measured_long_fraction"] = len(scopes["long"]) / len(rows) if rows else None
    if include_wall:
        arrivals = [1000 * float(item.get("arrival_offset_s") or 0) for item in rows]
        finishes = [
            arrival + finish
            for item, arrival in zip(rows, arrivals, strict=False)
            if (finish := safe_float(item.get("finish_latency_ms"))) is not None
        ]
        wall = max(finishes) - min(arrivals) if finishes and arrivals else None
        output.update(
            round_wall_ms=wall,
            throughput_rps=len(rows) * 1000 / wall if wall and wall > 0 else None,
        )
    for scope, items in scopes.items():
        for metric, key in (("ttft", "first_latency_ms"), ("completion", "finish_latency_ms")):
            values = [value for item in items if (value := safe_float(item.get(key))) is not None]
            for label, percent in (("p50", 50), ("p90", 90), ("p99", 99)):
                output[f"{scope}_{metric}_{label}_ms"] = percentile(values, percent)
    return output


def density_info(density: str) -> tuple[str, int]:
    if "_l" not in density:
        return "", 0
    level, percent = density.rsplit("_l", 1)
    try:
        return level, int(percent)
    except ValueError:
        return level, 0


def comma_list(value: str) -> list[str]:
    return list(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))


def parse_map(value: str, cast: Callable[[str], Any], *, label: str) -> dict[str, Any]:
    output = {}
    for item in comma_list(value):
        if "=" not in item:
            raise ValueError(f"expected model=value in {label} item: {item}")
        key, raw = (part.strip() for part in item.split("=", 1))
        if not key:
            raise ValueError(f"empty model key in {label} item: {item}")
        output[key] = cast(raw)
    return output
