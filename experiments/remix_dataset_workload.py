"""Remix a frozen dataset workload into a new size and arrival schedule."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from experiments.build_dataset_workload import _arrival_order, _assign_poisson_arrivals
from experiments.result_io import read_json, write_json


def _typed_pool(items: list[dict], *, is_short: bool) -> list[dict]:
    return sorted(
        (dict(item) for item in items if bool(item.get("is_short")) is is_short),
        key=lambda item: (int(item.get("tokens") or 0), str(item.get("req_id") or "")),
    )


def _cycle_sample(pool: list[dict], count: int, *, seed: int) -> list[dict]:
    if count <= 0:
        return []
    if not pool:
        raise ValueError("cannot sample from an empty request pool")
    rng, order, output = random.Random(seed), list(pool), []
    while len(output) < count:
        rng.shuffle(order)
        output.extend(dict(item) for item in order[: count - len(output)])
    return output


def _copy_request(item: dict, request_id: str, short: bool, lora: str | None = None) -> dict:
    output = {"req_id": request_id, "prompt": item["prompt"], "is_short": short}
    if lora:
        output["lora_tag"] = lora
    output.update(source=item.get("source", ""), tokens=item.get("tokens"))
    return output


def _phase1_from_pool(
    requests: list[dict], *, short_count: int, long_count: int, seed: int
) -> list[dict]:
    shorts, longs = _typed_pool(requests, is_short=True), _typed_pool(requests, is_short=False)
    if len(shorts) < 2 or not longs:
        raise ValueError(
            "source phase1 workload must include at least two short and one long requests"
        )
    output = [_copy_request(shorts[0], "short_a", True), _copy_request(shorts[1], "short_b", True)]
    output += [
        _copy_request(item, f"short_{index:02d}", True)
        for index, item in enumerate(_cycle_sample(shorts, max(0, short_count), seed=seed + 11))
    ]
    output += [_copy_request(longs[-1], "long_b", False)]
    output += [
        _copy_request(item, f"long_{index:02d}", False)
        for index, item in enumerate(_cycle_sample(longs, max(0, long_count), seed=seed + 23))
    ]
    return output


def _phase2_from_pool(
    requests: list[dict], *, short_count: int, long_count: int, seed: int
) -> list[dict]:
    shorts, longs = _typed_pool(requests, is_short=True), _typed_pool(requests, is_short=False)
    if len(shorts) < 2 or len(longs) < 2:
        raise ValueError(
            "source phase2 workload must include at least two short and two long requests"
        )
    output = [
        _copy_request(shorts[0], "short_a", True, "A"),
        _copy_request(shorts[1], "mid_b", True, "B"),
        _copy_request(longs[0], "long_a", False, "A"),
        _copy_request(longs[-1], "long_b", False, "B"),
    ]
    output += [
        _copy_request(item, f"short_extra_{index:02d}", True, "A" if index % 2 == 0 else "B")
        for index, item in enumerate(_cycle_sample(shorts, max(0, short_count), seed=seed + 101))
    ]
    output += [
        _copy_request(item, f"long_extra_{index:02d}", False, "A" if index % 2 == 0 else "B")
        for index, item in enumerate(_cycle_sample(longs, max(0, long_count), seed=seed + 211))
    ]
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-prefix", required=True)
    parser.add_argument("--out-prefix", required=True)
    for phase in (1, 2):
        parser.add_argument(f"--phase{phase}-short-count", type=int)
        parser.add_argument(f"--phase{phase}-long-count", type=int)
        parser.add_argument(f"--phase{phase}-arrival-rate", type=float, default=6.0)
        parser.add_argument(
            f"--phase{phase}-arrival-layout",
            choices=["grouped", "mixed", "beneficiary_rich"],
            default="beneficiary_rich",
        )
        parser.add_argument(
            f"--phase{phase}-early-short-frac", type=float, default=0.25 if phase == 1 else 0.20
        )
        parser.add_argument(
            f"--phase{phase}-post-long-short-bias", type=float, default=0.70 if phase == 1 else 0.60
        )
    parser.add_argument("--arrival-mode", choices=["burst", "poisson"], default="poisson")
    parser.add_argument("--arrival-seed", type=int, default=7)
    return parser


def main() -> int:
    args = _parser().parse_args()
    source, output = Path(args.src_prefix), Path(args.out_prefix)
    requests = read_json(Path(str(source) + "_requests.json"))
    lora_requests = read_json(Path(str(source) + "_lora_requests.json"))
    meta_path = Path(str(source) + "_meta.json")
    metadata = read_json(meta_path) if meta_path.exists() else {}
    workloads = [requests, lora_requests]
    for phase, key, minimum in ((1, "phase1", (2, 1)), (2, "phase2", (2, 2))):
        short_arg, long_arg = (
            getattr(args, f"{key}_short_count"),
            getattr(args, f"{key}_long_count"),
        )
        if short_arg is not None or long_arg is not None:
            short_count = max(
                minimum[0],
                int(
                    short_arg
                    if short_arg is not None
                    else metadata.get(f"{key}_config_short_count", 24)
                ),
            )
            long_count = max(
                minimum[1],
                int(
                    long_arg
                    if long_arg is not None
                    else metadata.get(f"{key}_config_long_count", 8 if phase == 1 else 12)
                ),
            )
            workloads[phase - 1] = (_phase1_from_pool if phase == 1 else _phase2_from_pool)(
                workloads[phase - 1],
                short_count=short_count,
                long_count=long_count,
                seed=args.arrival_seed,
            )
        workloads[phase - 1] = _arrival_order(
            workloads[phase - 1],
            seed=args.arrival_seed + (17 if phase == 1 else 1017),
            layout=getattr(args, f"{key}_arrival_layout"),
            early_short_frac=getattr(args, f"{key}_early_short_frac"),
            post_long_short_bias=getattr(args, f"{key}_post_long_short_bias"),
        )
        if args.arrival_mode == "poisson":
            workloads[phase - 1] = _assign_poisson_arrivals(
                workloads[phase - 1],
                rate_per_s=getattr(args, f"{key}_arrival_rate"),
                seed=args.arrival_seed + (0 if phase == 1 else 1009),
            )
        else:
            workloads[phase - 1] = [
                dict(item, arrival_offset_s=0.0) for item in workloads[phase - 1]
            ]
    requests, lora_requests = workloads
    actual = {}
    for phase, items in ((1, requests), (2, lora_requests)):
        short = sum(bool(item.get("is_short")) for item in items)
        actual[phase] = (short, len(items) - short)
    metadata.update(
        {
            "arrival_mode": args.arrival_mode,
            "phase1_arrival_layout": args.phase1_arrival_layout,
            "phase2_arrival_layout": args.phase2_arrival_layout,
            "phase1_arrival_rate": args.phase1_arrival_rate,
            "phase2_arrival_rate": args.phase2_arrival_rate,
        }
    )
    for phase, items in ((1, requests), (2, lora_requests)):
        key = f"phase{phase}"
        metadata.update(
            {
                f"{key}_config_short_count": getattr(args, f"{key}_short_count")
                if getattr(args, f"{key}_short_count") is not None
                else metadata.get(f"{key}_config_short_count"),
                f"{key}_config_long_count": getattr(args, f"{key}_long_count")
                if getattr(args, f"{key}_long_count") is not None
                else metadata.get(f"{key}_config_long_count"),
                f"{key}_short_count": actual[phase][0],
                f"{key}_long_count": actual[phase][1],
                f"{key}_long_fraction": actual[phase][1] / len(items) if items else 0.0,
                f"{key}_request_count": len(items),
                f"{key}_last_arrival_s": max(
                    (float(item.get("arrival_offset_s", 0.0)) for item in items), default=0.0
                ),
            }
        )
    metadata["source_workload_prefix"] = args.src_prefix
    paths = [
        Path(str(output) + suffix)
        for suffix in ("_requests.json", "_lora_requests.json", "_meta.json")
    ]
    for path, payload in zip(paths, (requests, lora_requests, metadata), strict=False):
        write_json(path, payload)
        print(f"[Saved] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
