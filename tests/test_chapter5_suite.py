from __future__ import annotations

from experiments.run_environment_preflight import _derive_densities, _derive_workload_overrides
from experiments.run_chapter5_suite import _resolve_stages


def test_resolve_stages_accepts_all_and_explicit_subset() -> None:
    assert _resolve_stages("all") == ["preflight", "main", "baseline", "figures", "partial-figures"]
    assert _resolve_stages("main,figures") == ["main", "figures"]


def test_preflight_workload_scaling_keeps_full_24gb_shape() -> None:
    config = {
        "eval": {"max_num_batched_tokens": 1536, "max_new_tokens": 64, "repeats": 2, "warmup_iters": 1},
        "workload": {"sample_count": 256},
    }
    densities = [
        {
            "name": "peak",
            "phase1_arrival_rate": 10.0,
            "phase2_arrival_rate": 10.0,
            "phase1_short_count": 24,
            "phase1_long_count": 8,
            "phase2_short_count": 24,
            "phase2_long_count": 12,
        }
    ]
    overrides, meta = _derive_workload_overrides(
        config=config,
        runtime_cfg={"max_num_batched_tokens": 1536},
        memory_gb=24.0,
    )
    resolved, density_meta = _derive_densities(
        densities,
        {"max_num_batched_tokens": 1536},
        config["eval"],
        meta,
    )

    assert overrides["eval"]["max_new_tokens"] == 64
    assert overrides["eval"]["repeats"] == 2
    assert overrides["workload"]["sample_count"] == 256
    assert [item["name"] for item in resolved] == ["peak"]
    assert resolved[0]["phase2_long_count"] == 12
    assert density_meta["dropped_densities"] == []


def test_preflight_workload_scaling_reduces_low_memory_shape() -> None:
    config = {
        "eval": {"max_num_batched_tokens": 1536, "max_new_tokens": 64, "repeats": 2, "warmup_iters": 1},
        "workload": {"sample_count": 256},
    }
    densities = [
        {
            "name": "low",
            "phase1_arrival_rate": 4.0,
            "phase2_arrival_rate": 4.0,
            "phase1_short_count": 24,
            "phase1_long_count": 8,
            "phase2_short_count": 24,
            "phase2_long_count": 12,
        },
        {
            "name": "peak",
            "phase1_arrival_rate": 10.0,
            "phase2_arrival_rate": 10.0,
            "phase1_short_count": 24,
            "phase1_long_count": 8,
            "phase2_short_count": 24,
            "phase2_long_count": 12,
        },
    ]
    overrides, meta = _derive_workload_overrides(
        config=config,
        runtime_cfg={"max_num_batched_tokens": 768},
        memory_gb=10.0,
    )
    resolved, density_meta = _derive_densities(
        densities,
        {"max_num_batched_tokens": 768},
        config["eval"],
        meta,
    )

    assert overrides["eval"]["max_new_tokens"] == 32
    assert overrides["eval"]["repeats"] == 1
    assert overrides["workload"]["sample_count"] == 90
    assert [item["name"] for item in resolved] == ["low"]
    assert resolved[0]["phase1_short_count"] == 8
    assert resolved[0]["phase2_long_count"] == 4
    assert density_meta["dropped_densities"] == ["peak"]
