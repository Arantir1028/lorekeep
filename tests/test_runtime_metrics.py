"""Contracts for runtime activation, metrics, and result extraction."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

from experiments.openworkload_support import extract_summary_from_result_json
from waveslice.metrics import RUNTIME_METRICS_FILE_ENV, WaveSliceMetrics
from waveslice.policy import WaveSlicePolicy
from waveslice.vllm import integration
from waveslice.vllm.subprocess import read_cross_process_metrics


def test_runtime_activation_and_deactivation_contract() -> None:
    class Scheduler:
        def schedule(self):
            return None

    policy = WaveSlicePolicy(enable_phase2_scheduler=False, enable_metrics_hook=False)
    with (
        mock.patch.object(
            integration, "load_scheduler_target", return_value=(Scheduler, "schedule")
        ),
        mock.patch.object(
            integration, "_build_scheduler_hook", side_effect=lambda state: state.original_schedule
        ),
        mock.patch.object(integration, "_install_scheduler_hooks"),
    ):
        try:
            integration.activate_wave_slice("Mistral-7B-v0.1", policy=policy)
            assert integration.is_wave_slice_active()
            assert "scheduler" in integration.get_wave_slice_metrics()
        finally:
            integration.deactivate_wave_slice()
    assert not integration.is_wave_slice_active()


def test_get_metrics_reset_clears_cross_process_events(tmp_path: Path, monkeypatch) -> None:
    class Scheduler:
        def schedule(self):
            return None

    metrics_path = tmp_path / "events.jsonl"
    monkeypatch.setenv(RUNTIME_METRICS_FILE_ENV, str(metrics_path))
    policy = WaveSlicePolicy(enable_phase2_scheduler=False, enable_metrics_hook=False)
    with (
        mock.patch.object(
            integration, "load_scheduler_target", return_value=(Scheduler, "schedule")
        ),
        mock.patch.object(
            integration, "_build_scheduler_hook", side_effect=lambda state: state.original_schedule
        ),
        mock.patch.object(integration, "_install_scheduler_hooks"),
    ):
        integration.activate_wave_slice("Mistral-7B-v0.1", policy=policy)
        try:
            metrics_path.write_text(
                json.dumps({"pid": 0, "payload": {"values": {"sched_total": 2}}}) + "\n",
                encoding="utf-8",
            )
            assert integration.get_wave_slice_metrics(reset=True)["scheduler"]["attempts"] == 2
            assert integration.get_wave_slice_metrics()["scheduler"]["attempts"] == 0
        finally:
            integration.deactivate_wave_slice()


def test_metrics_and_cross_process_merge_preserve_schema(tmp_path: Path, monkeypatch) -> None:
    metrics_path = tmp_path / "events.jsonl"
    monkeypatch.setenv(RUNTIME_METRICS_FILE_ENV, str(metrics_path))
    metrics = WaveSliceMetrics()
    metrics.record_scheduler_decision(True)
    metrics.record_phase1_choice(chosen_chunk=384, baseline_chunk=1536, explicit_plan=True)
    metrics.record_phase1_probe(
        short_len=96,
        long_len=1536,
        baseline_chunk=1536,
        best_chunk=384,
        queue_len=3,
        wait_us=25,
        slice_eligible=True,
    )
    metrics.record_phase2_decision(True, "scheduler_cashout")
    metrics.record_priority_lane_activation(active_ids=["short"], deferred_ids=["long"], lane_ttl=2)
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"kind": "aggregate", "pid": 0, "payload": {"values": {"sched_total": 1}}})
            + "\n"
        )
    merged = metrics.summary(read_cross_process_metrics())
    scheduler, phase2 = merged["scheduler"], merged["phase2"]
    assert (
        scheduler["attempts"],
        scheduler["applied"],
        scheduler["chosen_chunk_avg"],
        scheduler["probe_best_avg"],
    ) == (2, 1, 384, 384)
    assert scheduler["explicit_plan_ratio"] == 0.5
    assert phase2["reasons"] == {"scheduler_cashout": 1}
    assert phase2["priority_lane"]["activations"] == 1


def test_openworkload_summary_keeps_all_three_phases(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text(
        json.dumps(
            {
                "phase1": {"ttft_improve_ratio": {"mean": 1.1}},
                "phase2": {
                    "ttft_improve_ratio": {"mean": 1.2},
                    "phase2_apply_ratio": {"mean": 0.3},
                },
                "phase12": {
                    "ttft_improve_ratio": {"mean": 2.5},
                    "phase2_priority_lane_activations": {"mean": 3},
                },
            }
        ),
        encoding="utf-8",
    )
    summary = extract_summary_from_result_json(path)
    assert (
        summary["phase1_ttft_improve_mean"],
        summary["phase2_ttft_improve_mean"],
        summary["phase12_ttft_improve_mean"],
    ) == (1.1, 1.2, 2.5)
    assert summary["phase12_priority_lane_activations_mean"] == 3
