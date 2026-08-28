"""Compact contracts for the maintained Phase I/II runtime surface."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from experiments.openworkload_support import extract_summary_from_result_json
from waveslice.metrics import RUNTIME_METRICS_FILE_ENV, WaveSliceMetrics
from waveslice.policy import WaveSlicePolicy
from waveslice.scheduling.slicer import WaveBaseSlicer
from waveslice.vllm import integration, runtime
from waveslice.vllm.phase1_math import (
    phase1_authoritative_chunk,
    phase1_effective_ingress_min_chunk,
    phase1_runtime_adapt_policy,
    phase1_runtime_pressure_meta,
)
from waveslice.vllm.phase1_state import phase1_maybe_seed_ingress_virtual
from waveslice.vllm.phase2_beneficiaries import phase2_beneficiary_signal
from waveslice.vllm.phase2_cashout import (
    phase12_cashout_candidate_id,
    phase12_scheduler_cashout_grade,
    phase12_scheduler_cashout_value_signal,
)
from waveslice.vllm.phase2_gates import phase12_joint_phase2_ready
from waveslice.vllm.phase2_priority import phase12_priority_bubble_waiting_queue
from waveslice.vllm.request_hooks import (
    _build_v1_request_num_tokens_hook,
    _build_v1_request_num_tokens_with_spec_hook,
)
from waveslice.vllm.state import Phase1CohortStats, Phase12BeneficiarySignal, ScheduledRequestInfo
from waveslice.vllm.subprocess import read_cross_process_metrics


def test_joint_hard_gate_requires_recent_phase1_signal() -> None:
    signal = Phase12BeneficiarySignal(None, 0, 0, [], {})
    state = SimpleNamespace(
        phase1_virtual_token_caps={},
        phase12_recent_phase1_apply_ttl=0,
        phase12_recent_phase1_strength=0,
        phase12_recent_phase1_chunk=0,
        phase12_recent_phase2_cashout_cooldown=0,
    )
    policy = WaveSlicePolicy(
        enable_phase1_scheduler=True,
        enable_phase2_scheduler=True,
        phase12_phase2_gate_mode="hard",
    )
    args = dict(
        state=state,
        policy=policy,
        prefill_lens=[128, 1024],
        num_decode_tokens=0,
        lora_ranks=[4, 8],
        signal=signal,
    )
    assert phase12_joint_phase2_ready(**args) == (False, "joint_waiting_for_phase1")
    state.phase12_recent_phase1_apply_ttl = 1
    assert phase12_joint_phase2_ready(**args) == (True, "joint_recent_phase1_ttl")


def test_standalone_phase2_selects_low_service_beneficiary() -> None:
    infos = [
        ScheduledRequestInfo("short", 64, 64, 64, 64, 0, True, 4),
        ScheduledRequestInfo("long", 1024, 1024, 256, 1024, 0, False, 16),
    ]
    signal = phase2_beneficiary_signal(
        policy=WaveSlicePolicy(phase2_min_hetero_ratio=4), req_infos=infos
    )
    assert signal.long_anchor_id == "long"
    assert signal.beneficiary_selected_ids == ["short"]


def test_phase2_cashout_defers_anchor_before_other_short_requests() -> None:
    infos = [
        ScheduledRequestInfo("selected", 64, 64, 64, 64, None, True, 4),
        ScheduledRequestInfo("other_short", 96, 96, 96, 96, None, True, 4),
        ScheduledRequestInfo("long", 2048, 2048, 1536, 2048, None, False, 16),
    ]
    signal = Phase12BeneficiarySignal(
        "long",
        1,
        1,
        ["selected"],
        {"selected": 1},
    )
    assert phase12_cashout_candidate_id(req_infos=infos, beneficiary_signal=signal) == "long"


def test_phase2_cashout_value_allows_one_bounded_anchor_window() -> None:
    infos = [
        ScheduledRequestInfo("short", 64, 64, 64, 64, None, True, 4),
        ScheduledRequestInfo("long", 2048, 2048, 1536, 2048, None, False, 16),
    ]
    signal = Phase12BeneficiarySignal("long", 1, 0.6, ["short"], {"short": 0.6})
    value = phase12_scheduler_cashout_value_signal(
        req_infos=infos,
        beneficiary_signal=signal,
        candidate_id="long",
    )
    grade = phase12_scheduler_cashout_grade(
        policy=WaveSlicePolicy(),
        candidate_id="long",
        selected_quality=0.6,
        value_signal=value,
    )
    assert value["net_value"] > 0 and grade and grade["allowed"]


def test_phase2_priority_promotes_selected_beneficiary() -> None:
    groups = [
        SimpleNamespace(request_id=name, is_prefill=lambda: True)
        for name in ("other_a", "other_b", "other_c", "selected")
    ]
    signal = Phase12BeneficiarySignal(
        "long",
        0.25,
        1,
        ["selected"],
        {"selected": 1},
    )
    reordered = phase12_priority_bubble_waiting_queue(
        groups,
        beneficiary_signal=signal,
        beneficiary_ids={"selected"},
        request_id_getter=lambda group: group.request_id,
        queue_rebuilder=lambda _queue, items: items,
    )
    assert reordered[0].request_id == "selected"


def test_runtime_pressure_moves_between_aggressive_and_conservative_targets() -> None:
    policy = WaveSlicePolicy(
        phase1_runtime_adaptive_enabled=True,
        phase1_runtime_queue_high_watermark=8,
        phase1_runtime_waiting_short_high_watermark=4,
        phase1_runtime_wait_us_high_watermark=1000,
        phase1_runtime_long_high_watermark=4096,
        phase1_runtime_aggressive_long_fraction=0.25,
        phase1_runtime_conservative_long_fraction=0.6,
        phase1_runtime_aggressive_ingress_target_chunk=256,
        phase1_runtime_conservative_ingress_target_chunk=1536,
    )
    cohort = Phase1CohortStats(128, 2, 256, [128, 128], 4096, "long", 3)
    low = phase1_runtime_pressure_meta(
        policy=policy,
        cohort=cohort,
        queue_len=1,
        waiting_short_count=4,
        max_wait_us=1000,
        virtual_cap_hit_rate=0,
    )
    high = phase1_runtime_pressure_meta(
        policy=policy,
        cohort=cohort,
        queue_len=8,
        waiting_short_count=0,
        max_wait_us=0,
        virtual_cap_hit_rate=1,
    )
    low_policy, _ = phase1_runtime_adapt_policy(policy, low)
    high_policy, _ = phase1_runtime_adapt_policy(policy, high)
    assert low["effective_pressure"] < high["effective_pressure"]
    assert low_policy.phase1_ingress_target_chunk < high_policy.phase1_ingress_target_chunk
    assert low_policy.phase1_target_long_fraction < high_policy.phase1_target_long_fraction


def test_ingress_seed_and_exact_chunk_contract() -> None:
    policy = WaveSlicePolicy(
        phase1_ingress_direct_authoritative=True,
        phase1_ingress_exact_chunk=True,
        phase1_ingress_target_chunk=128,
        phase1_ingress_min_chunk=256,
    )
    state = SimpleNamespace(
        policy=policy,
        metrics=WaveSliceMetrics(),
        slicer=WaveBaseSlicer(),
        phase1_active_prompt_tokens={},
        phase1_ingress_virtuals={},
        phase1_virtual_token_caps={},
        phase1_explicit_plans={},
    )
    for request_id, tokens in (("short_a", 67), ("short_b", 156), ("long_b", 2268)):
        phase1_maybe_seed_ingress_virtual(state, request_id=request_id, input_tokens=tokens)
    assert state.phase1_virtual_token_caps["long_b"] > 0
    assert phase1_effective_ingress_min_chunk(policy, target=128) == 128
    assert (
        phase1_authoritative_chunk(policy, state.slicer, target=257, short_len=157, upper=780)
        == 128
    )


def test_v1_request_properties_cap_prefill_only() -> None:
    class Request:
        request_id = "long_b"
        num_prompt_tokens = 2268
        num_output_tokens = num_output_placeholders = 0
        num_computed_tokens = 256
        num_tokens = property(lambda self: 2268)
        num_tokens_with_spec = property(lambda self: 2268)

    state = SimpleNamespace(
        phase1_virtual_token_caps={"long_b": 256},
        metrics=WaveSliceMetrics(),
        original_v1_request_num_tokens=Request.num_tokens,
        original_v1_request_num_tokens_with_spec=Request.num_tokens_with_spec,
    )
    Request.num_tokens = _build_v1_request_num_tokens_hook(state)
    Request.num_tokens_with_spec = _build_v1_request_num_tokens_with_spec_hook(state)
    request = Request()
    assert (request.num_tokens, request.num_tokens_with_spec) == (512, 512)
    request.request_id = "other"
    assert (request.num_tokens, request.num_tokens_with_spec) == (2268, 2268)


def test_scheduler_hook_restores_runtime_policy_on_native_early_return(monkeypatch) -> None:
    base = WaveSlicePolicy(enable_sjf_reorder=False, phase1_ingress_direct_authoritative=False)
    adapted = replace(base, phase1_ingress_target_chunk=768)
    cohort = Phase1CohortStats(128, 1, 128, [128], 2048, "long", 2)
    state = SimpleNamespace(
        original_schedule=lambda _scheduler: "native",
        policy=base,
        metrics=WaveSliceMetrics(),
        brain=SimpleNamespace(),
        phase1_virtual_token_caps={},
        phase1_shadow_seq_lens={},
        phase1_public_skip_rewrite_requests=set(),
    )
    scheduler = SimpleNamespace(
        waiting=[object()],
        running=[],
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=1536, long_prefill_token_threshold=0
        ),
    )
    monkeypatch.setattr(runtime, "_observe_scheduler_requests", lambda *_args: None)
    monkeypatch.setattr(runtime, "_phase12_tick_recent_phase1", lambda *_args: None)
    monkeypatch.setattr(runtime, "_phase12_tick_recent_phase2", lambda *_args: None)
    monkeypatch.setattr(runtime, "_collect_live_snapshot", lambda *_args: ([], 0.0))
    monkeypatch.setattr(
        runtime, "_select_phase1_cohort", lambda *_args, **_kwargs: (cohort, [], object())
    )

    def adapt(*_args, **_kwargs):
        state.policy = adapted
        return base

    monkeypatch.setattr(runtime, "_adapt_phase1_policy", adapt)
    monkeypatch.setattr(
        runtime, "_choose_phase1_chunk", lambda *_args, **_kwargs: (cohort.long_len, None, None)
    )
    assert runtime._build_scheduler_hook(state)(scheduler) == "native"
    assert state.policy is base


def test_scheduler_hook_records_one_probe_for_applied_slice(monkeypatch) -> None:
    policy = WaveSlicePolicy(
        enable_sjf_reorder=False,
        phase1_ingress_direct_authoritative=False,
        enable_phase1_dynamic_threshold=False,
        enable_phase1_budget_guidance=False,
    )
    cohort = Phase1CohortStats(128, 1, 128, [128], 2048, "long", 2)
    state = SimpleNamespace(
        original_schedule=lambda _scheduler: "scheduled",
        policy=policy,
        metrics=WaveSliceMetrics(),
        brain=SimpleNamespace(),
        phase1_virtual_token_caps={},
        phase1_shadow_seq_lens={},
        phase1_public_skip_rewrite_requests=set(),
        phase12_recent_phase1_apply_ttl=0,
        phase12_recent_phase1_strength=0.0,
        phase12_recent_phase1_chunk=0,
        phase12_last_phase1_req_id=None,
    )
    scheduler = SimpleNamespace(
        waiting=[object()],
        running=[],
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=1536, long_prefill_token_threshold=0
        ),
    )
    monkeypatch.setattr(runtime, "_observe_scheduler_requests", lambda *_args: None)
    monkeypatch.setattr(runtime, "_phase12_tick_recent_phase1", lambda *_args: None)
    monkeypatch.setattr(runtime, "_phase12_tick_recent_phase2", lambda *_args: None)
    monkeypatch.setattr(runtime, "_collect_live_snapshot", lambda *_args: ([], 0.0))
    monkeypatch.setattr(
        runtime, "_select_phase1_cohort", lambda *_args, **_kwargs: (cohort, [], object())
    )
    monkeypatch.setattr(runtime, "_adapt_phase1_policy", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runtime, "_choose_phase1_chunk", lambda *_args, **_kwargs: (512, 1536, None)
    )
    monkeypatch.setattr(runtime, "_phase1_apply_sequence_len_shadow", lambda **_kwargs: False)
    monkeypatch.setattr(
        runtime,
        "_phase12_apply_scheduler_cashout_to_queues",
        lambda **kwargs: (kwargs["running"], kwargs["waiting"], [], [], False),
    )
    monkeypatch.setattr(
        runtime,
        "_phase1_rewrite_scheduler_outputs",
        lambda **kwargs: (kwargs["outputs"], False, 0, 0, 0, 0),
    )
    assert runtime._build_scheduler_hook(state)(scheduler) == "scheduled"
    assert state.metrics.summary()["scheduler"]["probe_total"] == 1


def test_scheduler_priority_cashout_removes_work_before_schedule(monkeypatch) -> None:
    beneficiary, deferred = SimpleNamespace(request_id="short"), SimpleNamespace(request_id="long")
    infos = [
        ScheduledRequestInfo("short", 64, 64, 64, 128, 0, True, 1),
        ScheduledRequestInfo("long", 1024, 1024, 1024, 2048, 0, False, 1),
    ]
    signal = Phase12BeneficiarySignal("long", 0.5, 1, ["short"], {"short": 1})
    policy = WaveSlicePolicy(enable_phase2_scheduler=True, phase2_enable_scheduler_cashout=True)
    state = SimpleNamespace(
        policy=policy,
        metrics=WaveSliceMetrics(),
        brain=SimpleNamespace(),
        phase12_recent_phase2_cashout_cooldown=0,
        phase12_recent_phase1_chunk=0,
        phase2_priority_active_ids=set(),
        phase2_priority_deferred_ids=set(),
        phase2_priority_lane_ttl=0,
    )
    remaining = {"short": 64, "long": 1024}
    monkeypatch.setattr(
        runtime, "_collect_live_snapshot", lambda *_args: ([(beneficiary, 64), (deferred, 1024)], 0)
    )
    monkeypatch.setattr(
        runtime, "_cashout_context", lambda *_args, **_kwargs: (infos, signal, None)
    )
    monkeypatch.setattr(runtime, "_safe_request_id", lambda group: group.request_id)
    monkeypatch.setattr(
        runtime, "_safe_prefill_uncomputed_tokens", lambda group: remaining[group.request_id]
    )
    monkeypatch.setattr(
        runtime, "_safe_remaining_tokens", lambda group: remaining[group.request_id]
    )
    monkeypatch.setattr(
        runtime,
        "_phase12_scheduler_cashout_grade",
        lambda **_kwargs: {"allowed": True, "strength": 1},
    )
    monkeypatch.setattr(runtime, "_phase12_scheduler_cashout_value_signal", lambda **_kwargs: 1)
    monkeypatch.setattr(
        runtime, "_phase12_scheduler_cashout_cooldown_for_grade", lambda **_kwargs: 1
    )
    monkeypatch.setattr(
        runtime, "_phase12_priority_bubble_waiting_queue", lambda queue, **_kwargs: queue
    )
    running, waiting, hidden_running, hidden_waiting, applied = (
        runtime._phase12_apply_scheduler_cashout_to_queues(
            state=state,
            running=[],
            waiting=[beneficiary, deferred],
        )
    )
    assert applied and running == [] and waiting == [beneficiary]
    assert hidden_running == [] and hidden_waiting == [deferred]
    assert state.phase2_priority_active_ids == {"short"}
    assert state.phase2_priority_deferred_ids == {"long"}


def test_scheduler_priority_cashout_can_pause_running_anchor(monkeypatch) -> None:
    beneficiary, anchor = SimpleNamespace(request_id="short"), SimpleNamespace(request_id="long")
    infos = [
        ScheduledRequestInfo("short", 64, 64, 64, 64, 0, True, 4),
        ScheduledRequestInfo("long", 1024, 1024, 1024, 2048, 0, False, 16),
    ]
    signal = Phase12BeneficiarySignal("long", 1, 1, ["short"], {"short": 1})
    policy = WaveSlicePolicy(enable_phase2_scheduler=True, phase2_enable_scheduler_cashout=True)
    state = SimpleNamespace(
        policy=policy,
        metrics=WaveSliceMetrics(),
        brain=SimpleNamespace(),
        phase12_recent_phase2_cashout_cooldown=0,
        phase12_recent_phase1_chunk=0,
        phase2_priority_active_ids=set(),
        phase2_priority_deferred_ids=set(),
        phase2_priority_lane_ttl=0,
    )
    remaining = {"short": 64, "long": 1024}
    monkeypatch.setattr(
        runtime, "_collect_live_snapshot", lambda *_args: ([(beneficiary, 64), (anchor, 1024)], 0)
    )
    monkeypatch.setattr(
        runtime, "_cashout_context", lambda *_args, **_kwargs: (infos, signal, None)
    )
    monkeypatch.setattr(runtime, "_safe_request_id", lambda group: group.request_id)
    monkeypatch.setattr(
        runtime, "_safe_prefill_uncomputed_tokens", lambda group: remaining[group.request_id]
    )
    monkeypatch.setattr(
        runtime, "_safe_remaining_tokens", lambda group: remaining[group.request_id]
    )
    monkeypatch.setattr(
        runtime,
        "_phase12_scheduler_cashout_grade",
        lambda **_kwargs: {"allowed": True, "strength": 1},
    )
    monkeypatch.setattr(runtime, "_phase12_scheduler_cashout_value_signal", lambda **_kwargs: 1)
    monkeypatch.setattr(
        runtime, "_phase12_scheduler_cashout_cooldown_for_grade", lambda **_kwargs: 1
    )
    monkeypatch.setattr(
        runtime, "_phase12_priority_bubble_waiting_queue", lambda queue, **_kwargs: queue
    )
    running, waiting, hidden_running, hidden_waiting, applied = (
        runtime._phase12_apply_scheduler_cashout_to_queues(
            state=state,
            running=[anchor],
            waiting=[beneficiary],
        )
    )
    assert applied and running == [] and waiting == [beneficiary]
    assert hidden_running == [anchor] and hidden_waiting == []


def test_native_cashout_restores_only_deferred_request(monkeypatch) -> None:
    beneficiary = SimpleNamespace(request_id="short")
    anchor = SimpleNamespace(request_id="long")
    scheduler = SimpleNamespace(running=[], waiting=[beneficiary, anchor])
    state = SimpleNamespace(
        policy=WaveSlicePolicy(enable_phase2_scheduler=True, phase2_enable_scheduler_cashout=True),
    )
    monkeypatch.setattr(
        runtime,
        "_phase12_apply_scheduler_cashout_to_queues",
        lambda **_kwargs: ([], [beneficiary], [], [anchor], True),
    )
    monkeypatch.setattr(runtime, "_post_schedule_cashout", lambda _state, outputs: outputs)

    def native_schedule(owner):
        owner.running.append(owner.waiting.pop(0))
        return "scheduled"

    assert runtime._run_native_schedule(state, scheduler, native_schedule, (), {}) == "scheduled"
    assert scheduler.running == [beneficiary]
    assert scheduler.waiting == [anchor]


def test_phase2_only_scheduler_ticks_priority_state(monkeypatch) -> None:
    ticks = []
    state = SimpleNamespace(
        original_schedule=lambda _scheduler: "scheduled",
        policy=WaveSlicePolicy(enable_phase1_scheduler=False, enable_phase2_scheduler=True),
        metrics=WaveSliceMetrics(),
    )
    scheduler = SimpleNamespace(waiting=[], running=[])
    monkeypatch.setattr(runtime, "_observe_scheduler_requests", lambda *_args: None)
    monkeypatch.setattr(runtime, "_phase12_tick_recent_phase1", lambda *_args: None)
    monkeypatch.setattr(
        runtime, "_phase12_tick_recent_phase2", lambda *_args: ticks.append("phase2")
    )
    monkeypatch.setattr(runtime, "_run_native_schedule", lambda *_args: "scheduled")
    assert runtime._build_scheduler_hook(state)(scheduler) == "scheduled"
    assert ticks == ["phase2"]


def test_public_inject_and_uninject_contract() -> None:
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
            integration.inject_wave_slice("fake-model", policy=policy, force=True)
            assert integration.is_wave_slice_injected()
            assert "scheduler" in integration.get_wave_slice_metrics()
        finally:
            integration.uninject_wave_slice()
    assert not integration.is_wave_slice_injected()


def test_nested_sessions_keep_outer_injection_and_reject_replacement() -> None:
    class Scheduler:
        def schedule(self):
            return None

    policy = WaveSlicePolicy(enable_phase2_scheduler=False, enable_metrics_hook=False)
    patches = (
        mock.patch.object(
            integration, "load_scheduler_target", return_value=(Scheduler, "schedule")
        ),
        mock.patch.object(
            integration, "_build_scheduler_hook", side_effect=lambda state: state.original_schedule
        ),
        mock.patch.object(integration, "_install_scheduler_hooks"),
    )
    with patches[0], patches[1], patches[2]:
        with integration.wave_slice_session("fake-model", policy=policy):
            outer = integration._runtime_state
            with integration.wave_slice_session("fake-model", policy=policy):
                assert integration._runtime_state is outer
            assert integration._runtime_state is outer
            try:
                with integration.wave_slice_session("other-model", policy=policy):
                    pass
            except RuntimeError:
                pass
            else:
                raise AssertionError("an incompatible nested session must fail")
        assert not integration.is_wave_slice_injected()


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
        integration.wave_slice_session("fake-model", policy=policy),
    ):
        metrics_path.write_text(
            json.dumps({"pid": 0, "payload": {"values": {"sched_total": 2}}}) + "\n",
            encoding="utf-8",
        )
        assert integration.get_wave_slice_metrics(reset=True)["scheduler"]["attempts"] == 2
        assert integration.get_wave_slice_metrics()["scheduler"]["attempts"] == 0


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
