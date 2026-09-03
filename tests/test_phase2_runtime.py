"""Contracts for Phase II gating, priority, and scheduler cashout."""

from __future__ import annotations

from types import SimpleNamespace

from waveslice.metrics import WaveSliceMetrics
from waveslice.policy import WaveSlicePolicy
from waveslice.vllm import phase2_runtime, runtime
from waveslice.vllm.phase2_beneficiaries import phase2_beneficiary_signal
from waveslice.vllm.phase2_cashout import (
    phase12_cashout_candidate_id,
    phase12_scheduler_cashout_grade,
    phase12_scheduler_cashout_value_signal,
)
from waveslice.vllm.phase2_gates import phase12_joint_phase2_ready
from waveslice.vllm.phase2_priority import phase12_priority_bubble_waiting_queue
from waveslice.vllm.state import Phase12BeneficiarySignal, ScheduledRequestInfo


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
        phase2_runtime,
        "_collect_live_snapshot",
        lambda *_args: ([(beneficiary, 64), (deferred, 1024)], 0),
    )
    monkeypatch.setattr(
        phase2_runtime, "_cashout_context", lambda *_args, **_kwargs: (infos, signal, None)
    )
    monkeypatch.setattr(phase2_runtime, "_safe_request_id", lambda group: group.request_id)
    monkeypatch.setattr(
        phase2_runtime,
        "_safe_prefill_uncomputed_tokens",
        lambda group: remaining[group.request_id],
    )
    monkeypatch.setattr(
        phase2_runtime, "_safe_remaining_tokens", lambda group: remaining[group.request_id]
    )
    monkeypatch.setattr(
        phase2_runtime,
        "_phase12_scheduler_cashout_grade",
        lambda **_kwargs: {"allowed": True, "strength": 1},
    )
    monkeypatch.setattr(
        phase2_runtime, "_phase12_scheduler_cashout_value_signal", lambda **_kwargs: 1
    )
    monkeypatch.setattr(
        phase2_runtime,
        "_phase12_scheduler_cashout_cooldown_for_grade",
        lambda **_kwargs: 1,
    )
    monkeypatch.setattr(
        phase2_runtime,
        "_phase12_priority_bubble_waiting_queue",
        lambda queue, **_kwargs: queue,
    )
    running, waiting, hidden_running, hidden_waiting, applied = (
        phase2_runtime._phase12_apply_scheduler_cashout_to_queues(
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
        phase2_runtime,
        "_collect_live_snapshot",
        lambda *_args: ([(beneficiary, 64), (anchor, 1024)], 0),
    )
    monkeypatch.setattr(
        phase2_runtime, "_cashout_context", lambda *_args, **_kwargs: (infos, signal, None)
    )
    monkeypatch.setattr(phase2_runtime, "_safe_request_id", lambda group: group.request_id)
    monkeypatch.setattr(
        phase2_runtime,
        "_safe_prefill_uncomputed_tokens",
        lambda group: remaining[group.request_id],
    )
    monkeypatch.setattr(
        phase2_runtime, "_safe_remaining_tokens", lambda group: remaining[group.request_id]
    )
    monkeypatch.setattr(
        phase2_runtime,
        "_phase12_scheduler_cashout_grade",
        lambda **_kwargs: {"allowed": True, "strength": 1},
    )
    monkeypatch.setattr(
        phase2_runtime, "_phase12_scheduler_cashout_value_signal", lambda **_kwargs: 1
    )
    monkeypatch.setattr(
        phase2_runtime,
        "_phase12_scheduler_cashout_cooldown_for_grade",
        lambda **_kwargs: 1,
    )
    monkeypatch.setattr(
        phase2_runtime,
        "_phase12_priority_bubble_waiting_queue",
        lambda queue, **_kwargs: queue,
    )
    running, waiting, hidden_running, hidden_waiting, applied = (
        phase2_runtime._phase12_apply_scheduler_cashout_to_queues(
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
