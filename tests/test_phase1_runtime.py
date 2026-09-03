"""Contracts for Phase I planning and scheduler integration."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from waveslice.metrics import WaveSliceMetrics
from waveslice.policy import WaveSlicePolicy
from waveslice.scheduling.slicer import WaveBaseSlicer
from waveslice.vllm import phase1_planning, phase1_runtime, runtime
from waveslice.vllm.phase1_math import (
    phase1_authoritative_chunk,
    phase1_effective_ingress_min_chunk,
    phase1_runtime_adapt_policy,
    phase1_runtime_pressure_meta,
)
from waveslice.vllm.phase1_state import phase1_maybe_seed_ingress_virtual
from waveslice.vllm.request_hooks import (
    _build_v1_request_num_tokens_hook,
    _build_v1_request_num_tokens_with_spec_hook,
)
from waveslice.vllm.state import Phase1CohortStats


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
    monkeypatch.setattr(phase1_planning, "_collect_live_snapshot", lambda *_args: ([], 0.0))
    monkeypatch.setattr(
        phase1_planning,
        "_select_phase1_cohort",
        lambda *_args, **_kwargs: (cohort, [], object()),
    )

    def adapt(*_args, **_kwargs):
        state.policy = adapted
        return base

    monkeypatch.setattr(phase1_planning, "_adapt_phase1_policy", adapt)
    monkeypatch.setattr(
        phase1_planning,
        "_choose_phase1_chunk",
        lambda *_args, **_kwargs: (cohort.long_len, None, None),
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
    monkeypatch.setattr(phase1_planning, "_collect_live_snapshot", lambda *_args: ([], 0.0))
    monkeypatch.setattr(
        phase1_planning,
        "_select_phase1_cohort",
        lambda *_args, **_kwargs: (cohort, [], object()),
    )
    monkeypatch.setattr(
        phase1_planning, "_adapt_phase1_policy", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        phase1_planning,
        "_choose_phase1_chunk",
        lambda *_args, **_kwargs: (512, 1536, None),
    )
    monkeypatch.setattr(
        phase1_runtime, "_phase1_apply_sequence_len_shadow", lambda **_kwargs: False
    )
    monkeypatch.setattr(
        phase1_runtime,
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
