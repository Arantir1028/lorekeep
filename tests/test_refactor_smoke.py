from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = ROOT / "tests"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

import evaluate_waveslice_claims as claims
import eval_config
from experiments.run_frozen_eval_config import build_eval_invocation
from engine.base_slicer import WaveBaseSlicer
from engine.hijack.autoinject import merge_cross_process_metrics
from engine.hijack.engine_hooks import (
    _extract_processed_prompt_tokens,
    build_add_processed_request_hook,
    build_add_request_hook,
    build_v1_engine_core_add_request_hook,
    build_v1_process_inputs_hook,
)
from engine.hijack.metrics import WaveSliceMetrics
from engine.hijack.phase1_math import (
    phase1_authoritative_chunk,
    phase1_effective_ingress_min_chunk,
)
from engine.hijack.phase1_selection import phase1_find_ingress_virtual_candidate
from engine.hijack.phase1_stateful import phase1_maybe_seed_ingress_virtual
from engine.hijack.policy import WaveSlicePolicy
from engine.hijack.types import (
    _Phase12BeneficiarySignal,
    _Phase1IngressVirtualSlice,
    _ScheduledReqInfo,
)
from engine.hijack.v1_merge import freeze_v1_runner_output, merge_v1_runner_outputs
from engine.hijack.v1_split import (
    v1_execution_escape_req_ids,
    v1_filter_cached_reqs,
    v1_partition_req_ids,
    v1_partition_req_ids_diagnostic,
)
from engine import vllm_hijacker as hijacker


def _fake_reqs() -> list[claims.Req]:
    return [
        claims.Req("short_a", "hello", True, arrival_offset_s=0.0),
        claims.Req("long_b", "world", False, arrival_offset_s=0.1),
    ]


def _make_phase1_row(ttft: float, wall: float, apply: float) -> dict:
    return {
        "ttft_short_p99_ms": ttft,
        "round_wall_ms": wall,
        "texts": {"short_a": "ok", "long_b": "ok"},
        "hook_report": {
            "scheduler": {
                "apply_ratio": apply,
                "applied": 1,
                "attempts": 2,
                "baseline_chunk_avg": 1536.0,
                "chosen_chunk_avg": 768.0,
                "chosen_vs_baseline_ratio_avg": 0.5,
                "explicit_plan_ratio": apply,
                "rewrite_apply_ratio": 0.0,
                "rewrite_old_chunk_avg": None,
                "rewrite_new_chunk_avg": None,
                "rewrite_token_delta_avg": None,
                "virtual_cap_apply_ratio": 0.5,
                "virtual_cap_old_avg": 600.0,
                "virtual_cap_new_avg": 300.0,
                "virtual_cap_target_set": 1,
                "virtual_cap_helper_calls": 1,
                "virtual_cap_prefill_calls": 1,
                "virtual_cap_target_hits": 1,
                "probe_total": 1,
                "probe_slice_eligible_ratio": 1.0,
                "probe_best_lt_long_ratio": 1.0,
                "probe_short_avg": 100.0,
                "probe_long_avg": 2000.0,
                "probe_baseline_avg": 1536.0,
                "probe_best_avg": 768.0,
                "probe_queue_avg": 2.0,
                "probe_wait_us_avg": 100.0,
                "probe_reasons": {"ok": 1},
            }
        },
        "timed_out": False,
    }


def _make_phase2_row(ttft: float, wall: float, apply: float) -> dict:
    return {
        "ttft_short_p99_ms": ttft,
        "round_wall_ms": wall,
        "texts": {"short_a": "ok", "long_b": "ok"},
        "hook_report": {
            "slowdown_short": {"p99": 10.0},
            "phase2": {
                "apply_ratio": apply,
                "applied": 1,
                "attempts": 2,
                "escape_lane": {
                    "activations": 1,
                    "active_count_avg": 1.0,
                    "deferred_count_avg": 0.0,
                    "seen_active_hits": 1,
                    "finished_active_hits": 1,
                },
            },
        },
        "request_timings": {},
        "timed_out": False,
    }


class RefactorSmokeTests(unittest.TestCase):
    def test_frozen_v1_regression_config_builds_expected_invocation(self) -> None:
        config_path = ROOT / "experiments" / "configs" / "frozen_v1_gemma_mid_global_activity_repro.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))

        cmd, env = build_eval_invocation(config)

        self.assertEqual(cmd[0], "/home/onceas/anaconda3/envs/sara/bin/python")
        self.assertEqual(cmd[1], "tests/evaluate_waveslice_claims.py")
        self.assertIn("--include-phase12", cmd)
        self.assertIn("--requests-json", cmd)
        self.assertIn("results/openworkload_execescape_tradeoff/20260408_185300_serialclean10_v9/workloads/mid/gemma-7b-it_requests.json", cmd)
        self.assertIn("--lora-requests-json", cmd)
        self.assertIn("--phase1-force-min-chunk", cmd)
        self.assertIn("128", cmd)
        self.assertIn("--phase1-target-long-fraction", cmd)
        self.assertIn("0.33", cmd)
        self.assertIn("--no-phase2-enable-mixed-prefill-decode", cmd)
        self.assertEqual(env["WAVESLICE_VLLM_MODE"], "v1")
        self.assertEqual(env["VLLM_USE_V1"], "1")

    def test_extract_processed_prompt_tokens_prefers_decoder_tokens(self) -> None:
        self.assertEqual(
            _extract_processed_prompt_tokens(
                {
                    "encoder": {"prompt_token_ids": [1] * 32},
                    "decoder": {"prompt_token_ids": [2] * 2268},
                }
            ),
            2268,
        )

    def test_phase1_ingress_seed_sets_direct_cap_for_long_request(self) -> None:
        state = SimpleNamespace(
            policy=WaveSlicePolicy(phase1_ingress_direct_authoritative=True),
            metrics=WaveSliceMetrics(),
            slicer=WaveBaseSlicer(),
            phase1_active_prompt_tokens={},
            phase1_ingress_virtuals={},
            phase1_virtual_token_caps={},
        )

        phase1_maybe_seed_ingress_virtual(
            state,
            request_id="short_a",
            input_tokens=67,
        )
        phase1_maybe_seed_ingress_virtual(
            state,
            request_id="short_b",
            input_tokens=156,
        )
        phase1_maybe_seed_ingress_virtual(
            state,
            request_id="long_b",
            input_tokens=2268,
        )

        self.assertIn("long_b", state.phase1_ingress_virtuals)
        self.assertGreater(state.phase1_virtual_token_caps.get("long_b", 0), 0)
        scheduler = state.metrics.summary()["scheduler"]
        self.assertGreaterEqual(scheduler["virtual_cap_target_set"], 1.0)
        self.assertEqual(
            scheduler["request_traces"]["long_b"][-1]["event"],
            "phase1_ingress_direct_cap_seeded",
        )

    def test_phase1_ingress_seed_prefers_direct_plan_chunk_when_available(self) -> None:
        state = SimpleNamespace(
            policy=WaveSlicePolicy(phase1_ingress_direct_authoritative=True),
            metrics=WaveSliceMetrics(),
            slicer=WaveBaseSlicer(),
            phase1_active_prompt_tokens={},
            phase1_ingress_virtuals={},
            phase1_virtual_token_caps={},
            phase1_explicit_plans={},
        )

        with mock.patch(
            "engine.hijack.phase1_stateful.phase1_build_direct_explicit_plans",
            return_value=[SimpleNamespace(chunk_len=320)],
        ):
            phase1_maybe_seed_ingress_virtual(
                state,
                request_id="short_a",
                input_tokens=67,
            )
            phase1_maybe_seed_ingress_virtual(
                state,
                request_id="short_b",
                input_tokens=156,
            )
            phase1_maybe_seed_ingress_virtual(
                state,
                request_id="long_b",
                input_tokens=2268,
            )

        self.assertEqual(state.phase1_virtual_token_caps.get("long_b"), 320)

    def test_lookup_engine_prompt_tokens_prefers_scheduler_state(self) -> None:
        class FakeSeq:
            def __init__(self, total_tokens: int) -> None:
                self._total_tokens = total_tokens
                self.data = SimpleNamespace(get_num_computed_tokens=lambda: 0)

            def get_len(self) -> int:
                return self._total_tokens

        class FakeSeqGroup:
            def __init__(self, request_id: str, total_tokens: int) -> None:
                self.request_id = request_id
                self._seq = FakeSeq(total_tokens)

            def get_seqs(self):
                return [self._seq]

        scheduler_obj = SimpleNamespace(
            waiting=[FakeSeqGroup("long_b", 2268)],
            running=[],
            swapped=[],
        )
        engine_self = SimpleNamespace(scheduler=[scheduler_obj])

        self.assertEqual(
            hijacker._lookup_engine_prompt_tokens(engine_self, request_id="long_b"),
            2268,
        )

    def test_add_processed_request_hook_uses_true_prompt_tokens(self) -> None:
        state = SimpleNamespace(
            original_add_processed_request=lambda *_args, **_kwargs: "ok",
            policy=WaveSlicePolicy(phase1_ingress_direct_authoritative=True),
            metrics=WaveSliceMetrics(),
            brain=mock.Mock(),
            slicer=WaveBaseSlicer(),
            phase1_active_prompt_tokens={},
            phase1_ingress_virtuals={},
            phase1_virtual_token_caps={},
        )
        hook = build_add_processed_request_hook(
            state,
            estimate_solo_us=lambda *_args, **_kwargs: 123.0,
            phase1_maybe_seed_ingress_virtual=phase1_maybe_seed_ingress_virtual,
        )

        hook(
            object(),
            "short_a",
            {"prompt_token_ids": [1] * 67},
            object(),
            1.0,
            None,
        )
        hook(
            object(),
            "long_b",
            {"prompt_token_ids": [2] * 2268},
            object(),
            2.0,
            None,
        )

        self.assertEqual(state.phase1_active_prompt_tokens["long_b"], 2268)
        self.assertEqual(
            state.metrics.snapshot_requests(["long_b"])["long_b"]["input_tokens"],
            2268,
        )
        self.assertEqual(
            state.metrics.summary()["scheduler"]["request_traces"]["long_b"][0]["event"],
            "phase1_add_processed_request_observed",
        )

    def test_add_request_hook_prefers_seeded_prompt_tokens_over_estimate(self) -> None:
        estimate_prompt_tokens = mock.Mock(return_value=1686)
        lookup_engine_prompt_tokens = mock.Mock(return_value=None)
        state = SimpleNamespace(
            original_add_request=lambda *_args, **_kwargs: "ok",
            policy=WaveSlicePolicy(phase1_ingress_direct_authoritative=True),
            metrics=WaveSliceMetrics(),
            brain=mock.Mock(),
            phase1_active_prompt_tokens={"long_b": 2268},
            phase1_ingress_virtuals={},
            phase1_virtual_token_caps={},
        )
        hook = build_add_request_hook(
            state,
            estimate_prompt_tokens=estimate_prompt_tokens,
            estimate_solo_us=lambda *_args, **_kwargs: 123.0,
            lookup_engine_prompt_tokens=lookup_engine_prompt_tokens,
            phase1_maybe_seed_ingress_virtual=phase1_maybe_seed_ingress_virtual,
        )

        hook(object(), "long_b", "pretend prompt", object(), None)

        self.assertEqual(
            state.metrics.snapshot_requests(["long_b"])["long_b"]["input_tokens"],
            2268,
        )
        estimate_prompt_tokens.assert_not_called()
        lookup_engine_prompt_tokens.assert_not_called()

    def test_v1_process_inputs_hook_uses_true_request_prompt_tokens(self) -> None:
        state = SimpleNamespace(
            original_v1_processor_process_inputs=lambda *_args, **_kwargs: (
                "prompt",
                SimpleNamespace(prompt_token_ids=[2] * 2268),
            ),
            policy=WaveSlicePolicy(phase1_ingress_direct_authoritative=True),
            metrics=WaveSliceMetrics(),
            brain=mock.Mock(),
            slicer=WaveBaseSlicer(),
            phase1_active_prompt_tokens={},
            phase1_ingress_virtuals={},
            phase1_virtual_token_caps={},
        )
        hook = build_v1_process_inputs_hook(
            state,
            estimate_solo_us=lambda *_args, **_kwargs: 123.0,
            phase1_maybe_seed_ingress_virtual=phase1_maybe_seed_ingress_virtual,
        )

        hook(object(), "short_a", "prompt-a", object(), 1.0, None)
        hook(object(), "long_b", "prompt-b", object(), 2.0, None)

        self.assertEqual(state.phase1_active_prompt_tokens["long_b"], 2268)
        self.assertEqual(
            state.metrics.snapshot_requests(["long_b"])["long_b"]["input_tokens"],
            2268,
        )
        trace = state.metrics.summary()["scheduler"]["request_traces"]["long_b"]
        self.assertEqual(trace[0]["event"], "phase1_v1_process_inputs_observed")

    def test_v1_engine_core_add_request_hook_seeds_child_process_cap(self) -> None:
        state = SimpleNamespace(
            original_v1_engine_core_add_request=lambda *_args, **_kwargs: "ok",
            policy=WaveSlicePolicy(phase1_ingress_direct_authoritative=True),
            metrics=WaveSliceMetrics(),
            brain=mock.Mock(),
            slicer=WaveBaseSlicer(),
            phase1_active_prompt_tokens={},
            phase1_ingress_virtuals={},
            phase1_virtual_token_caps={},
        )
        hook = build_v1_engine_core_add_request_hook(
            state,
            estimate_solo_us=lambda *_args, **_kwargs: 123.0,
            phase1_maybe_seed_ingress_virtual=phase1_maybe_seed_ingress_virtual,
        )

        hook(object(), SimpleNamespace(request_id="short_a", prompt_token_ids=[1] * 67, arrival_time=1.0))
        hook(object(), SimpleNamespace(request_id="long_b", prompt_token_ids=[2] * 2268, arrival_time=2.0))

        self.assertEqual(state.phase1_active_prompt_tokens["long_b"], 2268)
        self.assertEqual(
            state.metrics.snapshot_requests(["long_b"])["long_b"]["input_tokens"],
            2268,
        )
        trace = state.metrics.summary()["scheduler"]["request_traces"]["long_b"]
        self.assertEqual(trace[0]["event"], "phase1_v1_engine_core_add_request_observed")

    def test_v1_scheduler_add_request_hook_sets_provisional_cap_for_long_prefill(self) -> None:
        state = SimpleNamespace(
            original_scheduler_add_request=lambda *_args, **_kwargs: "ok",
            policy=WaveSlicePolicy(
                phase1_ingress_direct_authoritative=True,
                phase1_ingress_exact_chunk=True,
                phase1_ingress_target_chunk=128,
                phase1_ingress_min_chunk=256,
            ),
            metrics=WaveSliceMetrics(),
            slicer=WaveBaseSlicer(),
            phase1_virtual_token_caps={},
        )

        request = SimpleNamespace(
            request_id="long_b",
            num_prompt_tokens=2268,
            num_output_tokens=0,
            num_computed_tokens=0,
        )

        hook = hijacker._build_v1_scheduler_add_request_hook(state)
        hook(object(), request)

        self.assertEqual(state.phase1_virtual_token_caps["long_b"], 128)
        trace = state.metrics.summary()["scheduler"]["request_traces"]["long_b"]
        self.assertEqual(trace[0]["event"], "phase1_v1_scheduler_add_request_provisional_cap")

    def test_v1_scheduler_add_request_hook_uses_lora_fractional_cap_for_moderate_long(self) -> None:
        state = SimpleNamespace(
            original_scheduler_add_request=lambda *_args, **_kwargs: "ok",
            policy=WaveSlicePolicy(
                phase1_ingress_direct_authoritative=True,
                phase1_ingress_exact_chunk=False,
                phase1_ingress_target_chunk=384,
                phase1_ingress_min_chunk=256,
                phase1_target_long_fraction=0.33,
                phase1_force_min_chunk=128,
            ),
            metrics=WaveSliceMetrics(),
            slicer=WaveBaseSlicer(),
            phase1_virtual_token_caps={},
        )

        request = SimpleNamespace(
            request_id="long_a",
            num_prompt_tokens=468,
            num_output_tokens=0,
            num_computed_tokens=0,
            lora_request=SimpleNamespace(lora_path="/tmp/adapter_a_rank8"),
        )

        hook = hijacker._build_v1_scheduler_add_request_hook(state)
        hook(object(), request)

        self.assertEqual(state.phase1_virtual_token_caps["long_a"], 154)

    def test_phase1_effective_ingress_min_chunk_allows_exact_target_below_floor(self) -> None:
        exact_policy = WaveSlicePolicy(
            phase1_ingress_exact_chunk=True,
            phase1_ingress_target_chunk=128,
            phase1_ingress_min_chunk=256,
        )
        mapped_policy = WaveSlicePolicy(
            phase1_ingress_exact_chunk=False,
            phase1_ingress_target_chunk=128,
            phase1_ingress_min_chunk=256,
        )

        self.assertEqual(
            phase1_effective_ingress_min_chunk(exact_policy, target=128),
            128,
        )
        self.assertEqual(
            phase1_effective_ingress_min_chunk(mapped_policy, target=128),
            256,
        )

    def test_phase1_authoritative_chunk_respects_exact_target_ceiling(self) -> None:
        policy = WaveSlicePolicy(
            phase1_ingress_exact_chunk=True,
            phase1_ingress_target_chunk=128,
            phase1_ingress_min_chunk=256,
            phase1_ingress_max_chunk=512,
        )

        chunk = phase1_authoritative_chunk(
            policy,
            WaveBaseSlicer(),
            target=257,
            short_len=157,
            upper=780,
        )

        self.assertEqual(chunk, 128)

    def test_phase12_joint_phase1_floor_skips_when_phase2_paths_disabled(self) -> None:
        policy = WaveSlicePolicy(
            enable_phase2_modelrunner=True,
            phase12_joint_coordination=True,
            phase2_enable_execution_escape=False,
            phase2_enable_scheduler_cashout=False,
            phase2_enable_v1_true_unbind=False,
        )
        snapshot = SimpleNamespace(running=["run"], waiting=["wait"])

        with mock.patch.object(
            hijacker,
            "_phase12_collect_prefill_lora_state",
            return_value=([780, 468], [16, 8]),
        ), mock.patch.object(
            hijacker,
            "_phase2_selective_gate",
            return_value=(True, 3.0, 2.0, True),
        ):
            floor = hijacker._phase12_joint_phase1_floor(
                state=SimpleNamespace(),
                snapshot=snapshot,
                policy=policy,
            )

        self.assertIsNone(floor)

    def test_v1_merge_keeps_generation_pooler_output_empty(self) -> None:
        class FakeOutput:
            def __init__(self, **kwargs) -> None:
                for key, value in kwargs.items():
                    setattr(self, key, value)

        out_a = FakeOutput(
            req_ids=["short_prefill"],
            req_id_to_index={"short_prefill": 0},
            sampled_token_ids=[[]],
            spec_token_ids=None,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
            kv_connector_output=None,
            num_nans_in_logits=None,
        )
        out_b = FakeOutput(
            req_ids=["long_prefill"],
            req_id_to_index={"long_prefill": 0},
            sampled_token_ids=[[42]],
            spec_token_ids=None,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
            kv_connector_output=None,
            num_nans_in_logits=None,
        )

        merged = merge_v1_runner_outputs(
            ["short_prefill", "long_prefill"],
            out_a,
            out_b,
        )

        self.assertEqual(merged.req_ids, ["short_prefill", "long_prefill"])
        self.assertEqual(merged.sampled_token_ids, [[], [42]])
        self.assertEqual(merged.pooler_output, [])

    def test_v1_freeze_runner_output_detaches_mutable_req_mappings(self) -> None:
        class FakeOutput:
            def __init__(self, **kwargs) -> None:
                for key, value in kwargs.items():
                    setattr(self, key, value)

        req_ids = ["short_prefill"]
        req_id_to_index = {"short_prefill": 0}
        output = FakeOutput(
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
            sampled_token_ids=[[11]],
            spec_token_ids=None,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
            kv_connector_output=None,
            num_nans_in_logits=None,
        )

        frozen = freeze_v1_runner_output(output)
        req_ids[:] = ["long_prefill"]
        req_id_to_index.clear()
        req_id_to_index["long_prefill"] = 0

        self.assertEqual(frozen.req_ids, ["short_prefill"])
        self.assertEqual(frozen.req_id_to_index, {"short_prefill": 0})

    def test_v1_partition_can_keep_decode_requests_on_long_side(self) -> None:
        class FakeModelInput:
            def __init__(self) -> None:
                self.num_scheduled_tokens = {
                    "short_prefill": 8,
                    "decode": 1,
                    "long_prefill": 64,
                }

        def collect_req_infos(*_args, **_kwargs) -> list[_ScheduledReqInfo]:
            return [
                _ScheduledReqInfo(
                    request_id="short_prefill",
                    scheduled_tokens=8,
                    remaining_tokens=8,
                    expected_chunk_tokens=8,
                    input_tokens=128,
                    arrival_s=0.0,
                    is_short=True,
                    lora_rank=1,
                ),
                _ScheduledReqInfo(
                    request_id="long_prefill",
                    scheduled_tokens=64,
                    remaining_tokens=64,
                    expected_chunk_tokens=64,
                    input_tokens=2048,
                    arrival_s=0.0,
                    is_short=False,
                    lora_rank=1,
                ),
            ]

        def build_beneficiary_signal(*_args, **_kwargs) -> _Phase12BeneficiarySignal:
            return _Phase12BeneficiarySignal(
                long_anchor_id=None,
                beneficiary_prefill_ids=[],
                beneficiary_prefill_count=0,
                beneficiary_fraction=0.0,
                beneficiary_wait_quality=0.0,
                beneficiary_size_quality=0.0,
                beneficiary_cashout_quality=0.0,
                beneficiary_selected_quality=0.0,
                beneficiary_selected_ids=[],
                beneficiary_score_map={},
            )

        short_ids, long_ids = v1_partition_req_ids(
            FakeModelInput(),
            WaveSlicePolicy(),
            req_info_collector=collect_req_infos,
            beneficiary_signal_builder=build_beneficiary_signal,
            include_non_prefill_in_long=True,
        ) or (None, None)

        self.assertEqual(short_ids, ["short_prefill"])
        self.assertEqual(long_ids, ["decode", "long_prefill"])

    def test_v1_partition_diagnostic_reports_ratio_gate(self) -> None:
        class FakeModelInput:
            def __init__(self) -> None:
                self.num_scheduled_tokens = {
                    "prefill_a": 64,
                    "prefill_b": 72,
                }

        def collect_req_infos(*_args, **_kwargs) -> list[_ScheduledReqInfo]:
            return [
                _ScheduledReqInfo(
                    request_id="prefill_a",
                    scheduled_tokens=64,
                    remaining_tokens=64,
                    expected_chunk_tokens=64,
                    input_tokens=1024,
                    arrival_s=0.0,
                    is_short=False,
                    lora_rank=1,
                ),
                _ScheduledReqInfo(
                    request_id="prefill_b",
                    scheduled_tokens=72,
                    remaining_tokens=72,
                    expected_chunk_tokens=72,
                    input_tokens=1152,
                    arrival_s=0.0,
                    is_short=False,
                    lora_rank=1,
                ),
            ]

        def build_beneficiary_signal(*_args, **_kwargs) -> _Phase12BeneficiarySignal:
            return _Phase12BeneficiarySignal(
                long_anchor_id=None,
                beneficiary_prefill_ids=[],
                beneficiary_prefill_count=0,
                beneficiary_fraction=0.0,
                beneficiary_wait_quality=0.0,
                beneficiary_size_quality=0.0,
                beneficiary_cashout_quality=0.0,
                beneficiary_selected_quality=0.0,
                beneficiary_selected_ids=[],
                beneficiary_score_map={},
            )

        split_ids, reason, debug = v1_partition_req_ids_diagnostic(
            FakeModelInput(),
            WaveSlicePolicy(phase2_min_hetero_ratio=4.0),
            req_info_collector=collect_req_infos,
            beneficiary_signal_builder=build_beneficiary_signal,
        )

        self.assertIsNone(split_ids)
        self.assertEqual(reason, "ratio_below_threshold")
        self.assertEqual(debug["prefill_req_count"], 2)
        self.assertGreater(debug["ratio_threshold_x1000"], debug["ratio_x1000"])

    def test_v1_execution_escape_falls_back_to_partition_when_beneficiary_empty(self) -> None:
        class FakeModelInput:
            def __init__(self) -> None:
                self.num_scheduled_tokens = {
                    "short_prefill": 16,
                    "decode": 1,
                    "long_prefill": 96,
                }

        def collect_req_infos(*_args, **_kwargs) -> list[_ScheduledReqInfo]:
            return [
                _ScheduledReqInfo(
                    request_id="short_prefill",
                    scheduled_tokens=16,
                    remaining_tokens=16,
                    expected_chunk_tokens=16,
                    input_tokens=256,
                    arrival_s=0.0,
                    is_short=True,
                    lora_rank=1,
                ),
                _ScheduledReqInfo(
                    request_id="long_prefill",
                    scheduled_tokens=96,
                    remaining_tokens=96,
                    expected_chunk_tokens=96,
                    input_tokens=2048,
                    arrival_s=0.0,
                    is_short=False,
                    lora_rank=1,
                ),
            ]

        def build_beneficiary_signal(*_args, **_kwargs) -> _Phase12BeneficiarySignal:
            return _Phase12BeneficiarySignal(
                long_anchor_id=None,
                beneficiary_prefill_ids=[],
                beneficiary_prefill_count=0,
                beneficiary_fraction=0.0,
                beneficiary_wait_quality=0.0,
                beneficiary_size_quality=0.0,
                beneficiary_cashout_quality=0.0,
                beneficiary_selected_quality=0.0,
                beneficiary_selected_ids=[],
                beneficiary_score_map={},
            )

        split_ids = v1_execution_escape_req_ids(
            FakeModelInput(),
            WaveSlicePolicy(phase2_execution_escape_mode="bounded_spillover"),
            state=SimpleNamespace(),
            req_info_collector=collect_req_infos,
            beneficiary_signal_builder=build_beneficiary_signal,
        )

        self.assertEqual(split_ids, (["short_prefill"], ["decode", "long_prefill"]))

    def test_v1_filter_cached_reqs_tolerates_sparse_optional_lists(self) -> None:
        class FakeCached:
            def __init__(
                self,
                *,
                req_ids,
                resumed_from_preemption,
                new_token_ids,
                new_block_ids,
                num_computed_tokens,
            ) -> None:
                self.req_ids = req_ids
                self.resumed_from_preemption = resumed_from_preemption
                self.new_token_ids = new_token_ids
                self.new_block_ids = new_block_ids
                self.num_computed_tokens = num_computed_tokens

        cached = FakeCached(
            req_ids=["decode", "long_prefill"],
            resumed_from_preemption=[False, False],
            new_token_ids=[],
            new_block_ids=[([1, 2],), ([3, 4],)],
            num_computed_tokens=[128, 256],
        )

        filtered = v1_filter_cached_reqs(cached, {"decode"})

        self.assertEqual(filtered.req_ids, ["decode"])
        self.assertEqual(filtered.resumed_from_preemption, [False])
        self.assertEqual(filtered.new_token_ids, [[]])
        self.assertEqual(filtered.new_block_ids, [([1, 2],)])
        self.assertEqual(filtered.num_computed_tokens, [128])

    def test_v1_request_token_hooks_cap_prefill_but_not_decode(self) -> None:
        class DummyRequest:
            def __init__(
                self,
                *,
                request_id: str,
                total_tokens: int,
                spec_tokens: int,
                prompt_tokens: int,
                output_tokens: int,
                computed_tokens: int,
            ) -> None:
                self.request_id = request_id
                self._total_tokens = total_tokens
                self._spec_tokens = spec_tokens
                self.num_prompt_tokens = prompt_tokens
                self._num_output_tokens = output_tokens
                self.num_computed_tokens = computed_tokens

            @property
            def num_tokens(self) -> int:
                return self._total_tokens

            @property
            def num_tokens_with_spec(self) -> int:
                return self._total_tokens + self._spec_tokens

            @property
            def num_output_tokens(self) -> int:
                return self._num_output_tokens

        state = SimpleNamespace(
            phase1_virtual_token_caps={"prefill": 128, "decode": 64},
            metrics=WaveSliceMetrics(),
            original_v1_request_num_tokens=DummyRequest.num_tokens,
            original_v1_request_num_tokens_with_spec=DummyRequest.num_tokens_with_spec,
        )

        original_num_tokens = DummyRequest.num_tokens
        original_num_tokens_with_spec = DummyRequest.num_tokens_with_spec
        DummyRequest.num_tokens = hijacker._build_v1_request_num_tokens_hook(state)
        DummyRequest.num_tokens_with_spec = hijacker._build_v1_request_num_tokens_with_spec_hook(state)
        try:
            prefill = DummyRequest(
                request_id="prefill",
                total_tokens=1024,
                spec_tokens=16,
                prompt_tokens=1024,
                output_tokens=0,
                computed_tokens=256,
            )
            decode = DummyRequest(
                request_id="decode",
                total_tokens=1025,
                spec_tokens=4,
                prompt_tokens=1024,
                output_tokens=1,
                computed_tokens=1024,
            )

            self.assertEqual(prefill.num_tokens, 384)
            self.assertEqual(prefill.num_tokens_with_spec, 384)
            self.assertEqual(decode.num_tokens, 1025)
            self.assertEqual(decode.num_tokens_with_spec, 1029)
        finally:
            DummyRequest.num_tokens = original_num_tokens
            DummyRequest.num_tokens_with_spec = original_num_tokens_with_spec

        scheduler = state.metrics.summary()["scheduler"]
        self.assertEqual(scheduler["virtual_cap_target_hits"], 2.0)
        self.assertGreater(scheduler["virtual_cap_helper_calls"], 0.0)
        self.assertGreater(scheduler["virtual_cap_prefill_calls"], 0.0)
        self.assertGreater(scheduler["virtual_cap_apply_ratio"], 0.0)

    def test_v1_request_token_hooks_use_effective_waiting_computed_tokens(self) -> None:
        class DummyRequest:
            def __init__(self) -> None:
                self.request_id = "prefill"
                self._total_tokens = 1024
                self.num_prompt_tokens = 1024
                self._num_output_tokens = 0
                self.num_computed_tokens = 0
                self.__wave_slice_effective_computed_tokens__ = 768

            @property
            def num_tokens(self) -> int:
                return self._total_tokens

            @property
            def num_output_tokens(self) -> int:
                return self._num_output_tokens

        state = SimpleNamespace(
            phase1_virtual_token_caps={"prefill": 128},
            metrics=WaveSliceMetrics(),
            original_v1_request_num_tokens=DummyRequest.num_tokens,
        )

        original_num_tokens = DummyRequest.num_tokens
        DummyRequest.num_tokens = hijacker._build_v1_request_num_tokens_hook(state)
        try:
            request = DummyRequest()
            self.assertEqual(request.num_tokens, 896)
            request.__wave_slice_skip_v1_cap__ = True
            self.assertEqual(request.num_tokens, 1024)
        finally:
            DummyRequest.num_tokens = original_num_tokens

    def test_v1_waiting_cap_runtime_hooks_track_effective_computed_tokens(self) -> None:
        class DummyRequest:
            def __init__(self) -> None:
                self.request_id = "prefill"

        class DummyKVCacheManager:
            def get_computed_blocks(self, request: DummyRequest) -> tuple[str, int]:
                self.seen_skip_flag = bool(getattr(request, "__wave_slice_skip_v1_cap__", False))
                return "blocks", 512

        class DummyConnector:
            def get_num_new_matched_tokens(
                self,
                request: DummyRequest,
                num_new_local_computed_tokens: int,
            ) -> tuple[int, bool]:
                self.seen_local = num_new_local_computed_tokens
                return 128, False

        request = DummyRequest()
        kv_cache_manager = DummyKVCacheManager()
        connector = DummyConnector()
        scheduler_obj = SimpleNamespace(
            kv_cache_manager=kv_cache_manager,
            connector=connector,
        )

        cleanup_requests, restore_specs = hijacker._install_v1_waiting_cap_runtime_hooks(
            SimpleNamespace(policy=WaveSlicePolicy()),
            scheduler_obj,
        )
        try:
            blocks, local_tokens = scheduler_obj.kv_cache_manager.get_computed_blocks(request)
            self.assertEqual(blocks, "blocks")
            self.assertEqual(local_tokens, 512)
            self.assertTrue(kv_cache_manager.seen_skip_flag)
            self.assertEqual(
                getattr(request, "__wave_slice_effective_computed_tokens__"),
                512,
            )

            external_tokens, load_async = scheduler_obj.connector.get_num_new_matched_tokens(
                request,
                local_tokens,
            )
            self.assertEqual((external_tokens, load_async), (128, False))
            self.assertEqual(connector.seen_local, 512)
            self.assertEqual(
                getattr(request, "__wave_slice_effective_computed_tokens__"),
                640,
            )
        finally:
            hijacker._restore_v1_waiting_cap_runtime_hooks(cleanup_requests, restore_specs)

        self.assertFalse(hasattr(request, "__wave_slice_effective_computed_tokens__"))
        self.assertFalse(hasattr(request, "__wave_slice_skip_v1_cap__"))

    def test_v1_update_after_schedule_hook_clamps_prefill_before_state_advance(self) -> None:
        class DummyRequest:
            def __init__(self, request_id: str) -> None:
                self.request_id = request_id
                self.num_output_tokens = 0
                self.num_prompt_tokens = 2268
                self.num_computed_tokens = 0

        class DummyScheduler:
            def __init__(self) -> None:
                self.requests = {
                    "long_b": DummyRequest("long_b"),
                    "short_a": DummyRequest("short_a"),
                }

            def _update_after_schedule(self, scheduler_output) -> None:
                for req_id, num_tokens in scheduler_output.num_scheduled_tokens.items():
                    self.requests[req_id].num_computed_tokens += int(num_tokens)

        state = SimpleNamespace(
            phase1_virtual_token_caps={"long_b": 256},
            metrics=WaveSliceMetrics(),
            original_scheduler_update_after_schedule=DummyScheduler._update_after_schedule,
            policy=WaveSlicePolicy(phase1_ingress_direct_authoritative=True),
        )

        original = DummyScheduler._update_after_schedule
        DummyScheduler._update_after_schedule = hijacker._build_v1_scheduler_update_after_schedule_hook(state)
        try:
            scheduler = DummyScheduler()
            scheduler_output = SimpleNamespace(
                num_scheduled_tokens={"long_b": 1536, "short_a": 64},
                total_num_scheduled_tokens=1600,
            )

            scheduler._update_after_schedule(scheduler_output)

            self.assertEqual(scheduler_output.num_scheduled_tokens["long_b"], 256)
            self.assertEqual(scheduler_output.num_scheduled_tokens["short_a"], 64)
            self.assertEqual(scheduler_output.total_num_scheduled_tokens, 320)
            self.assertEqual(scheduler.requests["long_b"].num_computed_tokens, 256)
            self.assertEqual(scheduler.requests["short_a"].num_computed_tokens, 64)
        finally:
            DummyScheduler._update_after_schedule = original

    def test_v1_waiting_num_new_tokens_helper_clamps_prefill(self) -> None:
        class DummyRequest:
            def __init__(self, request_id: str) -> None:
                self.request_id = request_id
                self.num_prompt_tokens = 2268
                self.num_output_tokens = 0
                self.num_computed_tokens = 0

            @property
            def num_tokens(self) -> int:
                return 2268

        state = SimpleNamespace(
            phase1_virtual_token_caps={"long_b": 256},
            metrics=WaveSliceMetrics(),
        )
        request = DummyRequest("long_b")

        clamped = hijacker._compute_v1_waiting_num_new_tokens(
            state,
            object(),
            request,
            0,
            1536,
        )

        self.assertEqual(clamped, 256)
        trace = state.metrics.summary()["scheduler"]["request_traces"]["long_b"]
        self.assertEqual(trace[0]["event"], "phase1_v1_schedule_waiting_num_new_tokens_seen")
        self.assertEqual(trace[1]["event"], "phase1_v1_schedule_waiting_num_new_tokens_clamp")

    def test_v1_running_num_new_tokens_helper_clamps_prefill(self) -> None:
        class DummyRequest:
            def __init__(self, request_id: str) -> None:
                self.request_id = request_id
                self.num_prompt_tokens = 2268
                self.num_output_tokens = 0
                self.num_computed_tokens = 512
                self.num_output_placeholders = 0

            @property
            def num_tokens_with_spec(self) -> int:
                return 2268

        state = SimpleNamespace(
            phase1_virtual_token_caps={"long_b": 256},
            metrics=WaveSliceMetrics(),
        )
        request = DummyRequest("long_b")

        clamped = hijacker._compute_v1_running_num_new_tokens(
            state,
            object(),
            request,
        )

        self.assertEqual(clamped, 256)
        trace = state.metrics.summary()["scheduler"]["request_traces"]["long_b"]
        self.assertEqual(trace[0]["event"], "phase1_v1_schedule_running_num_new_tokens_seen")
        self.assertEqual(trace[1]["event"], "phase1_v1_schedule_running_num_new_tokens_clamp")

    def test_phase1_tick_hide_keep_ceiling_preserves_medium_shorts(self) -> None:
        cohort = hijacker._Phase1CohortStats(
            representative_short_len=22,
            short_count=2,
            short_token_mass=178,
            short_lengths=[22, 156],
            long_len=780,
            long_req_id="long_b",
            total_count=3,
        )

        keep_ceiling = hijacker._phase1_tick_hide_keep_ceiling(
            best_chunk=128,
            cohort=cohort,
        )

        self.assertEqual(keep_ceiling, 156)

    def test_phase12_force_lora_tick_hide_only_for_joint_modelrunner_lora(self) -> None:
        state = SimpleNamespace(
            policy=WaveSlicePolicy(
                phase12_joint_coordination=True,
                enable_phase2_modelrunner=True,
            )
        )

        self.assertTrue(
            hijacker._phase12_should_force_lora_tick_hide(
                state=state,
                lora_mode_enabled=True,
                waiting_short_count=1,
            )
        )
        self.assertFalse(
            hijacker._phase12_should_force_lora_tick_hide(
                state=state,
                lora_mode_enabled=False,
                waiting_short_count=1,
            )
        )
        self.assertFalse(
            hijacker._phase12_should_force_lora_tick_hide(
                state=SimpleNamespace(
                    policy=WaveSlicePolicy(
                        phase12_joint_coordination=False,
                        enable_phase2_modelrunner=True,
                    )
                ),
                lora_mode_enabled=True,
                waiting_short_count=1,
            )
        )

    def test_phase1_filter_snapshot_for_lora_cohort_prefers_same_adapter(self) -> None:
        class DummyLora:
            def __init__(self, path: str) -> None:
                self.lora_path = path

        class DummySeqGroup:
            def __init__(self, request_id: str, remaining: int, lora_path: str) -> None:
                self.request_id = request_id
                self.num_tokens_with_spec = remaining
                self.num_prompt_tokens = remaining
                self.num_computed_tokens = 0
                self.lora_request = DummyLora(lora_path)

            def is_prefill(self) -> bool:
                return True

        snapshot = [
            (DummySeqGroup("short_a", 22, "/tmp/adapter_a_rank8"), 22),
            (DummySeqGroup("mid_b", 156, "/tmp/adapter_b_rank16"), 156),
            (DummySeqGroup("long_a", 468, "/tmp/adapter_a_rank8"), 468),
            (DummySeqGroup("long_b", 780, "/tmp/adapter_b_rank16"), 780),
        ]

        filtered = hijacker._phase1_filter_snapshot_for_lora_cohort(
            snapshot,
            preferred_request_id="long_b",
        )

        self.assertEqual(
            [_sg.request_id for _sg, _ in filtered],
            ["mid_b", "long_b"],
        )

    def test_phase1_find_ingress_virtual_candidate_prefers_largest_live_match(self) -> None:
        ingress_virtuals = {
            "long_a": _Phase1IngressVirtualSlice(
                long_req_id="long_a",
                representative_short_len=96,
                short_count=2,
                short_token_mass=192,
                short_lengths=[96, 96],
                original_long_len=2048,
                active_count=3,
            ),
            "long_b": _Phase1IngressVirtualSlice(
                long_req_id="long_b",
                representative_short_len=96,
                short_count=2,
                short_token_mass=192,
                short_lengths=[96, 96],
                original_long_len=3072,
                active_count=3,
            ),
        }
        snapshot = [
            (SimpleNamespace(request_id="long_a"), 512),
            (SimpleNamespace(request_id="long_b"), 1024),
        ]

        candidate = phase1_find_ingress_virtual_candidate(
            ingress_virtuals,
            snapshot=snapshot,
            request_id_getter=lambda seq_group: str(getattr(seq_group, "request_id", "")),
        )

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate[0].long_req_id, "long_b")
        self.assertEqual(candidate[2], 1024)

    def test_phase1_build_ingress_fallback_cohort_uses_live_long_with_seeded_history(self) -> None:
        state = SimpleNamespace(
            policy=WaveSlicePolicy(
                phase1_force_min_chunk=128,
                min_long_seq=256,
                min_hetero_ratio=2.0,
            ),
            phase1_ingress_virtuals={
                "long_b": _Phase1IngressVirtualSlice(
                    long_req_id="long_b",
                    representative_short_len=96,
                    short_count=2,
                    short_token_mass=192,
                    short_lengths=[96, 96],
                    original_long_len=2900,
                    active_count=3,
                )
            },
        )
        snapshot = [(SimpleNamespace(request_id="long_b"), 448)]

        cohort = hijacker._phase1_build_ingress_fallback_cohort(state, snapshot)

        self.assertIsNotNone(cohort)
        assert cohort is not None
        self.assertEqual(cohort.long_req_id, "long_b")
        self.assertEqual(cohort.long_len, 448)
        self.assertEqual(cohort.representative_short_len, 96)

    def test_phase1_build_global_activity_cohort_uses_active_prompt_view(self) -> None:
        state = SimpleNamespace(
            policy=WaveSlicePolicy(
                phase1_force_min_chunk=128,
                min_long_seq=256,
                min_hetero_ratio=2.0,
            ),
            phase1_active_prompt_tokens={
                "short_a": 82,
                "short_b": 156,
                "long_b": 2268,
            },
            phase1_ingress_virtuals={
                "long_b": _Phase1IngressVirtualSlice(
                    long_req_id="long_b",
                    representative_short_len=119,
                    short_count=2,
                    short_token_mass=238,
                    short_lengths=[82, 156],
                    original_long_len=2268,
                    active_count=3,
                )
            },
        )
        snapshot = [(SimpleNamespace(request_id="long_b"), 320)]

        cohort = hijacker._phase1_build_global_activity_cohort(state, snapshot)

        self.assertIsNotNone(cohort)
        assert cohort is not None
        self.assertEqual(cohort.long_req_id, "long_b")
        self.assertEqual(cohort.long_len, 320)
        self.assertEqual(cohort.short_lengths, [82])
        self.assertGreaterEqual(cohort.total_count, 3)

    def test_phase1_collect_secondary_lora_caps_targets_other_adapter_long(self) -> None:
        class DummyLora:
            def __init__(self, path: str) -> None:
                self.lora_path = path

        class DummySeqGroup:
            def __init__(self, request_id: str, remaining: int, lora_path: str) -> None:
                self.request_id = request_id
                self.num_tokens_with_spec = remaining
                self.num_prompt_tokens = remaining
                self.num_computed_tokens = 0
                self.lora_request = DummyLora(lora_path)

            def is_prefill(self) -> bool:
                return True

        snapshot = [
            (DummySeqGroup("short_a", 22, "/tmp/adapter_a_rank8"), 22),
            (DummySeqGroup("mid_b", 156, "/tmp/adapter_b_rank16"), 156),
            (DummySeqGroup("long_a", 468, "/tmp/adapter_a_rank8"), 468),
            (DummySeqGroup("long_b", 780, "/tmp/adapter_b_rank16"), 780),
        ]
        state = SimpleNamespace(
            policy=WaveSlicePolicy(phase1_ingress_direct_authoritative=True),
            slicer=SimpleNamespace(
                choose_dynamic_chunk=lambda **kwargs: 384
            ),
            brain=None,
            phase1_explicit_plans={},
            phase1_ingress_virtuals={},
        )

        with mock.patch.object(hijacker, "_phase1_explicit_chunk_from_plan", return_value=None), \
             mock.patch.object(hijacker, "_phase1_direct_chunk_candidate", return_value=None), \
             mock.patch.object(hijacker, "_phase1_baseline_chunk_proxy", return_value=None):
            caps = hijacker._phase1_collect_secondary_lora_caps(
                state=state,
                snapshot=snapshot,
                primary_request_id="long_b",
                max_wait_us=0.0,
                queue_len=4,
                scheduler_cfg=None,
                original_budget=None,
                original_threshold=None,
                )

        self.assertEqual(caps, {"long_a": 128})

    def test_phase1_prune_ingress_virtual_caps_keeps_active_provisional_cap(self) -> None:
        state = SimpleNamespace(
            phase1_virtual_token_caps={"long_b": 256, "stale": 128},
            phase1_active_prompt_tokens={"long_b": 2268},
            phase1_ingress_virtuals={},
        )

        pruned = hijacker._phase1_prune_ingress_virtual_caps(state)

        self.assertEqual(pruned, {"long_b": 256})

    def test_v1_schedule_waiting_patch_rewrites_hot_path(self) -> None:
        class DummyRequest:
            def __init__(self, request_id: str) -> None:
                self.request_id = request_id
                self.num_prompt_tokens = 2268
                self.num_output_tokens = 0
                self.num_computed_tokens = 256
                self.num_output_placeholders = 0

            @property
            def num_tokens(self) -> int:
                return 2268

            @property
            def num_tokens_with_spec(self) -> int:
                return 2268

        class DummyScheduler:
            def __init__(self) -> None:
                self.scheduler_config = SimpleNamespace(long_prefill_token_threshold=0)

            def schedule(self, request: DummyRequest, num_computed_tokens: int = 0, token_budget: int = 1536) -> tuple[int, int]:
                running_tokens = (request.num_tokens_with_spec +
                                  request.num_output_placeholders -
                                  request.num_computed_tokens)
                if (0 < self.scheduler_config.long_prefill_token_threshold < running_tokens):
                    running_tokens = self.scheduler_config.long_prefill_token_threshold
                running_tokens = min(running_tokens, token_budget)

                num_new_tokens = request.num_tokens - num_computed_tokens
                if (0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens):
                    num_new_tokens = self.scheduler_config.long_prefill_token_threshold
                num_new_tokens = min(num_new_tokens, token_budget)
                return running_tokens, num_new_tokens

        state = SimpleNamespace(
            phase1_virtual_token_caps={"long_b": 256},
            metrics=WaveSliceMetrics(),
        )
        patched_schedule = hijacker._build_v1_schedule_waiting_patch(
            state,
            DummyScheduler.schedule,
        )

        scheduler = DummyScheduler()
        request = DummyRequest("long_b")
        self.assertEqual(
            patched_schedule(scheduler, request, num_computed_tokens=0, token_budget=1536),
            (1536, 256),
        )
        self.assertEqual(
            patched_schedule(scheduler, DummyRequest("other"), num_computed_tokens=0, token_budget=1536),
            (1536, 1536),
        )

    def test_evaluate_claims_main_smoke_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_json = Path(tmpdir) / "smoke_eval.json"

            def fake_run_series(args, *, mode, **kwargs):
                if mode == "baseline":
                    return [_make_phase1_row(100.0, 1000.0, 0.0) for _ in range(args.repeats)]
                if mode == "phase1_only":
                    return [_make_phase1_row(50.0, 980.0, 0.2) for _ in range(args.repeats)]
                if mode == "baseline_lora_compat":
                    return [_make_phase2_row(100.0, 1000.0, 0.0) for _ in range(args.repeats)]
                if mode in {"phase2_lora", "phase12_lora"}:
                    ttft = 50.0 if mode == "phase12_lora" else 80.0
                    return [_make_phase2_row(ttft, 990.0, 0.4) for _ in range(args.repeats)]
                raise AssertionError(f"unexpected mode: {mode}")

            argv = [
                "evaluate_waveslice_claims.py",
                "--model-path",
                "fake-model",
                "--model-name",
                "fake-model",
                "--requests-json",
                "fake_requests.json",
                "--lora-requests-json",
                "fake_lora_requests.json",
                "--adapter-a",
                "adapter_a",
                "--adapter-b",
                "adapter_b",
                "--repeats",
                "2",
                "--warmup-iters",
                "0",
                "--include-phase12",
                "--out-json",
                str(out_json),
            ]

            with mock.patch.object(claims, "load_reqs_json", side_effect=lambda path: _fake_reqs()), \
                 mock.patch.object(
                     claims,
                     "measure_input_tokens",
                     side_effect=lambda model_path, reqs, trust_remote_code=False: {
                         r.req_id: 16 if r.is_short else 512 for r in reqs
                     },
                 ), \
                 mock.patch.object(claims, "_run_series", side_effect=fake_run_series), \
                 mock.patch.object(claims, "_ensure_eval_adapters", return_value=("adapter_a", "adapter_b")), \
                 mock.patch.object(sys, "argv", argv):
                rc = claims.main()

            self.assertEqual(rc, 0)
            self.assertTrue(out_json.exists())
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["config"]["requests_json"], "fake_requests.json")
            self.assertEqual(payload["config"]["lora_requests_json"], "fake_lora_requests.json")
            self.assertIsNotNone(payload["phase12"]["ttft_improve_ratio"]["mean"])
            self.assertIsNotNone(payload["phase2"]["phase2_apply_ratio"]["mean"])

    def test_inject_wave_slice_smoke_builds_patch_state(self) -> None:
        class FakeScheduler:
            def schedule(self, *args, **kwargs):
                return None

        policy = WaveSlicePolicy(
            enable_phase2_modelrunner=False,
            enable_metrics_hook=False,
            enable_vllm_lora_compat_patch=False,
            enable_v1_runtime_lifecycle_patch=False,
        )

        with mock.patch.object(hijacker, "_load_scheduler_target", return_value=(FakeScheduler, "schedule")), \
             mock.patch.object(hijacker, "_build_scheduler_hook", side_effect=lambda state: state.original_schedule), \
             mock.patch.object(hijacker, "_load_sequence_data_cls", side_effect=RuntimeError("skip")):
            try:
                hijacker.inject_wave_slice("fake-model", gamma=1.0, policy=policy, force=True)
                self.assertTrue(hijacker.is_wave_slice_injected())
            finally:
                hijacker.uninject_wave_slice()

        self.assertFalse(hijacker.is_wave_slice_injected())

    def test_scheduler_hook_phase1_disabled_still_runs_preschedule_cashout(self) -> None:
        seen_waiting = []

        class FakeScheduler:
            def __init__(self) -> None:
                self.running = ["run_a"]
                self.waiting = ["keep_prefill", "hide_prefill"]

            def schedule(self, *args, **kwargs):
                seen_waiting.append(list(self.waiting))
                return SimpleNamespace(scheduled_seq_groups=[])

        scheduler = FakeScheduler()
        state = SimpleNamespace(
            original_schedule=FakeScheduler.schedule,
            scheduler_method_name="schedule",
            policy=WaveSlicePolicy(
                enable_phase1_scheduler=False,
                enable_phase2_modelrunner=True,
                phase2_enable_scheduler_cashout=True,
            ),
            metrics=WaveSliceMetrics(),
        )

        hook = hijacker._build_scheduler_hook(state)

        with mock.patch.object(
            hijacker,
            "_phase12_apply_scheduler_cashout_to_queues",
            side_effect=lambda **kwargs: (
                kwargs["running"],
                ["keep_prefill"],
                [],
                ["hide_prefill"],
                True,
            ),
        ) as prehide_mock, mock.patch.object(
            hijacker,
            "_phase12_scheduler_cashout_rewrite",
            side_effect=lambda **kwargs: (kwargs["scheduler_outputs"], False),
        ):
            outputs = hook(scheduler)

        self.assertEqual(seen_waiting, [["keep_prefill"]])
        self.assertEqual(list(scheduler.waiting), ["keep_prefill", "hide_prefill"])
        self.assertEqual(list(scheduler.running), ["run_a"])
        self.assertEqual(outputs.scheduled_seq_groups, [])
        self.assertEqual(prehide_mock.call_count, 1)

    def test_configure_mode_phase12_propagates_v1_toggles(self) -> None:
        with mock.patch.object(eval_config, "inject_wave_slice") as inject_mock:
            eval_config.configure_mode(
                model_name="fake-model",
                mode="phase12_lora",
                queue_reorder_mode="sjf",
                queue_reorder_aging_quantum_us=20_000.0,
                phase2_dispatch_mode="synchronized",
                phase1_objective_mode="fair_escape",
                phase1_gamma=2.0,
                phase1_ingress_target_chunk=384,
                phase1_ingress_direct_authoritative=False,
                phase1_ingress_exact_chunk=False,
                phase12_phase2_gate_mode="soft",
                phase12_phase2_soft_ratio_scale=1.15,
                phase12_phase2_soft_pressure_scale=1.10,
                phase12_phase2_soft_min_long_prefill=512,
                phase12_phase2_soft_allow_mixed_decode=True,
                phase12_phase2_soft_recent_strength_floor=0.08,
                phase12_phase2_soft_require_cashout_signal=True,
                phase12_phase2_soft_recent_chunk_match_scale=1.5,
                phase12_phase2_soft_window_score_threshold=0.95,
                phase12_phase2_soft_window_recent_weight=0.40,
                phase12_phase2_soft_window_chunk_weight=0.25,
                phase12_phase2_soft_window_pressure_weight=0.20,
                phase12_phase2_soft_window_ratio_weight=0.10,
                phase12_phase2_soft_window_decode_bonus=0.10,
                phase12_phase2_scheduler_cashout_soft_floor=0.55,
                phase12_phase2_scheduler_cashout_quality_floor=0.78,
                phase12_phase2_scheduler_cashout_cooldown_ticks=2,
                phase2_enable_scheduler_cashout=True,
                phase2_enable_execution_escape=False,
                phase2_enable_v1_true_unbind=True,
            )

        policy = inject_mock.call_args.kwargs["policy"]
        self.assertTrue(policy.enable_phase2_modelrunner)
        self.assertTrue(policy.phase2_enable_scheduler_cashout)
        self.assertFalse(policy.phase2_enable_execution_escape)
        self.assertTrue(policy.phase2_enable_v1_true_unbind)
        self.assertFalse(policy.enable_tick_hide)
        self.assertTrue(policy.allow_phase1_tick_hide_with_lora)

    def test_configure_mode_phase12_enables_tick_hide_when_phase2_paths_disabled(self) -> None:
        with mock.patch.object(eval_config, "inject_wave_slice") as inject_mock:
            eval_config.configure_mode(
                model_name="fake-model",
                mode="phase12_lora",
                queue_reorder_mode="sjf",
                queue_reorder_aging_quantum_us=20_000.0,
                phase2_dispatch_mode="synchronized",
                phase1_objective_mode="fair_escape",
                phase1_gamma=2.0,
                phase1_ingress_target_chunk=128,
                phase1_ingress_direct_authoritative=True,
                phase1_ingress_exact_chunk=True,
                phase12_phase2_gate_mode="soft",
                phase12_phase2_soft_ratio_scale=1.15,
                phase12_phase2_soft_pressure_scale=1.10,
                phase12_phase2_soft_min_long_prefill=512,
                phase12_phase2_soft_allow_mixed_decode=True,
                phase12_phase2_soft_recent_strength_floor=0.08,
                phase12_phase2_soft_require_cashout_signal=True,
                phase12_phase2_soft_recent_chunk_match_scale=1.5,
                phase12_phase2_soft_window_score_threshold=0.95,
                phase12_phase2_soft_window_recent_weight=0.40,
                phase12_phase2_soft_window_chunk_weight=0.25,
                phase12_phase2_soft_window_pressure_weight=0.20,
                phase12_phase2_soft_window_ratio_weight=0.10,
                phase12_phase2_soft_window_decode_bonus=0.10,
                phase12_phase2_scheduler_cashout_soft_floor=0.55,
                phase12_phase2_scheduler_cashout_quality_floor=0.78,
                phase12_phase2_scheduler_cashout_cooldown_ticks=2,
                phase2_enable_scheduler_cashout=False,
                phase2_enable_execution_escape=False,
                phase2_enable_v1_true_unbind=False,
            )

        policy = inject_mock.call_args.kwargs["policy"]
        self.assertTrue(policy.enable_tick_hide)
        self.assertTrue(policy.allow_phase1_tick_hide_with_lora)

    def test_phase2_debug_summary_reports_v1_probes(self) -> None:
        metrics = WaveSliceMetrics()
        metrics.record_phase2_debug_counter("execution_escape_v1_output_hits")
        metrics.record_phase2_debug_counter("execution_escape_v1_output_misses", amount=2)
        metrics.record_phase2_true_unbind_gate("non_v1_scheduler_output")
        metrics.record_phase2_v1_unbind()

        report = metrics.summary()
        phase2 = report["phase2"]
        debug = phase2["debug"]

        self.assertEqual(phase2["v1_true_unbind_applied"], 1)
        self.assertEqual(debug["counters"]["execution_escape_v1_output_hits"], 1)
        self.assertEqual(debug["counters"]["execution_escape_v1_output_misses"], 2)
        self.assertEqual(
            debug["true_unbind_gate_reasons"]["non_v1_scheduler_output"],
            1,
        )

    def test_merge_cross_process_metrics_includes_phase1_detail_events(self) -> None:
        base_report = {
            "scheduler": {
                "attempts": 0,
                "applied": 0,
                "apply_ratio": 0.0,
                "baseline_chunk_avg": None,
                "chosen_chunk_avg": None,
                "chosen_vs_baseline_ratio_avg": None,
                "explicit_plan_ratio": 0.0,
                "rewrite_applied": 0,
                "rewrite_apply_ratio": 0.0,
                "rewrite_group_count": 0,
                "rewrite_old_chunk_avg": None,
                "rewrite_new_chunk_avg": None,
                "rewrite_token_delta_avg": None,
                "virtual_cap_apply_ratio": 0.0,
                "virtual_cap_old_avg": None,
                "virtual_cap_new_avg": None,
                "virtual_cap_target_set": 0.0,
                "virtual_cap_helper_calls": 0.0,
                "virtual_cap_prefill_calls": 0.0,
                "virtual_cap_target_hits": 0.0,
            },
            "phase2": {"attempts": 0, "applied": 0, "reasons": {}, "debug": {}, "escape_lane": {}},
        }
        records = [
            {"kind": "scheduler_decision", "payload": {"applied": True}},
            {"kind": "phase1_choice", "payload": {"baseline_chunk": 1536, "chosen_chunk": 384, "explicit_plan": True}},
            {"kind": "phase1_probe", "payload": {"reason": "apply", "short_len": 96, "long_len": 1536, "baseline_chunk": 1536, "best_chunk": 384, "queue_len": 3, "wait_us": 25.0, "slice_eligible": True}},
            {"kind": "phase1_proposal", "payload": {"scheduler_chunk": 512, "direct_chunk": 384, "cohort_target": 320, "direct_won": True}},
            {"kind": "phase1_rewrite", "payload": {"rewritten_groups": 1, "old_chunk_sum": 1536, "new_chunk_sum": 384, "token_delta_sum": 1152}},
            {"kind": "phase1_virtual_cap_probe", "payload": {"target_set": True, "helper_called": True, "prefill_call": True, "target_hit": True}},
            {"kind": "phase1_virtual_cap", "payload": {"old_total_tokens": 1536, "new_total_tokens": 384, "applied": True}},
            {"kind": "phase1_step_trace", "payload": {"request_id": "long_b", "event": "phase1_target_set", "is_prefill": True, "num_computed_tokens": 512, "uncached": 1024, "cached": 0, "target_chunk": 384}},
        ]

        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as fh:
            metrics_path = fh.name
            for rec in records:
                fh.write(json.dumps(rec) + "\n")
        try:
            with mock.patch.dict("os.environ", {"WAVESLICE_AUTOINJECT_METRICS_FILE": metrics_path}, clear=False):
                merged = merge_cross_process_metrics(base_report)
        finally:
            Path(metrics_path).unlink(missing_ok=True)

        scheduler = merged["scheduler"]
        self.assertEqual(scheduler["attempts"], 1)
        self.assertEqual(scheduler["applied"], 1)
        self.assertEqual(scheduler["baseline_chunk_avg"], 1536.0)
        self.assertEqual(scheduler["chosen_chunk_avg"], 384.0)
        self.assertEqual(scheduler["chosen_vs_baseline_ratio_avg"], 0.25)
        self.assertEqual(scheduler["explicit_plan_ratio"], 1.0)
        self.assertEqual(scheduler["rewrite_applied"], 1)
        self.assertEqual(scheduler["rewrite_old_chunk_avg"], 1536.0)
        self.assertEqual(scheduler["rewrite_new_chunk_avg"], 384.0)
        self.assertEqual(scheduler["rewrite_token_delta_avg"], 1152.0)
        self.assertEqual(scheduler["virtual_cap_apply_ratio"], 1.0)
        self.assertEqual(scheduler["virtual_cap_old_avg"], 1536.0)
        self.assertEqual(scheduler["virtual_cap_new_avg"], 384.0)
        self.assertEqual(scheduler["virtual_cap_target_set"], 1.0)
        self.assertEqual(scheduler["virtual_cap_helper_calls"], 1.0)
        self.assertEqual(scheduler["virtual_cap_prefill_calls"], 1.0)
        self.assertEqual(scheduler["virtual_cap_target_hits"], 1.0)
        self.assertEqual(scheduler["probe_total"], 1.0)
        self.assertEqual(scheduler["probe_slice_eligible_ratio"], 1.0)
        self.assertEqual(scheduler["probe_best_lt_long_ratio"], 1.0)
        self.assertEqual(scheduler["probe_short_avg"], 96.0)
        self.assertEqual(scheduler["probe_long_avg"], 1536.0)
        self.assertEqual(scheduler["probe_baseline_avg"], 1536.0)
        self.assertEqual(scheduler["probe_best_avg"], 384.0)
        self.assertEqual(scheduler["probe_queue_avg"], 3.0)
        self.assertEqual(scheduler["probe_wait_us_avg"], 25.0)
        self.assertEqual(scheduler["proposal_scheduler_avg"], 512.0)
        self.assertEqual(scheduler["proposal_direct_avg"], 384.0)
        self.assertEqual(scheduler["proposal_cohort_target_avg"], 320.0)
        self.assertEqual(scheduler["proposal_direct_win_ratio"], 1.0)
        self.assertEqual(scheduler["probe_reasons"]["apply"], 1)
        self.assertEqual(scheduler["request_traces"]["long_b"][0]["event"], "phase1_target_set")
        self.assertEqual(scheduler["request_traces"]["long_b"][0]["num_computed_tokens"], 512)


if __name__ == "__main__":
    unittest.main()
