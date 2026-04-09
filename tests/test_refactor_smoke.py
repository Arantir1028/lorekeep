from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = ROOT / "tests"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

import evaluate_waveslice_claims as claims
from engine.hijack.policy import WaveSlicePolicy
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


if __name__ == "__main__":
    unittest.main()
