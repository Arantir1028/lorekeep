"""Model-free scheduler scenarios shared by developers and contract tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from waveslice.policy import WaveSlicePolicy
from waveslice.vllm.phase1_math import need_wave_slice
from waveslice.vllm.phase1_selection import phase1_live_cohort_from_snapshot
from waveslice.vllm.phase2_beneficiaries import phase2_beneficiary_signal
from waveslice.vllm.state import ScheduledRequestInfo

SCENARIOS = json.loads(
    (Path(__file__).parent / "fixtures" / "scheduler_scenarios.json").read_text(encoding="utf-8")
)


def test_phase1_scheduler_scenarios() -> None:
    for scenario in SCENARIOS["phase1"]:
        policy = WaveSlicePolicy(**scenario["policy"])
        snapshot = [
            (SimpleNamespace(request_id=item["request_id"]), item["remaining_tokens"])
            for item in scenario["requests"]
        ]
        lengths = [remaining for _, remaining in snapshot]
        expected = scenario["expected"]
        assert need_wave_slice(lengths, policy) is expected["eligible"], scenario["name"]
        if not expected["eligible"]:
            continue
        cohort = phase1_live_cohort_from_snapshot(
            snapshot, request_id_getter=lambda request: request.request_id
        )
        assert cohort is not None
        assert cohort.representative_short_len == expected["representative_short_len"]
        assert cohort.long_len == expected["long_len"]
        assert cohort.long_req_id == expected["long_request_id"]


def test_phase2_scheduler_scenarios() -> None:
    for scenario in SCENARIOS["phase2"]:
        signal = phase2_beneficiary_signal(
            policy=WaveSlicePolicy(**scenario["policy"]),
            req_infos=[ScheduledRequestInfo(**item) for item in scenario["requests"]],
        )
        expected = scenario["expected"]
        assert signal.long_anchor_id == expected["anchor_request_id"], scenario["name"]
        assert signal.beneficiary_selected_ids == expected["beneficiary_request_ids"]
