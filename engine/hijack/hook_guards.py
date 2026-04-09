from __future__ import annotations

from typing import Any


def has_running_waiting_queues(obj: Any) -> bool:
    return hasattr(obj, "running") and hasattr(obj, "waiting")


def phase2_scheduler_cashout_enabled(policy: Any) -> bool:
    return bool(getattr(policy, "phase2_enable_scheduler_cashout", False))


def phase2_modelrunner_passthrough(policy: Any) -> bool:
    return (
        phase2_scheduler_cashout_enabled(policy)
        and not bool(getattr(policy, "phase2_enable_v1_true_unbind", False))
        and not bool(getattr(policy, "phase2_enable_execution_escape", False))
    )
