from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict
from typing import Any

from engine.hijack.runtime_state import (
    AUTO_ENV_METRICS_FILE,
    WaveSlicePolicy,
)

logger = logging.getLogger("WaveSlice")
logger.addHandler(logging.NullHandler())

AUTO_ENV_ENABLED = "WAVESLICE_AUTOINJECT_ENABLED"
AUTO_ENV_MODEL = "WAVESLICE_AUTOINJECT_MODEL_NAME"
AUTO_ENV_GAMMA = "WAVESLICE_AUTOINJECT_GAMMA"
AUTO_ENV_POLICY = "WAVESLICE_AUTOINJECT_POLICY_JSON"
AUTO_ENV_PREV_PYTHONPATH = "WAVESLICE_AUTOINJECT_PREV_PYTHONPATH"
AUTO_ENV_PREV_VLLM_PLUGINS = "WAVESLICE_AUTOINJECT_PREV_VLLM_PLUGINS"


def publish_autoinject_env(model_name: str, gamma: float, policy: WaveSlicePolicy) -> None:
    try:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        prev_pythonpath = os.environ.get("PYTHONPATH")
        prev_vllm_plugins = os.environ.get("VLLM_PLUGINS")
        metrics_file = ensure_cross_process_metrics_file()
        os.environ[AUTO_ENV_ENABLED] = "1"
        os.environ[AUTO_ENV_MODEL] = str(model_name)
        os.environ[AUTO_ENV_GAMMA] = str(float(gamma))
        os.environ[AUTO_ENV_POLICY] = json.dumps(asdict(policy), sort_keys=True)
        os.environ[AUTO_ENV_METRICS_FILE] = metrics_file
        os.environ[AUTO_ENV_PREV_PYTHONPATH] = prev_pythonpath if prev_pythonpath is not None else ""
        os.environ[AUTO_ENV_PREV_VLLM_PLUGINS] = prev_vllm_plugins if prev_vllm_plugins is not None else ""

        py_entries = [p for p in (prev_pythonpath or "").split(os.pathsep) if p]
        if repo_root not in py_entries:
            py_entries.insert(0, repo_root)
        os.environ["PYTHONPATH"] = os.pathsep.join(py_entries)

        plugin_entries = [p.strip() for p in (prev_vllm_plugins or "").split(",") if p.strip()]
        if "waveslice_autoinject" not in plugin_entries:
            plugin_entries.append("waveslice_autoinject")
        os.environ["VLLM_PLUGINS"] = ",".join(plugin_entries)
    except Exception:
        logger.exception("[Wave-Slice] failed to publish child auto-inject env.")


def clear_autoinject_env() -> None:
    prev_pythonpath = os.environ.pop(AUTO_ENV_PREV_PYTHONPATH, None)
    prev_vllm_plugins = os.environ.pop(AUTO_ENV_PREV_VLLM_PLUGINS, None)
    if prev_pythonpath is not None:
        if prev_pythonpath:
            os.environ["PYTHONPATH"] = prev_pythonpath
        else:
            os.environ.pop("PYTHONPATH", None)
    if prev_vllm_plugins is not None:
        if prev_vllm_plugins:
            os.environ["VLLM_PLUGINS"] = prev_vllm_plugins
        else:
            os.environ.pop("VLLM_PLUGINS", None)
    for key in (
        AUTO_ENV_ENABLED,
        AUTO_ENV_MODEL,
        AUTO_ENV_GAMMA,
        AUTO_ENV_POLICY,
        AUTO_ENV_METRICS_FILE,
    ):
        os.environ.pop(key, None)


def ensure_cross_process_metrics_file() -> str:
    path = os.environ.get(AUTO_ENV_METRICS_FILE, "").strip()
    if path:
        return path
    fd, path = tempfile.mkstemp(prefix="waveslice_metrics_", suffix=".jsonl", dir="/tmp")
    os.close(fd)
    os.environ[AUTO_ENV_METRICS_FILE] = path
    return path


def reset_cross_process_metrics_file() -> None:
    path = ensure_cross_process_metrics_file()
    try:
        with open(path, "w", encoding="utf-8"):
            pass
    except Exception:
        logger.exception("[Wave-Slice] failed to reset cross-process metrics file.")


def merge_cross_process_metrics(report: dict[str, Any]) -> dict[str, Any]:
    path = os.environ.get(AUTO_ENV_METRICS_FILE, "").strip()
    if not path or not os.path.exists(path):
        return report
    try:
        with open(path, "r", encoding="utf-8") as fh:
            lines = [line.strip() for line in fh if line.strip()]
    except Exception:
        logger.exception("[Wave-Slice] failed to read cross-process metrics file.")
        return report

    if not lines:
        return report

    scheduler = dict(report.get("scheduler") or {})
    phase2 = dict(report.get("phase2") or {})
    reasons = dict(phase2.get("reasons") or {})
    escape_lane = dict(phase2.get("escape_lane") or {})
    debug = dict(phase2.get("debug") or {})

    sched_total = int(scheduler.get("attempts") or 0)
    sched_applied = int(scheduler.get("applied") or 0)
    phase2_total = int(phase2.get("attempts") or 0)
    phase2_applied = int(phase2.get("applied") or 0)
    lane_activations = int(escape_lane.get("activations") or 0)
    lane_active_sum = 0.0
    lane_active_count = 0
    lane_deferred_sum = 0.0
    lane_deferred_count = 0
    lane_ttl_sum = 0.0
    lane_ttl_count = 0
    lane_seen_events = int(escape_lane.get("seen_events") or 0)
    lane_seen_active_hits = int(escape_lane.get("seen_active_hits") or 0)
    lane_finished_events = int(escape_lane.get("finished_events") or 0)
    lane_finished_active_hits = int(escape_lane.get("finished_active_hits") or 0)
    lane_last_active_ids = list(escape_lane.get("last_active_ids") or [])
    lane_last_deferred_ids = list(escape_lane.get("last_deferred_ids") or [])
    current_bridge_active_ids: set[str] = set(lane_last_active_ids)
    schedule_hook_enter = int(debug.get("schedule_hook_enter") or 0)
    phase2_sched_pre_enter = int(debug.get("phase2_sched_pre_enter") or 0)
    phase2_sched_post_enter = int(debug.get("phase2_sched_post_enter") or 0)
    phase1_public_rewrite_applied = int(debug.get("phase1_public_rewrite_applied") or 0)
    schedule_hook_early_reasons = dict(debug.get("schedule_hook_early_reasons") or {})
    no_need_count_sum = float((debug.get("no_need_wave_slice_summary") or {}).get("count_avg") or 0.0)
    no_need_min_sum = float((debug.get("no_need_wave_slice_summary") or {}).get("min_len_avg") or 0.0)
    no_need_max_sum = float((debug.get("no_need_wave_slice_summary") or {}).get("max_len_avg") or 0.0)
    no_need_ratio_sum = float((debug.get("no_need_wave_slice_summary") or {}).get("hetero_ratio_avg") or 0.0)
    no_need_stats_count = 1 if "no_need_wave_slice_summary" in debug else 0
    low_quality_selected_sum = float((debug.get("scheduler_cashout_low_quality_summary") or {}).get("selected_quality_avg") or 0.0)
    low_quality_floor_sum = float((debug.get("scheduler_cashout_low_quality_summary") or {}).get("quality_floor_avg") or 0.0)
    low_quality_count_sum = float((debug.get("scheduler_cashout_low_quality_summary") or {}).get("selected_count_avg") or 0.0)
    low_quality_value_sum = float((debug.get("scheduler_cashout_low_quality_summary") or {}).get("value_score_avg") or 0.0)
    low_quality_net_sum = float((debug.get("scheduler_cashout_low_quality_summary") or {}).get("net_value_avg") or 0.0)
    low_quality_gain_sum = float((debug.get("scheduler_cashout_low_quality_summary") or {}).get("gain_score_avg") or 0.0)
    low_quality_cost_sum = float((debug.get("scheduler_cashout_low_quality_summary") or {}).get("cost_score_avg") or 0.0)
    low_quality_wait_gap_sum = float((debug.get("scheduler_cashout_low_quality_summary") or {}).get("wait_gap_avg") or 0.0)
    low_quality_candidate_wait_sum = float((debug.get("scheduler_cashout_low_quality_summary") or {}).get("candidate_wait_quality_avg") or 0.0)
    low_quality_candidate_size_sum = float((debug.get("scheduler_cashout_low_quality_summary") or {}).get("candidate_size_quality_avg") or 0.0)
    low_quality_candidate_shape_sum = float((debug.get("scheduler_cashout_low_quality_summary") or {}).get("candidate_shape_penalty_avg") or 0.0)
    low_quality_small_bonus_sum = float((debug.get("scheduler_cashout_low_quality_summary") or {}).get("small_candidate_bonus_avg") or 0.0)
    low_quality_medium_penalty_sum = float((debug.get("scheduler_cashout_low_quality_summary") or {}).get("medium_candidate_penalty_avg") or 0.0)
    low_quality_stats_count = 1 if "scheduler_cashout_low_quality_summary" in debug else 0
    apply_selected_sum = float((debug.get("scheduler_cashout_apply_summary") or {}).get("selected_quality_avg") or 0.0)
    apply_floor_sum = float((debug.get("scheduler_cashout_apply_summary") or {}).get("quality_floor_avg") or 0.0)
    apply_count_sum = float((debug.get("scheduler_cashout_apply_summary") or {}).get("selected_count_avg") or 0.0)
    apply_strength_sum = float((debug.get("scheduler_cashout_apply_summary") or {}).get("strength_avg") or 0.0)
    apply_value_sum = float((debug.get("scheduler_cashout_apply_summary") or {}).get("value_score_avg") or 0.0)
    apply_net_sum = float((debug.get("scheduler_cashout_apply_summary") or {}).get("net_value_avg") or 0.0)
    apply_gain_sum = float((debug.get("scheduler_cashout_apply_summary") or {}).get("gain_score_avg") or 0.0)
    apply_cost_sum = float((debug.get("scheduler_cashout_apply_summary") or {}).get("cost_score_avg") or 0.0)
    apply_wait_gap_sum = float((debug.get("scheduler_cashout_apply_summary") or {}).get("wait_gap_avg") or 0.0)
    apply_candidate_wait_sum = float((debug.get("scheduler_cashout_apply_summary") or {}).get("candidate_wait_quality_avg") or 0.0)
    apply_candidate_size_sum = float((debug.get("scheduler_cashout_apply_summary") or {}).get("candidate_size_quality_avg") or 0.0)
    apply_candidate_shape_sum = float((debug.get("scheduler_cashout_apply_summary") or {}).get("candidate_shape_penalty_avg") or 0.0)
    apply_small_bonus_sum = float((debug.get("scheduler_cashout_apply_summary") or {}).get("small_candidate_bonus_avg") or 0.0)
    apply_medium_penalty_sum = float((debug.get("scheduler_cashout_apply_summary") or {}).get("medium_candidate_penalty_avg") or 0.0)
    apply_stats_count = 1 if "scheduler_cashout_apply_summary" in debug else 0
    execution_escape_exception_types = dict(debug.get("execution_escape_exception_types") or {})
    execution_escape_exception_messages = dict(debug.get("execution_escape_exception_messages") or {})

    for line in lines:
        try:
            rec = json.loads(line)
        except Exception:
            continue
        kind = str(rec.get("kind") or "")
        payload = rec.get("payload") or {}
        if kind == "scheduler_decision":
            sched_total += 1
            if bool(payload.get("applied")):
                sched_applied += 1
        elif kind == "phase2_decision":
            phase2_total += 1
            if bool(payload.get("applied")):
                phase2_applied += 1
            reason = str(payload.get("reason") or "")
            if reason:
                reasons[reason] = int(reasons.get(reason, 0)) + 1
            if reason == "execution_escape_exception":
                exc_type = str(payload.get("exception_type") or "")
                exc_msg = str(payload.get("exception_message") or "")
                if exc_type:
                    execution_escape_exception_types[exc_type] = int(execution_escape_exception_types.get(exc_type, 0)) + 1
                if exc_msg:
                    execution_escape_exception_messages[exc_msg] = int(execution_escape_exception_messages.get(exc_msg, 0)) + 1
            if reason == "scheduler_cashout_low_quality":
                low_quality_selected_sum += float(payload.get("selected_quality") or 0.0)
                low_quality_floor_sum += float(payload.get("quality_floor") or 0.0)
                low_quality_count_sum += float(payload.get("selected_count") or 0.0)
                low_quality_value_sum += float(payload.get("value_score") or 0.0)
                low_quality_net_sum += float(payload.get("net_value") or 0.0)
                low_quality_gain_sum += float(payload.get("gain_score") or 0.0)
                low_quality_cost_sum += float(payload.get("cost_score") or 0.0)
                low_quality_wait_gap_sum += float(payload.get("wait_gap") or 0.0)
                low_quality_candidate_wait_sum += float(payload.get("candidate_wait_quality") or 0.0)
                low_quality_candidate_size_sum += float(payload.get("candidate_size_quality") or 0.0)
                low_quality_candidate_shape_sum += float(payload.get("candidate_shape_penalty") or 0.0)
                low_quality_small_bonus_sum += float(payload.get("small_candidate_bonus") or 0.0)
                low_quality_medium_penalty_sum += float(payload.get("medium_candidate_penalty") or 0.0)
                low_quality_stats_count += 1
            elif reason == "scheduler_cashout_beneficiary":
                apply_selected_sum += float(payload.get("selected_quality") or 0.0)
                apply_floor_sum += float(payload.get("quality_floor") or 0.0)
                apply_count_sum += float(payload.get("selected_count") or 0.0)
                apply_strength_sum += float(payload.get("strength") or 0.0)
                apply_value_sum += float(payload.get("value_score") or 0.0)
                apply_net_sum += float(payload.get("net_value") or 0.0)
                apply_gain_sum += float(payload.get("gain_score") or 0.0)
                apply_cost_sum += float(payload.get("cost_score") or 0.0)
                apply_wait_gap_sum += float(payload.get("wait_gap") or 0.0)
                apply_candidate_wait_sum += float(payload.get("candidate_wait_quality") or 0.0)
                apply_candidate_size_sum += float(payload.get("candidate_size_quality") or 0.0)
                apply_candidate_shape_sum += float(payload.get("candidate_shape_penalty") or 0.0)
                apply_small_bonus_sum += float(payload.get("small_candidate_bonus") or 0.0)
                apply_medium_penalty_sum += float(payload.get("medium_candidate_penalty") or 0.0)
                apply_stats_count += 1
        elif kind == "escape_lane_activation":
            lane_activations += 1
            lane_active_sum += float(payload.get("active_count") or 0.0)
            lane_active_count += 1
            lane_deferred_sum += float(payload.get("deferred_count") or 0.0)
            lane_deferred_count += 1
            lane_ttl_sum += float(payload.get("lane_ttl") or 0.0)
            lane_ttl_count += 1
            lane_last_active_ids = list(payload.get("active_ids") or [])[:16]
            lane_last_deferred_ids = list(payload.get("deferred_ids") or [])[:16]
            current_bridge_active_ids = {str(rid) for rid in lane_last_active_ids if str(rid)}
        elif kind == "escape_lane_observation":
            explicit_active_ids = {str(rid) for rid in (payload.get("active_ids") or []) if str(rid)}
            active_for_match = explicit_active_ids or current_bridge_active_ids
            seen_ids = [str(rid) for rid in (payload.get("seen_ids") or []) if str(rid)]
            finished_ids = [str(rid) for rid in (payload.get("finished_ids") or []) if str(rid)]
            lane_seen_events += 1
            lane_seen_active_hits += sum(1 for rid in seen_ids if rid in active_for_match)
            finished_count = int(payload.get("finished_count") or 0)
            if finished_count > 0:
                lane_finished_events += 1
                finished_hits = sum(1 for rid in finished_ids if rid in active_for_match)
                lane_finished_active_hits += finished_hits
                if active_for_match and finished_hits > 0:
                    current_bridge_active_ids.difference_update({rid for rid in finished_ids if rid in active_for_match})
        elif kind == "schedule_hook_enter":
            schedule_hook_enter += 1
        elif kind == "phase2_sched_pre_enter":
            phase2_sched_pre_enter += 1
        elif kind == "phase2_sched_post_enter":
            phase2_sched_post_enter += 1
        elif kind == "phase1_public_rewrite_applied":
            phase1_public_rewrite_applied += 1
        elif kind == "schedule_hook_early_return":
            reason = str(payload.get("reason") or "")
            if reason:
                schedule_hook_early_reasons[reason] = int(schedule_hook_early_reasons.get(reason, 0)) + 1
            if reason == "no_need_wave_slice":
                no_need_count_sum += float(payload.get("snapshot_count") or 0.0)
                no_need_min_sum += float(payload.get("min_len") or 0.0)
                no_need_max_sum += float(payload.get("max_len") or 0.0)
                no_need_ratio_sum += float(payload.get("hetero_ratio") or 0.0)
                no_need_stats_count += 1

    scheduler["attempts"] = sched_total
    scheduler["applied"] = sched_applied
    scheduler["apply_ratio"] = (float(sched_applied) / float(sched_total)) if sched_total else 0.0

    phase2["attempts"] = phase2_total
    phase2["applied"] = phase2_applied
    phase2["apply_ratio"] = (float(phase2_applied) / float(phase2_total)) if phase2_total else 0.0
    phase2["reasons"] = reasons

    escape_lane["activations"] = lane_activations
    escape_lane["active_count_avg"] = lane_active_sum / float(lane_active_count) if lane_active_count else escape_lane.get("active_count_avg")
    escape_lane["deferred_count_avg"] = lane_deferred_sum / float(lane_deferred_count) if lane_deferred_count else escape_lane.get("deferred_count_avg")
    escape_lane["ttl_avg"] = lane_ttl_sum / float(lane_ttl_count) if lane_ttl_count else escape_lane.get("ttl_avg")
    escape_lane["seen_events"] = lane_seen_events
    escape_lane["seen_active_hits"] = lane_seen_active_hits
    escape_lane["seen_active_hits_per_event"] = float(lane_seen_active_hits) / float(lane_seen_events) if lane_seen_events else None
    escape_lane["finished_events"] = lane_finished_events
    escape_lane["finished_active_hits"] = lane_finished_active_hits
    escape_lane["finished_active_hits_per_event"] = float(lane_finished_active_hits) / float(lane_finished_events) if lane_finished_events else None
    escape_lane["last_active_ids"] = lane_last_active_ids
    escape_lane["last_deferred_ids"] = lane_last_deferred_ids
    phase2["escape_lane"] = escape_lane
    phase2["debug"] = {
        "schedule_hook_enter": schedule_hook_enter,
        "phase2_sched_pre_enter": phase2_sched_pre_enter,
        "phase2_sched_post_enter": phase2_sched_post_enter,
        "phase1_public_rewrite_applied": phase1_public_rewrite_applied,
        "schedule_hook_early_reasons": schedule_hook_early_reasons,
        "no_need_wave_slice_summary": (
            {
                "count_avg": (no_need_count_sum / float(no_need_stats_count)),
                "min_len_avg": (no_need_min_sum / float(no_need_stats_count)),
                "max_len_avg": (no_need_max_sum / float(no_need_stats_count)),
                "hetero_ratio_avg": (no_need_ratio_sum / float(no_need_stats_count)),
            }
            if no_need_stats_count
            else {}
        ),
        "scheduler_cashout_low_quality_summary": (
            {
                "selected_quality_avg": (low_quality_selected_sum / float(low_quality_stats_count)),
                "quality_floor_avg": (low_quality_floor_sum / float(low_quality_stats_count)),
                "selected_count_avg": (low_quality_count_sum / float(low_quality_stats_count)),
                "value_score_avg": (low_quality_value_sum / float(low_quality_stats_count)),
                "net_value_avg": (low_quality_net_sum / float(low_quality_stats_count)),
                "gain_score_avg": (low_quality_gain_sum / float(low_quality_stats_count)),
                "cost_score_avg": (low_quality_cost_sum / float(low_quality_stats_count)),
                "wait_gap_avg": (low_quality_wait_gap_sum / float(low_quality_stats_count)),
                "candidate_wait_quality_avg": (low_quality_candidate_wait_sum / float(low_quality_stats_count)),
                "candidate_size_quality_avg": (low_quality_candidate_size_sum / float(low_quality_stats_count)),
                "candidate_shape_penalty_avg": (low_quality_candidate_shape_sum / float(low_quality_stats_count)),
                "small_candidate_bonus_avg": (low_quality_small_bonus_sum / float(low_quality_stats_count)),
                "medium_candidate_penalty_avg": (low_quality_medium_penalty_sum / float(low_quality_stats_count)),
            }
            if low_quality_stats_count
            else {}
        ),
        "scheduler_cashout_apply_summary": (
            {
                "selected_quality_avg": (apply_selected_sum / float(apply_stats_count)),
                "quality_floor_avg": (apply_floor_sum / float(apply_stats_count)),
                "selected_count_avg": (apply_count_sum / float(apply_stats_count)),
                "strength_avg": (apply_strength_sum / float(apply_stats_count)),
                "value_score_avg": (apply_value_sum / float(apply_stats_count)),
                "net_value_avg": (apply_net_sum / float(apply_stats_count)),
                "gain_score_avg": (apply_gain_sum / float(apply_stats_count)),
                "cost_score_avg": (apply_cost_sum / float(apply_stats_count)),
                "wait_gap_avg": (apply_wait_gap_sum / float(apply_stats_count)),
                "candidate_wait_quality_avg": (apply_candidate_wait_sum / float(apply_stats_count)),
                "candidate_size_quality_avg": (apply_candidate_size_sum / float(apply_stats_count)),
                "candidate_shape_penalty_avg": (apply_candidate_shape_sum / float(apply_stats_count)),
                "small_candidate_bonus_avg": (apply_small_bonus_sum / float(apply_stats_count)),
                "medium_candidate_penalty_avg": (apply_medium_penalty_sum / float(apply_stats_count)),
            }
            if apply_stats_count
            else {}
        ),
        "execution_escape_exception_types": execution_escape_exception_types,
        "execution_escape_exception_messages": execution_escape_exception_messages,
    }

    report["scheduler"] = scheduler
    report["phase2"] = phase2
    return report
