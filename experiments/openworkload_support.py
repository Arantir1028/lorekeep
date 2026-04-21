from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Optional


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            quoting=csv.QUOTE_MINIMAL,
            escapechar="\\",
        )
        writer.writeheader()
        writer.writerows(rows)


def workload_meta_matches_model(
    *,
    meta_path: Path,
    model: Any,
    model_path: str,
    local_snapshot: Optional[str],
    density: dict[str, Any],
    workload_cfg: dict[str, Any],
    require_density_match: bool = True,
) -> bool:
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    allowed_paths = {str(model.model_id), str(model_path)}
    if local_snapshot:
        allowed_paths.add(str(local_snapshot))
    if str(meta.get("model_path")) not in allowed_paths:
        return False
    if bool(meta.get("trust_remote_code", False)) != bool(model.trust_remote_code):
        return False
    if str(meta.get("arrival_mode")) != str(workload_cfg.get("arrival_mode", "poisson")):
        return False
    if str(meta.get("phase1_arrival_layout")) != str(workload_cfg.get("phase1_arrival_layout", "beneficiary_rich")):
        return False
    if str(meta.get("phase2_arrival_layout")) != str(workload_cfg.get("phase2_arrival_layout", "beneficiary_rich")):
        return False
    if require_density_match:
        if float(meta.get("phase1_arrival_rate", -1)) != float(density["phase1_arrival_rate"]):
            return False
        if float(meta.get("phase2_arrival_rate", -1)) != float(density["phase2_arrival_rate"]):
            return False
    return True


def load_existing_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


def completed_case_keys(rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for row in rows:
        if str(row.get("status", "")).strip().lower() != "ok":
            continue
        density = str(row.get("density") or "").strip()
        model_key = str(row.get("model_key") or "").strip()
        if density and model_key:
            keys.add((density, model_key))
    return keys


def float_or_none(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def list_gpu_compute_pids() -> list[int]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return []
    if proc.returncode != 0:
        return []
    pids: list[int] = []
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if not line or line.lower().startswith("no running"):
            continue
        try:
            pids.append(int(line))
        except Exception:
            continue
    return sorted(set(pids))


def clear_gpu_lock(gpu_lock_path: Path) -> None:
    try:
        gpu_lock_path.unlink(missing_ok=True)
    except Exception:
        pass


def list_waveslice_processes(
    *,
    experiment_proc_patterns: tuple[str, ...],
) -> list[dict[str, Any]]:
    try:
        proc = subprocess.run(
            ["ps", "-eo", "pid=,ppid=,pgid=,args="],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return []
    if proc.returncode != 0:
        return []
    rows: list[dict[str, Any]] = []
    current_pid = os.getpid()
    current_ppid = os.getppid()
    for line in (proc.stdout or "").splitlines():
        raw = line.strip()
        if not raw:
            continue
        parts = raw.split(None, 3)
        if len(parts) < 4:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
            pgid = int(parts[2])
        except Exception:
            continue
        cmd = parts[3]
        if pid in {current_pid, current_ppid}:
            continue
        if any(pattern in cmd for pattern in experiment_proc_patterns):
            rows.append({"pid": pid, "ppid": ppid, "pgid": pgid, "cmd": cmd})
    return rows


def kill_process_group(pgid: int) -> None:
    try:
        os.killpg(int(pgid), signal.SIGKILL)
    except Exception:
        pass


def purge_experiment_processes(
    *,
    reason: str,
    gpu_lock_path: Path,
    experiment_proc_patterns: tuple[str, ...],
    preserve_pid: Optional[int] = None,
) -> list[int]:
    victims = list_waveslice_processes(experiment_proc_patterns=experiment_proc_patterns)
    killed_pgids: set[int] = set()
    killed_pids: list[int] = []
    for item in victims:
        pid = int(item["pid"])
        pgid = int(item["pgid"])
        if preserve_pid is not None and pid == preserve_pid:
            continue
        if pgid not in killed_pgids and pgid > 0:
            kill_process_group(pgid)
            killed_pgids.add(pgid)
        else:
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
        killed_pids.append(pid)
    if killed_pids:
        print(
            f"[SupplementSuite] purged lingering experiment processes ({reason}): {sorted(set(killed_pids))}",
            flush=True,
        )
    clear_gpu_lock(gpu_lock_path)
    return killed_pids


def pid_is_alive(pid: int) -> bool:
    try:
        proc = subprocess.run(
            ["ps", "-p", str(int(pid)), "-o", "pid="],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return False
    return proc.returncode == 0 and bool((proc.stdout or "").strip())


def wait_for_clean_gpu(
    *,
    label: str,
    gpu_lock_path: Path,
    timeout_sec: int = 1800,
    poll_interval_s: float = 5.0,
) -> None:
    clear_gpu_lock(gpu_lock_path)
    deadline = time.time() + max(1, int(timeout_sec))
    announced = False
    stable_active: list[int] = []
    while True:
        active = [pid for pid in list_gpu_compute_pids() if pid_is_alive(pid)]
        if not active:
            return
        if stable_active != active:
            stable_active = list(active)
            time.sleep(0.5)
            continue
        if time.time() >= deadline:
            raise TimeoutError(
                f"GPU did not become idle before starting {label}; active_pids={active}"
            )
        if not announced:
            print(
                f"[SupplementSuite] waiting for idle GPU before {label}; active_pids={active}",
                flush=True,
            )
            announced = True
        time.sleep(max(0.5, float(poll_interval_s)))


def maybe_rel_to(base: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def extract_summary_from_result_json(result_json: Path) -> dict[str, Any]:
    if not result_json.exists():
        return {}
    data = json.loads(result_json.read_text(encoding="utf-8"))
    phase1 = data.get("phase1") or {}
    phase2 = data.get("phase2") or {}
    phase12 = data.get("phase12") or {}

    def _mean(payload: dict[str, Any], key: str) -> Optional[float]:
        raw = payload.get(key)
        if isinstance(raw, dict):
            return float_or_none(raw.get("mean"))
        return float_or_none(raw)

    return {
        "phase1_ttft_improve_mean": _mean(phase1, "ttft_improve_ratio"),
        "phase1_wall_improve_mean": _mean(phase1, "round_wall_improve_ratio"),
        "phase1_error_rate_mean": _mean(phase1, "error_rate"),
        "phase1_scheduler_apply_mean": _mean(phase1, "scheduler_apply_ratio"),
        "phase2_ttft_improve_mean": _mean(phase2, "ttft_improve_ratio"),
        "phase2_wall_improve_mean": _mean(phase2, "round_wall_improve_ratio"),
        "phase2_slowdown_improve_mean": _mean(phase2, "slowdown_improve_ratio"),
        "phase2_error_rate_mean": _mean(phase2, "wave_error_rate"),
        "phase2_apply_ratio_mean": _mean(phase2, "phase2_apply_ratio"),
        "phase12_ttft_improve_mean": float_or_none((phase12.get("ttft_improve_ratio") or {}).get("mean")),
        "phase12_wall_improve_mean": float_or_none((phase12.get("round_wall_improve_ratio") or {}).get("mean")),
        "phase12_slowdown_improve_mean": float_or_none((phase12.get("slowdown_improve_ratio") or {}).get("mean")),
        "phase12_incremental_error_mean": float_or_none((phase12.get("incremental_error_rate") or {}).get("mean")),
        "phase12_scheduler_apply_mean": float_or_none((phase12.get("phase2_apply_ratio") or {}).get("mean")),
        "phase12_escape_lane_activations_mean": float_or_none((phase12.get("phase2_escape_lane_activations") or {}).get("mean")),
        "phase12_escape_lane_seen_hits_mean": float_or_none((phase12.get("phase2_escape_lane_seen_active_hits") or {}).get("mean")),
        "phase12_escape_lane_finished_hits_mean": float_or_none((phase12.get("phase2_escape_lane_finished_active_hits") or {}).get("mean")),
    }


def build_dataset_source_payload(config: dict[str, Any]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for ds in config.get("datasets", []):
        if not isinstance(ds, dict):
            continue
        entries.append(
            {
                "key": ds["key"],
                "dataset_id": ds["dataset_id"],
                "split": ds["split"],
                "extractor": ds["extractor"],
                "streaming": bool(ds.get("streaming", False)),
                "label": ds.get("label"),
                "role": ds.get("role"),
                "reason": ds.get("reason"),
            }
        )
    return {"datasets": entries}


def resolve_existing_result_json(row: dict[str, Any], reuse_results_dir: Optional[Path], *, safe_key_fn) -> Optional[Path]:
    candidates: list[Path] = []
    value = row.get("result_json")
    if value:
        p = Path(str(value))
        if p.exists():
            candidates.append(p)
        elif reuse_results_dir is not None:
            candidates.append(reuse_results_dir / p.name)
    model = str(row.get("model") or "")
    model_key = str(row.get("model_key") or safe_key_fn(model))
    if reuse_results_dir is not None and model_key:
        candidates.append(reuse_results_dir / f"{safe_key_fn(model_key)}_dataset_eval.json")
    for p in candidates:
        if p.exists():
            return p.resolve()
    return None


def resolve_existing_meta_json(row: dict[str, Any], repo_root: Path) -> Optional[Path]:
    value = row.get("workload_meta_json")
    if not value:
        return None
    p = maybe_rel_to(repo_root, str(value))
    if p.exists():
        return p
    return None
