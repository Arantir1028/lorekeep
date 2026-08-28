from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.result_io import (
    comma_list,
    density_info,
    mean_values,
    parse_map,
    ratio,
    read_csv,
    read_json,
    resolve,
    timing_summary,
    write_csv,
    write_json,
)
from experiments.run_frozen_eval_config import build_eval_invocation, eval_section_keys

DEFAULT_SOURCE = (
    ROOT / "results/openworkload_ratio_sweep_lora8/ratio_sweep_20step_5models_a100_overnight_main"
)
DEFAULT_DISTSERVE = (
    ROOT
    / "results/chapter5_exports/distserve_functional_repro_ratio_sweep/distserve_method_metrics.csv"
)
DEFAULT_OUT = ROOT / "results/openworkload_ratio_sweep_lora8/cucumis_2a100_dispatch_real_split"
DEFAULT_MODELS = (
    "baichuan2-7b-chat,gemma-2-9b-it,gemma-7b-it,mistral-7b-instruct-v0.2,qwen2.5-7b-instruct"
)
DEFAULT_DENSITIES = (
    "mid_l10,mid_l30,mid_l50,mid_l70,mid_l90,high_l10,high_l30,high_l50,high_l70,high_l90"
)


def _resolve(value: str) -> Path:
    return resolve(ROOT, value)


def _parse_model_float_map(value: str) -> dict[str, float]:
    return parse_map(value, float, label="gpu memory map")


def _parse_model_int_map(value: str) -> dict[str, int]:
    return parse_map(value, int, label="integer map")


def _case_source_config(source: Path, density: str, model_key: str) -> dict[str, Any]:
    return dict(read_json(source / "raw" / density / f"{model_key}_dataset_eval.json")["config"])


def _split_requests(
    items: list[dict[str, Any]], dispatcher: str, *, output_tokens: int
) -> list[list[dict[str, Any]]]:
    ordered = sorted(
        (dict(item) for item in items),
        key=lambda item: (float(item.get("arrival_offset_s") or 0), str(item.get("req_id") or "")),
    )
    replicas: list[list[dict[str, Any]]] = [[], []]
    if dispatcher == "round_robin":
        for index, item in enumerate(ordered):
            replicas[index % 2].append(item)
        return replicas
    if dispatcher != "least_backlog":
        raise ValueError(f"unknown dispatcher: {dispatcher}")
    available = [0.0, 0.0]
    for item in ordered:
        arrival = 1000 * float(item.get("arrival_offset_s") or 0)
        service = float(max(1, int(item.get("tokens") or 1)) + output_tokens * 16)
        loads = [max(0.0, ready - arrival) for ready in available]
        replica = min(range(2), key=lambda index: (loads[index], index))
        replicas[replica].append(item)
        available[replica] = max(arrival, available[replica]) + service
    return replicas


def _source_section(source: dict[str, Any], prefix: str, section: str) -> dict[str, Any]:
    marker = prefix + "_"
    values = {key[len(marker) :]: value for key, value in source.items() if key.startswith(marker)}
    if section == "phase12_soft_gate" and "gate_mode" in values:
        values["phase2_gate_mode"] = values.pop("gate_mode")
    allowed = eval_section_keys(section)
    return {key: value for key, value in values.items() if key in allowed}


def _eval_config_from_source(
    source: dict[str, Any], req_path: Path, lora_path: Path
) -> dict[str, Any]:
    runtime = {
        "python_bin": source.get("python_bin") or sys.executable,
        "trust_remote_code": bool(source.get("trust_remote_code", False)),
        "warmup_iters": int(source.get("warmup_iters", 1)),
        "repeats": int(source.get("repeats", 2)),
        "timeout_sec": int(source.get("timeout_sec", 240)),
        "max_new_tokens": int(source.get("max_new_tokens", 64)),
        "max_model_len": int(source.get("max_model_len", 3072)),
        "max_num_batched_tokens": int(source.get("max_num_batched_tokens", 1536)),
        "gpu_memory_utilization": float(source.get("gpu_memory_utilization", 0.7)),
        "queue_reorder_mode": str(source.get("queue_reorder_mode", "sjf")),
        "queue_reorder_aging_quantum_us": int(
            float(source.get("queue_reorder_aging_quantum_us", 20000))
        ),
    }
    soft = _source_section(source, "phase12_phase2", "phase12_soft_gate")
    soft.setdefault("phase2_gate_mode", "soft")
    return {
        "evaluator": "tests/evaluate_waveslice_claims.py",
        "include_phase12": True,
        "model": {"name": source.get("model_name"), "path": source.get("model_path")},
        "workload": {"requests_json": str(req_path), "lora_requests_json": str(lora_path)},
        "adapters": {"adapter_a": source.get("adapter_a"), "adapter_b": source.get("adapter_b")},
        "runtime": runtime,
        "phase1": _source_section(source, "phase1", "phase1"),
        "phase12_soft_gate": soft,
        "phase2": _source_section(source, "phase2", "phase2"),
    }


def _prepare_case(
    *,
    source: Path,
    out_dir: Path,
    density: str,
    model_key: str,
    dispatcher: str,
    output_tokens: int,
    gpu_memory_utilization_override: float,
    gpu_memory_utilization_by_model: dict[str, float],
    max_num_batched_tokens_by_model: dict[str, int],
) -> list[Path]:
    root = out_dir / "workloads" / density / model_key / dispatcher
    request_replicas = _split_requests(
        read_json(source / "workloads" / density / f"{model_key}_requests.json"),
        dispatcher,
        output_tokens=output_tokens,
    )
    lora_replicas = _split_requests(
        read_json(source / "workloads" / density / f"{model_key}_lora_requests.json"),
        dispatcher,
        output_tokens=output_tokens,
    )
    source_cfg = _case_source_config(source, density, model_key)
    memory = float(gpu_memory_utilization_by_model.get(model_key, gpu_memory_utilization_override))
    batch_tokens = int(max_num_batched_tokens_by_model.get(model_key, 0))
    if memory > 0:
        source_cfg["gpu_memory_utilization"] = memory
    if batch_tokens > 0:
        source_cfg["max_num_batched_tokens"] = batch_tokens
    configs = []
    for replica in range(2):
        prefix = root / f"replica{replica}"
        request_path, lora_path, meta_path = (
            prefix.with_name(f"{prefix.name}_{suffix}.json")
            for suffix in ("requests", "lora_requests", "meta")
        )
        write_json(request_path, request_replicas[replica])
        write_json(lora_path, lora_replicas[replica])
        write_json(
            meta_path,
            {
                "source_main_run": str(source),
                "density": density,
                "model_key": model_key,
                "dispatcher": dispatcher,
                "replica_id": replica,
                "phase1_request_count": len(request_replicas[replica]),
                "phase2_request_count": len(lora_replicas[replica]),
                "gpu_memory_utilization_effective": source_cfg.get("gpu_memory_utilization"),
                "gpu_memory_utilization_by_model": dict(
                    sorted(gpu_memory_utilization_by_model.items())
                ),
                "max_num_batched_tokens_effective": source_cfg.get("max_num_batched_tokens"),
                "max_num_batched_tokens_by_model": dict(
                    sorted(max_num_batched_tokens_by_model.items())
                ),
            },
        )
        config_path = (
            out_dir / "configs" / density / model_key / dispatcher / f"replica{replica}.json"
        )
        write_json(config_path, _eval_config_from_source(source_cfg, request_path, lora_path))
        configs.append(config_path)
    return configs


def _replica_invocation(
    config_path: Path, out_json: Path, *, cuda_device: str | None = None
) -> tuple[list[str], dict[str, str]]:
    cmd, env = build_eval_invocation(read_json(config_path), out_json_override=str(out_json))
    env.setdefault("VLLM_NO_USAGE_STATS", "1")
    if cuda_device and cuda_device.strip():
        device = cuda_device.strip()
        env["CUDA_VISIBLE_DEVICES"] = device
        env["WAVESLICE_GPU_LOCK_PATH"] = (
            f"/tmp/waveslice_gpu_experiment_cuda_{''.join(char if char.isalnum() else '_' for char in device)}.lock"
        )
        index = int(device.split(",", 1)[0])
        env["VLLM_PORT"] = str(43000 + 1000 * index)
    return cmd, env


def _outer_timeout_sec(config_path: Path) -> int:
    timeout = int((read_json(config_path).get("runtime") or {}).get("timeout_sec") or 240)
    return max(timeout + 240, int(timeout * 1.35))


def _terminate_process_group(proc: subprocess.Popen[str], *, grace_sec: int = 10) -> None:
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=grace_sec)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def _run_replicas(
    *,
    replica_ids: list[int],
    config_paths: list[Path],
    out_paths: list[Path],
    log_root: Path,
    cuda_devices: list[str],
) -> dict[int, int]:
    log_root.mkdir(parents=True, exist_ok=True)
    launched = []
    returncodes, pending = {}, set(replica_ids)
    try:
        for replica in replica_ids:
            device = cuda_devices[replica] if replica < len(cuda_devices) else None
            cmd, env = _replica_invocation(
                config_paths[replica], out_paths[replica], cuda_device=device
            )
            stdout = (log_root / f"replica{replica}.stdout.log").open("w", encoding="utf-8")
            stderr = (log_root / f"replica{replica}.stderr.log").open("w", encoding="utf-8")
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(ROOT),
                    env=env,
                    text=True,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                )
            except BaseException:
                stdout.close()
                stderr.close()
                raise
            launched.append(
                (
                    replica,
                    proc,
                    stdout,
                    stderr,
                    time.monotonic() + _outer_timeout_sec(config_paths[replica]),
                )
            )
        while pending:
            now = time.monotonic()
            for replica, proc, _, stderr, deadline in launched:
                if replica not in pending:
                    continue
                code = proc.poll()
                if code is not None:
                    returncodes[replica] = code
                    pending.remove(replica)
                elif now >= deadline:
                    stderr.write(
                        f"\n[CUCUMIS2Dispatch] outer timeout; terminating replica {replica} process group\n"
                    )
                    stderr.flush()
                    _terminate_process_group(proc)
                    returncodes[replica] = 124
                    pending.remove(replica)
            if pending:
                time.sleep(1)
    finally:
        for _, proc, stdout, stderr, _ in launched:
            if proc.poll() is None:
                _terminate_process_group(proc)
            stdout.close()
            stderr.close()
    return returncodes


def _merge_case(
    out_dir: Path,
    density: str,
    model_key: str,
    dispatcher: str,
    *,
    hardware_profile: str,
    hardware_label: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payloads = [
        read_json(out_dir / "raw" / density / model_key / dispatcher / f"replica{replica}.json")
        for replica in range(2)
    ]
    phase12 = [list((payload.get("per_repeat") or {}).get("phase12") or []) for payload in payloads]
    repeat_count = min(map(len, phase12))
    variant = f"CUCUMIS-2GPU-{'RR' if dispatcher == 'round_robin' else 'LB'}" + (
        f" ({hardware_label})" if hardware_label else ""
    )
    execution = f"real_split_workload_parallel_{hardware_profile or 'current_gpu'}"
    summaries, requests = [], []
    for repeat in range(repeat_count):
        combined = {}
        for replica, rows in enumerate(phase12):
            for req_id, timing in (rows[repeat].get("wave_request_timings") or {}).items():
                item = dict(timing) | {"replica_id": replica}
                combined[str(req_id)] = item
                requests.append(
                    {
                        "density": density,
                        "model_key": model_key,
                        "method": "CUCUMIS",
                        "method_variant": variant,
                        "dispatcher": dispatcher,
                        "hardware_profile": hardware_profile,
                        "hardware_label": hardware_label,
                        "gpu_count": 2.0,
                        "replica_count": 2,
                        "replica_execution": execution,
                        "repeat_index": repeat,
                        "replica_id": replica,
                        "req_id": req_id,
                        **item,
                    }
                )
        summaries.append(timing_summary(combined, include_fraction=True, include_wall=True))
    level, percent = density_info(density)
    method = {
        "density": density,
        "density_level": level,
        "target_long_fraction_pct": percent,
        "model_key": model_key,
        "method": "CUCUMIS",
        "method_variant": variant,
        "dispatcher": dispatcher,
        "hardware_profile": hardware_profile,
        "hardware_label": hardware_label,
        "gpu_count": 2.0,
        "replica_count": 2,
        "replica_execution": execution,
        "repeat_count": repeat_count,
    }
    method.update(
        {
            key: mean_values([summary.get(key) for summary in summaries])
            for key in sorted({key for summary in summaries for key in summary})
        }
    )
    return method, requests


def build_equal_resource_comparison(
    distserve_path: Path, method_rows: list[dict[str, Any]], *, direct_output: bool = False
) -> list[dict[str, Any]]:
    distserve = {
        (row["density"], row["model_key"]): row
        for row in read_csv(distserve_path)
        if row.get("resource_profile") == "distserve_2a100"
        and row.get("kv_profile") == "realistic_pcie"
    }
    output = []
    metrics = (
        "all_ttft_p99_ms",
        "short_ttft_p99_ms",
        "long_ttft_p99_ms",
        "all_completion_p99_ms",
        "short_completion_p99_ms",
        "long_completion_p99_ms",
        "round_wall_ms",
        "throughput_rps",
    )
    for cucumis in method_rows:
        dist = distserve.get((str(cucumis.get("density")), str(cucumis.get("model_key"))))
        if not dist:
            continue
        hardware_profile = cucumis.get("hardware_profile") or dist.get("hardware_profile")
        hardware_label = cucumis.get("hardware_label") or dist.get("hardware_label")
        row = {
            "density": cucumis.get("density"),
            "density_level": cucumis.get("density_level"),
            "target_long_fraction_pct": cucumis.get("target_long_fraction_pct"),
            "model_key": cucumis.get("model_key"),
        }
        if direct_output:
            row.update(
                distserve_variant=dist.get("method_variant") or "DistServe-2GPU",
                cucumis_variant=cucumis.get("method_variant"),
                cucumis_dispatcher=cucumis.get("dispatcher"),
                comparison_scope="equal_resource_2xgpu_real_split",
            )
        else:
            row.update(
                distserve_variant=dist.get("method_variant"),
                distserve_decode_replay_mode=dist.get("decode_replay_mode"),
                distserve_decode_batch_size=dist.get("decode_batch_size"),
                distserve_decode_batch_alpha=dist.get("decode_batch_alpha"),
                cucumis_variant=cucumis.get("method_variant"),
                cucumis_dispatcher=cucumis.get("dispatcher"),
                comparison_scope="equal_resource_2xgpu_real_split"
                if hardware_profile or hardware_label
                else "equal_resource_2xa100_real_split",
            )
            if hardware_profile or hardware_label:
                row.update(hardware_profile=hardware_profile, hardware_label=hardware_label)
        for metric in metrics:
            row[f"distserve_{metric}"] = dist.get(metric)
            row[f"cucumis2_{metric}"] = cucumis.get(metric)
            row[f"distserve_vs_cucumis2_{metric}_ratio"] = ratio(
                dist.get(metric), cucumis.get(metric)
            )
        output.append(row)
    return output


def _discover_complete_cases(out_dir: Path) -> list[tuple[str, str, str]]:
    root = out_dir / "raw"
    return (
        [
            (density.name, model.name, dispatcher.name)
            for density in sorted(path for path in root.iterdir() if path.is_dir())
            for model in sorted(path for path in density.iterdir() if path.is_dir())
            for dispatcher in sorted(path for path in model.iterdir() if path.is_dir())
            if all((dispatcher / f"replica{replica}.json").exists() for replica in range(2))
        ]
        if root.exists()
        else []
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run real split-workload CUCUMIS-2A100 dispatcher sweep."
    )
    defaults = {
        "source-main-run": str(DEFAULT_SOURCE),
        "distserve-method-metrics": str(DEFAULT_DISTSERVE),
        "out-dir": str(DEFAULT_OUT),
        "models": DEFAULT_MODELS,
        "densities": DEFAULT_DENSITIES,
        "dispatchers": "round_robin,least_backlog",
        "gpu-memory-utilization-by-model": "",
        "max-num-batched-tokens-by-model": "",
        "hardware-profile": "",
        "hardware-label": "",
        "replica-cuda-devices": "",
    }
    for name, default in defaults.items():
        parser.add_argument(f"--{name}", default=default)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization-override", type=float, default=0.0)
    parser.add_argument("--limit-cases", type=int, default=0)
    for name in ("prepare-only", "merge-only", "parallel-replicas"):
        parser.add_argument(f"--{name}", action="store_true")
    return parser


def _run_case(
    args: argparse.Namespace,
    source: Path,
    out_dir: Path,
    case: tuple[str, str, str],
    devices: list[str],
    memory: dict[str, float],
    batches: dict[str, int],
) -> list[dict[str, Any]]:
    density, model, dispatcher = case
    configs = _prepare_case(
        source=source,
        out_dir=out_dir,
        density=density,
        model_key=model,
        dispatcher=dispatcher,
        output_tokens=args.output_tokens,
        gpu_memory_utilization_override=args.gpu_memory_utilization_override,
        gpu_memory_utilization_by_model=memory,
        max_num_batched_tokens_by_model=batches,
    )
    if args.prepare_only:
        return [
            {"density": density, "model_key": model, "dispatcher": dispatcher, "status": "prepared"}
        ]
    outputs = [
        out_dir / "raw" / density / model / dispatcher / f"replica{replica}.json"
        for replica in range(2)
    ]
    missing = [replica for replica, path in enumerate(outputs) if not path.exists()]
    logs = out_dir / "logs" / density / model / dispatcher
    run_groups = [missing] if args.parallel_replicas else [[replica] for replica in missing]
    codes = {}
    for replica_ids in run_groups:
        codes.update(
            _run_replicas(
                replica_ids=replica_ids,
                config_paths=configs,
                out_paths=outputs,
                log_root=logs,
                cuda_devices=devices,
            )
        )
    return [
        {
            "density": density,
            "model_key": model,
            "dispatcher": dispatcher,
            "replica_id": replica,
            "status": "ok" if output.exists() and codes.get(replica, 0) == 0 else "failed",
            "returncode": codes.get(replica, 0),
            "cuda_device": devices[replica] if replica < len(devices) else "",
            "parallel_replicas": bool(args.parallel_replicas),
            "result_json": str(output),
        }
        for replica, output in enumerate(outputs)
    ]


def main() -> int:
    args = _parser().parse_args()
    source, distserve, out_dir = map(
        _resolve, (args.source_main_run, args.distserve_method_metrics, args.out_dir)
    )
    devices = comma_list(args.replica_cuda_devices)
    if args.parallel_replicas and len(devices) < 2:
        raise ValueError(
            "--parallel-replicas requires --replica-cuda-devices with at least two device ids"
        )
    cases = [
        (density, model, dispatcher)
        for model in comma_list(args.models)
        for density in comma_list(args.densities)
        for dispatcher in comma_list(args.dispatchers)
    ]
    cases = cases[: args.limit_cases] if args.limit_cases else cases
    memory = _parse_model_float_map(args.gpu_memory_utilization_by_model)
    batches = _parse_model_int_map(args.max_num_batched_tokens_by_model)
    progress = []
    for case in cases:
        rows = (
            [
                {
                    "density": case[0],
                    "model_key": case[1],
                    "dispatcher": case[2],
                    "status": "merge_only",
                }
            ]
            if args.merge_only
            else _run_case(args, source, out_dir, case, devices, memory, batches)
        )
        for row in rows:
            progress.append(row)
            write_csv(out_dir / "metadata/progress.csv", progress)
            if row["status"] == "failed":
                write_json(out_dir / "metadata/progress.json", progress)
                return 1
    if args.prepare_only:
        write_json(out_dir / "metadata/progress.json", progress)
        print(f"[CUCUMIS2Dispatch] prepared out_dir={out_dir}")
        return 0
    merge_cases = list(dict.fromkeys([*cases, *_discover_complete_cases(out_dir)]))
    method_rows, request_rows = [], []
    for density, model, dispatcher in merge_cases:
        if not all(
            (out_dir / "raw" / density / model / dispatcher / f"replica{replica}.json").exists()
            for replica in range(2)
        ):
            continue
        method, requests = _merge_case(
            out_dir,
            density,
            model,
            dispatcher,
            hardware_profile=args.hardware_profile,
            hardware_label=args.hardware_label,
        )
        method_rows.append(method)
        request_rows.extend(requests)
    comparison = build_equal_resource_comparison(distserve, method_rows, direct_output=True)
    outputs = {
        "metadata/progress": progress,
        "cucumis_2a100_real_method_metrics": method_rows,
        "cucumis_2a100_real_request_metrics": request_rows,
        "distserve_equal_resource_real_comparison": comparison,
    }
    for name, rows in outputs.items():
        write_csv(out_dir / f"{name}.csv", rows)
        write_json(out_dir / f"{name}.json", rows)
    print(f"[CUCUMIS2Dispatch] out_dir={out_dir}")
    print(f"[CUCUMIS2Dispatch] method_rows={len(method_rows)} comparison_rows={len(comparison)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
