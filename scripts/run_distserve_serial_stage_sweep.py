from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from eval_support import Req, load_reqs_json  # noqa: E402
from evaluate_waveslice_claims import _build_engine, _cleanup_engine, _run_round  # noqa: E402

from experiments.distserve_functional import (
    DECODE_REPLAY_FORMULA_VERSION,
    KvTransferProfile,
    ResourceProfile,
)
from experiments.distserve_serial import (
    DECODE_REPLAY_MODE,
    _distserve_requests,
    _infer_kv_model,
    _kv_profile,
    _payload_to_rows,
    _replay_payload,
    _resource_profile,
    _simulate_repeats,
    _stage_costs_for_repeat,
)
from experiments.result_io import (
    comma_list,
    parse_map,
    read_json,
    resolve,
    write_csv,
    write_json,
)

DEFAULT_SOURCE = (
    ROOT / "results/openworkload_ratio_sweep_lora8/ratio_sweep_20step_5models_a100_overnight_main"
)
DEFAULT_CONFIG = ROOT / "experiments/configs/distserve_functional_repro_ratio_sweep.json"
DEFAULT_OUT = ROOT / "results/openworkload_ratio_sweep_lora8/distserve_serial_stage_sweep"
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


def _load_lora_request_rows(source: Path, density: str, model_key: str) -> list[dict[str, Any]]:
    return [
        dict(item)
        for item in read_json(source / "workloads" / density / f"{model_key}_lora_requests.json")
    ]


def _load_model_config(model_path: Path) -> dict[str, Any]:
    return read_json(model_path / "config.json")


def _cfg_bool(cfg: dict[str, Any], key: str, default: bool) -> bool:
    return default if cfg.get(key) is None else bool(cfg[key])


def _cfg_int(cfg: dict[str, Any], key: str, default: int) -> int:
    return default if cfg.get(key) is None else int(cfg[key])


def _engine_args(source: dict[str, Any]) -> argparse.Namespace:
    defaults = {
        "max_num_batched_tokens": 1536,
        "max_model_len": 3072,
        "max_num_partial_prefills": 1,
        "max_long_partial_prefills": 1,
        "gpu_memory_utilization": 0.7,
        "trust_remote_code": False,
    }
    return argparse.Namespace(**(defaults | source))


def _zero_arrival(request: Req) -> Req:
    return Req(request.req_id, request.prompt, request.is_short, request.lora_tag, 0.0)


def _measure_individual_stage(
    *,
    source_cfg: dict[str, Any],
    reqs: list[Req],
    max_new_tokens: int,
    warmup_iters: int,
    repeats: int,
    timeout_sec: int,
    ignore_eos: bool,
    enable_chunked_prefill: bool,
    stage_name: str,
) -> list[dict[str, dict[str, Any]]]:
    if not reqs:
        return []
    engine = None
    try:
        args = _engine_args(source_cfg)
        engine, lora_map = _build_engine(
            args,
            mode="baseline_lora_compat",
            enable_lora=True,
            enable_chunked_prefill=enable_chunked_prefill,
            adapter_a=str(source_cfg.get("adapter_a") or ""),
            adapter_b=str(source_cfg.get("adapter_b") or ""),
        )
        for index in range(max(0, int(warmup_iters))):
            _run_round(
                engine=engine,
                reqs=[_zero_arrival(reqs[0])],
                max_new_tokens=max_new_tokens,
                ignore_eos=ignore_eos,
                timeout_sec=timeout_sec,
                enable_lora=True,
                lora_map=lora_map,
                run_tag=f"warmup_{stage_name}_{index}",
            )
        repeats_output = []
        for repeat in range(max(1, int(repeats))):
            timings = {}
            for index, request in enumerate(reqs):
                result = _run_round(
                    engine=engine,
                    reqs=[_zero_arrival(request)],
                    max_new_tokens=max_new_tokens,
                    ignore_eos=ignore_eos,
                    timeout_sec=timeout_sec,
                    enable_lora=True,
                    lora_map=lora_map,
                    run_tag=f"{stage_name}_r{repeat}_q{index}",
                )
                timing = (result.get("request_timings") or {}).get(request.req_id)
                if result.get("timed_out") or not isinstance(timing, dict):
                    error = "timed out" if result.get("timed_out") else "produced no timing"
                    raise RuntimeError(f"{stage_name} {error} for request {request.req_id}")
                timings[request.req_id] = dict(timing)
            repeats_output.append(timings)
        return repeats_output
    finally:
        _cleanup_engine(engine)


def _run_case(
    *,
    source: Path,
    out_dir: Path,
    config: dict[str, Any],
    density: str,
    model_key: str,
    resource: ResourceProfile,
    kv_profile: KvTransferProfile,
    output_tokens: int,
    ttft_mode: str,
    stage_warmup_iters: int | None,
    stage_repeats: int | None,
    min_service_ms: float,
    decode_batch_size: int,
    decode_batch_alpha: float,
    gpu_memory_utilization_override: float,
    gpu_memory_utilization_by_model: dict[str, float],
    max_num_batched_tokens_by_model: dict[str, int],
    hardware_profile: str,
    hardware_label: str,
    force: bool,
) -> Path:
    path = out_dir / "raw" / density / model_key / "distserve_serial.json"
    if path.exists() and not force:
        return path
    source_cfg = _case_source_config(source, density, model_key)
    memory = float(gpu_memory_utilization_by_model.get(model_key, gpu_memory_utilization_override))
    batch_tokens = int(max_num_batched_tokens_by_model.get(model_key, 0))
    if memory > 0:
        source_cfg["gpu_memory_utilization"] = memory
    if batch_tokens > 0:
        source_cfg["max_num_batched_tokens"] = batch_tokens
    request_rows = _load_lora_request_rows(source, density, model_key)
    requests = _distserve_requests(request_rows)
    lora_path = source / "workloads" / density / f"{model_key}_lora_requests.json"
    lora_reqs = [_zero_arrival(request) for request in load_reqs_json(str(lora_path))]
    kv_model = _infer_kv_model(
        _load_model_config(Path(str(source_cfg.get("model_path") or ""))),
        int((config.get("simulation") or {}).get("dtype_bytes") or 2),
    )
    repeats = stage_repeats if stage_repeats is not None else _cfg_int(source_cfg, "repeats", 2)
    warmups = (
        stage_warmup_iters
        if stage_warmup_iters is not None
        else _cfg_int(source_cfg, "warmup_iters", 1)
    )
    measure = dict(
        source_cfg=source_cfg,
        reqs=lora_reqs,
        warmup_iters=warmups,
        repeats=repeats,
        timeout_sec=_cfg_int(source_cfg, "timeout_sec", 240),
        ignore_eos=_cfg_bool(source_cfg, "ignore_eos", False),
        enable_chunked_prefill=_cfg_bool(
            source_cfg, "phase2_baseline_enable_chunked_prefill", True
        ),
    )
    one = _measure_individual_stage(
        max_new_tokens=1, stage_name="distserve_prefill_proxy", **measure
    )
    full = _measure_individual_stage(
        max_new_tokens=output_tokens, stage_name="distserve_full_decode", **measure
    )
    costs = [
        _stage_costs_for_repeat(
            one_token_timings=one_timings, full_timings=full_timings, min_service_ms=min_service_ms
        )
        for one_timings, full_timings in zip(one, full, strict=True)
    ]
    payload = {
        "density": density,
        "model_key": model_key,
        "model_label": str(source_cfg.get("model_name") or model_key),
        "source_main_run": str(source),
        "source_lora_requests_json": str(lora_path),
        "source_config": source_cfg,
        "resource_profile": asdict(resource),
        "kv_profile": asdict(kv_profile),
        "kv_model": asdict(kv_model),
        "output_tokens": output_tokens,
        "ttft_mode": ttft_mode,
        "decode_replay_mode": DECODE_REPLAY_MODE,
        "decode_replay_formula_version": DECODE_REPLAY_FORMULA_VERSION,
        "decode_batch_size": decode_batch_size,
        "decode_batch_alpha": decode_batch_alpha,
        "gpu_memory_utilization_effective": source_cfg.get("gpu_memory_utilization"),
        "gpu_memory_utilization_by_model": dict(sorted(gpu_memory_utilization_by_model.items())),
        "max_num_batched_tokens_effective": source_cfg.get("max_num_batched_tokens"),
        "max_num_batched_tokens_by_model": dict(sorted(max_num_batched_tokens_by_model.items())),
        "hardware_profile": hardware_profile,
        "hardware_label": hardware_label,
        "stage_measurement": f"real_{hardware_profile}_individual_vllm_lora"
        if hardware_profile
        else "real_current_gpu_individual_vllm_lora",
        "execution_mode": f"stage_measurement_logical_pd_replay_{DECODE_REPLAY_MODE}",
        "per_repeat": _simulate_repeats(
            requests,
            costs,
            output_tokens=output_tokens,
            kv_model=kv_model,
            kv_profile=kv_profile,
            resource=resource,
            ttft_mode=ttft_mode,
            decode_batch_size=decode_batch_size,
            decode_batch_alpha=decode_batch_alpha,
        ),
    }
    write_json(path, payload)
    return path


def _replay_case(
    *,
    replay_source_dir: Path,
    out_dir: Path,
    density: str,
    model_key: str,
    resource: ResourceProfile,
    kv_profile: KvTransferProfile,
    output_tokens: int,
    ttft_mode: str,
    decode_batch_size: int,
    decode_batch_alpha: float,
    hardware_profile: str,
    hardware_label: str,
    force: bool,
) -> Path:
    path = out_dir / "raw" / density / model_key / "distserve_serial.json"
    if path.exists() and not force:
        return path
    source = replay_source_dir / "raw" / density / model_key / "distserve_serial.json"
    write_json(
        path,
        _replay_payload(
            payload=read_json(source),
            source_payload_path=source,
            resource=resource,
            kv_profile=kv_profile,
            output_tokens=output_tokens,
            ttft_mode=ttft_mode,
            decode_batch_size=decode_batch_size,
            decode_batch_alpha=decode_batch_alpha,
            hardware_profile=hardware_profile,
            hardware_label=hardware_label,
        ),
    )
    return path


def _discover_payloads(out_dir: Path) -> list[Path]:
    return sorted((out_dir / "raw").glob("*/*/distserve_serial.json"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DistServe P/D replay from real single-A100 stage measurements."
    )
    defaults = {
        "source-main-run": str(DEFAULT_SOURCE),
        "distserve-config": str(DEFAULT_CONFIG),
        "out-dir": str(DEFAULT_OUT),
        "models": DEFAULT_MODELS,
        "densities": DEFAULT_DENSITIES,
        "resource-profile": "distserve_2a100",
        "kv-profile": "realistic_pcie",
        "ttft-mode": "prefill_finish",
        "gpu-memory-utilization-by-model": "",
        "max-num-batched-tokens-by-model": "",
        "hardware-profile": "",
        "hardware-label": "",
        "replay-source-dir": "",
    }
    for name, default in defaults.items():
        parser.add_argument(f"--{name}", default=default)
    for name, default in (
        ("output-tokens", 64),
        ("decode-batch-size", 16),
        ("stage-warmup-iters", -1),
        ("stage-repeats", -1),
        ("limit-cases", 0),
    ):
        parser.add_argument(f"--{name}", type=int, default=default)
    for name, default in (
        ("decode-batch-alpha", 0.08),
        ("min-service-ms", 0.01),
        ("gpu-memory-utilization-override", 0.0),
    ):
        parser.add_argument(f"--{name}", type=float, default=default)
    parser.add_argument(
        "--decode-replay-mode", choices=(DECODE_REPLAY_MODE,), default=DECODE_REPLAY_MODE
    )
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    source, out_dir = _resolve(args.source_main_run), _resolve(args.out_dir)
    config = read_json(_resolve(args.distserve_config))
    resource, kv = (
        _resource_profile(config, args.resource_profile),
        _kv_profile(config, args.kv_profile),
    )
    cases = [
        (density, model)
        for model in comma_list(args.models)
        for density in comma_list(args.densities)
    ]
    cases = cases[: args.limit_cases] if args.limit_cases else cases
    common = dict(
        out_dir=out_dir,
        resource=resource,
        kv_profile=kv,
        output_tokens=args.output_tokens,
        ttft_mode=args.ttft_mode,
        decode_batch_size=args.decode_batch_size,
        decode_batch_alpha=args.decode_batch_alpha,
        hardware_profile=args.hardware_profile,
        hardware_label=args.hardware_label,
        force=args.force,
    )
    progress = []
    if not args.merge_only:
        for density, model in cases:
            if args.replay_source_dir.strip():
                path = _replay_case(
                    replay_source_dir=_resolve(args.replay_source_dir),
                    density=density,
                    model_key=model,
                    **common,
                )
            else:
                path = _run_case(
                    source=source,
                    config=config,
                    density=density,
                    model_key=model,
                    stage_warmup_iters=None
                    if args.stage_warmup_iters < 0
                    else args.stage_warmup_iters,
                    stage_repeats=None if args.stage_repeats < 0 else args.stage_repeats,
                    min_service_ms=args.min_service_ms,
                    gpu_memory_utilization_override=args.gpu_memory_utilization_override,
                    gpu_memory_utilization_by_model=_parse_model_float_map(
                        args.gpu_memory_utilization_by_model
                    ),
                    max_num_batched_tokens_by_model=_parse_model_int_map(
                        args.max_num_batched_tokens_by_model
                    ),
                    **common,
                )
            progress.append(
                {"density": density, "model_key": model, "status": "ok", "result_json": str(path)}
            )
            write_csv(out_dir / "metadata/progress.csv", progress)
    method_rows, request_rows, stage_rows = [], [], []
    for path in _discover_payloads(out_dir):
        method, requests, stages = _payload_to_rows(read_json(path))
        method_rows.append(method)
        request_rows.extend(requests)
        stage_rows.extend(stages)
    outputs = {
        "metadata/progress": progress,
        "distserve_serial_method_metrics": method_rows,
        "distserve_serial_request_metrics": request_rows,
        "distserve_serial_stage_costs": stage_rows,
    }
    for name, rows in outputs.items():
        write_csv(out_dir / f"{name}.csv", rows)
        write_json(out_dir / f"{name}.json", rows)
    print(f"[DistServeSerial] out_dir={out_dir}")
    print(
        f"[DistServeSerial] method_rows={len(method_rows)} request_rows={len(request_rows)} stage_rows={len(stage_rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
