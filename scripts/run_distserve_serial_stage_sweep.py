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
    DistServeRequest,
    DistServeStageCost,
    KvModel,
    KvTransferProfile,
    ResourceProfile,
    simulate_distserve_from_stage_costs,
    summarize_timings,
    timing_dict_for_json,
)
from experiments.result_io import (
    comma_list,
    density_info,
    mean_values,
    parse_map,
    read_json,
    resolve,
    safe_float,
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
DECODE_REPLAY_MODE = "continuous_batching"


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


def _distserve_requests(rows: list[dict[str, Any]]) -> list[DistServeRequest]:
    requests = []
    for row in rows:
        requests.append(
            DistServeRequest(
                req_id=str(row["req_id"]),
                arrival_ms=1000 * float(row.get("arrival_offset_s") or 0),
                prompt_tokens=int(row["tokens"]),
                is_short=bool(row["is_short"]),
            )
        )
    return sorted(requests, key=lambda request: (request.arrival_ms, request.req_id))


def _infer_kv_model(config: dict[str, Any], dtype_bytes: int) -> KvModel:
    def first(*keys: str) -> Any:
        return next((config[key] for key in keys if config.get(key) is not None), None)

    heads = first("num_attention_heads", "n_head", "num_heads")
    hidden = first("hidden_size", "n_embd", "d_model", "model_dim", "n_embed")
    kv_heads = first("num_key_value_heads", "n_head_kv", "multi_query_group_num")
    layers = first("num_hidden_layers", "n_layer", "num_layers")
    kv_heads = (1 if config.get("multi_query") else heads) if kv_heads is None else kv_heads
    if any(value is None for value in (heads, hidden, kv_heads, layers)):
        raise ValueError("cannot infer KV model parameters from model config")
    heads, hidden = int(heads), int(hidden)
    if heads <= 0 or hidden % heads:
        raise ValueError(f"invalid attention shape q_heads={heads} hidden={hidden}")
    return KvModel(int(layers), int(kv_heads), hidden // heads, int(dtype_bytes))


def _load_model_config(model_path: Path) -> dict[str, Any]:
    return read_json(model_path / "config.json")


def _resource_profile(config: dict[str, Any], key: str) -> ResourceProfile:
    for row in config["simulation"]["resource_profiles"]:
        if row["key"] == key:
            return ResourceProfile(
                key=row["key"],
                label=row["label"],
                method_label=row["method_label"],
                prefill_workers=int(row["prefill_workers"]),
                decode_workers=int(row["decode_workers"]),
                prefill_service_scale=float(row["prefill_service_scale"]),
                decode_service_scale=float(row["decode_service_scale"]),
                gpu_count=float(row["gpu_count"]),
            )
    raise KeyError(f"resource profile not found: {key}")


def _kv_profile(config: dict[str, Any], key: str) -> KvTransferProfile:
    for row in config["simulation"]["kv_profiles"]:
        if row["key"] == key:
            return KvTransferProfile(
                key=row["key"],
                label=row["label"],
                bandwidth_gbps=float(row["bandwidth_gbps"]),
                fixed_ms=float(row["fixed_ms"]),
                paper_default=bool(row.get("paper_default", False)),
            )
    raise KeyError(f"KV profile not found: {key}")


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


def _stage_costs_for_repeat(
    *,
    one_token_timings: dict[str, dict[str, Any]],
    full_timings: dict[str, dict[str, Any]],
    min_service_ms: float,
) -> dict[str, DistServeStageCost]:
    costs = {}
    for req_id, full in sorted(full_timings.items()):
        one = one_token_timings[req_id]
        one_finish = safe_float(one.get("finish_latency_ms"))
        full_finish, full_first = (
            safe_float(full.get("finish_latency_ms")),
            safe_float(full.get("first_latency_ms")),
        )
        if one_finish is None or full_first is None or full_finish is None:
            raise RuntimeError(f"incomplete stage timing for {req_id}")
        prefill = max(float(min_service_ms), one_finish)
        decode = max(float(min_service_ms), full_finish - prefill)
        first = max(float(min_service_ms), full_first - prefill)
        costs[req_id] = DistServeStageCost(req_id, prefill, decode, first, full_first, full_finish)
    return costs


def _method_row(
    *,
    density: str,
    model_key: str,
    model_label: str,
    resource: ResourceProfile,
    kv_profile: KvTransferProfile,
    summary: dict[str, Any],
    repeat_count: int,
    output_tokens: int,
    ttft_mode: str,
    kv_model: KvModel,
    decode_replay_mode: str,
    decode_batch_size: int,
    decode_batch_alpha: float,
    hardware_profile: str,
    hardware_label: str,
) -> dict[str, Any]:
    level, percent = density_info(density)
    measurement = (
        f"real_{hardware_profile}_individual_vllm_lora"
        if hardware_profile
        else "real_current_gpu_individual_vllm_lora"
    )
    row = {
        "density": density,
        "density_level": level,
        "target_long_fraction_pct": percent,
        "model_key": model_key,
        "model_label": model_label,
        "method": "DistServe",
        "method_variant": f"DistServe-2GPU ({hardware_label})"
        if hardware_label
        else resource.label,
        "resource_profile": resource.key,
        "logical_resource_profile": resource.key,
        "hardware_profile": hardware_profile,
        "hardware_label": hardware_label,
        "kv_profile": kv_profile.key,
        "kv_profile_label": kv_profile.label,
        "paper_default_kv": kv_profile.paper_default,
        "ttft_mode": ttft_mode,
        "gpu_count": resource.gpu_count,
        "physical_measurement_gpu_count": 1,
        "prefill_workers": resource.prefill_workers,
        "decode_workers": resource.decode_workers,
        "output_tokens": output_tokens,
        "decode_replay_mode": decode_replay_mode,
        "decode_replay_formula_version": DECODE_REPLAY_FORMULA_VERSION,
        "decode_batch_size": decode_batch_size,
        "decode_batch_alpha": decode_batch_alpha,
        "repeat_count": repeat_count,
        "stage_measurement": measurement,
        "execution_mode": f"stage_measurement_logical_pd_replay_{decode_replay_mode}",
        "kv_bytes_per_token": kv_model.bytes_per_token,
    }
    row.update(summary)
    return row


def _repeat_payload(
    index: int, costs: dict[str, DistServeStageCost], result: Any
) -> dict[str, Any]:
    return {
        "repeat_index": index,
        "stage_costs": {key: asdict(value) for key, value in sorted(costs.items())},
        "request_timings": {
            key: timing_dict_for_json(value)
            for key, value in sorted(result.request_timings.items())
        },
        "summary": summarize_timings(result.request_timings, result.round_wall_ms),
    }


def _row_context(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], ResourceProfile, KvTransferProfile, KvModel]:
    density, model_key = str(payload["density"]), str(payload["model_key"])
    resource, kv, model = (
        ResourceProfile(**payload["resource_profile"]),
        KvTransferProfile(**payload["kv_profile"]),
        KvModel(**payload["kv_model"]),
    )
    level, percent = density_info(density)
    replay = str(payload["decode_replay_mode"])
    if replay != DECODE_REPLAY_MODE:
        raise ValueError(f"unsupported DistServe replay mode in payload: {replay}")
    hardware_profile, hardware_label = (
        str(payload.get("hardware_profile") or ""),
        str(payload.get("hardware_label") or ""),
    )
    context = {
        "density": density,
        "density_level": level,
        "target_long_fraction_pct": percent,
        "model_key": model_key,
        "model_label": str(payload["model_label"]),
        "method": "DistServe",
        "method_variant": f"DistServe-2GPU ({hardware_label})"
        if hardware_label
        else resource.label,
        "resource_profile": resource.key,
        "logical_resource_profile": resource.key,
        "kv_profile": kv.key,
        "kv_profile_label": kv.label,
        "hardware_profile": hardware_profile,
        "hardware_label": hardware_label,
        "gpu_count": resource.gpu_count,
        "physical_measurement_gpu_count": 1,
        "decode_replay_mode": replay,
        "decode_batch_size": int(payload["decode_batch_size"]),
        "decode_replay_formula_version": str(
            payload.get("decode_replay_formula_version") or "legacy_unversioned"
        ),
        "decode_batch_alpha": float(payload["decode_batch_alpha"]),
        "stage_measurement": f"real_{hardware_profile}_individual_vllm_lora"
        if hardware_profile
        else "real_current_gpu_individual_vllm_lora",
        "execution_mode": f"stage_measurement_logical_pd_replay_{replay}",
    }
    return context, resource, kv, model


def _payload_to_rows(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    context, resource, kv, model = _row_context(payload)
    repeats = list(payload["per_repeat"])
    summary = {
        key: mean_values([item["summary"].get(key) for item in repeats])
        for key in sorted({key for item in repeats for key in item["summary"]})
    }
    method = _method_row(
        density=context["density"],
        model_key=context["model_key"],
        model_label=context["model_label"],
        resource=resource,
        kv_profile=kv,
        summary=summary,
        repeat_count=len(repeats),
        output_tokens=int(payload["output_tokens"]),
        ttft_mode=str(payload["ttft_mode"]),
        kv_model=model,
        decode_replay_mode=context["decode_replay_mode"],
        decode_batch_size=context["decode_batch_size"],
        decode_batch_alpha=context["decode_batch_alpha"],
        hardware_profile=context["hardware_profile"],
        hardware_label=context["hardware_label"],
    )
    request_rows, stage_rows = [], []
    for repeat in repeats:
        prefix = context | {"repeat_index": int(repeat["repeat_index"])}
        for source, target in (
            (repeat["request_timings"], request_rows),
            (repeat["stage_costs"], stage_rows),
        ):
            for req_id, values in sorted(source.items()):
                target.append(prefix | {"req_id": req_id} | dict(values))
    return method, request_rows, stage_rows


def _simulate_repeats(
    requests: list[DistServeRequest],
    repeats: list[dict[str, DistServeStageCost]],
    *,
    output_tokens: int,
    kv_model: KvModel,
    kv_profile: KvTransferProfile,
    resource: ResourceProfile,
    ttft_mode: str,
    decode_batch_size: int,
    decode_batch_alpha: float,
) -> list[dict[str, Any]]:
    output = []
    for index, costs in enumerate(repeats):
        result = simulate_distserve_from_stage_costs(
            requests,
            costs,
            output_tokens=output_tokens,
            kv_model=kv_model,
            kv_profile=kv_profile,
            resource_profile=resource,
            ttft_mode=ttft_mode,
            decode_replay_mode=DECODE_REPLAY_MODE,
            decode_batch_size=decode_batch_size,
            decode_batch_alpha=decode_batch_alpha,
        )
        output.append(_repeat_payload(index, costs, result))
    return output


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


def _requests_from_payload(payload: dict[str, Any]) -> list[DistServeRequest]:
    repeats = list(payload["per_repeat"])
    if not repeats:
        raise ValueError("cannot replay payload without per_repeat entries")
    requests = [
        DistServeRequest(
            str(req_id),
            1000 * float(row["arrival_offset_s"]),
            int(row["prompt_tokens"]),
            bool(row["is_short"]),
        )
        for req_id, row in sorted(repeats[0]["request_timings"].items())
    ]
    if not requests:
        raise ValueError("cannot replay payload without request timings")
    return sorted(requests, key=lambda request: (request.arrival_ms, request.req_id))


def _stage_costs_from_payload_repeat(item: dict[str, Any]) -> dict[str, DistServeStageCost]:
    costs = {
        str(req_id): DistServeStageCost(
            str(row["req_id"]),
            float(row["prefill_service_ms"]),
            float(row["decode_service_ms"]),
            safe_float(row.get("decode_first_token_ms")),
            safe_float(row.get("source_first_latency_ms")),
            safe_float(row.get("source_finish_latency_ms")),
        )
        for req_id, row in sorted(item["stage_costs"].items())
    }
    if not costs:
        raise ValueError("cannot replay payload repeat without stage_costs")
    return costs


def _replay_payload(
    *,
    payload: dict[str, Any],
    source_payload_path: Path,
    resource: ResourceProfile,
    kv_profile: KvTransferProfile,
    output_tokens: int,
    ttft_mode: str,
    decode_batch_size: int,
    decode_batch_alpha: float,
    hardware_profile: str,
    hardware_label: str,
) -> dict[str, Any]:
    source_repeats = payload["per_repeat"]
    costs = [_stage_costs_from_payload_repeat(item) for item in source_repeats]
    out = dict(payload)
    out.update(
        resource_profile=asdict(resource),
        kv_profile=asdict(kv_profile),
        output_tokens=output_tokens,
        ttft_mode=ttft_mode,
        decode_replay_mode=DECODE_REPLAY_MODE,
        decode_replay_formula_version=DECODE_REPLAY_FORMULA_VERSION,
        decode_batch_size=decode_batch_size,
        decode_batch_alpha=decode_batch_alpha,
        hardware_profile=hardware_profile or str(payload.get("hardware_profile") or ""),
        hardware_label=hardware_label or str(payload.get("hardware_label") or ""),
        execution_mode=f"stage_measurement_logical_pd_replay_{DECODE_REPLAY_MODE}",
        replay_source_payload=str(source_payload_path),
        per_repeat=_simulate_repeats(
            _requests_from_payload(payload),
            costs,
            output_tokens=output_tokens,
            kv_model=KvModel(**payload["kv_model"]),
            kv_profile=kv_profile,
            resource=resource,
            ttft_mode=ttft_mode,
            decode_batch_size=decode_batch_size,
            decode_batch_alpha=decode_batch_alpha,
        ),
    )
    for index, item in enumerate(out["per_repeat"]):
        item["repeat_index"] = int(source_repeats[index]["repeat_index"])
    return out


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
