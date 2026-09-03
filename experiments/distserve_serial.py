"""Pure data transformation and replay helpers for DistServe stage sweeps."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

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
from experiments.result_io import density_info, mean_values, safe_float

DECODE_REPLAY_MODE = "continuous_batching"


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
