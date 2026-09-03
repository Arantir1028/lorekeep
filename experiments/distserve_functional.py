from __future__ import annotations

import heapq
from dataclasses import dataclass

from experiments.result_io import percentile

DECODE_REPLAY_FORMULA_VERSION = "continuous_batching_v2_conserved_decode_service"


@dataclass(frozen=True)
class DistServeRequest:
    req_id: str
    arrival_ms: float
    prompt_tokens: int
    is_short: bool


@dataclass(frozen=True)
class ResourceProfile:
    key: str
    label: str
    method_label: str
    prefill_workers: int
    decode_workers: int
    prefill_service_scale: float
    decode_service_scale: float
    gpu_count: float


@dataclass(frozen=True)
class KvTransferProfile:
    key: str
    label: str
    bandwidth_gbps: float
    fixed_ms: float
    paper_default: bool = False


@dataclass(frozen=True)
class KvModel:
    layers: int
    kv_heads: int
    head_dim: int
    dtype_bytes: int = 2

    @property
    def bytes_per_token(self) -> int:
        return int(2 * self.layers * self.kv_heads * self.head_dim * self.dtype_bytes)

    def transfer_ms(self, tokens: int, profile: KvTransferProfile) -> float:
        bytes_per_ms = float(profile.bandwidth_gbps) * 1e6
        return float(profile.fixed_ms) + (
            max(0, tokens) * self.bytes_per_token / bytes_per_ms if bytes_per_ms > 0 else 0
        )


@dataclass(frozen=True)
class DistServeStageCost:
    req_id: str
    prefill_service_ms: float
    decode_service_ms: float
    decode_first_token_ms: float | None = None
    source_first_latency_ms: float | None = None
    source_finish_latency_ms: float | None = None


@dataclass(frozen=True)
class SimulatedRequestTiming:
    req_id: str
    is_short: bool
    ttft_mode: str
    arrival_offset_s: float
    prompt_tokens: int
    prefill_start_ms: float
    prefill_finish_ms: float
    kv_transfer_ms: float
    decode_start_ms: float
    decode_finish_ms: float
    prefill_ttft_ms: float
    decode_ready_ttft_ms: float
    decode_start_ttft_ms: float
    decode_first_token_ttft_ms: float
    decode_queue_first_token_ttft_ms: float
    first_latency_ms: float
    finish_latency_ms: float
    tpot_ms: float


@dataclass(frozen=True)
class SimulationResult:
    request_timings: dict[str, SimulatedRequestTiming]
    round_wall_ms: float
    throughput_rps: float


_TTFT_MODES = {
    "prefill_finish",
    "decode_ready",
    "decode_first_token",
    "decode_queue_start",
    "decode_queue_first_token",
    "decode_start",
}


def _empty_result() -> SimulationResult:
    return SimulationResult({}, 0.0, 0.0)


def _validate_mode(mode: str) -> None:
    if mode not in _TTFT_MODES:
        raise ValueError(f"unknown DistServe TTFT mode: {mode}")


def _finalize_distserve(
    ordered: list[DistServeRequest],
    prefill: dict[str, tuple[float, float, float]],
    decode: dict[str, tuple[float, float, float, float]],
    ttft_mode: str,
) -> SimulationResult:
    timings = {}
    for request in ordered:
        prefill_start, prefill_finish, transfer = prefill[request.req_id]
        decode_start, decode_finish, tpot, first_delta = decode[request.req_id]
        values = {
            "prefill_finish": prefill_finish - request.arrival_ms,
            "decode_ready": prefill_finish + transfer - request.arrival_ms,
            "decode_start": decode_start - request.arrival_ms,
        }
        values["decode_first_token"] = decode_start + first_delta - request.arrival_ms
        values["decode_queue_first_token"] = values["decode_first_token"]
        first_key = "decode_start" if ttft_mode == "decode_queue_start" else ttft_mode
        timings[request.req_id] = SimulatedRequestTiming(
            req_id=request.req_id,
            is_short=request.is_short,
            ttft_mode=ttft_mode,
            arrival_offset_s=request.arrival_ms / 1000,
            prompt_tokens=request.prompt_tokens,
            prefill_start_ms=prefill_start,
            prefill_finish_ms=prefill_finish,
            kv_transfer_ms=transfer,
            decode_start_ms=decode_start,
            decode_finish_ms=decode_finish,
            prefill_ttft_ms=values["prefill_finish"],
            decode_ready_ttft_ms=values["decode_ready"],
            decode_start_ttft_ms=values["decode_start"],
            decode_first_token_ttft_ms=values["decode_first_token"],
            decode_queue_first_token_ttft_ms=values["decode_queue_first_token"],
            first_latency_ms=values[first_key],
            finish_latency_ms=decode_finish - request.arrival_ms,
            tpot_ms=tpot,
        )
    wall = max(
        0.0,
        max(item.decode_finish_ms for item in timings.values())
        - min(request.arrival_ms for request in ordered),
    )
    return SimulationResult(timings, wall, len(ordered) * 1000 / wall if wall > 0 else 0.0)


def _stage_decode_costs(
    cost: DistServeStageCost, *, output_tokens: int, decode_service_scale: float
) -> tuple[float, float, float]:
    raw = max(0.0, float(cost.decode_service_ms))
    total = raw * decode_service_scale
    token_count = max(1, int(output_tokens))
    raw_first = cost.decode_first_token_ms
    if raw_first is None:
        raw_first = raw / token_count if raw > 0 else 0.0
    first = min(total, max(0.0, float(raw_first) * decode_service_scale))
    tail = (total - first) / (token_count - 1) if token_count > 1 else 0.0
    return total, first, tail


def _simulate_stage_cost_continuous_decode(
    arrivals: list[tuple[float, str, DistServeRequest]],
    costs: dict[str, DistServeStageCost],
    *,
    output_tokens: int,
    decode_service_scale: float,
    decode_batch_size: int,
    decode_batch_alpha: float,
) -> dict[str, tuple[float, float, float, float]]:
    pending = sorted(arrivals, key=lambda item: (item[0], item[1]))
    if not pending:
        return {}
    batch_size, output_tokens, alpha = (
        max(1, int(decode_batch_size)),
        max(1, int(output_tokens)),
        max(0.0, decode_batch_alpha),
    )
    now, cursor, ready, generated = pending[0][0], 0, [], {}
    starts, first_deltas, first_times, finishes = {}, {}, {}, {}

    def admit() -> int:
        index = cursor
        while index < len(pending) and pending[index][0] <= now + 1e-9:
            request = pending[index][2]
            ready.append(request)
            generated[request.req_id] = 0
            index += 1
        return index

    while cursor < len(pending) or ready:
        if not ready:
            now = max(now, pending[cursor][0])
        cursor = admit()
        if not ready:
            continue
        batch = ready[:batch_size]
        step_costs = []
        for request in batch:
            _, first, per_token = _stage_decode_costs(
                costs[request.req_id],
                output_tokens=output_tokens,
                decode_service_scale=decode_service_scale,
            )
            if request.req_id not in starts:
                starts[request.req_id], first_deltas[request.req_id] = now, first
            step_costs.append(first if generated[request.req_id] <= 0 else per_token)
        now += max(step_costs or [0.0]) * (1 + alpha * max(0, len(batch) - 1))
        finished = set()
        for request in batch:
            req_id = request.req_id
            generated[req_id] += 1
            if generated[req_id] == 1:
                first_times[req_id] = now
            if generated[req_id] >= output_tokens:
                finishes[req_id] = now
                finished.add(req_id)
        ready[:] = [request for request in ready if request.req_id not in finished]
    return {
        req_id: (
            starts[req_id],
            finishes[req_id],
            (finishes[req_id] - starts[req_id]) / output_tokens,
            first_times[req_id] - starts[req_id],
        )
        for _, req_id, _ in pending
    }


def simulate_distserve_from_stage_costs(
    requests: list[DistServeRequest],
    stage_costs: dict[str, DistServeStageCost],
    *,
    output_tokens: int,
    kv_model: KvModel,
    kv_profile: KvTransferProfile,
    resource_profile: ResourceProfile,
    ttft_mode: str = "prefill_finish",
    decode_replay_mode: str = "continuous_batching",
    decode_batch_size: int = 16,
    decode_batch_alpha: float = 0.08,
) -> SimulationResult:
    ordered = sorted(requests, key=lambda request: (request.arrival_ms, request.req_id))
    if not ordered:
        return _empty_result()
    _validate_mode(ttft_mode)
    missing = [request.req_id for request in ordered if request.req_id not in stage_costs]
    if missing:
        raise KeyError(f"missing DistServe stage costs for {len(missing)} requests: {missing[:5]}")
    workers = [0.0] * max(1, int(resource_profile.prefill_workers))
    heapq.heapify(workers)
    prefill = {}
    for request in ordered:
        start = max(request.arrival_ms, heapq.heappop(workers))
        finish = (
            start
            + max(0.0, stage_costs[request.req_id].prefill_service_ms)
            * resource_profile.prefill_service_scale
        )
        prefill[request.req_id] = (
            start,
            finish,
            kv_model.transfer_ms(request.prompt_tokens + 1, kv_profile),
        )
        heapq.heappush(workers, finish)
    arrivals = sorted(
        (prefill[request.req_id][1] + prefill[request.req_id][2], request.req_id, request)
        for request in ordered
    )
    output_tokens = max(1, int(output_tokens))
    if decode_replay_mode != "continuous_batching" or int(resource_profile.decode_workers) != 1:
        raise ValueError(
            "maintained DistServe replay requires continuous batching and one decode worker"
        )
    decode = _simulate_stage_cost_continuous_decode(
        arrivals,
        stage_costs,
        output_tokens=output_tokens,
        decode_service_scale=resource_profile.decode_service_scale,
        decode_batch_size=decode_batch_size,
        decode_batch_alpha=decode_batch_alpha,
    )
    return _finalize_distserve(ordered, prefill, decode, ttft_mode)


def summarize_timings(
    timings: dict[str, SimulatedRequestTiming], round_wall_ms: float
) -> dict[str, float | int | None]:
    rows = list(timings.values())
    scopes = {
        "all": rows,
        "short": [item for item in rows if item.is_short],
        "long": [item for item in rows if not item.is_short],
    }

    def values(scope, attr):
        return [float(getattr(item, attr)) for item in scope]

    output: dict[str, float | int | None] = {
        "request_count": len(rows),
        "short_request_count": len(scopes["short"]),
        "long_request_count": len(scopes["long"]),
        "measured_long_fraction": len(scopes["long"]) / len(rows) if rows else None,
        "round_wall_ms": round_wall_ms,
        "throughput_rps": len(rows) * 1000 / round_wall_ms if round_wall_ms > 0 else None,
        "tpot_mean_ms": sum(values(rows, "tpot_ms")) / len(rows) if rows else None,
        "kv_transfer_mean_ms": sum(values(rows, "kv_transfer_ms")) / len(rows) if rows else None,
    }
    for scope_name, scope in scopes.items():
        for metric, attr in (
            ("ttft", "first_latency_ms"),
            ("completion", "finish_latency_ms"),
            ("tpot", "tpot_ms"),
        ):
            samples = values(scope, attr)
            for label, pct in (("p50", 50), ("p90", 90), ("p99", 99)):
                output[f"{scope_name}_{metric}_{label}_ms"] = percentile(samples, pct)
    return output


def timing_dict_for_json(timing: SimulatedRequestTiming) -> dict[str, float | int | bool]:
    before_aliases = (
        "arrival_offset_s",
        "prompt_tokens",
        "ttft_mode",
        "first_latency_ms",
        "prefill_ttft_ms",
        "decode_ready_ttft_ms",
        "decode_start_ttft_ms",
        "decode_first_token_ttft_ms",
        "decode_queue_first_token_ttft_ms",
        "finish_latency_ms",
    )
    result = {key: getattr(timing, key) for key in before_aliases}
    result.update(
        scheduled_first_latency_ms=timing.first_latency_ms,
        scheduled_finish_latency_ms=timing.finish_latency_ms,
    )
    for key in (
        "prefill_start_ms",
        "prefill_finish_ms",
        "kv_transfer_ms",
        "decode_start_ms",
        "decode_finish_ms",
        "tpot_ms",
        "is_short",
    ):
        result[key] = getattr(timing, key)
    return result
