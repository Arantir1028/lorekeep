import re
from argparse import Namespace
from dataclasses import fields
from pathlib import Path

import pytest

from experiments.distserve_functional import (
    DistServeRequest,
    DistServeStageCost,
    KvModel,
    KvTransferProfile,
    ResourceProfile,
    simulate_distserve_from_stage_costs,
)
from experiments.openworkload_models import resolve_model_entry
from experiments.openworkload_support import load_config
from experiments.run_chapter5_suite import _stages
from experiments.run_environment_preflight import _derive_densities, _derive_workload_overrides
from experiments.run_frozen_eval_config import (
    apply_eval_config,
    build_eval_invocation,
    validate_eval_config,
)
from scripts import (
    run_cucumis_2a100_dispatch_sweep as cucumis_runner,
    run_hardware_portability_experiment as hardware_runner,
)
from waveslice.lut import config as lut_config
from waveslice.lut.loader import load_model_luts
from waveslice.policy import WaveSlicePolicy
from waveslice.scheduling.scheduler import WaveScheduler
from waveslice.scheduling.slicer import WaveBaseSlicer


def test_runtime_sources_do_not_pin_a_user_home_directory() -> None:
    user_home = re.compile(r"/(?:home|Users)/[^/\s\"']+/|/root/")
    roots = (
        Path("waveslice"),
        Path("experiments"),
        Path("profiler"),
        Path("scripts"),
        Path("tools"),
    )
    suffixes = {".py", ".sh", ".json", ".toml", ".yaml", ".yml"}
    offenders = []
    for root in roots:
        for path in root.rglob("*"):
            if (
                path.is_file()
                and path.suffix in suffixes
                and user_home.search(path.read_text(encoding="utf-8"))
            ):
                offenders.append(str(path))
    assert not offenders, f"machine-specific home paths found: {offenders}"


def test_policy_reference_covers_every_runtime_field() -> None:
    reference = Path("docs/configuration.md").read_text(encoding="utf-8")
    missing = [
        field.name
        for field in fields(WaveSlicePolicy)
        if f"`{field.name}`" not in reference
    ]
    assert not missing, f"policy fields missing from configuration reference: {missing}"


def test_scheduler_and_slicer_contract() -> None:
    scheduler = WaveScheduler(
        model_name="Mistral-7B-v0.1", gamma=0.5, max_queue_depth=100
    )
    for wait, depth in ((0, 0), (50, 50), (500, 100)):
        chunk = scheduler.schedule_real(S_s=45, S_l=2048, t_wait_us=wait, queue_length=depth)
        assert 45 <= chunk <= 2048
    plans = WaveBaseSlicer().build_long_prefill_plan(
        short_len=96, long_total_len=1500, scheduler=scheduler, t_wait_us=800, queue_length=4
    )
    assert sum(plan.chunk_len for plan in plans) == 1500
    assert [plan.long_offset for plan in plans] == [0, plans[0].chunk_len]


def test_lut_loader_does_not_fall_back_to_generic_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "raw_profile.json").write_text(
        '{"T_solo": {"256": 1.0}}', encoding="utf-8"
    )
    (tmp_path / "lut_gain.json").write_text("{}", encoding="utf-8")
    (tmp_path / "lut_penalty.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(lut_config, "DATA_DIR", str(tmp_path))

    with pytest.raises(FileNotFoundError, match="missing WaveSlice LUT files"):
        load_model_luts("missing-model")


def test_resource_catalog_keeps_full10_model_resolution_stable() -> None:
    entries = load_config("experiments/configs/openworkload_resources_full10.json")["models"]
    models = [resolve_model_entry(entry) for entry in entries]

    assert [(model.key, model.model_id, model.lut_name) for model in models] == [
        (
            "mistral-7b-v0.1",
            "mistralai/Mistral-7B-v0.1",
            "mistralai--Mistral-7B-v0.1",
        ),
        (
            "mistral-7b-instruct-v0.2",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "mistralai--Mistral-7B-Instruct-v0.2",
        ),
        (
            "zephyr-7b-beta",
            "HuggingFaceH4/zephyr-7b-beta",
            "HuggingFaceH4--zephyr-7b-beta",
        ),
        (
            "openchat-3.5-0106",
            "openchat/openchat-3.5-0106",
            "openchat--openchat-3.5-0106",
        ),
        ("gemma-7b-it", "google/gemma-7b-it", "google--gemma-7b-it"),
        (
            "baichuan2-7b-chat",
            "baichuan-inc/Baichuan2-7B-Chat",
            "baichuan-inc--Baichuan2-7B-Chat",
        ),
        ("gpt-j-6b", "EleutherAI/gpt-j-6b", "EleutherAI--gpt-j-6b"),
        ("pythia-6.9b", "EleutherAI/pythia-6.9b", "EleutherAI--pythia-6.9b"),
        (
            "qwen2.5-7b-instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen--Qwen2.5-7B-Instruct",
        ),
        ("gemma-2-9b-it", "google/gemma-2-9b-it", "google--gemma-2-9b-it"),
    ]


def test_hardware_lut_selection_uses_the_shared_model_catalog() -> None:
    args = Namespace(
        lut_models="selected",
        models=(
            "baichuan2-7b-chat,gemma-2-9b-it,gemma-7b-it,"
            "mistral-7b-instruct-v0.2,qwen2.5-7b-instruct"
        ),
    )

    assert hardware_runner._lut_models(args) == (
        "baichuan-inc/Baichuan2-7B-Chat,google/gemma-2-9b-it,google/gemma-7b-it,"
        "mistralai/Mistral-7B-Instruct-v0.2,Qwen/Qwen2.5-7B-Instruct"
    )


def test_continuous_distserve_contract() -> None:
    requests = [DistServeRequest(key, 0.0, 10, key == "b") for key in ("a", "b")]
    costs = {key: DistServeStageCost(key, 1.0, 8.0, 2.0) for key in ("a", "b")}
    result = simulate_distserve_from_stage_costs(
        requests,
        costs,
        output_tokens=4,
        kv_model=KvModel(1, 1, 1),
        kv_profile=KvTransferProfile("test", "test", 0, 0),
        resource_profile=ResourceProfile(
            "distserve_2a100", "DistServe-2A100", "DistServe", 2, 1, 1, 1, 2
        ),
        decode_batch_size=2,
        decode_batch_alpha=0,
    )
    assert result.round_wall_ms == 9
    assert all(
        timing.decode_start_ms == 1 and timing.finish_latency_ms == 9
        for timing in result.request_timings.values()
    )


def test_distserve_first_token_split_conserves_total_decode_service() -> None:
    request = DistServeRequest("only", 0.0, 10, True)
    result = simulate_distserve_from_stage_costs(
        [request],
        {"only": DistServeStageCost("only", 1.0, 10.0, 4.0)},
        output_tokens=4,
        kv_model=KvModel(1, 1, 1),
        kv_profile=KvTransferProfile("test", "test", 0, 0),
        resource_profile=ResourceProfile("test", "test", "test", 1, 1, 1, 1, 1),
        decode_batch_size=1,
        decode_batch_alpha=0,
    )
    timing = result.request_timings["only"]
    assert timing.decode_first_token_ttft_ms == 5
    assert timing.finish_latency_ms == 11


def test_chapter5_config_and_capacity_contract() -> None:
    assert _stages("all") == ["preflight", "main", "baseline"]
    config = load_config("experiments/configs/openworkload_ratio_sweep_lora8.json")
    assert "vllm_mode" not in config["eval"]
    assert [item["name"] for item in config["workload"]["densities"]] == [
        f"{level}_l{pct}" for level in ("mid", "high") for pct in (10, 30, 50, 70, 90)
    ]
    capacity = {
        "eval": {
            "max_num_batched_tokens": 1536,
            "max_new_tokens": 64,
            "repeats": 2,
            "warmup_iters": 1,
        },
        "workload": {"sample_count": 256},
    }
    density = {
        "name": "low",
        "phase1_arrival_rate": 4,
        "phase2_arrival_rate": 4,
        "phase1_short_count": 24,
        "phase1_long_count": 8,
        "phase2_short_count": 24,
        "phase2_long_count": 12,
    }
    overrides, meta = _derive_workload_overrides(
        config=capacity, runtime_cfg={"max_num_batched_tokens": 768}, memory_gb=10
    )
    resolved, _ = _derive_densities(
        [density], {"max_num_batched_tokens": 768}, capacity["eval"], meta
    )
    assert (
        overrides["eval"]["max_new_tokens"],
        overrides["eval"]["repeats"],
        overrides["workload"]["sample_count"],
    ) == (32, 1, 90)
    assert (resolved[0]["phase1_short_count"], resolved[0]["phase2_long_count"]) == (8, 4)


def test_replica_partial_launch_failure_terminates_started_process(
    tmp_path: Path, monkeypatch
) -> None:
    launched = []

    class Process:
        alive = True

        def poll(self):
            return None if self.alive else 0

    def popen(*_args, **kwargs):
        if launched:
            raise OSError("second launch failed")
        process = Process()
        process.stdout, process.stderr = kwargs["stdout"], kwargs["stderr"]
        launched.append(process)
        return process

    monkeypatch.setattr(cucumis_runner.subprocess, "Popen", popen)
    monkeypatch.setattr(
        cucumis_runner, "_replica_invocation", lambda *_args, **_kwargs: (["fake"], {})
    )
    monkeypatch.setattr(cucumis_runner, "_outer_timeout_sec", lambda _path: 10)
    monkeypatch.setattr(
        cucumis_runner, "_terminate_process_group", lambda process: setattr(process, "alive", False)
    )
    with pytest.raises(OSError, match="second launch failed"):
        cucumis_runner._run_replicas(
            replica_ids=[0, 1],
            config_paths=[Path("a"), Path("b")],
            out_paths=[Path("x"), Path("y")],
            log_root=tmp_path,
            cuda_devices=["0", "1"],
        )
    assert not launched[0].alive
    assert launched[0].stdout.closed and launched[0].stderr.closed


def test_eval_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="repeatz"):
        build_eval_invocation({"runtime": {"repeatz": 2}})


def test_eval_config_maps_model_schema_to_runtime_names() -> None:
    args = type("Args", (), {})()
    apply_eval_config(args, {"model": {"name": "gemma", "path": "/models/gemma"}})
    assert (args.model_name, args.model_path) == ("gemma", "/models/gemma")
    assert not hasattr(args, "name") and not hasattr(args, "path")


def test_legacy_cucumis_source_drops_retired_policy_fields(tmp_path: Path) -> None:
    source = {
        "model_name": "model",
        "model_path": "/model",
        "phase1_gamma": 0.5,
        "phase12_phase2_gate_mode": "soft",
        "phase2_enable_scheduler_cashout": True,
        "phase2_enable_execution_escape": True,
        "phase2_execution_escape_max_active": 2,
    }
    config = cucumis_runner._eval_config_from_source(
        source, tmp_path / "requests.json", tmp_path / "lora.json"
    )
    validate_eval_config(config)
    assert config["phase1"]["gamma"] == 0.5
    assert config["phase12_soft_gate"]["phase2_gate_mode"] == "soft"
    assert config["phase2"] == {"enable_scheduler_cashout": True}
