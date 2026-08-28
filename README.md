# WaveSlice

[English](./README.md) | [简体中文](./README.zh-CN.md)

WaveSlice is a V1 scheduler extension for vLLM. It is installed as a normal
Python package, leaves the vLLM source tree untouched, and is enabled
declaratively through `EngineArgs`. This repository also contains the CUCUMIS
paper's Chapter 2 and Chapter 5 experiment workflows.

## Environment

Run all commands from the repository root. Activate the target environment before running experiments, for example:

```bash
conda activate sara
```

The experiment drivers use the active Python interpreter by default. Config files should use repository-relative paths and should not contain machine-local paths such as a user home directory, a Conda installation path, or a fixed Hugging Face cache path.

The recorded experiment environment uses the following versions. The WaveSlice
package itself does not enforce a vLLM version number; the current integration
targets the V1 APIs present in this environment.

- Python `3.10`
- PyTorch `2.7.1+cu126`
- vLLM `0.10.1`
- Transformers `4.57.6`
- Datasets `4.8.4`
- PEFT `0.18.1`
- Safetensors `0.5.3`
- Hugging Face Hub `0.36.2`
- NumPy `2.2.6`
- tqdm `4.67.1`

Install the repository package in the active environment:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Python API

WaveSlice extends vLLM's `EngineArgs`; it does not require a patched vLLM
checkout or a separate injection call. WaveSlice currently supports the vLLM
V1 engine only.

```python
from waveslice import EngineArgs, WaveSliceConfig
from vllm.engine.llm_engine import LLMEngine

engine_args = EngineArgs(
    model="/models/Mistral-7B-v0.1",
    enable_wave_slice=True,
    wave_slice_config=WaveSliceConfig(
        # Used to select LUT files; this is not the model checkpoint path.
        lut_model="Mistral-7B-v0.1",
        gamma=2.0,
    ),
    enable_chunked_prefill=True,
    max_num_batched_tokens=1536,
    max_num_partial_prefills=1,
    max_long_partial_prefills=1,
    max_model_len=4096,
    enforce_eager=True,
)
engine = LLMEngine.from_engine_args(engine_args)
```

Set `enable_wave_slice=False` to use the native vLLM scheduler path. If
`lut_model` is omitted, WaveSlice infers it from the last component of
`model`. Runtime metrics remain available through
`waveslice.get_wave_slice_metrics()`.

Models and datasets are selected by config. With `resource_selection.auto_download=true`, missing Hugging Face assets are downloaded automatically. Set `resource_selection.offline=true` only when the target machine already has a complete local cache. If a gated model is not accessible, the Hugging Face or vLLM error should be handled by granting access or logging in with the right token.

## Chapter 5

On a new GPU, Chapter 5 preflight checks the hardware fingerprint stored in each LUT. Missing, stale, or mismatched LUTs are rebuilt automatically before the main experiment starts. The LUT builder uses the active Python environment and local Hugging Face snapshots; if a model is not cached, the normal model download policy in the config applies.

```bash
python experiments/run_chapter5_suite.py \
  --config experiments/configs/chapter5_pipeline_default.json \
  --stages preflight \
  --run-tag chapter5_demo
```

After preflight succeeds, Chapter 5 is run through the same driver:

```bash
python experiments/run_chapter5_suite.py \
  --config experiments/configs/chapter5_pipeline_default.json \
  --run-tag chapter5_demo
```

The pipeline has five stages:

- `preflight`: detect the current GPU/software environment, probe runnable model settings, and write a resolved config.
- `main`: run the dataset-backed open-workload evaluation.
- `baseline`: rerun baseline and ablation variants on the exact workloads produced by `main`.
- `figures`: regenerate the main Chapter 5 figures, tables, and summary markdown.
- `partial-figures`: regenerate optional density-sweep and LoRA-dispersion figures.

Useful configs:

- Pipeline config: `experiments/configs/chapter5_pipeline_default.json`
- Main experiment config: `experiments/configs/openworkload_v1_local_realworld_lora8.json`
- Baseline/ablation config: `experiments/configs/chapter5_baseline_variants_lora7.json`

The LUT builder writes or refreshes:

- `waveslice/data/lut_tables/raw_profile_<lut_name>.json`
- `waveslice/data/lut_tables/lut_gain_<lut_name>.json`
- `waveslice/data/lut_tables/lut_penalty_<lut_name>.json`
- `waveslice/data/lut_tables/runtime_calibration_<lut_name>.json`
- `waveslice/data/lut_tables/runtime_sanity_<lut_name>.json`

Use `--skip-preflight-lut-rebuild` only for debugging stale-LUT detection. With that flag, preflight refuses to write a usable resolved config when the selected LUTs do not match the current GPU fingerprint.

Preflight also scales workload pressure for smaller GPUs. It writes the final values into `metadata/resolved_config.json` and records the decision in `metadata/workload_capacity.json`. The automatic adjustments cover `max_new_tokens`, `repeats`, `sample_count`, per-density request counts, arrival-rate scale, and dropping `peak` when capacity is too low.

Run only the environment preflight:

```bash
python experiments/run_chapter5_suite.py \
  --config experiments/configs/chapter5_pipeline_default.json \
  --stages preflight \
  --run-tag chapter5_demo
```

For a fast metadata/config check without loading a vLLM engine:

```bash
python experiments/run_chapter5_suite.py \
  --config experiments/configs/chapter5_pipeline_default.json \
  --stages preflight \
  --run-tag chapter5_demo \
  --skip-preflight-engine-smoke
```

Run a small trial before the full sweep:

```bash
python experiments/run_chapter5_suite.py \
  --config experiments/configs/chapter5_pipeline_default.json \
  --run-tag chapter5_trial \
  --model-keys mistral-7b-v0.1 \
  --dataset-keys ultrachat200k \
  --densities mid
```

Run only the main stage:

```bash
python experiments/run_chapter5_suite.py \
  --config experiments/configs/chapter5_pipeline_default.json \
  --stages main \
  --run-tag chapter5_demo \
  --model-keys mistral-7b-v0.1,gemma-7b-it \
  --dataset-keys ultrachat200k,longbench \
  --densities mid,high
```

Run only baseline/ablation variants from an existing main run:

```bash
python experiments/run_chapter5_suite.py \
  --config experiments/configs/chapter5_pipeline_default.json \
  --stages baseline \
  --main-run-root results/openworkload_v1_local_realworld_lora8/chapter5_demo_main \
  --run-tag chapter5_demo \
  --variants priority_no_chunk \
  --model-keys mistral-7b-v0.1 \
  --densities mid
```

Regenerate figures from existing runs:

```bash
python experiments/run_chapter5_suite.py \
  --config experiments/configs/chapter5_pipeline_default.json \
  --stages figures \
  --main-run-root results/openworkload_v1_local_realworld_lora8/chapter5_demo_main \
  --baseline-run-root results/chapter5_baseline_variants/chapter5_demo_baseline \
  --export-name chapter5_demo
```

Common filters:

- `--model-keys`: comma-separated model keys from the main config.
- `--dataset-keys`: comma-separated dataset keys from the main config.
- `--densities`: comma-separated workload densities.
- `--variants`: comma-separated baseline/ablation variants for the `baseline` stage.

Expected Chapter 5 paths:

- Preflight run: `results/openworkload_v1_local_realworld_lora8/<run_tag>_preflight/`
- Main run: `results/openworkload_v1_local_realworld_lora8/<run_tag>_main/`
- Baseline run: `results/chapter5_baseline_variants/<run_tag>_baseline/`
- Figure/table export: `results/chapter5_exports/<run_tag>/`

Important preflight artifacts:

- `metadata/resolved_environment.json`: detected Python, CUDA, package, and GPU information.
- `metadata/model_preflight.json`: per-model smoke/capacity status.
- `metadata/runtime_capacity.json`: selected runtime capacity settings.
- `metadata/workload_capacity.json`: selected workload scale, decode length, repeats, sample count, and density/request-count changes.
- `metadata/resolved_config.json`: config consumed by the Chapter 5 main stage.
- `metadata/preflight_summary.json`: short machine-readable summary.

## Chapter 2

The maintained Chapter 2 path rebuilds fixed-vLLM observations without loading
CUCUMIS hooks:

- Driver: `experiments/run_chapter2_observations.py`
- Config: `experiments/configs/chapter2_observations_v2.json`

Run both retained observations:

```bash
python experiments/run_chapter2_observations.py \
  --config experiments/configs/chapter2_observations_v2.json \
  --observations all \
  --run-name chapter2_demo
```

Use `--observations obs1` for the one-vs-two-long comparison or `obs2` for the
fixed-token-budget sweep. Outputs are written below
`results/chapter2_observations_v2/<run_name>/` with a manifest, workloads, raw
request timings, logs, and `summary.json`.

## Path Rules

- Run commands from the repository root.
- Keep experiment paths repository-relative.
- Configure persistent model/dataset choices in JSON configs.
- Use CLI filters only for temporary subsets or smoke runs.
- Let preflight produce the portable Chapter 5 resolved config on a new machine.
