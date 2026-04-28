# CUCUMIS Experiment Workflow

[English](./README.md) | [简体中文](./README.zh-CN.md)

This README only documents the paper experiment path: Chapter 2 prestudy runs and Chapter 5 evaluation runs.

## Environment

Run all commands from the repository root. Activate the target environment before running experiments, for example:

```bash
conda activate sara
```

The experiment drivers use the active Python interpreter by default. Config files should use repository-relative paths and should not contain machine-local paths such as a user home directory, a Conda installation path, or a fixed Hugging Face cache path.

The current experiment stack expects:

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

- `data/lut_tables/raw_profile_<lut_name>.json`
- `data/lut_tables/lut_gain_<lut_name>.json`
- `data/lut_tables/lut_penalty_<lut_name>.json`
- `data/lut_tables/runtime_calibration_<lut_name>.json`
- `data/lut_tables/runtime_sanity_<lut_name>.json`

Use `--skip-preflight-lut-rebuild` only for debugging stale-LUT detection. With that flag, preflight refuses to write a usable resolved config when the selected LUTs do not match the current GPU fingerprint.

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
  --variants strict_no_chunk \
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
- `metadata/resolved_config.json`: config consumed by the Chapter 5 main stage.
- `metadata/preflight_summary.json`: short machine-readable summary.

## Chapter 2

Chapter 2 prestudy runs use:

- Driver: `experiments/chapter2_prestudy.py`
- Config: `experiments/configs/chapter2_prestudy_v1.json`

Run the full retained prestudy set:

```bash
python experiments/chapter2_prestudy.py all \
  --config experiments/configs/chapter2_prestudy_v1.json \
  --out-root results/chapter2_prestudy_v1/<run_id>
```

Run one case at a time:

```bash
python experiments/chapter2_prestudy.py e1 --config experiments/configs/chapter2_prestudy_v1.json
python experiments/chapter2_prestudy.py e2 --config experiments/configs/chapter2_prestudy_v1.json
python experiments/chapter2_prestudy.py e3 --config experiments/configs/chapter2_prestudy_v1.json
python experiments/chapter2_prestudy.py e3paper --config experiments/configs/chapter2_prestudy_v1.json
python experiments/chapter2_prestudy.py e4 --config experiments/configs/chapter2_prestudy_v1.json
python experiments/chapter2_prestudy.py e5 --config experiments/configs/chapter2_prestudy_v1.json
```

Retained Chapter 2 cases:

- `E1 Motivating Microbenchmark`: long-first and short-arrival motivating case.
- `E2 Arrival-Order Sensitivity`: arrival-order sensitivity under contention.
- `E3 Fixed Chunking vs Online Control`: no chunking, fixed chunking, and online control comparison.
- `E3 Paper Case`: stable paper-facing export from an existing beneficiary-rich result.
- `E4 Density Sweep`: TTFT and slowdown behavior as load increases.
- `E5 LoRA Multi-Tenancy Relevance`: non-LoRA, homogeneous LoRA, and mixed-adapter LoRA comparison.

Expected Chapter 2 path:

- `results/chapter2_prestudy_v1/<run_id>/`

The Chapter 2 output tree contains per-case summaries, request-level timing data, and figure files used for the paper motivation/observation section.

## Path Rules

- Run commands from the repository root.
- Keep experiment paths repository-relative.
- Configure persistent model/dataset choices in JSON configs.
- Use CLI filters only for temporary subsets or smoke runs.
- Let preflight produce the portable Chapter 5 resolved config on a new machine.
