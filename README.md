# CUCUMIS

[English](./README.md) | [简体中文](./README.zh-CN.md)

CUCUMIS is a runtime scheduling layer for heterogeneous LLM serving on top of vLLM.

## Current Scope

## Current Method Definition

The current paper and experiment path is:

- **Phase I:** control long-prefill chunk size so long requests return to the scheduler boundary earlier.
- **Phase II:** use execution escape / priority promotion to reshape the next scheduled window after that boundary is exposed.
- **Adaptation:** use runtime queue-pressure adaptation inside the scheduler to avoid over-triggering under low pressure while still reacting to live queues.

This is not the abandoned `true_unbind` / dual-stream path. Current results and paper text should not describe execution-level long/short co-running.

The adaptive policy is runtime queue-pressure adaptation. Configured density controls the workload, while live queue length, waiting short requests, wait urgency, long remaining tokens, and virtual-cap hits drive the active chunk/gate choice. See `docs/current_method.md` for the precise definition.

Supported experiment entrypoints:

- `experiments/run_chapter5_suite.py`
  - Chapter 5 pipeline driver. Can run the main suite, baseline variants, figure regeneration, or the full end-to-end workflow.
- `experiments/chapter2_prestudy.py`
  - Chapter 2 prestudy with public cases `E1`-`E5` plus `e3paper` for figure export.
- `tests/evaluate_waveslice_claims.py`
  - Core repeated evaluation harness that emits TTFT, slowdown, wall time, per-repeat metrics, and request-level timings.
- `experiments/run_frozen_eval_config.py`
  - Frozen single-case regression runner.
- `experiments/run_openworkload_execescape_suite.py`
  - Chapter 5 main dataset-backed open-workload suite for the current v1 path.
- `experiments/run_chapter5_baseline_variants.py`
  - Chapter 5 baseline and mechanism-ablation runner that reuses workloads from an existing main-suite run.
- `scripts/regenerate_chapter5_main_outputs.py`
  - Rebuilds the main Chapter 5 figures, tables, and summary markdown from a main run plus a baseline-variant run.
- `scripts/regenerate_chapter5_partial_figures.py`
  - Rebuilds the density-sweep and optional LoRA-latency-dispersion figures used alongside the Chapter 5 study.

## Repository Layout

- `engine/vllm_hijacker.py`: runtime injection entrypoint and vLLM hook coordination.
- `engine/runtime_bootstrap.py`: runtime bootstrap for vLLM mode and CUDA platform fixes.
- `tests/evaluate_waveslice_claims.py`: core harness for TTFT, slowdown, wall time, and mismatch reporting.
- `experiments/run_chapter5_suite.py`: Chapter 5 all-in-one and stage-by-stage orchestrator.
- `experiments/chapter2_prestudy.py`: current Chapter 2 prestudy driver.
- `experiments/run_openworkload_execescape_suite.py`: current real-workload v1 suite.
- `experiments/run_chapter5_baseline_variants.py`: Chapter 5 baseline and ablation runner.
- `experiments/build_dataset_workload.py`: request builder used by the dataset-backed suites.
- `config/experiment_catalog.py`: supported model catalog.

## Environment

Run all commands from the repository root.

Minimum software versions used by the current codebase:

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

Typical setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

You also need:

- a CUDA-visible GPU
- a working CUDA/PyTorch installation compatible with your GPU
- access to the target Hugging Face model weights, either as a local snapshot or a Hugging Face model ID

## Quick Start

Validate that the hooks load correctly:

```bash
python tests/validate_wave_slice.py
```

Run the preferred frozen v1 regression target:

```bash
python experiments/run_frozen_eval_config.py \
  --config experiments/configs/frozen_v1_gemma_mid_global_activity_repro.json
```

Run the current open-workload suite:

```bash
python experiments/run_openworkload_execescape_suite.py \
  --config experiments/configs/openworkload_execescape_default.json
```

Run the core harness directly on a single case:

```bash
python tests/evaluate_waveslice_claims.py \
  --model-name mistralai--Mistral-7B-v0.1 \
  --model-path mistralai/Mistral-7B-v0.1 \
  --max-new-tokens 64 \
  --warmup-iters 2 \
  --repeats 3 \
  --max-model-len 3072 \
  --max-num-batched-tokens 1536 \
  --gpu-memory-utilization 0.6 \
  --phase2-dispatch-mode synchronized \
  --include-phase12 \
  --out-json results/mistral_eval.json
```

## Chapter 5 Workflow

Chapter 5 includes four stages:

1. `main`: run the dataset-backed open-workload sweep across the selected models and densities.
2. `baseline`: rerun the comparable baseline and mechanism-ablation variants on the exact workloads produced by `main`.
3. `figures`: regenerate the main Chapter 5 figures, tables, and summary markdown from `main` + `baseline`.
4. `partial-figures`: optionally regenerate the density sweep and LoRA latency dispersion figures.

Relevant configs:

- pipeline config: `experiments/configs/chapter5_pipeline_default.json`
- main suite config: `experiments/configs/openworkload_v1_local_realworld_lora8.json`
- baseline config: `experiments/configs/chapter5_baseline_variants_lora7.json`

Path convention:

- Output roots, run roots, and example commands in this section use project-relative paths from the repository root.
- The default Chapter 5 main config uses the configured `eval.python_bin` from the suite config. On this machine the current v1 open-workload config points at the `sara` conda environment.

Model and dataset configuration:

- Edit the main-suite config for a persistent setup.
  The most important fields are `resource_selection`, `models`, `datasets`, and `workload.densities`.
- Or use CLI overrides:
  `--model-keys`, `--dataset-keys`, and `--densities`.
- The baseline stage needs a `source_run_root`, provided either by `--source-run-root` or by `experiments/run_chapter5_suite.py`.

Run the full Chapter 5 workflow:

```bash
python experiments/run_chapter5_suite.py \
  --config experiments/configs/chapter5_pipeline_default.json \
  --run-tag chapter5_demo
```

Run only the Chapter 5 main suite:

```bash
python experiments/run_chapter5_suite.py \
  --stages main \
  --run-tag chapter5_demo \
  --model-keys mistral-7b-v0.1,gemma-7b-it \
  --dataset-keys ultrachat200k,longbench \
  --densities mid,high
```

Run the Chapter 5 baseline variants on an existing main run:

```bash
python experiments/run_chapter5_suite.py \
  --stages baseline \
  --main-run-root results/openworkload_v1_local_realworld_lora8/chapter5_demo_main \
  --run-tag chapter5_demo \
  --variants strict_no_chunk \
  --model-keys mistral-7b-v0.1 \
  --densities mid
```

Regenerate the main Chapter 5 figures and tables from existing runs:

```bash
python experiments/run_chapter5_suite.py \
  --stages figures \
  --main-run-root results/openworkload_v1_local_realworld_lora8/chapter5_demo_main \
  --baseline-run-root results/chapter5_baseline_variants/chapter5_demo_baseline \
  --export-name chapter5_demo
```

Regenerate the optional partial figures:

```bash
python experiments/run_chapter5_suite.py \
  --stages partial-figures \
  --main-run-root results/openworkload_v1_local_realworld_lora8/chapter5_demo_main \
  --e5-summary results/chapter2_prestudy_v1/<run_id>/E5_lora_multitenancy_relevance/summary_all_models.json \
  --export-name chapter5_demo
```

You can also run each stage directly:

```bash
python experiments/run_openworkload_execescape_suite.py \
  --config experiments/configs/openworkload_v1_local_realworld_lora8.json \
  --run-name chapter5_demo_main \
  --model-keys mistral-7b-v0.1,gemma-7b-it \
  --dataset-keys ultrachat200k,longbench \
  --densities mid
```

```bash
python experiments/run_chapter5_baseline_variants.py \
  --config experiments/configs/chapter5_baseline_variants_lora7.json \
  --source-run-root results/openworkload_v1_local_realworld_lora8/chapter5_demo_main \
  --run-name chapter5_demo_baseline \
  --variants fixed_chunk_vs_sarathi \
  --model-keys mistral-7b-v0.1 \
  --densities mid
```

```bash
python scripts/regenerate_chapter5_main_outputs.py \
  --main-run results/openworkload_v1_local_realworld_lora8/chapter5_demo_main \
  --baseline-run results/chapter5_baseline_variants/chapter5_demo_baseline \
  --out-dir results/chapter5_exports/chapter5_demo
```

```bash
python scripts/regenerate_chapter5_partial_figures.py \
  --main-run results/openworkload_v1_local_realworld_lora8/chapter5_demo_main \
  --out-dir results/chapter5_exports/chapter5_demo \
  --e5-summary results/chapter2_prestudy_v1/<run_id>/E5_lora_multitenancy_relevance/summary_all_models.json
```

Expected outputs:

- main suite: `<main out_root>/<run_tag>_main/`
- baseline suite: `<baseline out_root>/<run_tag>_baseline/`
- exported Chapter 5 figures/tables: `<figures_out_root>/<run_tag>/`

The export directory contains `chapter5_manifest.json`, `chapter5_summary.md`, per-repeat CSV/JSON exports, and the regenerated figures/tables.

## Chapter 2 Prestudy

Chapter 2 prestudy entrypoint:

- `experiments/chapter2_prestudy.py`
- config: `experiments/configs/chapter2_prestudy_v1.json`

Run the full public prestudy set:

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

Supported prestudy cases:

- `E1 Motivating Microbenchmark`: long-first then short-arrivals motivating example.
  Outputs include timelines, short-request TTFT/completion plots, wall-time plots, and `summary.json`.
- `E2 Arrival-Order Sensitivity`: shows how arrival order changes interference severity.
  Outputs include TTFT and completion-slowdown comparisons.
- `E3 Fixed Chunking vs Online Control`: compares no chunking, fixed chunking, and online control.
  Outputs include TTFT and wall-time comparisons.
- `E3 Paper Case`: re-exports the stable beneficiary-rich figure from an existing result.
- `E4 Density Sweep`: shows how TTFT inflation and slowdown evolve with contention.
- `E5 LoRA Multi-Tenancy Relevance`: compares non-LoRA, homogeneous LoRA, and mixed-adapter LoRA.

## Outputs

The core harness writes a summary JSON containing the metrics that matter for the current project:

- `phase1`, `phase2`, `phase12`: aggregate TTFT, slowdown, wall-time, and error summaries
- `per_repeat`: repeat-level raw metrics for the same blocks
- `request_timings`: request-level first-token and finish latencies for each repeat
- `hook_report`: runtime instrumentation exported from the hijacker

The open-workload suite writes:

- `metadata/`: resolved config, manifest CSV/JSON, and aggregated summaries
- `raw/`: per-model per-density result JSONs
- `workloads/`: generated request JSONs
- `figures/`: suite-level plots

## Reading Order

If you want the shortest useful reading path, start with:

1. `engine/vllm_hijacker.py`
2. `tests/evaluate_waveslice_claims.py`
3. `experiments/chapter2_prestudy.py`
4. `experiments/run_openworkload_execescape_suite.py`
