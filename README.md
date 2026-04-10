# Wave-Slice

[English](./README.md) | [简体中文](./README.zh-CN.md)

Wave-Slice is a runtime scheduling layer for heterogeneous LoRA serving on top of vLLM.

At a high level, the current codebase contains two cooperating mechanisms:

1. `Phase-I`: dynamic slicing for long base-prefill requests.
2. `Phase-II`: selective scheduling for adapter-stage execution.

The implementation is injected at runtime and does not require modifying the upstream vLLM source tree.

## Repository Layout

- `engine/vllm_hijacker.py`
  - Main runtime injection entrypoint.
  - Contains the runtime hooks, coordination logic, and most runtime metrics.
- `engine/base_slicer.py`
  - Builds chunk proposals and slicing plans for long-prefill requests.
- `scheduler/wave_scheduler.py`
  - Contains the chunk selection logic used by Phase-I.
- `profiler/offline_profiler.py`
  - Collects raw latency metadata for LUT construction.
- `profiler/lut_generator.py`
  - Builds `lut_gain` and `lut_penalty` from raw profiling data.
- `tests/validate_wave_slice.py`
  - Sanity checks for hook installation and optional live LoRA validation.
- `tests/evaluate_waveslice_claims.py`
  - Main repeated experiment harness.
- `experiments/waveslice_a100_suite.py`
  - Multi-model synthetic workload suite.
- `experiments/build_dataset_workload.py`
  - Builds request JSON files from public datasets.
- `experiments/waveslice_dataset_suite.py`
  - Multi-model dataset-driven experiment suite.
- `experiments/profile_real_model_metadata.py`
  - Collects reusable model metadata.
- `tools/synthetic_lora_builder.py`
  - Builds synthetic adapters used by the experiment suites.

## Environment Requirements

Run all commands from the repository root.

Minimum software requirements used by the current codebase:

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

The repository now includes a `requirements.txt` with the Python dependencies used by the current experiments.

If your machine requires a specific CUDA wheel for PyTorch, install the correct PyTorch build first from the official PyTorch instructions for your CUDA version, then install the remaining packages from `requirements.txt`.

Typical setup from a fresh clone:

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
- access to the target Hugging Face model weights, either as:
  - a local path such as `/path/to/model_snapshot`, or
  - a Hugging Face model ID such as `mistralai/Mistral-7B-v0.1`

## Runtime Model Inputs

Most experiment scripts accept either a local model path or a Hugging Face model ID through `--model-path`.

For example, either of the following is valid:

```bash
--model-path /path/to/local/model_snapshot
```

```bash
--model-path mistralai/Mistral-7B-v0.1
```

Synthetic LoRA adapters are usually created automatically by the experiment scripts. If you want to provide them manually, pass them with:

```bash
--adapter-a /path/to/adapter_a
--adapter-b /path/to/adapter_b
```

## Lowest-Level Runtime API

If you want to inject Wave-Slice manually in your own vLLM program:

```python
from engine.vllm_hijacker import inject_wave_slice, uninject_wave_slice

inject_wave_slice("Mistral-7B-v0.1")
# create and run your vLLM engine here
uninject_wave_slice()
```

Most collaborators should use the provided scripts instead of calling this entrypoint directly.

## Recommended Order For A New Contributor

1. Validate that the hooks install correctly.
2. Build or inspect the raw profiling data and LUTs.
3. Run the repeated single-model harness.
4. Run the synthetic multi-model suite.
5. Run the dataset-driven suite.

## 1) Validate Hook Installation

Basic validation:

```bash
python tests/validate_wave_slice.py
```

Optional live LoRA validation:

```bash
python tests/validate_wave_slice.py --run-lora-live
```

## 2) Build Raw Profiles And LUTs

Run the offline profiler for one model:

```bash
python profiler/offline_profiler.py --model Mistral-7B-v0.1
```

Generate LUTs for all configured models:

```bash
python profiler/lut_generator.py --models all
```

Generate LUTs for a specific model:

```bash
python profiler/lut_generator.py --models Mistral-7B-v0.1
```

Override bucket granularity:

```bash
python profiler/lut_generator.py --models Mistral-7B-v0.1 --buckets 32,64,128,256,288,320,352,384,448,512,768,1024,1536,2048,3072,4096
```

Outputs are written under:

- `data/lut_tables/raw_profile_*.json`
- `data/lut_tables/lut_gain_*.json`
- `data/lut_tables/lut_penalty_*.json`

## 3) Main Repeated Experiment Harness

The main experiment entrypoint is:

- `tests/evaluate_waveslice_claims.py`

It supports:

- Phase-I only
- Phase-II only
- joint Phase-I + Phase-II evaluation
- synthetic prompt workloads
- external dataset-derived request JSON files
- warmup and repeated runs

### Example: synthetic repeated run

```bash
python tests/evaluate_waveslice_claims.py \
  --model-name Mistral-7B-v0.1 \
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

Useful flags:

- `--include-phase12`
- `--include-strict`
- `--skip-phase2`
- `--short-repeat`
- `--long-repeat`
- `--phase1-objective-mode`
- `--phase2-dispatch-mode synchronized|async_experimental`
- `--trust-remote-code`

### Example: dataset-driven run with external request files

```bash
python tests/evaluate_waveslice_claims.py \
  --model-name Mistral-7B-v0.1 \
  --model-path mistralai/Mistral-7B-v0.1 \
  --max-new-tokens 64 \
  --warmup-iters 2 \
  --repeats 3 \
  --max-model-len 3072 \
  --max-num-batched-tokens 1536 \
  --gpu-memory-utilization 0.6 \
  --phase2-dispatch-mode synchronized \
  --include-phase12 \
  --requests-json results/mistral_dataset_requests.json \
  --lora-requests-json results/mistral_dataset_lora_requests.json \
  --out-json results/mistral_phase12_dataset.json
```

### Preferred V1 Main Regression

When judging the current V1 path, prefer the frozen real `openworkload mid`
regression target instead of the tiny synthetic repeated default case:

```bash
python experiments/run_frozen_eval_config.py \
  --config experiments/configs/frozen_v1_gemma_mid_global_activity_repro.json
```

This uses:

- real open-source dataset-derived request JSON files
- mixed short and long requests
- Poisson arrivals
- the same `mid` workload family used for the V0 regression comparison

## 4) Build Dataset Workloads

Dataset workload builder:

- `experiments/build_dataset_workload.py`

The current builder samples from:

- `LongBench`
- `UltraChat200k`

Typical command:

```bash
python experiments/build_dataset_workload.py \
  --model-path mistralai/Mistral-7B-v0.1 \
  --out-prefix results/mistral_dataset_mix \
  --max-prompt-tokens 3008 \
  --sample-count 96
```

Outputs:

- `results/<prefix>_requests.json`
- `results/<prefix>_lora_requests.json`
- `results/<prefix>_meta.json`

## 5) Synthetic Multi-Model Suite

Synthetic suite runner:

- `experiments/waveslice_a100_suite.py`

List supported models:

```bash
python experiments/waveslice_a100_suite.py --list-models
```

Run the full synthetic suite:

```bash
python experiments/waveslice_a100_suite.py \
  --repeats 3 \
  --warmup-iters 2 \
  --max-new-tokens 64 \
  --include-phase12 \
  --timeout-sec 240
```

Run a subset of models:

```bash
python experiments/waveslice_a100_suite.py \
  --models mistral-7b-v0.1,mistral-7b-instruct-v0.2,zephyr-7b-beta \
  --repeats 3 \
  --warmup-iters 2 \
  --max-new-tokens 64 \
  --include-phase12 \
  --timeout-sec 240
```

Outputs:

- `results/waveslice_a100_suite_<timestamp>/`
- `results/waveslice_a100_suite_<timestamp>.csv`

## 6) Dataset Multi-Model Suite

Dataset suite runner:

- `experiments/waveslice_dataset_suite.py`

Run the full dataset suite:

```bash
python experiments/waveslice_dataset_suite.py \
  --repeats 3 \
  --timeout-sec 240
```

Run a subset:

```bash
python experiments/waveslice_dataset_suite.py \
  --models mistral-7b-v0.1,mistral-7b-instruct-v0.2,zephyr-7b-beta \
  --repeats 3 \
  --timeout-sec 240
```

Outputs:

- `results/waveslice_dataset_suite_<timestamp>/`
- `results/waveslice_dataset_suite_<timestamp>.csv`

## 7) Real-Model Metadata Profiling

Metadata profiler:

- `experiments/profile_real_model_metadata.py`

This script collects reusable model metadata such as:

- model architecture fields
- attention layout
- tokenizer information
- existing profiling artifact links
- dataset token-length summaries

Run it with:

```bash
python experiments/profile_real_model_metadata.py
```

Output example:

- `results/real_model_metadata_with_datasets.json`

## Reading Order For The Core Code

If you want a short reading path, start with:

1. `engine/vllm_hijacker.py`
2. `tests/evaluate_waveslice_claims.py`
3. `experiments/waveslice_a100_suite.py`

Those three files cover:

- runtime injection and coordination
- the main evaluation harness
- multi-model orchestration
