# Wave-Slice

[English](./README.md) | [简体中文](./README.zh-CN.md)

Wave-Slice 是一个构建在 vLLM 之上的运行时调度层，用来处理异构 LoRA 推理场景。

从当前代码结构来看，主线包含两个相互配合的机制：

1. `Phase-I`：针对长 Base-prefill 请求的动态切分。
2. `Phase-II`：针对适配器阶段的选择性调度。

整个实现通过运行时注入完成，不需要修改上游 vLLM 源码。

## 仓库结构

- `engine/vllm_hijacker.py`
  - 运行时注入主入口。
  - 包含主要 hook、联合协调逻辑和大部分运行时指标。
- `engine/base_slicer.py`
  - 为长 prefill 请求生成 chunk proposal 和切分计划。
- `scheduler/wave_scheduler.py`
  - Phase-I 使用的 chunk 选择逻辑。
- `profiler/offline_profiler.py`
  - 采集用于构建 LUT 的原始延迟元数据。
- `profiler/lut_generator.py`
  - 从 raw profiling 数据生成 `lut_gain` 和 `lut_penalty`。
- `tests/validate_wave_slice.py`
  - hook 安装检查和可选的 live LoRA 验证。
- `tests/evaluate_waveslice_claims.py`
  - 主重复实验脚本。
- `experiments/waveslice_a100_suite.py`
  - 多模型 synthetic workload 套件。
- `experiments/build_dataset_workload.py`
  - 从公开数据集构建 requests JSON。
- `experiments/waveslice_dataset_suite.py`
  - 多模型 dataset-driven 套件。
- `experiments/profile_real_model_metadata.py`
  - 收集可复用模型元数据。
- `tools/synthetic_lora_builder.py`
  - 构建实验中使用的 synthetic adapter。

## 环境要求


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


如果你的机器需要特定 CUDA 版本的 PyTorch 轮子，建议先按照 PyTorch 官方安装说明装好匹配的 PyTorch，再安装 `requirements.txt` 中剩余依赖。

从零开始的典型安装方式：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

除此之外，你还需要：

- 一张 CUDA 可见的 GPU
- 与本机 GPU 兼容的 CUDA / PyTorch 环境
- 目标 Hugging Face 模型权重，可通过以下两种形式提供：
  - 本地路径，例如 `/path/to/model_snapshot`
  - Hugging Face 模型 ID，例如 `mistralai/Mistral-7B-v0.1`

## 模型输入方式

大部分实验脚本的 `--model-path` 同时支持：

```bash
--model-path /path/to/local/model_snapshot
```

或者：

```bash
--model-path mistralai/Mistral-7B-v0.1
```

如果希望手动指定 LoRA adapter，可以通过下面两个参数传入：

```bash
--adapter-a /path/to/adapter_a
--adapter-b /path/to/adapter_b
```

如果不手动提供，实验套件通常会自动构建 synthetic adapter。

## 最底层运行接口

如果你想在自己的 vLLM 程序里手动注入 Wave-Slice，可以这样写：

```python
from engine.vllm_hijacker import inject_wave_slice, uninject_wave_slice

inject_wave_slice("Mistral-7B-v0.1")
# 在这里创建并运行你的 vLLM engine
uninject_wave_slice()
```

大多数协作者更适合直接使用仓库脚本，而不是从这个接口开始。

## 建议的新协作者上手顺序

1. 先验证 hook 是否安装成功。
2. 构建或检查 raw profile 和 LUT。
3. 运行单模型 repeated harness。
4. 运行 synthetic 多模型套件。
5. 运行 dataset-driven 套件。

## 1）验证 Hook 安装

基础验证：

```bash
python tests/validate_wave_slice.py
```

可选的 live LoRA 验证：

```bash
python tests/validate_wave_slice.py --run-lora-live
```

## 2）构建 Raw Profile 和 LUT

为单个模型运行离线 profiler：

```bash
python profiler/offline_profiler.py --model Mistral-7B-v0.1
```

为所有已配置模型生成 LUT：

```bash
python profiler/lut_generator.py --models all
```

为单个模型生成 LUT：

```bash
python profiler/lut_generator.py --models Mistral-7B-v0.1
```

覆盖 bucket 粒度：

```bash
python profiler/lut_generator.py --models Mistral-7B-v0.1 --buckets 32,64,128,256,288,320,352,384,448,512,768,1024,1536,2048,3072,4096
```

输出目录：

- `data/lut_tables/raw_profile_*.json`
- `data/lut_tables/lut_gain_*.json`
- `data/lut_tables/lut_penalty_*.json`

## 3）主重复实验脚本

主实验入口：

- `tests/evaluate_waveslice_claims.py`

它支持：

- 只运行 Phase-I
- 只运行 Phase-II
- 运行联合的 Phase-I + Phase-II
- synthetic prompt workload
- 外部 dataset-derived request JSON
- warmup 和 repeated runs

### 示例：synthetic repeated

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

常用参数：

- `--include-phase12`
- `--include-strict`
- `--skip-phase2`
- `--short-repeat`
- `--long-repeat`
- `--phase1-objective-mode`
- `--phase2-dispatch-mode synchronized|async_experimental`
- `--trust-remote-code`

### 示例：使用外部 requests JSON 运行 dataset workload

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

### 推荐的 V1 主回归

如果要判断当前 V1 路径，优先使用冻结的真实 `openworkload mid`
回归目标，而不是默认那个很小的 synthetic repeated case：

```bash
python experiments/run_frozen_eval_config.py \
  --config experiments/configs/frozen_v1_gemma_mid_global_activity_repro.json
```

这条回归会固定使用：

- 来自真实开源数据集构建的 request JSON
- 长短混合请求
- 泊松到达分布
- 与 V0 回归对照相同的 `mid` workload 家族

## 4）构建 Dataset Workload

数据集 workload 构建脚本：

- `experiments/build_dataset_workload.py`

当前构建器会从以下公开数据集抽样：

- `LongBench`
- `UltraChat200k`

典型命令：

```bash
python experiments/build_dataset_workload.py \
  --model-path mistralai/Mistral-7B-v0.1 \
  --out-prefix results/mistral_dataset_mix \
  --max-prompt-tokens 3008 \
  --sample-count 96
```

输出文件：

- `results/<prefix>_requests.json`
- `results/<prefix>_lora_requests.json`
- `results/<prefix>_meta.json`

## 5）Synthetic 多模型套件

脚本：

- `experiments/waveslice_a100_suite.py`

查看支持的模型：

```bash
python experiments/waveslice_a100_suite.py --list-models
```

运行完整 synthetic suite：

```bash
python experiments/waveslice_a100_suite.py \
  --repeats 3 \
  --warmup-iters 2 \
  --max-new-tokens 64 \
  --include-phase12 \
  --timeout-sec 240
```

运行一个子集：

```bash
python experiments/waveslice_a100_suite.py \
  --models mistral-7b-v0.1,mistral-7b-instruct-v0.2,zephyr-7b-beta \
  --repeats 3 \
  --warmup-iters 2 \
  --max-new-tokens 64 \
  --include-phase12 \
  --timeout-sec 240
```

输出目录：

- `results/waveslice_a100_suite_<timestamp>/`
- `results/waveslice_a100_suite_<timestamp>.csv`

## 6）Dataset 多模型套件

脚本：

- `experiments/waveslice_dataset_suite.py`

运行完整 dataset suite：

```bash
python experiments/waveslice_dataset_suite.py \
  --repeats 3 \
  --timeout-sec 240
```

运行一个子集：

```bash
python experiments/waveslice_dataset_suite.py \
  --models mistral-7b-v0.1,mistral-7b-instruct-v0.2,zephyr-7b-beta \
  --repeats 3 \
  --timeout-sec 240
```

输出目录：

- `results/waveslice_dataset_suite_<timestamp>/`
- `results/waveslice_dataset_suite_<timestamp>.csv`

## 7）真实模型元数据 Profiling

脚本：

- `experiments/profile_real_model_metadata.py`

这个脚本会收集可复用模型元数据，例如：

- 模型结构字段
- attention 形式
- tokenizer 信息
- 现有 profiling artifact 链接
- 数据集 token-length 统计

运行方式：

```bash
python experiments/profile_real_model_metadata.py
```

输出示例：

- `results/real_model_metadata_with_datasets.json`

## 核心代码阅读顺序

如果只想先抓主线，建议按这个顺序阅读：

1. `engine/vllm_hijacker.py`
2. `tests/evaluate_waveslice_claims.py`
3. `experiments/waveslice_a100_suite.py`

这三个文件分别对应：

- 运行时注入与联合协调逻辑
- 主评测脚本
- 多模型实验编排
