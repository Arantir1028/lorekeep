# CUCUMIS

[English](./README.md) | [简体中文](./README.zh-CN.md)

CUCUMIS 是一个构建在 vLLM 之上的异构 LLM 推理运行时调度层。

## 当前保留的入口

## 当前方法定义

当前论文和实验主线固定为：

- **Phase I：** 通过控制 long prefill 的 chunk size，让长请求更早返回调度边界。
- **Phase II：** 在调度边界暴露后，通过 execution escape / priority promotion 重塑下一个调度窗口。
- **自适应策略：** 在 scheduler 内基于运行时队列压力做 runtime queue-pressure adaptation，避免低压误触发，同时响应真实队列状态。

这不是已经放弃的 `true_unbind` / dual-stream 路线。当前结果和论文表述都不应该再写成长短请求在执行层双流并行。

当前 adaptive policy 是 runtime queue-pressure adaptation。配置里的 density 只负责生成 workload，真正的 chunk/gate 选择由在线 queue length、waiting short、等待时间、long remaining tokens 和 virtual-cap hit 信号驱动。精确定义见 `docs/current_method.md`。

支持的实验入口：

- `experiments/run_chapter5_suite.py`
  - Chapter 5 总控入口。可以分阶段跑主实验、基线/机制消融、图表导出，也可以一键跑完整流程。
- `experiments/chapter2_prestudy.py`
  - Chapter 2 预实验，公开保留 `E1`-`E5`，以及 `e3paper` 图导出。
- `tests/evaluate_waveslice_claims.py`
  - 核心重复评测 harness，输出 TTFT、slowdown、wall time、per-repeat 指标和 request-level timing。
- `experiments/run_frozen_eval_config.py`
  - 冻结单点回归入口。
- `experiments/run_openworkload_execescape_suite.py`
  - 当前 v1 主线的 Chapter 5 真实 workload 主实验。
- `experiments/run_chapter5_baseline_variants.py`
  - Chapter 5 的基线对照与机制消融入口，复用已有主实验 workload。
- `scripts/regenerate_chapter5_main_outputs.py`
  - 基于主实验和基线实验结果，重建 Chapter 5 主图、表格和汇总 markdown。
- `scripts/regenerate_chapter5_partial_figures.py`
  - 重建 Chapter 5 中配套使用的 density sweep 图，以及可选的 LoRA latency dispersion 图。

## 仓库结构

- `engine/vllm_hijacker.py`：运行时注入入口和 vLLM hook 协调逻辑。
- `engine/runtime_bootstrap.py`：vLLM 模式和 CUDA 平台 bootstrap。
- `tests/evaluate_waveslice_claims.py`：核心 TTFT / slowdown / wall-time 评测 harness。
- `experiments/run_chapter5_suite.py`：Chapter 5 一键式和分阶段总控入口。
- `experiments/chapter2_prestudy.py`：当前 Chapter 2 预实验入口。
- `experiments/run_openworkload_execescape_suite.py`：当前真实 workload v1 主实验。
- `experiments/run_chapter5_baseline_variants.py`：Chapter 5 基线与消融入口。
- `experiments/build_dataset_workload.py`：dataset-driven suite 使用的请求构造器。
- `config/experiment_catalog.py`：当前支持的模型目录。

## 环境

所有命令都从仓库根目录执行。

当前代码使用的最小软件版本：

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

典型安装方式：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

此外还需要：

- 一张 CUDA 可见的 GPU
- 与 GPU 匹配的 CUDA / PyTorch 环境
- 目标 Hugging Face 模型权重，本地 snapshot 或 HF model ID 都可以

## 快速开始

先检查 hook 是否能正常加载：

```bash
python tests/validate_wave_slice.py
```

运行当前推荐的冻结 v1 回归点：

```bash
python experiments/run_frozen_eval_config.py \
  --config experiments/configs/frozen_v1_gemma_mid_global_activity_repro.json
```

运行当前 open-workload 主实验：

```bash
python experiments/run_openworkload_execescape_suite.py \
  --config experiments/configs/openworkload_execescape_default.json
```

直接运行单 case 核心 harness：

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

## Chapter 5 实验工作流

Chapter 5 包含四个阶段：

1. `main`：跑 dataset-backed 的 open-workload 主实验，覆盖选中的模型和 density。
2. `baseline`：在 `main` 产出的同一批 workload 上复跑基线对照和机制消融。
3. `figures`：基于 `main` + `baseline` 结果重建 Chapter 5 主图、表格和汇总 markdown。
4. `partial-figures`：可选地重建 density sweep 和 LoRA latency dispersion 图。

相关配置文件：

- pipeline 配置：`experiments/configs/chapter5_pipeline_default.json`
- 主实验配置：`experiments/configs/openworkload_v1_local_realworld_lora8.json`
- 基线配置：`experiments/configs/chapter5_baseline_variants_lora7.json`

路径约定：

- 这一节里的输出目录、run 目录和命令示例，默认都按仓库根目录下的相对路径来写。
- Chapter 5 主实验使用 suite 配置里的 `eval.python_bin`。在这台机器上，当前 v1 open-workload 配置指向 `sara` conda 环境。

模型和数据集配置：

- 如果希望长期复用一套配置，直接改主实验配置文件。
  最关键的是 `resource_selection`、`models`、`datasets`、`workload.densities`。
- 如果只是临时筛选，直接走 CLI 覆盖：
  `--model-keys`、`--dataset-keys`、`--densities`。
- 基线阶段需要 `source_run_root`，可以用 `--source-run-root` 传入，也可以让 `experiments/run_chapter5_suite.py` 自动串起来。

一条命令跑完整套 Chapter 5：

```bash
python experiments/run_chapter5_suite.py \
  --config experiments/configs/chapter5_pipeline_default.json \
  --run-tag chapter5_demo
```

只跑 Chapter 5 主实验：

```bash
python experiments/run_chapter5_suite.py \
  --stages main \
  --run-tag chapter5_demo \
  --model-keys mistral-7b-v0.1,gemma-7b-it \
  --dataset-keys ultrachat200k,longbench \
  --densities mid,high
```

基于已有主实验结果，只跑基线/机制消融：

```bash
python experiments/run_chapter5_suite.py \
  --stages baseline \
  --main-run-root results/openworkload_v1_local_realworld_lora8/chapter5_demo_main \
  --run-tag chapter5_demo \
  --variants strict_no_chunk \
  --model-keys mistral-7b-v0.1 \
  --densities mid
```

基于已有 run，只重建 Chapter 5 主图和表格：

```bash
python experiments/run_chapter5_suite.py \
  --stages figures \
  --main-run-root results/openworkload_v1_local_realworld_lora8/chapter5_demo_main \
  --baseline-run-root results/chapter5_baseline_variants/chapter5_demo_baseline \
  --export-name chapter5_demo
```

重建可选的 partial figures：

```bash
python experiments/run_chapter5_suite.py \
  --stages partial-figures \
  --main-run-root results/openworkload_v1_local_realworld_lora8/chapter5_demo_main \
  --e5-summary results/chapter2_prestudy_v1/<run_id>/E5_lora_multitenancy_relevance/summary_all_models.json \
  --export-name chapter5_demo
```

如果想把每块实验拆开单独跑，也可以直接调用底层脚本：

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

输出目录约定：

- 主实验：`<main out_root>/<run_tag>_main/`
- 基线实验：`<baseline out_root>/<run_tag>_baseline/`
- Chapter 5 图表导出：`<figures_out_root>/<run_tag>/`

导出目录里会包含 `chapter5_manifest.json`、`chapter5_summary.md`、per-repeat CSV/JSON，以及重建后的图表和表格。

## Chapter 2 预实验

入口：

- `experiments/chapter2_prestudy.py`
- 配置：`experiments/configs/chapter2_prestudy_v1.json`

运行当前公开保留的整套预实验：

```bash
python experiments/chapter2_prestudy.py all \
  --config experiments/configs/chapter2_prestudy_v1.json \
  --out-root results/chapter2_prestudy_v1/<run_id>
```

单独运行某个实验：

```bash
python experiments/chapter2_prestudy.py e1 --config experiments/configs/chapter2_prestudy_v1.json
python experiments/chapter2_prestudy.py e2 --config experiments/configs/chapter2_prestudy_v1.json
python experiments/chapter2_prestudy.py e3 --config experiments/configs/chapter2_prestudy_v1.json
python experiments/chapter2_prestudy.py e3paper --config experiments/configs/chapter2_prestudy_v1.json
python experiments/chapter2_prestudy.py e4 --config experiments/configs/chapter2_prestudy_v1.json
python experiments/chapter2_prestudy.py e5 --config experiments/configs/chapter2_prestudy_v1.json
```

当前保留的预实验：

- `E1 Motivating Microbenchmark`：长请求先到、短请求后到的动机例子。
  会导出 timeline、短请求 TTFT / completion 图、wall-time 图和 `summary.json`。
- `E2 Arrival-Order Sensitivity`：展示到达顺序如何改变干扰强度。
- `E3 Fixed Chunking vs Online Control`：比较 no chunking、fixed chunking 和 online control。
- `E3 Paper Case`：从已有 beneficiary-rich 结果中导出稳定的正文图。
- `E4 Density Sweep`：展示随着负载上升，TTFT 膨胀和 slowdown 如何变化。
- `E5 LoRA Multi-Tenancy Relevance`：比较 non-LoRA、homogeneous LoRA、mixed-adapter LoRA。

## 输出说明

核心 harness 的 summary JSON 会包含当前项目最关心的指标：

- `phase1`、`phase2`、`phase12`：聚合后的 TTFT、slowdown、wall-time、error 指标
- `per_repeat`：repeat 级别的原始指标
- `request_timings`：每个 request 的首 token 延迟和完成延迟
- `hook_report`：runtime hijacker 导出的 instrumentation

open-workload suite 会写出：

- `metadata/`：resolved config、manifest CSV/JSON、聚合摘要
- `raw/`：每个模型、每个 density 的结果 JSON
- `workloads/`：生成的请求 JSON
- `figures/`：suite 级图表

## 建议阅读顺序

如果只想先抓主线，建议按下面顺序读：

1. `engine/vllm_hijacker.py`
2. `tests/evaluate_waveslice_claims.py`
3. `experiments/chapter2_prestudy.py`
4. `experiments/run_openworkload_execescape_suite.py`
