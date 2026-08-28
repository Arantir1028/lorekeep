# WaveSlice

[English](./README.md) | [简体中文](./README.zh-CN.md)

WaveSlice 是面向 vLLM V1 的调度扩展。它作为普通 Python 包安装，不修改 vLLM
源码，并通过 `EngineArgs` 声明式启用。本仓库同时保留 CUCUMIS 论文的 Chapter 2
与 Chapter 5 实验工作流。

## 环境

所有命令都从仓库根目录执行。先激活目标环境，例如：

```bash
conda activate sara
```

实验脚本默认使用当前已经激活的 Python 解释器。配置文件里不应该写机器本地路径，例如用户 home、Conda 安装目录、固定 Hugging Face cache 目录等；实验输出、资源目录和 run root 都应使用仓库相对路径。

已有实验记录使用下面这组环境版本。WaveSlice 包本身不固定 vLLM 版本号；当前实现
针对这套环境中的 V1 API。

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

在激活的环境里安装本仓库：

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Python API

WaveSlice 扩展了 vLLM 的 `EngineArgs`，不要求修改 vLLM 源码，也不需要额外调用
注入函数。WaveSlice 目前只支持 vLLM V1 引擎。

```python
from waveslice import EngineArgs, WaveSliceConfig
from vllm.engine.llm_engine import LLMEngine

engine_args = EngineArgs(
    model="/models/Mistral-7B-v0.1",
    enable_wave_slice=True,
    wave_slice_config=WaveSliceConfig(
        # 用于选择 LUT 文件，不是模型权重路径。
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

设置 `enable_wave_slice=False` 时会走原生 vLLM 调度路径。若省略
`lut_model`，WaveSlice 会从 `model` 的最后一级名称推断 LUT 名称。运行时指标仍可
通过 `waveslice.get_wave_slice_metrics()` 获取。

模型和数据集由配置文件选择。`resource_selection.auto_download=true` 时，本地缺失的 Hugging Face 资源会自动下载；只有目标机器已经有完整 cache 时，才把 `resource_selection.offline` 设为 `true`。如果模型需要授权但当前账号没有权限，Hugging Face 或 vLLM 会直接报错，需要用户自己处理授权或 token。

## Chapter 5 主实验

新 GPU 上，Chapter 5 preflight 会检查每个 LUT 里记录的硬件 fingerprint。如果 LUT 缺失、过期，或者和当前 GPU 不匹配，preflight 会在主实验开始前自动重建。LUT builder 使用当前激活的 Python 环境和本地 Hugging Face snapshot；如果模型还没缓存，则遵循配置里的正常下载策略。

```bash
python experiments/run_chapter5_suite.py \
  --config experiments/configs/chapter5_pipeline_default.json \
  --stages preflight \
  --run-tag chapter5_demo
```

preflight 成功后，Chapter 5 继续通过同一个入口运行：

```bash
python experiments/run_chapter5_suite.py \
  --config experiments/configs/chapter5_pipeline_default.json \
  --run-tag chapter5_demo
```

这个命令会按顺序执行：

- `preflight`：检测当前 GPU / 软件环境，探测可运行的模型参数，并生成 resolved config。
- `main`：运行 dataset-backed open-workload 主实验。
- `baseline`：在 `main` 产出的同一批 workload 上复跑基线和机制消融。
- `figures`：重建 Chapter 5 主图、表格和汇总 markdown。
- `partial-figures`：可选地重建 density sweep 和 LoRA dispersion 图。

相关配置：

- Pipeline 配置：`experiments/configs/chapter5_pipeline_default.json`
- 主实验配置：`experiments/configs/openworkload_v1_local_realworld_lora8.json`
- 基线/消融配置：`experiments/configs/chapter5_baseline_variants_lora7.json`

LUT builder 会写入或刷新：

- `waveslice/data/lut_tables/raw_profile_<lut_name>.json`
- `waveslice/data/lut_tables/lut_gain_<lut_name>.json`
- `waveslice/data/lut_tables/lut_penalty_<lut_name>.json`
- `waveslice/data/lut_tables/runtime_calibration_<lut_name>.json`
- `waveslice/data/lut_tables/runtime_sanity_<lut_name>.json`

`--skip-preflight-lut-rebuild` 只建议用于调试 stale-LUT 检测；如果选中的 LUT 和当前 GPU fingerprint 不匹配，preflight 在这个模式下不会写出可用的 resolved config。

preflight 也会针对较小显存自动缩放 workload pressure。最终值会写进 `metadata/resolved_config.json`，决策记录在 `metadata/workload_capacity.json`。自动调整范围包括 `max_new_tokens`、`repeats`、`sample_count`、各 density 的请求数量、arrival-rate scale，以及在容量过低时丢掉 `peak`。

只跑环境 preflight：

```bash
python experiments/run_chapter5_suite.py \
  --config experiments/configs/chapter5_pipeline_default.json \
  --stages preflight \
  --run-tag chapter5_demo
```

如果只想快速检查配置解析和路径，不加载 vLLM engine：

```bash
python experiments/run_chapter5_suite.py \
  --config experiments/configs/chapter5_pipeline_default.json \
  --stages preflight \
  --run-tag chapter5_demo \
  --skip-preflight-engine-smoke
```

新机器上建议先跑一个小 trial：

```bash
python experiments/run_chapter5_suite.py \
  --config experiments/configs/chapter5_pipeline_default.json \
  --run-tag chapter5_trial \
  --model-keys mistral-7b-v0.1 \
  --dataset-keys ultrachat200k \
  --densities mid
```

只跑主实验阶段：

```bash
python experiments/run_chapter5_suite.py \
  --config experiments/configs/chapter5_pipeline_default.json \
  --stages main \
  --run-tag chapter5_demo \
  --model-keys mistral-7b-v0.1,gemma-7b-it \
  --dataset-keys ultrachat200k,longbench \
  --densities mid,high
```

基于已有主实验 run，只跑基线/机制消融：

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

基于已有 run 重建图表：

```bash
python experiments/run_chapter5_suite.py \
  --config experiments/configs/chapter5_pipeline_default.json \
  --stages figures \
  --main-run-root results/openworkload_v1_local_realworld_lora8/chapter5_demo_main \
  --baseline-run-root results/chapter5_baseline_variants/chapter5_demo_baseline \
  --export-name chapter5_demo
```

常用筛选参数：

- `--model-keys`：主实验配置里的模型 key，逗号分隔。
- `--dataset-keys`：主实验配置里的数据集 key，逗号分隔。
- `--densities`：workload density，逗号分隔。
- `--variants`：`baseline` 阶段使用的基线/消融 variant，逗号分隔。

Chapter 5 输出路径：

- Preflight run：`results/openworkload_v1_local_realworld_lora8/<run_tag>_preflight/`
- 主实验 run：`results/openworkload_v1_local_realworld_lora8/<run_tag>_main/`
- 基线/消融 run：`results/chapter5_baseline_variants/<run_tag>_baseline/`
- 图表导出：`results/chapter5_exports/<run_tag>/`

Preflight 关键产物：

- `metadata/resolved_environment.json`：当前 Python、CUDA、依赖包和 GPU 信息。
- `metadata/model_preflight.json`：每个模型的 smoke/capacity 探测状态。
- `metadata/runtime_capacity.json`：选中的 runtime capacity 参数。
- `metadata/workload_capacity.json`：选中的 workload scale、decode 长度、repeats、sample count，以及 density/request-count 调整。
- `metadata/resolved_config.json`：后续 Chapter 5 主实验实际消费的配置。
- `metadata/preflight_summary.json`：简短机器可读汇总。

## Chapter 2 观察实验

当前维护的 Chapter 2 路径只重建固定 vLLM 观察，不加载 CUCUMIS hook：

- 入口：`experiments/run_chapter2_observations.py`
- 配置：`experiments/configs/chapter2_observations_v2.json`

运行两个保留观察：

```bash
python experiments/run_chapter2_observations.py \
  --config experiments/configs/chapter2_observations_v2.json \
  --observations all \
  --run-name chapter2_demo
```

`--observations obs1` 只运行一个/两个长请求对比，`obs2` 只运行固定 token budget
sweep。输出位于 `results/chapter2_observations_v2/<run_name>/`，包含 manifest、
workload、原始请求时延、日志与 `summary.json`。

## 路径规则

- 所有命令从仓库根目录执行。
- 实验路径保持仓库相对路径。
- 长期使用的模型、数据集和 density 写进 JSON 配置。
- 临时小规模试跑再用 CLI filter。
- 新机器上先让 preflight 生成 Chapter 5 的 portable resolved config，再跑主实验。
