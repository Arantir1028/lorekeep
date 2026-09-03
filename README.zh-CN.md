# WaveSlice

[English](README.md)

WaveSlice 是面向 vLLM V1 的调度扩展，用于处理 prompt 长度和 LoRA 服务成本存在
差异的在线请求。它在 scheduler 边界控制长 prefill 的分块和请求优先级，以普通
Python 包的形式安装，并通过扩展后的 `EngineArgs` 启用，不需要修改 vLLM 源码。

本仓库同时包含 CUCUMIS 研究使用的实验程序和精简结果数据。

## 运行要求

- Python 3.10 或更高版本
- vLLM 支持的 Linux 和 CUDA 环境
- 提供 V1 scheduler API 的 vLLM
- 与目标模型匹配的 WaveSlice LUT

项目不固定 vLLM 版本号。接入层使用 vLLM V1 的内部接口，升级 vLLM 后需要重新
验证兼容性。WaveSlice 不支持旧的 V0 引擎。

## 安装

在已经配置好 vLLM 的环境中安装 WaveSlice 包：

```bash
python -m pip install -e .
```

也可以通过可选依赖安装 vLLM：

```bash
python -m pip install -e ".[vllm]"
```

只阅读代码和运行 CPU 测试时，使用轻量开发环境：

```bash
make install-dev
make check
```

完整实验环境使用 `requirements.txt` 引用的已验证约束：

```bash
make install-runtime
```

环境和验证等级见[本地开发指南](docs/development.md)。运行时版本记录在
`constraints/validated-vllm.txt` 中，WaveSlice 运行时代码不会写死 vLLM 版本判断。

## 名词说明

WaveSlice 包含两种 scheduler 机制。配置字段和运行指标沿用以下内部阶段名称：

- **Prefill slicing（Phase I）**：限制一次 scheduler 调用接纳的长 prefill token
  数量，使等待请求更早获得重新调度的机会。启用 WaveSlice 后默认开启。
- **Priority cashout（Phase II）**：在一次 scheduler 调用内提升一个服务成本较低的
  prefill，并暂缓一个与其竞争、服务成本较高的 prefill。该机制默认关闭。

## 快速开始

必须从 `waveslice` 导入 `EngineArgs`，不能给
`vllm.engine.arg_utils.EngineArgs` 直接增加 WaveSlice 参数。应当在创建 vLLM
引擎之前完成该导入，以便进程使用 V1。

```python
from waveslice import EngineArgs, get_wave_slice_metrics

from vllm.engine.llm_engine import LLMEngine
from vllm.sampling_params import SamplingParams

engine_args = EngineArgs(
    model="mistralai/Mistral-7B-v0.1",
    enable_wave_slice=True,
    enable_chunked_prefill=True,
    max_num_batched_tokens=1536,
    max_num_partial_prefills=1,
    max_long_partial_prefills=1,
    max_model_len=4096,
    enforce_eager=True,
)
engine = LLMEngine.from_engine_args(engine_args)

engine.add_request(
    request_id="req-001",
    prompt="Explain scheduler-level prefill chunking.",
    params=SamplingParams(max_tokens=64, temperature=0.0),
)

while engine.has_unfinished_requests():
    for output in engine.step():
        if output.finished:
            result = output.outputs[0]
            print(result.text)

print(get_wave_slice_metrics(reset=True))
```

WaveSlice 默认根据 `EngineArgs.model` 推导 LUT 名称，通常不需要重复填写
`lut_model`。例如，Hugging Face 模型标识 `mistralai/Mistral-7B-v0.1` 会映射为
`mistralai--Mistral-7B-v0.1`，与 LUT 生成程序采用的名称一致。使用本地模型目录时，
默认采用目录名；Hugging Face cache 中的 snapshot 路径会根据
`models--org--model` 目录恢复模型名。

只有本地目录名称存在歧义，或者需要显式选择另一份兼容 profile 时，才需要设置
`WaveSliceConfig(lut_model=...)`。Prefill slicing 应与
`enable_chunked_prefill=True` 配合使用，其余 vLLM 参数仍按普通 `EngineArgs`
字段配置。

目标模型对应的 raw、gain 和 penalty LUT 必须同时存在。缺少模型专用 LUT 时，
WaveSlice 会直接报告错误，不会替换为通用 LUT。

### 关闭 WaveSlice

继续使用同一个 `EngineArgs` 类，将开关设为 `False`：

```python
from waveslice import EngineArgs

engine_args = EngineArgs(
    model="mistralai/Mistral-7B-v0.1",
    enable_wave_slice=False,
)
```

此时使用原生 vLLM V1 scheduler。声明式 `EngineArgs` 是应用入口，WaveSlice 不再
提供单独的 inject 或 uninject 接口。

### 启用 priority cashout（Phase II）

默认策略启用 Phase I 和运行时指标。Phase II priority cashout 需要显式开启：

```python
from waveslice import EngineArgs, WaveSliceConfig, WaveSlicePolicy

engine_args = EngineArgs(
    model="mistralai/Mistral-7B-v0.1",
    enable_wave_slice=True,
    wave_slice_config=WaveSliceConfig(
        policy=WaveSlicePolicy(
            enable_phase1_scheduler=True,
            enable_phase2_scheduler=True,
            phase2_enable_scheduler_cashout=True,
        ),
    ),
    enable_chunked_prefill=True,
)
```

Phase II 每次最多暂缓一个竞争 prefill，范围仅限一次 scheduler 调用，随后恢复该
请求。默认联合策略会参考最近的 Phase I 活动，并在连续 cashout 之间设置 cooldown。

## 配置

`WaveSliceConfig` 包含三个公共字段：

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `lut_model` | 从 `EngineArgs.model` 推断 | LUT 选择的可选覆盖值 |
| `gamma` | `2.0` | LUT 目标函数中的队列压力惩罚系数 |
| `policy` | `WaveSlicePolicy()` | Prefill slicing、priority cashout、队列排序和指标配置 |

模型 LUT 位于 `waveslice/data/lut_tables/`。正式实验应使用针对目标模型、且硬件
指纹兼容的 profile。Chapter 5 preflight 会检查硬件指纹，并在 LUT 缺失或过期时
重新生成。

维护中的 LUT 生成程序使用同一套模型命名规则：

```bash
python experiments/build_hybrid_checkpoint_runtime_luts.py \
  --models mistralai/Mistral-7B-v0.1
```

目标 checkpoint 必须已经位于当前 Hugging Face 缓存中。该命令会执行 GPU profile
和运行时校准。

以上模型会生成以 `mistralai--Mistral-7B-v0.1` 为共同名称的 raw profile、gain
LUT 和 penalty LUT；运行时也会从 `EngineArgs.model` 推导出相同名称。

WaveSlice 不能与另一个自定义 `scheduler_cls` 同时启用。运行时状态在进程内共享；
需要使用不同 WaveSlice 配置的多个引擎应放在不同进程中运行。

## 运行时指标

`get_wave_slice_metrics()` 会合并主进程和 vLLM worker 进程收集的指标，包括：

- 请求数量、TTFT 和 completion slowdown；
- Phase I 触发条件、chunk 选择、virtual cap 和运行时自适应；
- Phase II 尝试次数、决策原因、cooldown 和 priority lane 状态。

传入 `reset=True` 会在返回当前报告后清空计数器。只需要清空时可调用
`reset_wave_slice_metrics()`。

## 方法概要

WaveSlice 只在 scheduler 边界执行两个阶段：

- **Phase I** 为长 prefill 选择受限 chunk，使 vLLM 更早返回 scheduler，重新考虑
  等待请求。
- **Phase II** 选择一个低服务成本受益请求和一个高服务成本竞争请求，提升前者，
  并将后者暂缓一次 scheduler 调用。

这两个阶段都不会修改模型执行、延迟已经计算出的输出、解绑请求或创建额外 CUDA
stream。具体规则和 vLLM 接入方式见[方法与接入设计](docs/method.md)。

内部调用路径和模块职责见[架构说明](docs/architecture.md)，所有策略字段见
[配置参考](docs/configuration.md)。

## 实验与结果

仓库提供 Chapter 2 观察实验、Chapter 5 open-workload 实验、请求比例扫描、
DistServe 对比以及 A100、RTX 4090、RTX 5090 硬件迁移实验。

- [实验指南](docs/experiments.md)
- [结果数据与来源说明](docs/results.md)

完整运行目录、日志、模型快照和生成的 LoRA adapter 默认不进入 Git。仓库只跟踪
随包发布的 LUT 和经过筛选的精简结果数据。

## 目录结构

| 路径 | 用途 |
| --- | --- |
| `waveslice/` | 可安装的 Python 包和 vLLM V1 接入 |
| `waveslice/scheduling/` | 基于 LUT 的 chunk 和 fairness 计算 |
| `waveslice/vllm/` | scheduler、request、进程和指标适配 |
| `waveslice/data/lut_tables/` | 随包发布的 profile 和 LUT |
| `experiments/` | workload、preflight 和实验入口 |
| `scripts/` | ratio、DistServe、双 GPU 和硬件迁移编排 |
| `tests/` | 单元测试、接入契约和实验 evaluator |
| `results/` | 本地运行目录和 Git 跟踪的精简导出 |
| `constraints/` | 已验证的运行时依赖组合 |

## 开发

安装轻量开发环境并运行全部非 GPU 检查：

```bash
make install-dev
make check
```

单独执行某类检查时可以使用：

```bash
make test
make lint
make validate-configs
make check-docs
make verify-results
```

测试套件不能替代真实 GPU 验证。修改 scheduler 行为或升级 vLLM 后，需要重新运行
对应的 GPU smoke test 和正式实验。
