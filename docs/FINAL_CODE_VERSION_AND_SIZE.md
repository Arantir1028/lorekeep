# WaveSlice 最终代码版本、项目功能与合理规模

日期：2026-08-11

> 历史说明：本文记录 2026-08-11 的紧凑版验收快照。2026-08-28 的 WaveSlice
> 包布局与可读性重构已经取代其中的“当前代码”表述；下述行数、测试数量和入口结构
> 仅用于追溯当时标签，不代表现在工作树。

## 1. 版本结论

现有论文结果与当前维护代码必须分开标识：

| 用途 | 固定版本 |
| --- | --- |
| 既有论文结果追溯 | `results-final-20260714` |
| 删减前完整审计 | `pre-10k-refactor-20260810` |
| 首个紧凑维护基线 | `compact-refactored-final-20260811` |
| 当前重构完成版 | `compact-refactored-final-v2-20260811` |

既有结果来自 2026-07-14 的结果树，不应伪装成由当前代码重新生成。当前版修正了
Phase II 队列语义和 DistServe decode 服务时间公式，因此它是后续维护与正式重跑
的代码基线；旧标签保持不变，用于恢复和审计。

## 2. 项目实现的功能

### vLLM 调度插件

- 提供 `waveslice` 公共 API、配置、状态和 metrics。
- 通过 `waveslice.EngineArgs` 声明式接入 vLLM V1，不修改 vLLM 源码。
- Phase I 根据请求长度、LoRA cohort、LUT 和实时队列压力选择长 prefill chunk，
  通过原生 scheduler cap 提前产生下一次调度机会。
- Phase II 在 scheduler 队列上选择一个低服务成本受益请求，暂缓一个竞争长
  prefill 一轮，并稳定提升受益请求；不再修改 ModelRunner 或延迟输出。
- 记录 TTFT、completion slowdown、Phase I/II 决策、原因与 priority lane 指标。

### Workload、画像与实验流水线

- 构造/remix dataset workload，准备合成 LoRA，选择和校验本地模型资源。
- 执行 runtime calibration、hybrid checkpoint LUT、硬件指纹和 preflight。
- Chapter 2 固定预算观察；Chapter 5 主实验、基线、消融、ratio sweep 与导出。
- DistServe 真实单 GPU stage-cost 测量与逻辑 2-GPU continuous-batching replay；
  CUCUMIS 真实双副本 RR/LB 对比。
- A100、RTX 4090、RTX 5090 portability 的预检、运行、恢复、汇总与异常导出。
- GPU lock、timeout、resume、manifest、逐 case JSON 以及 CSV/JSON/LaTeX 导出。

DistServe 必须描述为“真实单 GPU stage-cost + 逻辑 2-GPU replay”，不能称为真实
双 GPU DistServe 部署。

## 3. 合理代码规模

统计 `config/`、`data/`、`engine/`、`experiments/`、`profiler/`、`scheduler/`、
`scripts/`、`tests/`、`tools/`、`waveslice/` 下全部 `.py` 物理行：

| 区域 | 文件 | 行数 |
| --- | ---: | ---: |
| runtime/config/scheduler/profiler | 36 | 3,519 |
| 实验实现 | 19 | 2,889 |
| CLI 与工具 | 9 | 1,807 |
| 契约测试 | 6 | 1,115 |
| **总计** | **70** | **9,330** |

其中空行 305、纯注释 3、非空且非纯注释 9,022；没有 minify、生成代码、改扩展名
或把实现移出统计目录。删减前为 32,660 行，当前减少 23,330 行（71.4%）。

当前 Sarathi-Serve HEAD 的 `sarathi/` Python 为 18,756 行，`csrc/` 为 1,145
行，核心运行时合计 19,901 行。NanoFlow README 将其 C++ backend + Python demo
概括为约 4K 行；按当前 HEAD 排除 `3rdparty` 后统计全部一方 Python/native 源码，
则为 125 个文件、23,907 行。两种 NanoFlow 数字说明统计边界比单一数字更重要。

WaveSlice 建立在 vLLM 上，同时包含研究插件、正式实验和硬件编排。保持这些功能时，
约 8,500--10,000 行是合理的人类手写维护范围；当前 9,330 行已经处于该范围，
继续机械追求更小会开始损害可读性或删除独立实验能力。

## 4. 本轮实质重构

- 删除 ModelRunner 执行逃逸、输出延迟和重复 lifecycle 路径，Phase II 只作用于
  scheduler 队列。
- 修复 cashout 在 native schedule 后恢复整份旧队列的问题；现在只恢复本轮
  暂缓请求，保留原生调度器的迁移结果。
- 删除碎片大小筛选等与“整轮暂缓请求”语义矛盾的字段，候选固定为一个长 anchor。
- 将十余项 weighted-window、sparse exception 和有限两位冒泡收敛为三步 gate
  与稳定 priority promotion；策略字段由 104 降至 79。
- 修复严格 evaluator 中 `model.name/path` 未映射到 runtime 字段的问题；配置拒绝
  未知字段，历史已退役字段只在旧结果解析处保留。
- 修复 DistServe 首 token 与其余 token 重复计费：余下服务时间按
  `(total - first) / (N - 1)` 分配，并写入公式版本。
- 合并重复脚本/测试/配置入口，删除已替代的一次性 prestudy、图表和 mock 网格。

## 5. 验证结果

- `pytest`：27/27 通过；`compileall`：通过；全部活动 JSON 配置可解析。
- vLLM V1 运行时接入和关闭均恢复原方法对象。
- A100 + Gemma-7B + LoRA 重叠请求验证：4/4 完成、无超时、Phase II applied=1、
  priority lane activations=1。
- DistServe 公式 v2 离线重放：50/50 case，4,000 request rows，旧结果未覆盖。
  相比旧公式，TTFT P99 不变；completion P99 中位 +1.63%，wall +1.36%，
  throughput -1.35%。
- AST 审计后，runtime 核心中最长热点函数为 64 行；未发现只定义未调用的私有
  实现。

真实 GPU 验证证明的是 hook、队列语义和完成性，不是统计性能结论。由于当前版
改变了 Phase II 实现，正式论文性能数字仍需用当前标签完整重跑后再更新。

## 6. 最终评估

没有剩余的高优先级结构性重构。`waveslice/vllm/runtime.py` 集中编排 V1
运行时生命周期。两个超过 100
行的函数位于实验 CLI 主入口，属于线性参数/阶段编排，不是复杂运行时算法。

后续只应在出现实际重复职责、vLLM 升级或新实验需求时继续重构。当前必要工作是
基于该固定版本重跑正式 Phase I/II sweep，而不是继续压缩代码。
