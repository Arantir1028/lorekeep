# WaveSlice 10k 紧凑维护版验收

日期：2026-08-11

> 历史说明：这是 2026-08-11 标签的验收记录。2026-08-28 的可读性与包结构重构
> 不再以 10k 物理行数为目标，因此下面的规模数字只描述旧快照。

## 结论

当前重构完成版为 `compact-refactored-final-v2-20260811`。按维护目录内全部 `.py`
物理行统计，共 70 个文件、9,330 行，低于 10,000 行目标。没有使用 minify、
生成代码、改扩展名或迁移到统计外目录。

| 区域 | 行数 |
| --- | ---: |
| runtime/config/scheduler/profiler | 3,519 |
| experiments | 2,889 |
| scripts/tools | 1,807 |
| tests | 1,115 |
| **总计** | **9,330** |

## 保留功能

1. `waveslice.EngineArgs`、配置和 metrics 公共 API。
2. vLLM V1 Phase I chunk boundary control。
3. Phase II beneficiary selection、单轮 scheduler cashout 与 priority lane。
4. workload/LoRA/LUT/preflight，以及 Chapter 2、Chapter 5、DistServe/CUCUMIS、
   A100/4090/5090 portability 主链路。
5. GPU lock、timeout、resume、manifest 和正式 CSV/JSON/LaTeX 导出。

历史一次性实验、已替代图表脚本、非 pinned vLLM 猜测兼容和 ModelRunner 输出
逃逸不属于当前维护功能，可从审计标签恢复。

## 验收证据

- 27/27 pytest；compileall 通过；活动 JSON 配置全部可解析。
- V1 `schedule` 运行时接入和恢复检查通过。
- A100 实际重叠 workload：4/4 请求完成，Phase II 与 priority lane 各触发 1 次。
- DistServe 公式 v2：50 个 case 离线重放完成，旧目录保留。
- 策略字段从 104 减到 79，runtime 核心无超过 64 行的函数热点。

既有论文结果继续由 `results-final-20260714` 追溯；当前代码因 Phase II 语义和
DistServe 公式已修正，不能声称与旧导出逐字节一致，正式性能结论需要完整重跑。
