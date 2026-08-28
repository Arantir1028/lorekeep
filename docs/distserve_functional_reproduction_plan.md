# DistServe 功能性复现实验方案

Status: fixed experiment plan; DistServe continuous-batching replay + CUCUMIS-2A100 comparison implemented. This is a planned paper addition, not yet inserted into the manuscript.

Date: 2026-07-03; updated 2026-07-06

## 目标

新增 `DistServe` baseline，直接回答 reviewer 对 prefill/decode disaggregation baseline 的要求。论文图表中使用名称：

`DistServe`

方法描述使用：

`DistServe baseline, functional reproduction`

不要给方法名添加额外后缀，也不要使用 `disaggregation-inspired` 或 `single-GPU proxy` 这类暧昧说法作为图表方法名。

## 正式比较口径

主比较只保留同资源口径：

- `DistServe-2A100`: one prefill A100 + one decode A100.
- `CUCUMIS-2A100-RR`: two independent CUCUMIS A100 replicas, round-robin dispatch.
- `CUCUMIS-2A100-LB`: two independent CUCUMIS A100 replicas, least-backlog dispatch.

不再把旧的不等资源比较作为 paper-facing 对比。旧的单卡 CUCUMIS 诊断表只可作为内部排查材料，不进入主图和主结论。

## 数据与负载

Source experiment:

- `ratio_sweep_20step_5models_a100_overnight`

Formal models:

- `baichuan2-7b-chat`
- `gemma-2-9b-it`
- `gemma-7b-it`
- `mistral-7b-instruct-v0.2`
- `qwen2.5-7b-instruct`

Formal workloads:

- `mid_l10`, `mid_l30`, `mid_l50`, `mid_l70`, `mid_l90`
- `high_l10`, `high_l30`, `high_l50`, `high_l70`, `high_l90`

数据集来源：

- short requests: UltraChat200k
- long requests: LongBench
- arrival process: Poisson arrivals

## DistServe 复现边界

功能性复现必须保留：

- independent prefill queue;
- independent decode queue;
- separate prefill/decode resources;
- KV handoff delay;
- TTFT, TPOT, completion, round time, throughput exports;
- KV transfer sensitivity.

它不声称逐行复现 DistServe implementation 或完整 placement optimizer。论文中的 claim 是：该 baseline 保留了与本实验相关的 P/D separation serving semantics。

## CUCUMIS-2A100 分发

两个 CUCUMIS 实例都使用同一个 CUCUMIS scheduler 配置，每个实例占 1 张 A100。分发器只决定新请求送到哪个实例，不改 scheduler 内部逻辑。

`round_robin`:

- 按到达时间排序；
- 请求依次分到 replica 0, replica 1, replica 0, replica 1；
- 用作简单负载均衡 baseline 和 ablation。

`least_backlog`:

- 每个请求到达时，估计两个 replica 的剩余 backlog；
- backlog = `max(0, replica_available_time - request_arrival_time)`；
- 选择 backlog 更小的 replica；
- 平局选择编号更小的 replica；
- 已分配请求的预计服务量使用 `prompt_tokens + 16 * output_tokens` 这个轻量 proxy，只用于分发决策。

因此 `least_backlog` 不是纯 shortest-queue count。它对长短请求异构更敏感，因为长请求会贡献更大的预计服务量。最终指标不使用这个 proxy 估算，而是合并两个 CUCUMIS replica 的真实运行结果。

## 当前实现

当前正式实现分成三层：

1. DistServe stage measurement：在单张 A100 上对每个 LoRA 请求逐请求真实推理两次，分别测 `1-token` prefill proxy 和 `64-token` full completion。
2. `DistServe-2A100` replay：使用真实测得的 per-request stage cost，按独立 prefill queue、KV handoff、独立 decode queue 做逻辑 P/D 分离回放；decode 阶段使用 token-level continuous batching，而不是 request-level serial decode。
3. `CUCUMIS-2A100-RR/LB`：把同一个 workload 按 RR 或 LB 拆成两个子 workload，在 A100 上分别运行两个独立 CUCUMIS replica，然后合并 request-level 和 method-level metrics。

早期 LUT/runtime-calibration DistServe export、trace-calibrated CUCUMIS replay、以及 request-level serial decode DistServe replay 只保留为内部诊断，不作为 paper-facing 结果。正式比较使用 continuous-batching DistServe replay 与真实 split-run 的 `CUCUMIS-2A100-RR/LB`。

## 指标

Primary:

- all-request p99 TTFT;
- short-request p99 TTFT;
- long-request p99 TTFT;
- `DistServe-2A100 / CUCUMIS-2A100` ratio.

Guardrails:

- p50/p90/p99 TTFT;
- all/short/long completion p99;
- TPOT;
- round completion time;
- throughput;
- KV transfer sensitivity.

## 输出

Output root:

Current paper-candidate run:

`results/openworkload_ratio_sweep_lora8/unified_distserve_continuous_cucumis_2a100_replay`

Paper-facing files:

- `distserve_serial/distserve_serial_method_metrics.csv`
- `cucumis_2a100/cucumis_2a100_real_method_metrics.csv`
- `cucumis_2a100/cucumis_2a100_real_request_metrics.csv`
- `cucumis_2a100/distserve_equal_resource_real_comparison.csv`
- `unified_method_metrics.csv`
- `unified_request_metrics.csv`
- `unified_equal_resource_comparison.csv`

Diagnostic-only files:

- `distserve_comparison.csv`
- `distserve_summary.csv`
- `distserve_sensitivity_*`
- `distserve_validation.csv`
- `distserve_resource_audit.csv`
- `results/chapter5_exports/distserve_functional_repro_ratio_sweep`
- `results/openworkload_ratio_sweep_lora8/cucumis_2a100_dispatch_real_split_formal`
- `results/openworkload_ratio_sweep_lora8/unified_distserve_serial_cucumis_2a100_formal`
- trace-calibrated CUCUMIS replay exports

## 论文使用建议

主图建议使用 `DistServe-2A100` vs `CUCUMIS-2A100-LB`。`CUCUMIS-2A100-RR` 可以作为 appendix/ablation，说明 naive round-robin 在异构请求下会被长请求拖累。

当前结果的可预期 narrative 是：

- CUCUMIS-2A100-LB 在 short-request p99 TTFT 上最稳定；
- DistServe continuous batching 明显缩小 completion/round/throughput 差距，但 CUCUMIS-2A100 仍在这些 guardrails 上更稳；
- low-long-fraction cases 中 DistServe 可能在 all/long TTFT 上更低；
- 因此不要把结论写成单边胜负，而应写成 request-sensitive latency 与 completion/throughput guardrail 的 trade-off。
