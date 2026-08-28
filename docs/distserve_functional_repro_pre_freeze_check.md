# DistServe 功能性复现冻结前检查

Status: DistServe continuous-batching replay + CUCUMIS-2A100 comparison complete; not frozen yet.

Date: 2026-07-03; updated 2026-07-06

This note reviews the current `DistServe` functional reproduction plan before we decide whether to freeze it as a formal paper result. It is not a manuscript section.

## 结论

当前实验已经改成同资源主口径：

- `DistServe-2A100` vs `CUCUMIS-2A100-RR`
- `DistServe-2A100` vs `CUCUMIS-2A100-LB`

不再使用旧的不等资源比较作为主比较，也不再导出 one-GPU-equivalent guardrail 作为 paper-facing 结果。

我的建议是：

- DistServe 先使用单 A100 逐请求真实推理测量 stage cost，再按独立 P/D 队列做逻辑回放，decode 阶段使用 token-level continuous batching；
- RR 和 LB 两种 CUCUMIS-2A100 分发都先真实跑完；
- 不再使用任何 1A100 vs 2A100 的对比作为 paper-facing 结果；
- 等真实 split-run 的完整结果出来后，再决定主图用 LB、RR，还是两者都展示。

## 输入与输出

Source experiment:

- `ratio_sweep_20step_5models_a100_overnight`

Current paper-candidate run:

- `results/openworkload_ratio_sweep_lora8/unified_distserve_continuous_cucumis_2a100_replay`

Diagnostic-only older runs:

- smoke: `results/openworkload_ratio_sweep_lora8/cucumis_2a100_dispatch_real_split_smoke_run`
- stopped formal: `results/openworkload_ratio_sweep_lora8/cucumis_2a100_dispatch_real_split_formal`
- LUT/runtime-calibrated DistServe export: `results/chapter5_exports/distserve_functional_repro_ratio_sweep`
- request-level serial decode replay: `results/openworkload_ratio_sweep_lora8/unified_distserve_serial_cucumis_2a100_formal`

Formal scope:

- models: Baichuan2-7B-Chat, Gemma-2-9B-IT, Gemma-7B-IT, Mistral-7B-Instruct-v0.2, Qwen2.5-7B-Instruct
- workloads: `mid_l10/30/50/70/90`, `high_l10/30/50/70/90`
- DistServe resource profile: `DistServe-2A100`
- CUCUMIS resource profiles: `CUCUMIS-2A100-RR`, `CUCUMIS-2A100-LB`
- KV profile: realistic PCIe
- default TTFT mode: `prefill_finish`
- stage measurement: real single-A100 individual request measurement, then logical P/D replay
- decode replay mode: `continuous_batching`, batch size 16, batch alpha 0.08

Expected rows from unified run:

- `distserve_serial/distserve_serial_method_metrics.csv`: 50 rows expected
- `cucumis_2a100/cucumis_2a100_real_method_metrics.csv`: 100 rows expected
- `cucumis_2a100/cucumis_2a100_real_request_metrics.csv`: 8000 rows expected
- `unified_equal_resource_comparison.csv`: 100 rows expected

## CUCUMIS-2A100 口径

当前正式口径是真实 split-run：

- 每个 workload 按 RR/LB 拆成两个子 workload；
- 两个子 workload 分别作为两个独立 CUCUMIS replica 运行；
- 合并两个 replica 的真实 request-level timings；
- RR/LB 只改变 front-end dispatch，不改变 CUCUMIS scheduler 内部逻辑。

`least_backlog` 的分发规则是：请求到达时选择预计 backlog 更小的 replica。backlog 由已分配请求的预计服务量维护，不是简单队列长度。预计服务量只用于分发，最终指标来自真实运行结果。

## Equal-Resource 当前状态

早期 LUT/runtime-calibrated DistServe export 和 trace-calibrated replay 暴露出过强的建模假设，不能作为论文结论。它们只作为内部诊断保留。

当前正式状态：

- smoke complete: `baichuan2-7b-chat`, `mid_l10`, RR and LB；
- old CUCUMIS-only formal stopped after `baichuan2-7b-chat/mid_l10/round_robin/replica0`；
- unified continuous replay complete: 5 models x 10 workloads x DistServe continuous-batching replay；
- CUCUMIS real split-run complete: 5 models x 10 workloads x 2 dispatchers x 2 replicas。

Ratio 定义仍为：`DistServe-2A100 / CUCUMIS-2A100`。小于 1 表示 DistServe 指标更低；大于 1 表示 CUCUMIS 指标更低。throughput ratio 小于 1 表示 DistServe throughput 更低。

Smoke result, diagnostic only:

| Dispatcher | all p99 TTFT | short p99 TTFT | round time | throughput |
| --- | ---: | ---: | ---: | ---: |
| `round_robin` | 469.8 ms | 159.8 ms | 10.14 s | 3.94 rps |
| `least_backlog` | 527.4 ms | 159.8 ms | 10.68 s | 3.75 rps |

In this single smoke case, RR is slightly better than LB on CUCUMIS-side round time and throughput. Do not generalize from this case; it is only a check that both dispatchers run and produce comparable outputs.

## 解释

暂不写结论。等真实 split-run 完整结果出来后，再判断：

- RR 是否只是 ablation；
- LB 是否适合作为主线；
- DistServe 的 TTFT 优势和 CUCUMIS 的 guardrail 表现是否稳定。

## Paper-Facing 建议

候选主表/主图：

- `DistServe-2A100`
- `CUCUMIS-2A100-LB`
- `CUCUMIS-2A100-RR`

推荐 appendix/ablation：

- `CUCUMIS-2A100-RR`
- KV transfer sensitivity
- TTFT semantic sensitivity

写法建议：

- 不写成 “CUCUMIS 全面优于 DistServe”。
- 先用真实 split-run 决定最终写法。
- 对 reviewer 的资源公平性质疑，用同资源对比回答。

## 冻结前 Checklist

Completed:

- [x] Remove one-GPU-equivalent guardrail from active resource profiles.
- [x] Export both CUCUMIS-2A100 dispatchers: RR and LB.
- [x] Implement real split-workload runner for both CUCUMIS-2A100 dispatchers.
- [x] Remove old single-resource PDF figures from the export directory.
- [x] Implement real single-A100 DistServe stage measurement and logical P/D replay.
- [x] Implement unified DistServe + CUCUMIS experiment runner.
- [x] Replace request-level serial DistServe decode replay with token-level continuous batching.
- [x] Regenerate equal-resource comparison from continuous-batching DistServe replay.

Required before freeze:

- [x] Finish smoke split-run for RR and LB.
- [x] Finish unified 5-model formal run/replay.
- [ ] Decide whether `CUCUMIS-2A100-LB` is the only paper-facing CUCUMIS-2A100 line or whether RR should stay in the main table.
- [ ] Decide final TTFT semantic mode for the manuscript.
- [ ] Mark this experiment as frozen in `docs/EXPERIMENT_INDEX.md` only after the above decisions are made.

Current recommendation:

Keep `DistServe` in `pre-freeze` state until the real RR/LB split-run results are available.
