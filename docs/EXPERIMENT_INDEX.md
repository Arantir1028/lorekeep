# Experiment Index

This file pins formal experiment artifacts that should not be confused with
smoke runs, aborted runs, or intermediate debugging outputs.

For a paper-writing-oriented summary, see
`docs/paper_facing_experiment_inventory_20260707.md`.

## Frozen Formal Results

| Experiment | Status | Primary Result |
| --- | --- | --- |
| `ratio_sweep_20step_5models_a100_overnight` | Complete, frozen; paper-facing request-heterogeneity sensitivity result | `results/chapter5_exports/ratio_sweep_20step_5models_a100_overnight` |
| `unified_distserve_continuous_cucumis_2a100_replay` | Complete; paper-facing DistServe equal-resource baseline with continuous-batching replay | `results/openworkload_ratio_sweep_lora8/unified_distserve_continuous_cucumis_2a100_replay` |
| `hardware_portability_a100_4090_5090` | Complete; paper-facing portability summary generated from copied-back full server runs | `results/chapter5_exports/hardware_portability_a100_4090_5090` |

## Paper-Facing Additions

| Section | Backing Experiment | Status |
| --- | --- | --- |
| Sensitivity to request heterogeneity | `ratio_sweep_20step_5models_a100_overnight` | Ready for evaluation text and figures |
| Guardrails beyond TTFT | `ratio_sweep_20step_5models_a100_overnight` | Ready; use compact table in main text and details in appendix |
| DistServe equal-resource comparison | `unified_distserve_continuous_cucumis_2a100_replay` | Ready; describe as continuous-batching replay from single-GPU physical stage costs under logical 2-GPU equal-resource comparison |
| Hardware portability | `hardware_portability_a100_4090_5090` | Ready as portability subsection or appendix table; use conservative wording for exception cases |

## Candidate / Auxiliary Results

| Experiment | Status | Use |
| --- | --- | --- |
| DistServe functional reproduction sensitivity outputs | Complete but auxiliary | Method calibration and sensitivity audit only |
| Official DistServe multi-GPU deployment reproduction | Candidate, not required for the current revision | High cost and implementation risk; do not block evaluation writing on it |

## Notes

- The superseded 10%-step run `ratio_sweep_formal_baichuan2_a100_r1`
  was stopped after switching the ratio sweep to five ratio points. Do not use
  it as a formal ratio-sweep result.
- For the frozen ratio-sweep card, see
  `docs/ratio_sweep_20step_5models_a100_overnight.md`.
- The earlier single-runtime proxy plan was removed. The active disaggregated
  comparison is the DistServe functional reproduction with token-level
  continuous-batching replay.
- The active DistServe comparison is equal-resource only:
  `DistServe-2A100` vs `CUCUMIS-2A100-RR/LB`. Do not use old unequal-resource
  diagnostics as paper-facing results.
- Trace-calibrated CUCUMIS replay is diagnostic only. The paper-facing
  CUCUMIS-2A100 comparison must come from the real RR/LB split-run outputs.
- The stopped CUCUMIS-only formal run
  `results/openworkload_ratio_sweep_lora8/cucumis_2a100_dispatch_real_split_formal`
  is diagnostic/partial only.
- The request-level serial DistServe replay
  `results/openworkload_ratio_sweep_lora8/unified_distserve_serial_cucumis_2a100_formal`
  is diagnostic only; its DistServe decode stage was too conservative.
- The active DistServe comparison uses token-level continuous-batching replay:
  `results/openworkload_ratio_sweep_lora8/unified_distserve_continuous_cucumis_2a100_replay`.
- The hardware portability sweep is tracked as
  `hardware_portability_a100_4090_5090`. Use the merged export under
  `results/chapter5_exports/hardware_portability_a100_4090_5090`, not partial
  server staging roots.
- Do not write that CUCUMIS wins every workload. Use average ratios, win counts,
  and exception-audit wording.

## Repository Data Policy

Git tracks the packaged LUT and runtime-calibration inputs under
`waveslice/data/lut_tables/` and these compact paper-facing artifacts:

- `results/chapter5_exports/ratio_sweep_20step_5models_a100_overnight/`
- `results/chapter5_exports/hardware_portability_a100_4090_5090/`
- `results/openworkload_ratio_sweep_lora8/unified_distserve_continuous_cucumis_2a100_replay/`

Full run trees, logs, generated LoRA adapters, local model snapshots, and
quarantined invalid results stay outside Git. They are too large or
machine-specific and can be regenerated from the tracked configs and scripts.
