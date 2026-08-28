# Refactor Validation — 2026-08-11

## Version boundary

- Frozen predecessor: `compact-refactored-final-20260811`
- Refactor completion tag: `compact-refactored-final-v2-20260811`
- Existing paper-result provenance remains `results-final-20260714`.

## Phase II real-GPU validation

Environment: vLLM 0.10.1 V1, A100-SXM4-80GB, Gemma-7B, two synthetic LoRA
adapters, one long and three short requests arriving together.

The diagnostic attempts were preserved rather than overwritten:

| Result root suffix | Outcome | Finding |
| --- | --- | --- |
| `scheduler_priority_smoke` | invalid first attempt | strict config mapped `model.name/path` incorrectly; fixed in evaluator |
| `scheduler_priority_overlap` | applied=0 | only waiting candidates were hidden |
| `scheduler_priority_overlap_v2` | applied=0 | value score rejected every real anchor |
| `scheduler_priority_overlap_v3` | applied=0 | fragment-size filter produced no actual candidate |
| `scheduler_priority_overlap_v4` | applied=1 | one-tick queue restoration fix validated |
| `scheduler_priority_overlap_v5` | applied=1 | final simplified gate and stable priority promotion validated |

Final v5 result: 4/4 requests finished, no timeout, 12 Phase II observations,
one cashout application, one cooldown observation, and one priority-lane
activation. This proves functional triggering and completion, not statistically
meaningful performance.

## DistServe formula replay

The maintained decode formula conserves measured service time:

```text
tail_token_cost = (total_decode_service - first_token_service) / (N - 1)
```

The corrected output is stored separately at
`results/openworkload_ratio_sweep_lora8/unified_distserve_continuous_cucumis_2a100_replay_formula_v2`.
It contains 50 method cases, 4,000 request rows, 4,000 stage-cost rows, and 100
equal-resource comparison rows. The manifest records
`continuous_batching_v2_conserved_decode_service`.

Relative to the preserved old replay, all-TTFT P99 is unchanged. Across 50
cases, median changes are +1.63% for completion P99, +1.36% for round wall time,
and -1.35% for throughput.

## Static and API validation

- 27 pytest contracts pass.
- `compileall` passes and all active JSON configs parse.
- vLLM V1 `schedule` hooks restore the exact original method
  objects after uninject.
- Maintained Python size is 9,330 physical lines across 70 files.
- The policy surface is 79 fields; runtime-core functions are at most 64 lines
  in the AST hotspot audit.
