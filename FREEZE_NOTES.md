# Freeze Notes

## Status

This repository is currently frozen on the post-refactor V0 validation baseline.

Do not change the main experiment logic in-place from this point onward.
Any new tuning, feature work, or risky cleanup should go through a new config,
new result path, or a separate branch.

## Frozen Baselines

### V0

- Config:
  - `experiments/configs/frozen_v0_gemma_mid_phase1conservative_v1_repro.json`
- Post-refactor validation result:
  - `results/v0_gemma_mid_sanity_r3_phase1conservative_v1_postrefactor_check.json`

### V1 Main Regression

- Config:
  - `experiments/configs/frozen_v1_gemma_mid_global_activity_repro.json`
- Preferred runner:
  - `experiments/run_frozen_eval_config.py`
- Validation result:
  - `results/v1_gemma_mid_global_activity_repro.json`

## Key Result

For the frozen `V0 gemma-mid` workpoint:

- `Phase-I + Phase-II TTFT mean = 4.5515x`
- `Phase-I + Phase-II wall mean = 0.9738x`

Reference result file:

- `results/v0_gemma_mid_sanity_r3_phase1conservative_v1_postrefactor_check.json`

For the frozen `V1 gemma-mid openworkload` workpoint:

- `Phase-I TTFT mean = 0.9315x`
- `Phase-I wall mean = 1.0010x`
- `Phase-I + Phase-II TTFT mean = 3.5327x`
- `Phase-I + Phase-II wall mean = 1.0113x`

Reference result file:

- `results/v1_gemma_mid_global_activity_repro.json`

## What Was Verified

The post-refactor validation run completed successfully and confirmed that:

- the evaluation entrypoint still runs end-to-end
- the hijack injection path still works
- summary construction still succeeds
- result JSON is written successfully

## Refactor Regressions Fixed Before Freeze

These refactor regressions were discovered during validation and fixed before
freezing the version:

1. `tests/evaluate_waveslice_claims.py`
   - restored `percentile as _percentile` import
2. `engine/vllm_hijacker.py`
   - restored explicit `WaveScheduler` import for `inject_wave_slice()`
3. `tests/eval_config.py`
   - made new provenance fields use `getattr(..., default)` so summary writing
     does not fail when a field is absent from a CLI path

## Minimal Regression Guard

Smoke test file:

- `tests/test_refactor_smoke.py`

Run:

```bash
python3 -m unittest tests.test_refactor_smoke -v
```

This smoke test is intended to catch the class of regressions where:

- code still compiles
- but the evaluation flow breaks midway
- or the inject/uninject path fails after refactoring

## Practical Rule Going Forward

Before trusting any future code change on this branch, run:

```bash
python3 -m unittest tests.test_refactor_smoke -v
```

If the change touches evaluation or runtime glue, also re-run the frozen
baseline configs and compare TTFT and wall time against:

- `results/v0_gemma_mid_sanity_r3_phase1conservative_v1_postrefactor_check.json`
- `results/v1_gemma_mid_global_activity_repro.json`

For V1 judgment specifically, prefer the real `openworkload mid` regression
target above instead of the tiny synthetic repeated default case.
