# Phase 13: long-run ERI auxiliary stability

## Pre-registered protocol

The sole primary question is whether the Phase 12 `lambda_eri=0.10` treatment advantage remains stable when training is extended from 5,000 to 15,000 episodes without changing the successful configuration. Secondary audits distinguish sustained improvement, plateau, deterioration, or reversal and track entropy, gradient clipping, DS1/DS2 heterogeneity, ERI mechanism quality, loss, auxiliary strength, and Frozen Greedy Baseline behavior.

The study is development-only. Control uses `lambda_eri=0`; treatment uses `lambda_eri=0.10`. Architecture, O2 observation, initialization logic, optimizer, learning rate, entropy coefficient, clipping threshold, FGB, sampler, DS1/DS2 sequence, scenarios, reward, environment, validation data, capacity, and budget are otherwise identical. Every arm starts from episode 0; no Phase 12 checkpoint or trajectory is continued.

The three seeds were frozen before Phase 13 outcomes: `20260816` and `20260818` represent clearly favorable Phase 12 seeds, while `20260819` is the only Phase 12 overall-unfavorable seed. This deliberately tests both retention of favorable behavior and possible reduction of unfavorable-seed variance. Seeds cannot be replaced, added, or removed after results are observed.

Each of six arms trains for exactly 15,000 episodes on `cuda:0`, for 90,000 formal episodes total. Validation checkpoints are frozen at 2,500, 5,000, 7,500, 10,000, 12,500, and 15,000 episodes. At every checkpoint, overall, DS1, and DS2 validation and paired hierarchical bootstrap intervals are computed at the `seed -> base layout -> paired arms` levels. Fixed public-state ERI diagnostics and per-window entropy, gradients, clipping, losses, auxiliary/RL gradient ratio, and FGB refreshes are recorded.

Trajectory classification is deterministic: a range of at least 1.0 or at least two sign changes is high variance; early-best degradation of at least 0.25 is early improvement followed by deterioration; an all-negative endpoint no worse than the first checkpoint is sustained improvement; otherwise an early favorable trajectory remaining favorable late is plateau; all deltas within +/-0.25 are no material difference. The success and independent optimization-warning thresholds are frozen in `experiments/configs/phase13_longrun_stability_v1.json`.

The eventual objective is to produce an RL policy that achieves lower paired relocation counts than ERI under the same public-information regime. Phase 13 only determines whether the current ERI-guided improvement is stable enough to justify a direct `RL_new - ERI` development comparison.

Scientific-integrity declarations:

- Phase 8 raw formal-test rows accessed = NO
- Formal test rerun = NO
- Test split accessed = NO
- Automatic 25k extension = NO
- Adaptive tuning or early stopping = NO

## Results

This section is populated from the compact result artifact after all six fixed runs complete.
