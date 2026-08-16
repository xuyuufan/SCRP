# Phase 7A - Formal Training Rehearsal

Status: bounded rehearsal only. No formal long training, checkpoint selection,
hyperparameter search, formal test access, PBFS/PBFSA, or performance claim was
made.

## Candidate configuration

Exactly one primary candidate was used. It is the Phase 6 audited configuration
with its lifecycle status changed to `CANDIDATE_FOR_REHEARSAL`:

| Field | Frozen value |
|---|---|
| Observation | O2, Mmax=6, 12 features |
| Sampling | mixed DS1+DS2, base-balanced, bucket by S |
| Optimizer | Adam, lr=0.00025, eps=1e-5 |
| Gamma / entropy | 1.0 / 0.01 |
| Batch size | 4 episodes |
| Gradient clip | 0.5 |
| Baseline | frozen greedy policy |
| Refresh | paired one-sided t-test, p<0.05 |
| Root seed | 20260816 |
| Checkpoint / validation interval | 10 / 10 iterations |
| Device | CPU |

No fallback configuration was activated. The candidate file SHA-256 is
`b284e0d1c9a8d750e4314b4bd7323af45f79d7cf2be3401ca22ec8b742793be3`.

## Rehearsal sequence and coverage

The first 500 train episodes passed the stability gate. Training stopped at a
temporary checkpoint, the full trainer/sampler/baseline/optimizer state was
loaded, and the same trajectory continued to 1,000 episodes. No single training
trajectory exceeded 1,000 episodes.

| Coverage | Measured value |
|---|---:|
| Unique train base layouts | 639 |
| S=5 / 6 / 7 / 8 / 9 / 10 episodes | 168 / 164 / 160 / 152 / 192 / 164 |
| DS1 / DS2 episodes | 499 / 501 |
| Unique scenario seeds | 1,000 |
| Bases visited more than once | 258 |
| Repeated visits beyond first visit | 361 |

The observed bucket proportions are within eight percentage points of their
base-population proportions. There is no severe S or variant skew.

## Numerical and rollout stability

| Diagnostic | Measured range/count |
|---|---|
| Loss | -0.374647 to 0.571266 |
| Policy loss | -0.365491 to 0.574207 |
| Entropy | 0.294150 to 2.087775 |
| Pre-clip gradient norm | 0.251934 to 4.171457 |
| Invalid actions | 0 |
| Truncations | 0 |
| Scenario mismatches | 0 |
| Zero-legal-action failures | 0 |
| Non-finite values | 0 |
| Exploding gradients (threshold 1,000) | 0 |
| Empty-decision episodes safely skipped | 8 |

Relocation and advantage values were recorded only as pipeline diagnostics and
are not interpreted as model effectiveness.

## Baseline refresh audit

Five statistically authorized refreshes occurred over 250 iterations, a 2%
refresh rate rather than an abnormal near-continuous refresh pattern.

| Iteration | n | Paired mean difference | t | one-sided p |
|---:|---:|---:|---:|---:|
| 28 | 4 | 2.25 | 2.3772 | 0.04893 |
| 37 | 4 | 2.50 | 5.0000 | 0.00770 |
| 86 | 4 | 0.75 | 3.0000 | 0.02883 |
| 192 | 4 | 0.75 | 3.0000 | 0.02883 |
| 235 | 4 | 2.25 | 3.0000 | 0.02883 |

The summary stores the old and new baseline state SHA-256 for every event.

## Checkpoint and split audit

The 500-episode stop/load check preserved iteration, episodes seen, the next
sampled batch, all per-base visit counters, baseline state and refresh history,
and optimizer state. No scenario was repeated because of resume.

Validation was limited to a `PIPELINE SMOKE ONLY`: 12 validation bases (two per
S bucket), two scenarios per base, 24/24 completed, zero invalid actions and
zero truncations. It was not used for selection, tuning, or config changes.
The test split usage assertion reports exactly zero.

Temporary 500/1,000 checkpoints, raw logs, and smoke outputs were deleted. Only
the candidate config, compact summary, documentation, runner, and tests remain.

## Measured compute cost

The RAM-instrumented CPU run measured 1,000 episodes in 113.05 seconds (8.846
episodes/s; 107.05 LOW decisions/s) with peak RAM 363,843,584 bytes and no VRAM
usage. An earlier identical-seed run under different host load took 380.20
seconds. Sampling coverage, numerical results, and baseline refresh history
were identical. Because the observed wall-time variance is material, budget
planning uses the slower measured rate rather than the optimistic repeat.

| Budget | Conservative measured extrapolation |
|---:|---:|
| 10,000 | 3,801.96 s (1.06 h) |
| 25,000 | 9,504.91 s (2.64 h) |
| 50,000 | 19,009.81 s (5.28 h) |
| 100,000 | 38,019.62 s (10.56 h) |

The recommended first formal budget is 25,000 episodes. This recommendation
does not start or authorize the run by itself.

## Gates

`REHEARSAL_PASS = YES`

`APPROVE_FORMAL_TRAINING = YES`

The frozen run identity is: O2, root seed 20260816, split manifest
`scrp-static-split-v1`, candidate config hash shown above, 25,000 planned
episodes, validation interval 10, checkpoint interval 10. The final code commit
SHA is recorded in the Phase 7A pull request and completion report.
