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

All 90,000 formal episodes completed from episode 0 on `cuda:0` using an NVIDIA GeForce RTX 4060 Laptop GPU, PyTorch `2.13.0+cu132`, CUDA runtime 13.2, and driver 595.97. Both models reported CUDA parameters at formal-run start; all paired scenario fingerprints matched. The CUDA smoke gate passed and `nvidia-smi` showed the Python process on GPU.

### Per-seed overall trajectories

| Seed | 2.5k | 5k | 7.5k | 10k | 12.5k | 15k | Frozen-rule classification |
|---:|---:|---:|---:|---:|---:|---:|---|
| 20260816 | -1.4792 | -0.7083 | +0.0313 | -0.1771 | +0.1250 | -0.1563 | high variance |
| 20260818 | +0.2500 | -0.7083 | -0.5208 | -0.1563 | -0.2188 | -0.2396 | early improvement then deterioration |
| 20260819 | -1.5729 | +0.0833 | -0.2500 | -0.5833 | -0.0417 | +0.0104 | high variance |

The two Phase 12 favorable seeds remain favorable at 15k, but their advantage is smaller than at 5k. Seed 20260819 remains marginally unfavorable at 15k rather than becoming favorable. Its magnitude falls from +0.0833 at 5k to +0.0104, but the direction still supports seed-dependent optimization instability.

### Pooled longitudinal results

| Episode | Overall delta | 95% CI | Favorable seeds |
|---:|---:|---:|---:|
| 2,500 | -0.9340 | `[-1.7049, 0.2222]` | 2/3 |
| 5,000 | -0.4444 | `[-0.8368, 0.0590]` | 2/3 |
| 7,500 | -0.2465 | `[-0.5486, 0.0417]` | 2/3 |
| 10,000 | -0.3056 | `[-0.6076, -0.0660]` | 3/3 |
| 12,500 | -0.0451 | `[-0.2500, 0.1563]` | 2/3 |
| 15,000 | -0.1285 | `[-0.2674, 0.0278]` | 2/3 |

The 15k primary endpoint remains favorable, but its interval includes zero and the magnitude is attenuated relative to 2.5k and 5k. The strongest late evidence occurs at 10k. There is no consecutive systematic positive reversal, but the pooled trajectory is best described as early improvement followed by deterioration/plateau rather than continued expansion.

At 15k, DS1 is -0.1944 with CI `[-0.4028, 0.0139]` and 3/3 favorable seeds. DS2 is -0.0625 with CI `[-0.2639, 0.1389]` and 2/3 favorable seeds. Neither dataset shows stable material degradation, but the DS2 long-run advantage is weak and uncertain.

### ERI mechanism trajectory

| Episode | Equivalent C/T | Strictly worse C/T | Mean penalty C/T |
|---:|---:|---:|---:|
| 2,500 | 79.533% / 92.287% | 20.467% / 7.713% | 0.21727 / 0.05926 |
| 5,000 | 87.096% / 93.755% | 12.904% / 6.245% | 0.11362 / 0.05173 |
| 7,500 | 85.779% / 93.454% | 14.221% / 6.546% | 0.11813 / 0.05681 |
| 10,000 | 84.048% / 93.190% | 15.952% / 6.810% | 0.18529 / 0.05530 |
| 12,500 | 91.648% / 93.115% | 8.352% / 6.885% | 0.07073 / 0.05850 |
| 15,000 | 92.476% / 93.679% | 7.524% / 6.321% | 0.06565 / 0.05587 |

Treatment retains better local ERI decision quality at 15k, but the gap narrows because control catches up. The treatment strictly-worse rate and penalty are approximately flat after 5k, while relocation improvement weakens. This is partial evidence that additional ERI consistency does not automatically translate into a growing long-horizon return advantage.

### Optimization trajectory

| Window ending | Entropy C/T | Clip frequency C/T | Pre-clip norm C/T | Weighted auxiliary/RL ratio T |
|---:|---:|---:|---:|---:|
| 2,500 | 1.3449 / 0.8924 | 95.893% / 98.987% | 1.4647 / 1.7962 | 0.1669 |
| 5,000 | 1.0890 / 0.8137 | 95.893% / 99.253% | 1.4650 / 2.1812 | 0.1572 |
| 7,500 | 1.1278 / 0.7507 | 96.853% / 99.200% | 1.6513 / 2.0619 | 0.1415 |
| 10,000 | 0.9793 / 0.7129 | 97.707% / 99.093% | 1.8734 / 1.9768 | 0.1495 |
| 12,500 | 0.8530 / 0.7273 | 97.280% / 98.773% | 1.8563 / 1.9217 | 0.1488 |
| 15,000 | 0.8707 / 0.7441 | 97.600% / 99.093% | 1.8715 / 2.1081 | 0.1508 |

Treatment entropy declines through 10k and then modestly rebounds; there is no monotonic collapse at 15k. Auxiliary gradient strength remains material after 5k rather than disappearing. Treatment clipping is approximately 99% throughout and reaches at least 99% in consecutive late windows, triggering `OPTIMIZATION_STABILITY_WARNING = YES`. Pre-clip norms are usually higher for treatment but do not satisfy the pre-registered monotonic late-rise warning.

FGB accepted refreshes were control/treatment `25/21`, `13/23`, and `32/29` for the three seeds, totaling `70/73`; rejected refreshes totaled `11180/11177`. Exact refresh episode positions are retained in the compact summary. There is no uniform arm direction. Invalid actions, truncations, numerical failures, and scenario mismatches are all zero across all windows and arms.

All pre-registered long-run success checks pass, so `ERI_AUX_LONGRUN_STABILITY_SUCCESS = YES`. This means the point-estimate benefit extends to 15k under the frozen gate; it does not mean the benefit expands, is statistically conclusive at 15k, or exceeds ERI. Given the attenuated endpoint, high per-seed volatility, and persistent clipping, automatically extending to 25k is not justified.

A bounded, separately pre-registered direct `RL_new - ERI` development comparison is now worthwhile because both Phase 12 replication and the Phase 13 endpoint gate pass. It should use identical public information, paired layouts/scenarios, explicit DS1/DS2 reporting, and no frozen formal-test access. Optimization stabilization should be treated as a parallel next priority before any larger training-budget escalation.
