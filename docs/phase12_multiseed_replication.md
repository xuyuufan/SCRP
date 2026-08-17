# Phase 12: multi-seed ERI auxiliary replication

## Pre-registered protocol

The sole research question is whether the Phase 11 improvement from the public-information, set-valued ERI auxiliary objective at `lambda_eri=0.10` reproduces across independent random seeds. This is a development-only train/validation study. It does not change the architecture, O2 observation, objective, optimizer, learning rate, entropy coefficient, Frozen Greedy Baseline, sampler, environment, or reward.

The five seeds were frozen before inspecting Phase 12 outcomes: `20260816`, `20260817`, `20260818`, `20260819`, and `20260820`. Each seed has a paired control (`lambda_eri=0`) and treatment (`lambda_eri=0.10`) starting at episode 0, with 5,000 training episodes per arm and identical initialization, base/variant/scenario schedules, validation layouts, and validation scenarios. The total fixed budget is 50,000 episodes. There is no performance-based early stopping or extension.

All formal Phase 12 training requires `cuda:0` and fails rather than falling back to CPU. Python, NumPy, PyTorch CPU, and PyTorch CUDA RNGs are seeded; deterministic PyTorch algorithms are enabled; cuDNN deterministic mode is enabled and benchmarking is disabled. GPU trajectories are not expected to be bitwise identical to historical CPU trajectories.

The pooled interval uses a hierarchical paired bootstrap that first resamples seed and then base layout, preserving the paired control/treatment and DS1/DS2 block. Scenario rows are not treated as independent observations. The success gate and its numerical thresholds are frozen in `experiments/configs/phase12_multiseed_v1.json` before the main experiment.

Scientific-integrity declarations:

- Phase 8 raw rows accessed = NO
- Formal test rerun = NO
- Test split used for training, tuning, selection, diagnostics, or debugging = NO

## Execution environment and results

Phase 12 was based on Phase 11 commit `adcffa7375cb37cdb5b8d652e003a894cd5af126`; the formal training code was commit `314fec5b24d9fe8e3d461206f539e02e5f3c8319`. Formal training used `cuda:0`, an NVIDIA GeForce RTX 4060 Laptop GPU (8,188 MiB), driver 595.97, PyTorch `2.13.0+cu132`, and PyTorch CUDA runtime 13.2. `torch.cuda.is_available()` was true and one CUDA device was visible. The smoke gate passed for both arms, including finite forward/backward/optimizer work, ERI loss, action and padding masks, and CUDA checkpoint restore. It reported zero invalid actions and truncations, and `nvidia-smi` showed the Python process on the GPU. The 100-episode-per-arm timing probe measured 18.932 seconds, 10.564 episodes/s, 24.1 MB peak allocated CUDA memory, and a 37% GPU-utilization snapshot.

| Seed | Control overall | Treatment overall | Delta | DS1 delta | DS2 delta |
|---:|---:|---:|---:|---:|---:|
| 20260816 | 11.1042 | 10.3958 | -0.7083 | -0.4583 | -0.9583 |
| 20260817 | 10.5833 | 10.4271 | -0.1563 | -0.1667 | -0.1458 |
| 20260818 | 11.0938 | 10.3854 | -0.7083 | -0.9583 | -0.4583 |
| 20260819 | 10.2188 | 10.3021 | +0.0833 | -0.0625 | +0.2292 |
| 20260820 | 10.8229 | 10.5208 | -0.3021 | -0.3750 | -0.2292 |

Overall treatment was favorable in 4/5 seeds, DS1 in 5/5, and DS2 in 4/5. The hierarchical paired bootstrap estimated an overall delta of -0.3583 with 95% CI `[-0.6583, -0.0708]`. DS1 was -0.4042, CI `[-0.7500, -0.1250]`; DS2 was -0.3125, CI `[-0.6958, 0.0375]`. Thus the Phase 11 DS2 deterioration did not reproduce as a stable cross-seed effect, although DS2 remained unfavorable for seed 20260819 and its pooled interval still included zero.

Across seeds, the mean ERI-score-equivalent rate increased from 88.600% to 92.867%, the strictly-worse rate decreased from 11.400% to 7.133%, mean ERI penalty decreased from 0.09752 to 0.06230, probability mass on the ERI-optimal set increased from 75.790% to 83.095%, and greedy action outside that set decreased from 11.400% to 7.133%. Strictly-worse rate and mean penalty improved in 4/5 seeds; seed 20260820 was the exception for both. Exact deterministic agreement increased from 35.147% to 39.661% but remains a secondary diagnostic.

Mean entropy was 1.1549 for control and 0.8331 for treatment and was lower under treatment in all five seeds. Mean pre-clip gradient norm was 1.5271 versus 1.9670. Gradient clipping remained extremely frequent: 96.592% for control and 99.104% for treatment, with treatment higher in every seed. The weighted auxiliary/RL gradient ratio averaged 0.1532 for treatment. FGB refresh counts by seed were control/treatment `15/11`, `9/11`, `5/16`, `25/20`, and `13/6`, averaging 13.4/12.8; there was no uniform treatment direction. Every arm had zero invalid actions, truncations, numerical failures, and scenario mismatches.

All frozen success checks passed, including the stronger evidence criterion that the overall pooled 95% CI upper bound is below zero. Therefore `ERI_AUX_MULTISEED_REPLICATION_SUCCESS = YES`.

The evidence supports cross-seed replication of the fixed ERI auxiliary mechanism in development validation, not a new formal-test claim. A separately authorized larger-budget development run (for example 15k–25k) is now reasonable. It must not start automatically, and the persistent clipping plus systematically lower treatment entropy should remain explicit optimization risks rather than being tuned within Phase 12.
