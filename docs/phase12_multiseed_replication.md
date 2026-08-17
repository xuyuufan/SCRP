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

This section is populated from the committed compact result artifact after the fixed experiment completes.
