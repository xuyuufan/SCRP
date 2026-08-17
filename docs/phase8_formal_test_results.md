# Phase 8 Formal Test Results

## Frozen run identity

- Run ID: `formal-test-o2-vs-eri-seed20260816-v1`
- Code SHA: `0a04690763865f5f28fc12bdf6127fbec7efbed3`
- Training run: `formal-o2-mixed-seed20260816-run1`
- Checkpoint: episode 15000
- Checkpoint SHA-256: `1dbcb20686840df3d392a89a66cb28b79a3a7531d4669300bf09818a714ed255`
- Configuration SHA-256: `b284e0d1c9a8d750e4314b4bd7323af45f79d7cf2be3401ca22ec8b742793be3`
- Observation: O2, `Mmax=6`
- Formal-test protocol: `scrp-formal-protocol-v1`
- Root and bootstrap seed: 20260816

The tracked worktree was clean when the identity was frozen. Three unrelated,
pre-existing paper PDFs were untracked and are recorded transparently in the
identity; they did not affect the run.

## Integrity and budget

The one-shot run completed 24,000 paired coordinates: 240 test base layouts,
DS1 and DS2 variants, and 50 scenarios per static variant. Both primary
algorithms therefore completed exactly 24,000 episodes, for 48,000 total raw
rows. Invalid actions, truncations, scenario-ID mismatches, duplicate rows,
missing coordinates, and missing algorithm pairs were all zero.

The scenario-level raw artifact is gitignored at
`experiments/raw_results/formal-test-o2-vs-eri-seed20260816-v1/primary-results.jsonl`.
It contains 48,000 rows (17,635,139 bytes) and has SHA-256
`9bf1b694c4c4568224dc3a4f8d92559829a9f08635e55828b1ea9319a6bc7366`.

## Primary results

Delta is `relocations_RL - relocations_ERI`; positive values favor ERI.

| Dataset | RL O2 mean | ERI mean | Mean delta | Hierarchical bootstrap 95% CI |
|---|---:|---:|---:|---:|
| DS1 | 10.0879167 | 8.9147500 | 1.1731667 | [1.0391604, 1.3119167] |
| DS2 | 9.9148333 | 9.0717500 | 0.8430833 | [0.7573313, 0.9303333] |

The frozen O2 RL checkpoint did not outperform ERI on either dataset. The
observed gap was smaller on DS2 than on DS1, but this comparison alone does not
establish that the merged/larger batch structure or O2 caused the difference.

## Robustness statistics

All robustness tests use the 240 per-instance mean paired differences for each
dataset, not 12,000 scenario rows treated as independent observations.

| Dataset | Wilcoxon statistic | Wilcoxon p | Paired t statistic | Paired t p | Cohen dz |
|---|---:|---:|---:|---:|---:|
| DS1 | 420.5 | 5.86456e-26 | 10.1603 | 2.16653e-20 | 0.655844 |
| DS2 | 705.5 | 5.28067e-29 | 10.4101 | 3.58994e-21 | 0.671971 |

The stratified hierarchical paired bootstrap used 10,000 repetitions with
seed 20260816. No naive scenario-row independent t-test was performed.

## Static-instance wins, ties, and losses

| Dataset | RL wins | Ties | ERI wins |
|---|---:|---:|---:|
| DS1 | 9 | 73 | 158 |
| DS2 | 22 | 46 | 172 |

## Descriptive parameter-group extremes

These are descriptive heterogeneity summaries only because every group has
five test base layouts.

- DS1 strongest RL group: `S06_T04_mu0.67`, mean delta -0.220.
- DS1 weakest RL group: `S09_T06_mu0.67`, mean delta 6.340.
- DS2 strongest RL group: `S08_T03_mu0.50`, mean delta -0.004.
- DS2 weakest RL group: `S09_T06_mu0.67`, mean delta 5.244.

## Reproducibility and finality

The full regression suite passed with 212 tests before formal execution. No
training, checkpoint selection, model change, hyperparameter adjustment, or
adaptive behavior occurred after opening the formal test results. Any future
model must be reported as a separate follow-up experiment, and this test split
must no longer be described as untouched for that follow-up.

`FORMAL_TEST_COMPLETE = YES`
