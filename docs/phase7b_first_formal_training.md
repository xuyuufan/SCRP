# Phase 7B - First Formal Training Run

Status: formal training complete. This phase used the train and validation
splits only. It did not access the formal test split and makes no comparison or
superiority claim against ERI.

## Frozen run identity

| Field | Frozen value |
|---|---|
| Run ID | `formal-o2-mixed-seed20260816-run1` |
| Code SHA | `f4140432ccc12a917e9cd2459fdb49bf1e1aff27` |
| Config SHA-256 | `b284e0d1c9a8d750e4314b4bd7323af45f79d7cf2be3401ca22ec8b742793be3` |
| Split / dataset | `scrp-static-split-v1` / `ku-galle-bacci-ds1-ds2-ec672df` |
| Observation | O2, Mmax=6 |
| Root seed | 20260816 |
| Planned / completed episodes | 25,000 / 25,000 |

The approved Adam configuration (learning rate 0.00025, gamma 1.0, entropy
coefficient 0.01, batch size 4, gradient clip 0.5), mixed base-balanced
DS1/DS2 sampler, S buckets, paired frozen greedy baseline, and refresh threshold
of one-sided p<0.05 remained unchanged throughout the run.

## Training coverage and stability

Training took 3,358.75 seconds, or 7.443 train episodes/second. Total wall time,
including ten full validation passes, was 7,795.07 seconds.

| Coverage | Episodes |
|---|---:|
| S=5 / 6 / 7 / 8 / 9 / 10 | 4,236 / 4,312 / 4,196 / 4,152 / 4,156 / 3,948 |
| DS1 / DS2 | 12,616 / 12,384 |
| Unique train base layouts | 960 |
| Unique dynamic scenario seeds | 25,000 |

All monitored loss, policy-loss, entropy, and gradient-norm values were finite.
There were zero invalid actions, truncations, and scenario mismatches. The
paired significance rule authorized 27 baseline refreshes; the completion
artifact records every iteration, statistic, p-value, and old/new state hash.

At episode 1,000 the run actively stopped, saved, loaded, and resumed. Policy,
baseline, sampler visit counts, iteration, episode count, and non-empty optimizer
state all matched after loading.

## Frozen validation and checkpoint selection

Validation ran every 2,500 training episodes. Each pass evaluated 20 scenarios
per static variant: 240 validation base layouts x DS1/DS2 x 20 = 9,600 rollout
episodes. The selection metric was frozen before training as:

`(mean(DS1 per-instance means) + mean(DS2 per-instance means)) / 2`

Lower is better.

| Training episode | Selection score |
|---:|---:|
| 2,500 | 10.616666667 |
| 5,000 | 10.551458333 |
| 7,500 | 10.587395833 |
| 10,000 | 10.462916667 |
| 12,500 | 10.439687500 |
| 15,000 | **10.292083333** |
| 17,500 | 10.420625000 |
| 20,000 | 10.499479167 |
| 22,500 | 10.342604167 |
| 25,000 | 10.347708333 |

The best validation checkpoint is episode 15,000, baseline state version 19,
with model-state SHA-256
`4f2333d26aaf312caa025b7bf9a7ed68bafd1f909519626503d9841d0350a125`
and checkpoint SHA-256
`1dbcb20686840df3d392a89a66cb28b79a3a7531d4669300bf09818a714ed255`.

The final checkpoint model-state SHA-256 is
`68ecf418f56877b27c05694a0379a74db9b8b8632d3e2314479973a61553fc34`;
its checkpoint SHA-256 is
`4138bd331d389d98ea6ef2e04df436715134d1a6ffedafe39cb2e1e1ccddad39`.

## Artifact and split audit

The local checkpoint directory retains only best, final, latest, and the six
non-duplicate required milestones (1,000, 2,500, 5,000, 10,000, 15,000, and
20,000). No temporary checkpoint or debug/cache artifact remains. Binary model
files are gitignored; their hashes and inventory are recorded in the completion
artifact.

The committed training artifact contains 100-episode aggregate windows, not
per-episode or per-iteration rows. The committed validation artifact contains
only the ten checkpoint-level DS1/DS2 distributions and selection records. Its
larger per-instance and parameter-group derivative is retained locally at
`experiments/raw_results/formal-o2-mixed-seed20260816-run1/validation-full-derived.json`,
which is gitignored. The authoritative local derivative is 2,288,965 bytes with
SHA-256
`be02cff39c29ce4d36cf9c843791c522385f9926b0790f1cfb514ef3709c5bca`;
it contains no scenario-level rows.

Formal test episode usage is exactly 0. ERI training or checkpoint-selection
usage is exactly 0. Formal test evaluation requires separate Phase 8
authorization and was not run.

`FORMAL_TRAINING_COMPLETE = YES`
