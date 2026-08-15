# SCRP formal experiment protocol v1

Status: design frozen for implementation review. No formal training or result
claim has been run under this protocol.

## Experimental unit and leakage boundary

The split unit is a **base static instance**, consisting of physical layout,
container identities, ordered batch membership, and precedence. A stochastic
scenario is a within-batch permutation realization generated from a scenario
seed. Static-instance counts and scenario-rollout counts must always be
reported separately.

DS1 and its deterministic DS2 derivative share one `base_instance_id` and one
split assignment. A DS1 layout in train can therefore never reappear as DS2 in
validation or test.

## Frozen split

The recommended split is stratified independently inside every one of the 48
`(S,T,fill_rate)` groups:

| Split | Per group | Total base layouts | DS1 variants | DS2 variants |
|---|---:|---:|---:|---:|
| Train | 20 | 960 | 960 | 960 |
| Validation | 5 | 240 | 240 | 240 |
| Test | 5 | 240 | 240 | 240 |

The alternative 18/6/6 split would provide 288 validation and 288 test
layouts, reducing estimation variance, but removes 96 training layouts. The
20/5/5 split is preferred because 240 held-out layouts per split already cover
all 48 parameter groups while retaining two-thirds of each group for training.

`scrp_split_v1.json` uses seed 352026 and deterministic SHA-256 ranking within
each group. Each entry stores original/base, DS1, and DS2 IDs. It deliberately
contains no scenario seed or hidden order.

## Scenario seed streams

Scenario seeds are derived separately from the manifest using:

```text
seed = split_base + sorted_global_base_rank * 1,000,000 + scenario_index
train split_base      = 1,000,000,000,000
validation split_base = 2,000,000,000,000
test split_base       = 3,000,000,000,000
```

The three streams are disjoint, deterministic, and auditable. DS1 and DS2
variants of one base layout use the same root seeds. Every compared algorithm
must use the same `(instance_id, scenario_seed)` so the existing sampler yields
the same `scenario_id` and hidden permutations for that static instance.

Training uses a dynamic large pool: `scenario_index` is the per-base visit
number, so revisiting a layout yields a new scenario without materializing
scenario files. This has higher stochastic diversity and lower overfitting
risk than a small fixed pool while remaining exactly reproducible from the
training trace/checkpoint. A fixed finite pool is simpler to cache but invites
scenario memorization and is retained only for debug reproductions.

Validation and test seed sets are fixed. They must not participate in gradient
updates. Test seeds cannot be read for model or hyperparameter selection.

## Development protocol

- Use the frozen static split; never create a debug-only split that overlaps
  train and test.
- Evaluate 10 fixed scenarios per validation/test static variant.
- O1 may be used, but metadata must record `observation_version="O1"`.
- Use for pipeline debugging and ablations only.
- Development results must not be presented as formal paper conclusions.

Across DS1+DS2 this costs 4,800 validation or test rollouts per algorithm:
`240 base layouts * 2 variants * 10 scenarios`.

## Formal paper protocol

- Freeze split manifest, dataset version, config, code commit, checkpoint, and
  test seed schedule before opening test results.
- Validation: 20 scenarios per static variant, 9,600 rollouts per checkpoint
  evaluation across DS1+DS2.
- Test: 50 scenarios per static variant, 24,000 rollouts per algorithm across
  DS1+DS2. Report DS1 and DS2 separately before any combined result.
- Preserve every scenario-level record. Aggregation cannot replace raw data.
- Apply common random numbers across RL and every baseline.
- Baselines must use the same environment and only public visible state; no
  future hidden order is available through their interface.

Test scenario-count comparison across both datasets, per algorithm:

| Scenarios/static variant | DS1 | DS2 | Combined |
|---:|---:|---:|---:|
| 10 | 2,400 | 2,400 | 4,800 |
| 20 | 4,800 | 4,800 | 9,600 |
| 50 | 12,000 | 12,000 | 24,000 |
| 100 | 24,000 | 24,000 | 48,000 |

K=50 is recommended for formal test: it gives five times the development
scenario coverage without the twofold cost of K=100. Precision must be
reported empirically via uncertainty intervals rather than assumed from K.
With RL plus 2 baselines, formal test costs 72,000 rollouts; plus 4 baselines,
120,000 rollouts. Validation cost is normally paid only for candidate RL
checkpoints, not repeatedly for fixed baselines.

## Metrics and output

Required scenario-level fields are:

```text
dataset, split, instance_id, base_instance_id, parameter_group,
scenario_seed, scenario_id, algorithm, relocations, terminated, truncated
```

Aggregate in this order while retaining raw rows:

1. per static instance: mean, sample standard deviation, standard error, and
   confidence interval across scenarios;
2. per parameter group: mean of per-instance means;
3. per dataset: equal-instance mean, with parameter-group detail;
4. overall only as a supplementary summary, never as the sole result.

Truncated or non-terminated episodes are protocol failures and must be reported,
not silently removed.

## Paired statistical analysis

For algorithms A and B, compute `relocations_A - relocations_B` only on matched
`(dataset, instance_id, scenario_seed, scenario_id)` rows.

Primary inference is a stratified hierarchical paired bootstrap: resample base
instances within parameter group, then resample paired scenarios within each
selected instance, and report a 95% confidence interval for the mean paired
difference. This respects clustering of scenarios within a static layout.

Robustness checks:

- Wilcoxon signed-rank test on per-instance mean paired differences;
- paired t-test on per-instance mean differences, with diagnostics and effect
  size, not a naive scenario-row t-test;
- parameter-group intervals and multiplicity-adjusted group comparisons when
  making group-specific claims.

Report effect sizes and intervals; a p-value alone is insufficient.

## Training episode budgets

Do not pre-expand scenarios. Sample a base instance and derive its dynamic seed
at episode creation.

For 960 training base layouts:

| Episodes | DS1-only average visits/base | Mixed DS1+DS2 average visits/variant |
|---:|---:|---:|
| 1,000 | 1.04 | 0.52 |
| 10,000 | 10.42 | 5.21 |
| 100,000 | 104.17 | 52.08 |

These are scale descriptions, not recommended stopping points. Final budgets
depend on measured learning curves and compute limits.

## DS1/DS2 training designs

- **A: DS1-only train, test DS1+DS2.** Answers cross-batch-size
  generalization; DS2 is a genuine distribution shift.
- **B: mixed DS1+DS2 train.** Answers whether one general policy can serve both
  regimes. Sample base layout first and variant second so duplicated physical
  layouts do not double-weight a base.
- **C: separate DS1/DS2 models.** Answers the value of specialization but
  doubles tuning/training and introduces more model-selection comparisons.

Design B is the recommended primary deployment-oriented experiment. Design A
is the principal generalization study, and C is a secondary specialization
ablation. All three inherit the same base split.

## Baseline interface and observation versions

The future baseline contract is:

```text
policy(instance, visible_state) -> legal_destination
```

All algorithms use the same environment, instance, scenario seed, and legal
action rules. They may inspect only the currently public state and must never
access future hidden permutations. PBFS/RIRH and other baselines are not
implemented in Phase 3.5.

O1 remains an approximation. O1 versus future O2 is an information/
architecture ablation; every config and result bundle must record observation
version. Phase 3.5 does not implement O2.

## Directory and retention policy

```text
experiments/
  protocols/   reviewed protocol documents
  splits/      frozen static split manifests
  configs/     versioned machine-readable protocol configs
  raw_results/ scenario-level outputs (ignored by Git)
  summaries/   reviewed small aggregates and provenance pointers
```

No formal training, baseline execution, hyperparameter tuning, or full
scenario evaluation is authorized by this document.
