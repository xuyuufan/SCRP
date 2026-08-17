# Phase 14: direct RL-vs-ERI development comparison

## Outcome

`RL_BEATS_ERI_DEVELOPMENT = NOT_EVALUATED`

The Phase 14 protocol and evaluation implementation are complete, but the
direct comparison did not run. Phase 13 recorded validation summaries at 5k,
10k, and 15k without persisting any Treatment policy checkpoint. Re-training
or replaying Phase 13 to reconstruct a policy is explicitly outside Phase 14,
so the evaluator stopped before its first ERI rollout.

This is a development-only result. Phase 8 raw formal-test rows were not
accessed, the formal test was not rerun, and the test split was not used.

## Git and protocol identity

- Base SHA: `6cc94b667f017044ab9cbf530f043088ec02cac0`
- Base branch: `phase/scrp-phase-13-longrun-stability`
- Stacked dependency: Draft PR #14, which was open and unmerged at preflight
- Phase 14 branch: `phase/scrp-phase-14-rl-vs-eri-development`
- Protocol: `phase14-rl-vs-eri-development-v1`
- Delta: `relocations_RL - relocations_ERI`; negative favors RL

The checkpoint selection and holdout definition were committed before any
Phase 14 ERI evaluation was attempted.

## Frozen checkpoint selection

The preferred rule was fixed in advance: for each seed, select the Treatment
checkpoint with minimum Phase 13 fixed-validation relocations among 5k, 10k,
and 15k, breaking exact ties toward the earlier checkpoint. Phase 14 ERI
results are forbidden as a selection input.

None of those policy states was persisted. The preregistered conservative
fallback therefore selected the Phase 13 15k primary endpoint for every seed:

| Policy seed | Frozen episode | Reason | Available |
|---:|---:|---|:---:|
| 20260816 | 15,000 | Phase 13 primary-endpoint fallback; reconstruction by retraining prohibited | No |
| 20260818 | 15,000 | Phase 13 primary-endpoint fallback; reconstruction by retraining prohibited | No |
| 20260819 | 15,000 | Phase 13 primary-endpoint fallback; reconstruction by retraining prohibited | No |

The evaluator expects these gitignored local files:

- `checkpoints/phase13-longrun-stability/seed20260816-treatment-15000.pt`
- `checkpoints/phase13-longrun-stability/seed20260818-treatment-15000.pt`
- `checkpoints/phase13-longrun-stability/seed20260819-treatment-15000.pt`

Before any rollout, all three files must exist and all metadata must agree on
the frozen seed, 15k episode count, lambda_eri=0.10, audited ERI auxiliary
version, and model state presence. A partial or corrupt set fails closed.

## Independent development holdout

The static split manifest contains 48 parameter groups and five validation
base layouts per group. Phases 11-13 used the lexicographically first
validation layout in each group for their fixed checkpoint probe. Phase 14
excludes those 48 layouts and freezes the other four per group:

- 192 unique validation base layouts
- 192 DS1 and 192 DS2 static variants
- scenario indices 0-19 for each static variant
- 7,680 paired RL/ERI coordinates per policy seed
- 23,040 total paired coordinates across three independently evaluated policies
- zero train layouts and zero test layouts

The holdout is substantially larger than the 48-layout, one-scenario probe
used for checkpoint selection and remains balanced across all parameter groups
and DS1/DS2.

Limitation: the legacy Phase 7B training workflow evaluated the full validation
split. Consequently these 192 layouts are not globally untouched development
data. They are, however, the maximum legal non-test pool, were not used by the
Phase 11/12/13 fixed probes, lambda selection, or current checkpoint selection,
and were never training layouts. Phase 14 records this limitation rather than
reusing the test split.

## Paired public-information evaluation

For every policy seed, dataset, base layout, static variant, and scenario
index, RL and ERI are independently reset with the identical instance and
scenario seed. The implementation verifies both scenario ID and a fingerprint
of the initial public state before accepting the pair.

RL uses only the O2 public observation and legal-action mask. Its action is the
lowest-index deterministic `argmax` under the repository convention. The
policy is placed in `eval()` mode, executed inside `torch.inference_mode()`,
has gradients disabled, and is checked by policy-state hash before and after
each paired rollout. There is no optimizer, FGB, auxiliary, online update,
sampling, exploration, fallback, beam search, reranking, or test-time
adaptation path in the Phase 14 module.

ERI receives the same live public state: current layout, public batch
precedence, and revealed current order. Neither method receives the scenario
seed, hidden future order, future identities, or test metadata as a decision
feature. The evaluator also rejects invalid actions, truncations, non-finite
statistics, scenario mismatches, and policy-state changes.

## Frozen analysis

When the original frozen policies are recoverable, the committed evaluator
will report:

- per-seed, pooled, DS1, and DS2 RL/ERI means and `RL - ERI` gaps;
- relative gap, paired wins/ties/losses, and seed consistency;
- hierarchical paired bootstrap over seed -> dataset -> base layout -> scenario;
- paired Wilcoxon, paired t-test, and Cohen's dz as secondary evidence;
- exact RL/ERI agreement, ERI-score equivalence, strictly worse ERI-score rate,
  mean ERI penalty, and outcomes after non-ERI-minimum actions;
- failure and success regimes by dataset, stack count, fill ratio, batch size,
  legal destinations, ERI-optimal ties, episode length, and retrieval stage.

`RL_BEATS_ERI_DEVELOPMENT = YES` requires a negative pooled gap, a bootstrap
95% CI wholly below zero, at least two favorable policy seeds, non-positive
DS1 and DS2 pooled gaps, a consistent paired win distribution, and every
integrity gate. A negative point estimate whose CI includes zero is
`INCONCLUSIVE`; a positive point estimate or integrity failure is `NO`.

## CUDA preflight

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- PyTorch: 2.13.0+cu132
- PyTorch CUDA runtime: 13.2
- Driver: 595.97
- `torch.cuda.is_available()`: true
- Requested device: `cuda:0`

No model was loaded and no Python CUDA inference process was expected in
`nvidia-smi`, because the frozen policy files are missing. A real run requires
all policy parameters on CUDA and verifies the Python process after loading
each checkpoint.

## Integrity and prohibited access

- Phase 8 raw formal-test rows accessed: **NO**
- Formal test rerun: **NO**
- Test split used: **NO**
- New RL training: **NO**
- Checkpoint reconstruction: **NO**
- ERI evaluation coordinates executed: **0**
- `OPTIMIZATION_STABILITY_WARNING = YES` (carried forward from Phase 13)
- Test suite: `256 passed` (`py -3.12 -m pytest -q`)

## Next recommendation

Recover the exact original Phase 13 15k Treatment checkpoint files from the
machine or artifact store on which Phase 13 ran, then place them at the frozen
relative paths and run the committed preflight. Their embedded seed, episode,
lambda, auxiliary version, and model state must pass as-is. Do not recreate
them by training. If the originals cannot be recovered, Phase 14 cannot answer
the research question without new authorization for a newly preregistered
replication phase; it must remain `NOT_EVALUATED`, and no formal test should be
attempted.
