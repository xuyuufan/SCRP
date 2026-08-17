# Phase 11 ERI Set-valued Auxiliary Objective

## Status and safeguards

Phase 11 is a development-only train/validation experiment starting from main
SHA `6e531a419857fa1474af3c4c01403e9065bc93c8`. Phase 10 remains separate as
open PR #11; none of its treatment code is used. Phase 8 raw formal-test rows
must not be accessed and the formal test must not be rerun.

The control and treatment retain the successful O2 shared policy, observation,
optimizer, learning rate, entropy term, frozen-greedy baseline (FGB), sampler,
scenario schedule, and 5,000-episode budget. The sole treatment difference is
`eri_aux_coefficient=0.10`; control uses zero. Both use one fixed train and one
fixed validation base per parameter group (48 each), both DS1 and DS2 variants,
and one validation scenario per static variant. The fixed endpoint at 5,000
episodes avoids adaptive checkpoint selection in this bounded comparison.

## Objective

At each live training decision, the existing exact public-state ERI scorer is
applied to every legal destination. If `A*` contains every action tied at the
minimum ERI score, the added loss is

`L_ERI_SET = -log(sum(a in A*) pi(a | s))`.

It is evaluated by `-logsumexp` over masked action log-probabilities. Illegal
actions cannot enter `A*` or receive probability. The total objective is

`L_TOTAL = [-mean(log pi(a_t|s_t) A_t) - 0.01 mean(H)] + lambda_eri L_ERI_SET`.

The RL return objective remains primary; policy logits are not detached.
Labels use only the live layout, batch precedence, and already revealed order.
They do not use scenario IDs/seeds, future permutations, private state, or test
artifacts.

## Frozen smoke and scale gate

Before the remainder of training, both arms run 16 paired episodes. Training
continues only if losses and gradients are finite, auxiliary gradients are
nonzero, parameters update, schedules pair exactly, invalid actions,
truncations, and mismatches are zero, checkpoint round-trip succeeds, and the
mean weighted-auxiliary/RL gradient-norm ratio is in `[0.02, 2.00]`.
Coefficient 0.10 is retained if this gate passes. If the scale gate fails, the
main comparison stops; at most one fixed scale-based revision may be justified
before a fresh run. Smoke checkpoints are deleted after verification.

## Frozen success gate

`ERI_AUX_PROTOTYPE_SUCCESS = YES` only if every condition holds:

1. Treatment overall validation relocations are lower.
2. The effect is not isolated: neither DS1 nor DS2 is worse by more than 0.25
   relocations and at least one improves.
3. The strictly-worse ERI action rate falls by at least one absolute percentage
   point.
4. DS1 and DS2 have no severe opposing behavior (the same 0.25 threshold).
5. There are no integrity, leakage, invalid-action, truncation, scenario, or
   numerical failures.
6. ERI-score equivalence does not fall while only exact deterministic tie-break
   imitation rises; relocation improvement remains mandatory.

The paired 95% interval uses 10,000 base-layout bootstrap resamples while
preserving the paired DS1/DS2 block. These gates are fixed before the main
result is observed and will not be changed afterward.

## Results

The 16-episode smoke passed. The treatment's mean weighted auxiliary/RL
gradient-norm ratio was 0.2404, inside the frozen range, so coefficient 0.10
was retained with no adjustment. Checkpoint round-trip passed and its temporary
checkpoint was deleted.

The fixed 5,000-episode result is:

| Metric | Control | Treatment | Treatment - control |
|---|---:|---:|---:|
| Validation relocations | 10.5000 | 10.3958 | -0.1042 |
| DS1 relocations | 10.6250 | 10.2500 | -0.3750 |
| DS2 relocations | 10.3750 | 10.5417 | +0.1667 |
| ERI-score-equivalent rate | 89.165% | 91.761% | +2.596 pp |
| Strictly-worse ERI rate | 10.835% | 8.239% | -2.596 pp |
| Mean ERI-score penalty | 0.08691 | 0.07280 | -0.01411 |
| Exact deterministic ERI agreement | 35.440% | 47.065% | +11.625 pp |
| Mean probability mass on `A*` | 0.85454 | 0.85506 | +0.00052 |

The paired base-layout bootstrap treatment-minus-control interval is
`[-0.4479, 0.2604]`; it includes zero, so the favorable point estimate is
uncertain. DS1 improved while DS2 worsened mildly, within the predeclared 0.25
severe-opposition threshold. The diagnostic covered 886 identical ERI-guided
public states. Multiple ERI-optimal destinations occurred in 67.494% of them.

Control/treatment mean policy losses were -0.02290/-0.02898, auxiliary losses
0.49702/0.33341, entropies 1.13863/0.80768, and pre-clip gradient norms
1.70494/1.82219. Clipping frequencies were 98.16%/98.96%; FGB refreshed 5/12
times. The treatment's full-run weighted auxiliary/RL gradient ratio was
0.15155. Invalid actions, truncations, numerical failures, and scenario
mismatches were all zero.

The optional downstream counterfactual for greedy actions outside `A*` was not
estimated: branching from ERI-controlled states would condition on realized
hidden future orders and introduce a high-variance post-hoc endpoint. This did
not affect the frozen gate.

All predeclared checks passed, therefore
`ERI_AUX_PROTOTYPE_SUCCESS = YES`. This is development evidence only. The
interval includes zero and DS2 moved adversely, so the evidence supports only
a separately authorized, better-powered development replication—not a formal
superiority claim and not an automatic larger run.
