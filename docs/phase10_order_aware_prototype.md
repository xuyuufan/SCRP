# Phase 10 Order-Aware Follow-up Prototype

## Scientific status

Phase 10 is a separate `DEVELOPMENT ONLY` follow-up. It uses only fixed train
and validation subsets. Phase 8 raw rows and the formal test split were not
accessed, and the formal test was not rerun. No formal claim or new untouched
test claim is made.

Only one primary variable changed: the policy architecture. The Phase 7B
optimizer, learning rate, gamma, entropy coefficient, batch size, sampler, and
FGB refresh rule remain unchanged. No ERI auxiliary objective was implemented.

## Architecture audit

The original `O2_SHARED_ENCODER_V1` flow is:

```text
stack nodes + order nodes + context node
                |
      shared Transformer encoder
                |
 global-only LOW decoder query/pooling
                |
 pointer scores over first S stack nodes
```

It has one encoder layer, four heads, embedding dimension 32, and FFN
dimension 64. O2 padding nodes are excluded by the node padding mask in the
shared encoder and LOW decoder. Order nodes can affect other nodes indirectly,
but no operation explicitly asks each stack representation to read the
revealed-order sequence.

The prototype `O2_ORDER_XATTN_V1` flow is:

```text
stack nodes + context -> stack/context encoder -> S stack queries
order nodes           -> separate order encoder -> real order keys/values
                                      |
             masked stack-to-order cross-attention
                                      |
                     order-aware stack representations
                                      |
            unchanged LOW pointer over first S stacks
```

The explicit O2 rank features remain in the order-node inputs; tensor position
alone is not used as the sequence signal. Raw container IDs are absent.
Padding order nodes cannot be keys or values in cross-attention.

The original class and default factory path remain unchanged, and an old-O2
state dict still loads strictly. The architecture versions are constructed
separately.

| Architecture | Parameters |
|---|---:|
| O2_SHARED_ENCODER_V1 | 21,312 |
| O2_ORDER_XATTN_V1 | 34,368 |
| Increase | 13,056 (+61.26%) |

## Frozen prototype gates

Before training, success thresholds were fixed as follows:

- treatment permutation action-change rate at least 1%;
- treatment permutation sensitivity at least 2x control;
- treatment order-ablation action-change rate at least 2%;
- treatment order-ablation sensitivity at least 2x control;
- strictly-worse ERI-score rate no more than one percentage point above control;
- validation mean no more than 0.25 relocations above control;
- padding perturbation effect exactly zero.

The prototype is successful only if all conditions pass. Exceeding ERI is not
required at this stage.

## Untrained controlled probe

- Revealed-order probability effect: 0.1212603.
- Revealed-order stack-embedding RMS effect: 0.1762124.
- Hidden future-order probability/embedding effect: exactly 0.
- Padding perturbation effect: exactly 0.

The architecture can express revealed-order differences without leaking hidden
future orders or using padding. This proves capacity, not learned use.

## 1,000-episode smoke

Both models used the same seed and identical sampler schedule. Both remained
finite with zero invalid actions, truncations, or scenario mismatches.

| Metric | Control | Treatment |
|---|---:|---:|
| Validation mean | 10.8750 | 11.5104 |
| DS1 validation | 10.7917 | 11.3125 |
| DS2 validation | 10.9583 | 11.7083 |
| Training entropy mean | 1.0462 | 1.3178 |
| Pre-clip gradient norm mean | 1.6109 | 1.4285 |
| FGB refreshes | 4 | 23 |

The stability gate passed, so the predeclared 5,000-episode comparison was
allowed to continue. Performance was already worse for the treatment.

## 5,000-episode development comparison

Each model trained for exactly 5,000 episodes on 48 fixed train base layouts
and used the same sample schedule, seed, optimizer/hyperparameters, and FGB
rule. Validation used 48 fixed validation bases, DS1 and DS2, one scenario per
static variant (96 paired episodes per model).

| Metric | Control | Treatment |
|---|---:|---:|
| Validation mean | 10.3542 | 10.7396 |
| DS1 validation | 10.4792 | 10.8958 |
| DS2 validation | 10.2292 | 10.5833 |
| Training entropy mean | 0.9962 | 1.2785 |
| Pre-clip gradient norm mean | 1.4847 | 1.4008 |
| FGB refreshes | 12 | 41 |

Treatment minus control is +0.3854 relocations. The exploratory paired
development bootstrap 95% interval is `[0.21875, 0.56250]` (10,000
repetitions, seed 20260816). Positive values are worse for the treatment. This
interval is development evidence, not paper-level inference.

## ERI-score and learned order use

All action diagnostics use ERI-guided public validation states.

| Metric | Control | Treatment |
|---|---:|---:|
| Public decision states | 886 | 886 |
| Exact ERI action agreement | 44.921% | 44.582% |
| ERI-score-equivalent actions | 92.889% | 91.084% |
| Strictly worse ERI-score actions | 7.111% | 8.916% |
| Mean ERI-score gap | 0.06038 | 0.07336 |
| Permutation action sensitivity | 0.000% | 1.290% |
| Permutation mean TV | 0.001763 | 0.002666 |
| Order-ablation action sensitivity | 2.903% | 1.935% |
| Order-ablation mean TV | 0.069397 | 0.052388 |
| Padding perturbation effect | 0 | 0 |

The explicit cross-attention increased permutation action sensitivity above
the absolute and relative thresholds. It did not increase order-ablation
sensitivity, and it made ERI-score errors and validation relocations worse.

## FGB diagnostics

The FGB rule itself is unchanged and every refresh test still uses four paired
episodes. Nevertheless, the treatment refreshed 41 times versus 12 for the
control under an identical sampler schedule. The new representation therefore
interacts strongly with the already noisy n=4 refresh mechanism. This is an
observed diagnostic, not authorization to change FGB within Phase 10.

## Decision

Frozen gate results:

- permutation absolute/relative: PASS;
- order-ablation absolute/relative: FAIL;
- ERI error not increased: FAIL;
- validation not worse: FAIL;
- padding invariant: PASS.

`ORDER_AWARE_PROTOTYPE_SUCCESS = NO`

The prototype is architecturally correct and leakage-safe but does not provide
acceptable development evidence. Per the predeclared protocol, work stops
here: no ERI auxiliary loss, no FGB change, no extra budget, and no model is
retained. It is **not** currently worth entering an auxiliary-objective phase
on top of this architecture. A future investigation would first need a new,
separately justified structural hypothesis rather than stacking another loss
onto a failed treatment.

Cleanup: no development checkpoint was written; zero models are retained.
