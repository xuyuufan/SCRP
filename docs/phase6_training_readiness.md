# Phase 6 - Training Readiness and Formal RL Audit

Status: bounded implementation audit and train-split sanity only. Formal long
training, validation selection, formal test, tuning, PBFS/PBFSA, and performance
claims remain outside this phase.

## 1. Frozen Greedy Baseline audit

The authoritative implementation inspected is `hier_pg/algorithm.py`, with the
policy and decoder behavior in `hier_pg/network.py`. The original code computes
discounted policy suffix returns, divides the paired greedy baseline episode
return by the number of policy decisions, subtracts that scalar at every policy
decision, and normalizes advantages across the iteration before REINFORCE.

| Semantic field | Original `hier_pg` | Phase 6 formal SCRP | Classification |
|---|---|---|---|
| Policy return | negative episode shifters | sum of `-1` relocation rewards | SEMANTICALLY EQUIVALENT |
| Baseline return | greedy negative episode shifters | paired greedy negative relocations | SAME |
| Credit granularity | per model decision | per LOW decision | SAME for LOW-only SCRP |
| Discounting | suffix return `G_t`, configurable gamma, default 1 | same | SAME |
| Advantage | `G_t - G_b / number_of_policy_decisions` | same | SAME |
| Normalization | iteration-level, separately for HIGH/LOW | iteration-level LOW only | SEMANTICALLY EQUIVALENT |
| Entropy | subtract coefficient times mean entropy | same | SAME |
| Initial baseline | deepcopy of initial policy, eval mode | same, plus explicit no-grad parameters | SEMANTICALLY EQUIVALENT |
| Baseline refresh | copy updated policy after paired one-sided t-test at p<0.05 | same | SAME |
| Scenario pairing | same environment seed | same static variant and scenario seed; scenario IDs asserted equal | SAME |
| Greedy semantics | pointer argmax under legal mask | LOW pointer argmax under legal stack mask | SAME |
| Batch aggregation | mean policy-gradient terms over iteration decisions | same | SAME |

The frozen formal formula is therefore

```text
G_t = sum_{k=t}^{T-1} gamma^(k-t) r_k
b = G_baseline / T_policy
A_t = G_t - b
```

followed by iteration-level standardization when the standard deviation is
nonzero. This is not the episode-level candidate `G_policy-G_baseline`.

Phase 2.5 already used the same advantage expression, so its formula was not
wrong. It was a sanity-only placeholder because its baseline remained fixed for
the entire run, its seeds were not driven by the formal manifest visit schedule,
it supported O1/fixed-shape batches only, and its checkpoint could not reproduce
the next base/variant/scenario sequence. Its historical behavior is unchanged.

## 2. Frozen baseline semantics

At trainer construction, the baseline is a deepcopy of the policy, put in eval
mode, and excluded from gradients. Policy and baseline receive the same static
DS1 or DS2 instance and numeric scenario seed. Actions, trajectory lengths, and
relocations may differ, but scenario IDs must match. After each iteration, the
paired policy and baseline episode returns are tested as in `hier_pg`; only a
significant one-sided improvement refreshes the baseline from the updated
policy. The refreshed copy is again eval-mode and no-grad.

## 3. O1/O2 policy factory and O2 padding

`make_scrp_policy` derives all observation metadata:

| Version | Nodes | Features | Candidates | Padding mask |
|---|---:|---:|---:|---|
| O1 | `S+1` | 12 | `S` | `None` |
| O2 | `S+6+1` | 12 | `S` | order-padding nodes only |

The network now accepts optional `node_padding_mask`. Its default is `None`, so
O1 and old call sites are backward compatible. For O2, masked nodes are excluded
from encoder key/value attention, decoder cross-attention, and the decoder's
global mean. Stack nodes, real order nodes, and context are never masked. The
LOW legal-action mask remains a separate `(B,S)` tensor.

A controlled same-weights test changes only invalid padding payloads. With the
old marker-only path, stack log probabilities change; with the attention mask,
they are exactly unchanged. The mask is therefore frozen for formal training.

## 4. Variable-S batching and mixed dataset sampling

Formal instances have `S=5..10`. Each batch first samples a train-split base;
that first base determines the S bucket, and remaining bases are sampled within
the same bucket. Thus action width is fixed within a batch and no fake stacks or
fake stack actions are introduced. `T` changes normalization values but not the
tensor shape, which depends only on S and the observation version.

For every episode the sampler order is:

1. sample a physical base instance;
2. sample DS1 or DS2 for that base;
3. allocate the next train-stream scenario seed from base ID and per-base visit.

It never samples uniformly from a flattened variant collection. Visit counters,
sampler RNG state, and the root seed are checkpointed. The existing disjoint
train/validation/test `ScenarioSeedSchedule` is preserved; Phase 6 sanity reads
only train assignments.

## 5. Checkpoint and deterministic resume

The checkpoint includes model and optimizer states, iteration, episodes seen,
root seed, torch RNG state, observation version, feature dimension, Mmax, S
buckets, dataset/split/training protocol versions, baseline type/state/update
count, per-base visit counters, full sampler state, and the config snapshot.

The deterministic CPU regression compares an uninterrupted two-iteration run
with a one-iteration save/load/resume run. Sampled base IDs, variants, scenario
seeds, second-iteration loss, and final model parameters are bit-identical.

## 6. Bounded real-data sanity

The committed summary covers 12 train-split base layouts across S=5,7,10. O2
runs 100 episodes and O1 runs four additional compatibility episodes. Both DS1
and DS2 occur. All recorded losses, entropy, and gradient norms are finite;
policy parameters change; baseline parameters receive no gradients; policy and
baseline scenario IDs are asserted equal; illegal actions and truncations are
zero. No validation or test instance is read and no relative performance claim
is made. Hyperparameters are explicitly `NOT FINAL HYPERPARAMETERS`.

## 7. Readiness gate

All twelve requested gates pass: audited advantage and frozen-baseline
semantics; O1/O2 factories; frozen padding decision; variable-S buckets;
base-balanced DS1/DS2 sampling; disjoint seeds; deterministic resume; bounded
real-data sanity; full regression; and no performance claim.

`READY_FOR_FORMAL_TRAINING = YES`

This means the implementation is safe to begin a separately authorized formal
training run. It does not authorize that run or freeze final hyperparameters.
