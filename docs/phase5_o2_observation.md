# Phase 5 - O2 Full Revealed-Order Observation

Status: implementation and diagnostic only. No training, hyperparameter search,
formal test evaluation, statistical comparison, PBFS, or PBFSA is included.

## 1. O1 information-loss audit

O1 has `S` stack nodes and one context node with 12 features. It represents
stack height/free space, top batch, current-target location/depth, current-batch
container fraction, the earliest remaining revealed rank in each stack, static
batch disorder, progress, and relocation/retrieval summaries.

O1 does not preserve the complete current-batch order. In the audited collision,
the physical layout is `[(1,5), (2,3,4), ()]`, target 1 and all other public
state except the order are fixed, and current-batch orders are `(1,2,3,4)` and
`(1,2,4,3)`. Containers 3 and 4 share a stack behind the unchanged earliest
member 2. O1 records the same count and minimum rank for that stack, producing
identical tensors. Under Galle revelation these states differ because the full
current-batch order is public. O2 distinguishes them.

## 2. Full DS1/DS2 batch-size audit

All 1,440 exact Ku/Galle source instances were parsed and their deterministic
DS2 variants derived with the frozen adjacent-pair merge rule.

| Universe | Maximum batch size |
|---|---:|
| DS1 | 4 |
| DS2 | 6 |
| Combined supported universe | **6** |

The formal O2 bound is therefore `Mmax=6`. The adapter rejects instances above
that audited universe instead of silently truncating their revealed order.

## 3. Node layout and fixed shape

```text
[S stack/action nodes] + [6 revealed-order slots] + [1 context node]
```

There are `(S + 7)` nodes and 12 features per node. The flattened observation
shape is `((S + 7) * 12,)`, dtype `float32`. The first `S` nodes remain the only
LOW pointer candidates. The last node remains context.

## 4. O2 feature table

Every feature is explicitly normalized to `[0,1]`, so the reviewed
`O2_FEATURE_SCALE` is twelve ones rather than an accidental reuse of O1 scale.
Feature meanings are node-type dependent and node type is always explicit.

| Index | Stack node | Revealed-order node | Context node | Range/scale |
|---:|---|---|---|---|
| 0 | type=0 | type=0.5 | type=1 | `[0,1] / 1` |
| 1 | stack index | full revealed rank | current batch rank | `[0,1] / 1` |
| 2 | height | container stack index | batch progress | `[0,1] / 1` |
| 3 | free space | container tier | batch remaining fraction | `[0,1] / 1` |
| 4 | top batch rank | blockers above container | target stack | `[0,1] / 1` |
| 5 | contains target/source | current batch rank | target depth | `[0,1] / 1` |
| 6 | top is target | is current target | relocation progress | `[0,1] / 1` |
| 7 | target tier | is currently top | total remaining | `[0,1] / 1` |
| 8 | target blockers | local stack height | batch size / Mmax | `[0,1] / 1` |
| 9 | current-batch fraction | local free space | remaining count / Mmax | `[0,1] / 1` |
| 10 | earliest rank summary | remaining-rank fraction | revealed flag | `[0,1] / 1` |
| 11 | padding=0 | padding marker | padding=0 | `{0,1} / 1` |

Raw container ID is never a feature. Each remaining order member is represented
losslessly by explicit full/remaining ranks and its unique public stack/tier
location. Full rank is explicit because the Transformer has no positional
encoding; tensor position is not the sole carrier of order.

## 5. Padding strategy

If fewer than six current-batch containers remain, unused order slots have
order-node type, all payload features zero, and `padding=1`. Real order nodes
have `padding=0`, so padding cannot be mistaken for a container.

The existing encoder has no node-padding mask. Padding embeddings can therefore
participate in attention and the global context, but they have a unique marker
and a fixed maximum count. Their count reveals only the already-public number
of remaining current-batch containers. Introducing a new mask API before any O2
training would expand the network architecture unnecessarily; a learned-mask or
explicit attention-mask ablation remains a pre-training design option.

## 6. Leakage guarantee

O2 reads only `SCRPState.revealed_orders[current_batch]`. It has no reference to
`Scenario`, the sampler, private scenario state, scenario seed/ID, or future
hidden mappings. Two scenarios with identical public history but different
future permutations produce bit-identical O2 tensors before reveal. Once that
batch is revealed, their O2 tensors may differ, as required by Galle's model.

## 7. Network integration and O1 compatibility

No `HierPolicyNetwork` code was modified. It already:

- accepts any number of 12-feature nodes;
- treats the last node as context; and
- takes action embeddings only from the first `len(action_mask)=S` nodes.

Thus O2 has input `(B,S+7,12)`, while LOW logits and masks remain `(B,S)`.
Revealed and padding nodes enrich encoder context but cannot become actions.

`SCRPRLAdapter` now accepts `observation_version="O1"|"O2"`; the default remains
O1. Existing O1 checkpoints, shapes, feature scales, and runner behavior are
unchanged. Future O2 metadata records problem type, observation version, stack/
tier sizes, feature dimension, `Mmax`, LOW decision mode, and dataset version.

## 8. Diagnostic and scope

The non-training diagnostic records the real DS1/DS2 shapes, maximum-batch
coverage, O1 collision, and O2 distinction in
`experiments/summaries/phase5_o2_diagnostic.json`. This is an information-
representation check only and contains no reward or performance comparison.

Before formal training, the remaining decisions are whether to add a true
attention padding mask, freeze the exact O2 policy/checkpoint construction API,
and run a separate training-sanity phase. Formal training and test access remain
unauthorized in Phase 5.
