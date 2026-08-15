# Phase 4.5 - Published ERI Baseline Reproduction

Status: reproduced and fidelity-tested. This is a clean-room, behavior-level
reproduction of the Expected Reshuffling Index (ERI), not a translation of the
public MATLAB source.

## 1. Sources and provenance

| Source | Role | Audit result |
|---|---|---|
| Ku and Arthanari (2016), *Container relocation problem with time windows for container departure*, Section 4 | Original ERI definition and derivation | ERI is the expected number of destination-stack containers departing before the relocated container. Same-window uncertainty contributes `n/2` for `n` such containers. |
| Galle et al., *The Stochastic Container Relocation Problem*, Section 2.1.1 | Batch-model restatement and deterministic tie-break | Gives the per-container form `1{ci<c} + 1{ci=c}/2`, then highest-stack and leftmost tie-breaks. |
| `https://github.com/vgalle/StochasticCRP` | Reference behavior inspection | `retrieveERI.m` was inspected at commit `ec672df26dae12de42ba3c4e95a4a9002e4410f6`. Its comparisons and tie behavior agree with Galle's paper. |

No `LICENSE` or `COPYING` file was present at the audited repository commit.
The repository was therefore used only to confirm externally observable
behavior. No MATLAB control structure or substantial expression was copied or
translated. The Python implementation is derived independently from the
mathematical specification below.

## 2. Specification audit

| Question | Finding | Evidence status |
|---|---|---|
| Full name and definition | Expected Reshuffling Index; expected count of containers already in a candidate destination that depart before the relocated blocker | [paper explicit] |
| Inputs | Static public batch membership/precedence, current public state, and legal destinations | [paper explicit] [inferred for project API] |
| Current blocker | Top container above the current target in its source stack; blockers are relocated one at a time | [paper explicit] [source-code confirmed] |
| Candidate score | Sum one for each definitely earlier container, one-half for each same-label unresolved container, and zero for each definitely later container | [paper explicit] [source-code confirmed] |
| Batch/time-window precedence | An earlier batch/time window is definitely earlier; a later batch/time window is definitely later | [paper explicit] |
| Current revealed order | Use the realized order to distinguish remaining members of the revealed current batch | [source-code confirmed] [inferred mapping to `SCRPState`] |
| Future batch internal order | Unknown. Two containers in the same unrevealed batch contribute one-half in expectation | [paper explicit] [source-code confirmed] |
| Primary choice | Minimum ERI score over non-full destinations other than the source | [paper explicit] [source-code confirmed] |
| Tie-break | Prefer the tallest minimizing stack, then the leftmost stack | [paper explicit in Galle] [source-code confirmed] |
| Determinism | Deterministic after Galle's leftmost final tie-break | [paper explicit in Galle] [source-code confirmed] |
| Variants | Ku presents ERI for the online CRPTW model; Galle applies the same local index rule in batch/online experiments | [paper explicit] |
| Ku versus Galle | Score definition agrees. Ku says final ties are broken arbitrarily; Galle fixes leftmost | [paper explicit] |
| Paper versus MATLAB | Score, taller-stack tie-break, and retained-leftmost behavior agree | [source-code confirmed] |

The only material textual ambiguity is Ku's arbitrary final tie. Version 1
follows Galle's explicit leftmost rule and records that choice in the algorithm
name and tests. No contradictory score rule was found.

## 3. Mathematical decision rule

Let `c` be the current topmost blocker and let `D` be the legal destination
set. For every live container `x` already in destination stack `d`, define the
public-information contribution

```text
q(x,c) = 1    if public precedence proves x departs before c
         0    if public precedence proves x departs after c
         1/2  if x and c are in the same unrevealed batch
```

If their shared batch has been revealed, its realized order replaces the
one-half case: `q(x,c)=1` exactly when `x` precedes `c` in that order, otherwise
zero. Then

```text
ERI(d,c) = sum over x in d of q(x,c)
```

and the selected destination is the lexicographic minimum of

```text
(ERI(d,c), -height(d), stack_id(d)).
```

An empty destination has score zero. Capacity affects candidacy through the
environment-provided legal set, not the ERI value.

## 4. Public-information boundary

`ERIBaseline` will implement the existing `SCRPBaseline` contract and receive
only `SCRPInstance`, a detached public `SCRPState`, and the legal destination
tuple. It will not receive or inspect a `Scenario`, scenario seed, scenario ID,
sampler, environment object, private hidden-order mapping, or scenario-bearing
metadata. Static batch membership and precedence are public. The complete
order of the current revealed batch is public. Future within-batch permutations
remain inaccessible.

## 5. Fidelity method

Octave/MATLAB is not available in the project environment. Fidelity validation
therefore uses hand-constructed golden cases derived from the paper equation
and independently checked against the audited reference behavior. Cases cover
strict precedence, revealed same-batch order, unrevealed one-half contributions,
empty/full-boundary destinations, both tie levels, current/future blockers, and
hidden-future invariance. This limitation prevents an executable cross-language
comparison but not a behavior-complete test of the local decision rule.

Golden-case evidence:

| Case | Expected published behavior | Python result |
|---|---|---|
| Strict earlier/later precedence | Scores `1` and `0`; choose the zero-score stack | Match |
| Two earlier containers | Additive score `2` | Match |
| Same unrevealed future batch | Contribution `1/2` | Match |
| Mixed earlier and unresolved | Score `1 + 1/2 = 3/2` | Match |
| Empty destination | Score `0` | Match |
| Equal score, unequal height | Choose tallest candidate | Match |
| Equal score and height | Choose leftmost/smallest stack ID | Match |
| Current-batch blocker | Use revealed realized order | Match |
| Later-batch blocker | Use public batch precedence | Match |
| Hidden future order changed | Identical public state gives identical action | Match |

All hand-crafted fidelity cases pass. The unified rollout additionally confirms
normal termination, zero invalid actions, reward equal to negative relocations,
and one decision per relocation on both DS1 and DS2 artifacts.

## 6. Development integration result

The existing Phase 4 subset was reused without expansion: 12 base layouts,
both DS1 and DS2 static variants, and three scenarios per variant. Random,
MinBlockingGreedy, ERI, and the current O1 LOW policy each ran 72 episodes, for
288 episodes total. All four algorithms received identical `scenario_id` values
within each `(dataset, instance_id, scenario_seed)` coordinate. DS1 and DS2
were not cross-paired.

| Algorithm | Episodes | Mean | Sample std | Min | Max |
|---|---:|---:|---:|---:|---:|
| `random_legal_v1` | 72 | 4.4583 | 2.6374 | 0 | 10 |
| `min_blocking_greedy_v1` | 72 | 3.1944 | 1.7169 | 0 | 6 |
| `eri_reproduction_v1` | 72 | 3.1944 | 1.7169 | 0 | 6 |
| `current_o1_low` | 72 | 4.3056 | 2.5211 | 0 | 10 |

Invalid actions and truncated episodes were both zero. ERI and the Phase 4
MinBlockingGreedy rule are mathematically equivalent under the present public
state model, so their matching development traces are expected. The ERI name
adds audited published provenance; it does not create a stronger rule.

**NOT A PERFORMANCE RESULT.** This small training-split development subset is
only a pipeline/fidelity check. It supports no statistical-significance,
paper-quality, or algorithm-superiority claim.

## 7. Known deviations and remaining ambiguity

- Ku leaves the final tie arbitrary; version 1 follows Galle's deterministic
  leftmost rule.
- Stack IDs are zero-based in this project and one-based in MATLAB; “leftmost”
  is preserved behaviorally.
- No executable MATLAB/Octave cross-language run was possible locally. Golden
  cases were independently derived from the equation and reference behavior.
- The audited reference repository has no explicit license. The implementation
  is clean-room and no reference code is redistributed.

## 8. Recommendation

ERI is sufficient as the thesis's first fidelity-audited published heuristic
baseline. PBFS/PBFSA would add exact/approximate search comparisons, but also
substantially increase implementation-fidelity burden, sampling/stopping-rule
surface area, and compute cost. Because ERI is now reproducible and the project
still lacks O2 and formal RL training, the recommended next step is to prioritize
O2/RL work rather than implement PBFS/PBFSA immediately. PBFS/PBFSA should be
reconsidered only if the thesis explicitly requires an exact-search reference
on small instances or reviewers require a stronger non-RL comparator.

Development evaluation remains explicitly **NOT A PERFORMANCE RESULT**. No
formal test split, formal RL training, O2 implementation, PBFS, or PBFSA is
authorized in this phase.
