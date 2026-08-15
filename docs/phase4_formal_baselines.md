# Phase 4 — Formal Baseline Layer

Status: implementation-complete development baseline layer. This phase does
not contain a formal test-set result, a training run, O2, or a claim of
published-algorithm reproduction.

## 1. Published-baseline audit

The audit used the three papers supplied with the project and the public
Galle repository at commit `ec672df26dae12de42ba3c4e95a4a9002e4410f6`.

| Work | Methods relevant to SCRP | Public implementation audit | Phase 4 decision |
|---|---|---|---|
| Ku and Arthanari | Random destination selection; Expected Reshuffling Index (ERI); search/abstraction methods | The paper defines Random and ERI. No independently verifiable, clearly licensed implementation of those methods was located in this audit. | Random is reimplemented from its simple mathematical definition. ERI/search are deferred. |
| Galle et al. | Random, Leveling (L), ERI, Expected MinMax (EM), Expected Group (EG), PBFS and PBFSA | `vgalle/StochasticCRP` contains MATLAB `retrieveRand.m`, `retrieveL.m`, `retrieveERI.m`, `retrieveEM.m`, `retrieveEG.m`, PBFS/PBFSA files, instances, and experiment scripts. No `LICENSE` or `COPYING` file is present at the audited commit, so code is inspected but not copied. | Only an independent Random baseline is in Tier 0. L/ERI/EM/EG and PBFS/PBFSA require Phase 4.5 fidelity and licensing decisions. |
| Bacci, Mattia and Ventura | Realization-Independent Reallocation Heuristic (RIRH) and RIRH-MinMax | The paper gives algorithmic pseudocode and reports a C implementation. The audit did not locate a public SCRP RIRH implementation with a clear license; the author's public page exposes other BRP code, not an identified RIRH package. | RIRH variants are deferred; their realization-independent construction is not equivalent to this phase's one-step greedy API. |

Availability conclusions are bounded to the sources checked on 2026-08-16;
“not located” is not a claim that no implementation exists anywhere.

## 2. Baseline contract and leakage boundary

`SCRPBaseline.select_destination(instance, state, legal_destinations)` receives:

- immutable public static instance data (layout, container IDs, batch
  membership and batch precedence);
- a detached `SCRPState` copy containing only orders revealed so far; and
- the core-computed legal destination tuple.

It does not receive `Scenario`, a scenario sampler, scenario ID, scenario seed,
the environment object, private state, or future within-batch permutations.
The rollout obtains a new detached state before every action and validates the
returned action against the legal tuple before calling the core transition.
Mutating the state supplied to a baseline therefore cannot mutate the core.

The static `instance` argument is intentional: batch membership and precedence
are public information in the formal protocol and are required by deterministic
rules. It contains no sampled permutation or scenario identifier.

## 3. Tier 0 definitions

### RandomLegalBaseline

At each decision, select uniformly from the legal destination stacks. Its
`random.Random` generator is initialized by an explicit `action_seed`. This
seed is separate from the scenario seed and has no path to scenario sampling.
The evaluation adapter derives it by SHA-256 from a versioned domain string,
an action-seed root, algorithm name, static instance ID, and scenario seed.

### MinBlockingGreedyBaseline

This is a transparent project baseline, not a reproduction of ERI, EM, EG,
MinMax, or RIRH.

For blocker `c`, destination `s`, and every container `x` already in `s`, define

`p(x < c) = 1` if public precedence proves `x` is retrieved before `c`, `0` if
it proves the opposite, and `1/2` if `x` and `c` share an unrevealed batch.
When that batch is the current revealed batch, the exact revealed ranks are
used. The primary score is

`B(s,c) = sum_{x in s} p(x < c)`.

The selected destination lexicographically minimizes:

1. `B(s,c)`;
2. free slots after the move, preserving emptier stacks as flexible capacity;
3. stable zero-based stack ID.

The rule is deterministic. A change only to a hidden future permutation cannot
change its action; a change to the currently revealed order can.

## 4. Rollout and evaluation

`run_baseline_episode` resets scenario and action randomness separately, runs
to normal termination, records actions/relocations/reward, and enforces:

- every selected destination is legal;
- total reward equals negative relocations;
- baseline decision count equals relocation count; and
- normal completion is `terminated=True`, `truncated=False`.

`evaluate_algorithm_on_schedule` is shared by Tier 0 baselines and the existing
O1 LOW policy through thin adapters. It emits the Phase 3.5 `ScenarioResult`
schema for every scheduled scenario and retains dataset, split, base-instance,
parameter-group, seed, scenario ID and algorithm provenance. The paired check
compares scenario IDs by `(dataset, instance_id, scenario_seed)`.

Raw JSONL output defaults to `experiments/raw_results/`, which remains ignored.
Aggregation reports count, sample mean, sample standard deviation, minimum and
maximum relocations without discarding scenario-level records.

## 5. Development integration run

The checked development run uses 12 base layouts from four parameter groups,
both DS1 and DS2 artifacts (24 static artifacts), three scenarios per artifact,
and all three algorithms: Random, MinBlockingGreedy, and the current checkpointed
O1 LOW policy. This gives 72 scenario results per algorithm.

All 216 episodes terminated, none truncated, no invalid action was emitted,
and all paired scenario IDs matched. Aggregate values are recorded in
`experiments/summaries/phase4_development_summary.json` solely as an integration
artifact. The subset is too small and was selected for development, so its
numbers are not a formal performance result and must not be used for a paper
claim.

## 6. Phase 4.5 recommendation

Before adding a stronger published baseline:

1. resolve source-code licensing and cite an immutable revision;
2. write a paper-to-code fidelity specification for L/ERI/EM/EG and separately
   for PBFS/PBFSA or RIRH;
3. define tie-breaking, sampling budgets, stopping criteria, batch versus online
   semantics, and numerical equivalence tests; and
4. validate against published small examples or original outputs before running
   any formal test split.

EM is the most practical next heuristic candidate because Galle supplies
inspectable code and the paper describes it directly. It still must not be
labelled reproduced until licensing and fidelity checks pass.
