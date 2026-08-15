# Version History

| Phase | Git Branch | Git Tag | Main Changes | Tests | Status |
|---|---|---|---|---|---|
| Phase 4.5 | `phase/scrp-phase-4.5-eri` | — | Clean-room published ERI reproduction, golden fidelity cases, four-algorithm CRN development check | 127 passed (16 Phase 4.5 cases) | Fidelity-audited; PR pending |
| Phase 4 | `phase/scrp-phase-4-baselines` | — | Leakage-safe Tier 0 baselines, unified rollout/evaluation, paired development verification | 110 passed (16 Phase 4 cases) | Validated development baseline layer; PR pending |
| Phase 3.5 | `phase/scrp-phase-3.5` | — | Frozen static split, DS1/DS2 linkage, scenario schedule, and formal evaluation protocol | 94 passed | Validated protocol design; untagged |
| Phase 2.5 | `phase/scrp-phase-2.5` | `phase-2.5` | SCRP core, O1 adapter, and LOW-only runner/training integration | 64 passed | Stable sanity checkpoint |

Tags are the canonical immutable recovery points. Use a phase branch for
continued work within that stage and an experiment branch for changes that have
not yet been validated as a stable research baseline.
