# Changelog

This file records research milestones rather than every file-level edit.

## Phase 2.5 — 2026-08-15

### Added

- Added the SCRP domain model, validated environment transitions, deterministic
  scenario sampling, staged batch revelation, and reproducible reset seeding.
- Added the O1 partial-information observation representation and RL adapter
  with legal-action masks, serializable metrics, and state snapshots.
- Added an independent LOW-only SCRP episode runner and tiny sanity-training
  workflow using the existing hierarchical policy network.
- Added policy/baseline scenario pairing, training metrics, finite-gradient
  checks, and a reproducible Phase 2.5 sanity checkpoint.

### Changed

- Reused the existing LOW decoder and REINFORCE update path for SCRP while
  keeping the existing network architecture intact.
- Used all-ones feature scaling for the 12-feature O1 SCRP observation.

### Fixed

- Enforced legal relocation destinations and transactional rejection of invalid
  actions.
- Kept future batch orders hidden from public observations, results, metrics,
  and snapshots until their batch is revealed.

### Tests

- `64 passed` with Python 3.12.3 and pytest 8.3.4 on 2026-08-15.
- Coverage includes entity invariants, deterministic seeding and revelation,
  transitions, observations, network integration, the RL adapter, the LOW-only
  runner, training updates, and checkpoint restoration.

### Notes

- Phase 2.5 is a sanity-scale integration checkpoint, not a full benchmark or
  paper-results training run.
- The frozen greedy baseline and policy evaluate identical scenarios derived
  from each episode seed.
- No HIGH pseudo-actions, HIGH loss samples, duplicate-dataset logic, or mutable
  environment seeds were introduced for SCRP.
- The tracked `scrp_phase_2_5_sanity.pt` file is intentionally retained as a
  small reproducibility artifact; other generated model checkpoints are ignored.
- The next phase should add full experiment configurations, larger datasets,
  benchmark evaluation, and repeated-seed result reporting on experiment or
  phase branches before promotion to `main`.
