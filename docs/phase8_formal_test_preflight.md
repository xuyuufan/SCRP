# Phase 8 Formal-Test Preflight

Phase 8 is frozen as evaluation infrastructure only. This preflight does not
run the formal test and does not access, enumerate, sample, or inspect the
formal test split.

The future runner is fixed to the Phase 7B episode-15000 O2 checkpoint and its
verified SHA-256, the frozen Phase 7B configuration hash, the existing split
and dataset protocol versions, root seed 20260816, 50 scenarios per static
variant, and the O2-policy-versus-ERI comparison. It performs inference only;
it contains no training or optimizer update.

Before any future split manifest access, the runner atomically writes a run
identity containing the Git commit and branch, tracked and complete worktree
status, run and checkpoint identities, frozen protocol values, UTC timestamp,
and Python/library runtime versions. It refuses to overwrite an existing run
identity or any existing partial, final, failure, or summary artifact.

During a future authorized run, paired raw results are written incrementally
to a gitignored `.partial` file. Failures are preserved separately. Final raw
and compact summary artifacts are created only after all frozen schedule,
pairing, coverage, and statistical checks succeed; the completion marker is
written last. No resume or test-result-dependent adaptive behavior is
implemented.

The formal-test command is intentionally not recorded as executed here. It
must be run only under a separate explicit authorization after this frozen
commit has been reviewed.
