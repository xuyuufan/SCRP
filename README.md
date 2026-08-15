# SCRP

Research code for the Stochastic Container Relocation Problem (SCRP). The
current checkpoint contains the SCRP core environment, partial-information
scenario handling, the O1 observation/RL adapter, and the Phase 2.5 LOW-only
policy-gradient sanity-training path built on the existing hierarchical policy
network.

## Validation

Run the test suite from the repository root:

```bash
python -m pytest -ra
```

On Windows, if `python` resolves to the Microsoft Store alias, use:

```bash
py -3.12 -m pytest -ra
```

## Branch workflow

- `main` contains tested checkpoints suitable as a stable research baseline.
- `phase/<phase-name>` records a major research-development stage.
- `experiment/<experiment-name>` isolates unvalidated experimental changes.
- `feature/<feature-name>` is used for focused implementation work.

Complete work is validated on its branch before it is merged into `main`.
Important paper checkpoints receive annotated tags and an entry in
[`docs/version_history.md`](docs/version_history.md).

## Reproducing historical versions

List all checkpoints:

```bash
git tag
```

Inspect a checkpoint:

```bash
git show phase-2.5
```

Temporarily inspect its files without moving a branch:

```bash
git switch --detach phase-2.5
```

Return to the latest stable version:

```bash
git switch main
```

Start a new experiment from the checkpoint:

```bash
git switch -c experiment/new-test phase-2.5
```
