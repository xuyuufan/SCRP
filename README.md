# SCRP

Research code for the Stochastic Container Relocation Problem (SCRP). The
current worktree contains the SCRP core environment, partial-information
scenario handling, the O1 observation/RL adapter, the Phase 2.5 LOW-only
policy-gradient sanity-training path, and the Phase 3 paper-instance data
loader/schema built on the original Ku/Galle benchmark files.

## Paper instances

The Phase 3 audit, Ku-to-Galle-to-Bacci dataset mapping, static JSON schema,
source findings, and small reproduction results are documented in
[`docs/phase3_dataset_reproduction.md`](docs/phase3_dataset_reproduction.md).
The public API in `scrp.datasets` parses original Ku CRPTW text files, derives
Galle/Bacci DS2 by merging adjacent batches, and saves scenario-free static
instances. A small converted set and separate scenario rollout report live in
[`data/phase3_sanity`](data/phase3_sanity).

## Formal experiment protocol

Phase 3.5 defines the static-instance split, DS1/DS2 linkage, disjoint scenario
seed streams, common-random-number evaluation records, and development/formal
rollout budgets. See
[`experiments/protocols/formal_protocol_v1.md`](experiments/protocols/formal_protocol_v1.md).
The frozen manifest is
[`experiments/splits/scrp_split_v1.json`](experiments/splits/scrp_split_v1.json).
No formal training or baseline implementation is included.

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
