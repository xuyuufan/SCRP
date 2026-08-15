# Phase 3 paper-instance reproduction audit

Audit date: 2026-08-15. This phase implements data ingestion and a small
end-to-end check only. It does not implement a baseline, train a policy, or
claim reproduction of any paper result.

Evidence labels used below:

- **[论文明确]** stated directly in the cited paper or its official repository.
- **[论文可推导]** follows mechanically from an explicit definition.
- **[当前不确定]** not specified precisely enough to regenerate independently.
- **[需要原数据确认]** paper prose is insufficient; exact files are authoritative.

## 1. Reproduction matrix

| Field | Ku & Arthanari (2016), CRPTW | Galle et al. (2017/2018) | Bacci et al. (2022) |
|---|---|---|---|
| Dataset / experiment | **[论文明确]** Randomly generated CRPTW test scenarios (Table 2); exact and heuristic experiments. | **[论文明确]** “existing dataset” = Ku data (Exp. 1 batch, Exp. 3 online); “modified dataset” (Exp. 2); one-batch variant (Exp. 4); separate 100-instance comparison set in Appendix A.2. | **[论文明确]** DS1 (small batches) and DS2 (large batches), Section 4.1. |
| Stacks S | **[论文明确]** 5..10. | **[论文明确]** Existing/modified: 5..10; Appendix A.2 only: S=4. | **[论文明确]** DS1/DS2: 5..10. |
| Max tiers T / h | **[论文明确]** 3..6. | **[论文明确]** Existing/modified: 3..6; Appendix A.2 only: T=4. | **[论文明确]** DS1/DS2: 3..6. |
| Fill / utilization | **[论文明确]** 50% and 67%. | **[论文明确]** 50% and 67% for existing/modified. | **[论文明确]** μ in {0.5, 0.67}. |
| Containers C / N | **[论文明确]** Table 2 gives every exact value: for each S=5..10 and T=3..6, the 50%/67% counts are those in the public files. | **[论文明确]** `C = round(μ*S*T)`, nearest integer (Section 5 dataset description); same C in modified data. | **[论文明确]** Same DS1 counts; DS2 uses all DS1 instances and does not alter N. |
| Batches / time windows | **[论文明确]** Number of distinct time windows is around C/2. **[需要原数据确认]** Exact count varies per file. | **[论文明确]** Existing data averages two containers/batch. **[论文可推导]** Modified count is `ceil(W/2)` after merging adjacent occupied batches. | **[论文明确]** DS1 small; DS2 doubles average batch size. **[论文可推导]** Via cited Galle scheme, DS2 count is `ceil(W/2)`. |
| Batch-size distribution | **[论文明确]** Average approximately 2; no parametric distribution is stated. **[需要原数据确认]** Exact sizes are label frequencies in each file. | **[论文明确]** Existing average 2; modified average 4 (`γ=2`). No fuller distribution is stated. | **[论文明确]** DS1 small; DS2 “basically doubled”; exact sizes follow adjacent-batch merging, not a fresh distribution. |
| Batch assignment rule | **[论文明确]** Each container is booked to a time window; different windows have precedence. **[当前不确定]** Random assignment algorithm and RNG are not disclosed. | **[论文明确]** Existing labels are interpreted as ordered batches. Modified data maps original batch `w` to `w'=ceil(w/γ)`, γ=2. | **[论文明确]** DS1 is the Ku/Galle data; DS2 uses Galle’s scheme. |
| Initial bay generation | **[论文明确]** 30 randomly generated configurations per `(S,T,fill)` setting. **[当前不确定]** Placement algorithm and seed are not stated. | **[论文明确]** Existing physical configurations are reused. Modified data has the same containers in the same configuration. | **[论文明确]** DS1 reuses the existing configurations; DS2 considers all DS1 instances and retains layout/N. |
| Within-batch order distribution | **[论文明确]** Same-window candidates are equally likely to be the next departure (online information model). Exact realized orders are not stored. | **[论文明确]** Uniform random permutation independently within each ordered batch (A5*). | **[论文明确]** All B-based orders have the same probability. |
| Revelation timing | **[论文明确]** Online: next target is revealed one container at a time at chance nodes. | **[论文明确]** Batch model: full permutation of batch w is revealed after batches 1..w-1 are retrieved (A6*); online experiments retain one-at-a-time revelation. | **[论文明确]** Full order of a batch becomes known when the last container of the previous batch is retrieved. |
| Random instances / parameter setting | **[论文明确]** 30. | **[论文明确]** 30 for each existing/modified `(S,T,μ)` group; Appendix A.2 separately has 100 total for one setting. | **[论文明确]** Each `(N,S,T)` group has exactly 30 for both DS1 and DS2. |
| Total static instances | **[论文可推导]** `6*4*2*30 = 1440`. | **[论文明确]** Existing 1440; modified has one derivative per existing instance, hence 1440; other experiments are distinct variants/sets. | **[论文明确]** DS1 1440; **[论文可推导]** DS2 1440 because every DS1 instance is transformed once. |
| Scenarios / simulations per static instance | **[论文明确]** Exact method once; ERI and random heuristic 5,000 stochastic simulations per instance. Static instance count remains 1440. | **[论文明确]** Heuristic estimates use 5,000 uniformly sampled retrieval orders per instance unless otherwise stated. Exact scenario space is **[论文可推导]** `Π_w |B_w|!`. | **[论文可推导]** Scenario space is `|Ω_B| = Π_w |B_w|!`; tables report group-average SC. The paper does not redefine each order as a static instance. |
| Train/test split | **[论文明确]** None; computational benchmark only. | **[论文明确]** None; computational benchmark only. | **[论文明确]** None; computational benchmark only. |
| Public source | **[论文明确]** `crp-timewindow.blogspot.com` links the test archive. | **[论文明确]** `github.com/vgalle/StochasticCRP`; repository README documents the 24 folders and 60 files/folder. | **[论文明确]** Data/code availability cites Galle’s public repository. |
| Exact original files available | **[需要原数据确认]** Yes: the public repository contains 1,440 Ku-format `.txt` files. All were parsed in this audit at commit `ec672df26dae12de42ba3c4e95a4a9002e4410f6`. | **[论文明确]** Yes for existing DS1; DS2 is an exact deterministic transformation in `Experiments_2.m`. | **[论文明确]** Yes/derivable: DS1 files plus the cited DS2 transformation. |
| Regeneration from prose only | **[当前不确定]** Not exact because layout/batch RNG algorithms and seeds are absent. | Not needed for DS1/DS2 because exact files and transformation code exist. Appendix A.2 random-generation details are insufficient for bit-identical regeneration. | Not needed: load DS1 and transform DS2. A fresh “similar” generator would not reproduce the benchmark files. |

The 48 principal groups are `(S,T,μ)`, not individual stochastic orders:
6 stack counts x 4 tier counts x 2 fill rates. The public tree contains 30
files in each group.

## 2. Dataset relationship

```text
Ku static CRPTW text file
  physical stack layout + time-window labels (no unique IDs/order realization)
        |
        | stable ID assignment; ascending time-window -> ordered batch_id
        v
Galle “existing dataset” SCRP static instance
  same layout, same membership/precedence
  batch model samples a uniform within-batch permutation per scenario
        |
        v
Bacci DS1
  the same 1,440 Ku/Galle static instances

DS1 / Galle existing instance
        |
        | keep every container and stack position
        | w' = ceil(w / 2), merging adjacent batches
        v
Galle “modified dataset” = Bacci DS2
  same N and layout, fewer batches, average batch size about 4
```

Ku’s raw files do not encode a realized retrieval order or unique container
identity. Each occupied tier carries a pair of equal integer labels; Galle’s
official `readInputFile.m` reads the first member of each pair as the layout
label. Across all 1,440 files, every pair is equal. This project assigns IDs
only after parsing.

## 3. Available-source audit

Found:

- Three paper PDFs supplied under `docs/`.
- Exact public `vgalle/StochasticCRP` repository, fetched read-only under the
  ignored `tmp/` directory for this audit. It contains 24 size folders, 60 raw
  instances per folder, 1,440 total.
- Exact Galle reader (`readInputFile.m`) and DS2 transformation
  (`Experiments_2.m`, `mergeTimeWindows=2`, elementwise `ceil(label/2)`).
- 18 converted sanity artifacts in `data/phase3_sanity/`: nine selected DS1
  instances and their nine DS2 derivatives. These are not 18 independent Ku
  configurations; each DS2 file has a DS1 parent.

Not found in the workspace before acquisition:

- Any Ku/Galle/Bacci raw benchmark archive or converted JSON.
- A separate Bacci-only instance encoding. The paper points back to the Galle
  source and transformation.
- Any evidence that an older duplicate-priority benchmark is this dataset.

Potential sources:

- Ku’s CRPTW blog/Google Drive archive.
- Galle’s GitHub/Zenodo repository (used here and sufficient).

Need download/manual acquisition:

- For full-scale local evaluation, pin/download the public repository at the
  audited commit. No manual acquisition is needed for the current sanity set.
- Independent paper-spec regeneration is neither needed nor exact; do not use
  it as a substitute for the original files.

### Raw-data findings

The parser audit found 185 files whose sixth header integer differs from the
number of non-empty labels observed in the layout, and one file with a gap in
the observed labels. Galle’s MATLAB reader ignores the sixth header field and
uses the layout. The implemented conversion therefore:

1. treats occupied-tier labels as authoritative;
2. sorts distinct labels to preserve precedence;
3. maps them to non-empty `batch_id = 1..K`;
4. records `original_header_field_6`, `observed_num_time_windows`, and
   `original_label_to_batch` for auditability.

It never creates an empty batch to match a disagreeing header.

## 4. Static JSON schema

```json
{
  "schema_version": "scrp-static-instance-v1",
  "instance_id": "ku2016-T271014_0503_001",
  "source_dataset": "Ku2016_CRPTW_Galle2017_existing_Bacci2022_DS1",
  "num_stacks": 5,
  "max_tiers": 3,
  "num_containers": 8,
  "batch_order": [1, 2, 3, 4],
  "stacks": [[1], [2, 3], [4], [5], [6, 7, 8]],
  "container_batch": {"1": 3, "2": 1, "3": 2},
  "metadata": {
    "fill_rate": 0.5,
    "paper": ["Ku & Arthanari (2016)"],
    "parameter_group": "S05_T03_mu0.5",
    "original_file": "T271014_0503_001.txt",
    "original_instance_id": "T271014_0503_001",
    "converted_instance_id": "ku2016-T271014_0503_001",
    "id_assignment_rule": "stack-major, then bottom-to-top, consecutive integers from 1",
    "stack_orientation": "bottom-to-top"
  }
}
```

`container_batch` above is abbreviated only for exposition; saved files contain
all IDs. Required semantics:

- `stacks` are bottom-to-top and contain every unique ID exactly once.
- `batch_order` is the known precedence; every listed batch is non-empty.
- `source_dataset` and provenance remain static metadata.
- `num_containers` is validated against both container mapping and layout.
- `hidden_orders`, permutations, order seeds, `scenario_seed`, and
  `scenario_id` are forbidden anywhere in a static record.
- A scenario remains the existing `Scenario(root_seed, order_seeds,
  hidden_orders, scenario_id)` object and is never written into these files.

The current tiny model needs no field or transition-semantic change. S/T are
native fields, N is derived from `containers`, `Container` supplies unique ID
and batch membership, `initial_stacks` supplies bottom-to-top layout,
`batch_order` supplies precedence, and metadata carries provenance.

## 5. Loader and DS1/DS2 plan

### DS1 (implemented)

- Input: exact Ku `.txt`; expected 1,440 files in 48 groups x 30.
- Parse: strict six-field header and S stack rows; validate row count, height,
  pair structure, equal pair values, N, capacity, and positive labels.
- Identity: assign integers 1..N by stack-major/bottom-to-top traversal. IDs do
  not enter O1 as ordinal features.
- Batch mapping: ascending observed time-window label to ordered non-empty
  `batch_id` while preserving the raw mapping in metadata.
- Validate: all base `SCRPInstance` invariants plus
  `N <= S*T-(T-1)` and future-order leakage rejection.
- Output: scenario-free schema v1 with original and converted IDs, parameter
  group, raw filename, label mapping, fill, and ID rule.
- Seed: none for static conversion; scenario seed is supplied only to
  `ScenarioSampler`/environment reset.

### DS2 (implemented)

- Input: a validated DS1 instance.
- Conversion: retain IDs, N, S/T, and exact stacks; merge adjacent batch
  positions with `new = ceil(old/2)`.
- Validate/output: same static checks; provenance adds parent instance,
  merge factor and formula. Expected full count is 1,440 derivatives.
- Seed: none; transformation is deterministic. Scenario seeds remain external.

No paper-spec static generator is implemented. Exact sources are available,
and the papers do not disclose enough RNG detail for bit-identical fresh bay
generation. A future non-benchmark synthetic generator must be clearly named
as synthetic and parameterize S, T, N/fill, batch sizes/count, instance count,
and seed; it must not claim to regenerate the original files.

## 6. Validation coverage

Automated tests cover:

1. declared/observed container count;
2. stack capacity;
3. unique deterministic IDs;
4. each ID appears exactly once;
5. every ID has a batch;
6. explicit ordered batch list;
7. no empty batch (observed labels are normalized, not header padding);
8. paper feasibility `N <= S*T-(T-1)`;
9. identical save/load round trip;
10. different scenario seeds leave static data identical and can differ in
    hidden permutations;
11. same seed reproduces the exact scenario;
12. no hidden permutation in a static record;
13. future-order/scenario fields in metadata are rejected.

In addition, the parser and DS2 transform were run over all 1,440 public raw
files: 1,440 parsed, 48 groups, exactly 30/group, DS1 batch-count range 3..24,
and DS2 range 2..12.

## 7. Small-scale sanity result

Selected original groups and files:

| Group | Original files | DS1 `(S,T,N)` | DS1 batch counts |
|---|---|---|---|
| S05 T03 μ=.50 | 001..003 | (5,3,8) | 4, 3, 4 |
| S07 T04 μ=.67 | 001..003 | (7,4,19) | 11, 10, 11 |
| S10 T06 μ=.50 | 001..003 | (10,6,30) | 18, 17, 16 |

Each DS1 and its DS2 derivative was saved, reloaded, and run with scenario
seeds 101, 202, and 303 using both a seeded random-legal rollout and the
current Phase 2.5 LOW checkpoint in greedy mode. Result: 18 static files, 54
scenario pairs, all random rollouts terminated, all LOW-network rollouts
terminated, zero truncations, and matching scenario IDs between policies for
every `(instance_id, scenario_seed)`. Detailed batch sizes, scenario IDs, and
relocation counts are in `data/phase3_sanity/sanity_results.json`.

## 8. Remaining paper-level uncertainties

- Ku states that configurations are randomly generated but does not specify
  the exact stack-placement algorithm, time-window assignment algorithm, RNG,
  or seed. Therefore exact independent regeneration from prose is impossible.
- The semantic names of both duplicated raw tier fields are not documented in
  Galle’s README/reader. This does not block exact conversion because all
  1,440 pairs are equal and the official reader selects one value, but a raw
  format specification from Ku would be needed to name both fields confidently.
- Ku’s header-field discrepancies are present in the original public data; the
  papers do not explain them. The official reader behavior and observed layout
  are used rather than silently inventing empty batches.

DS2’s construction itself is not unresolved: Galle gives the exact formula
and code, and Bacci explicitly adopts that scheme.
