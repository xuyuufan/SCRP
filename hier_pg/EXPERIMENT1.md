# Experiment-1: CRP-D on dup_dataset

This document defines a full, reproducible pipeline for **Experiment-1**:

- Train `PG-FGB-D` on medium-scale CRP-D instances.
- Test in-distribution (same scale) and out-of-distribution (larger scale).
- Compare against `GLAH` on the exact same test files.

## 1) Goal and Setup

- Problem: `CRP-D` (duplicate groups, stack+tier focus)
- Dataset: `benchmark/dup_dataset`
- RL algorithm: `algorithms/CRP_D/rl/pg_fgb`
- Baseline: `algorithms/CRP_D/heuristic/glah`

Recommended primary protocol:

1. Train on **S=6** (medium).
2. Test on **S=6** (in-distribution) and **S=8,10** (scale generalization).
3. Report by `S × alpha` and overall.

## 2) Training

Default training script:

```bash
cd /data/liuw2/Platform_CRISP
python -m algorithms.CRP_D.rl.pg_fgb.run
```

Model output:

- `algorithms/CRP_D/rl/pg_fgb/trained_models/pg_fgb_crpd.pt`

Default training settings (from `run.py`):

- `DUP_STACKS=6`
- `num_iterations=1000`
- `episodes_per_iter=16`
- `eval_episodes=16`
- `greedy_high=True`
- `feature_scale="10,80,10,1,10"`

## 3) Quick Sanity Test (step-by-step)

Use one known instance and inspect behavior:

```bash
cd /data/liuw2/Platform_CRISP
python -m algorithms.CRP_D.rl.pg_fgb.test_policy \
  --model algorithms/CRP_D/rl/pg_fgb/trained_models/pg_fgb_crpd.pt \
  --file benchmark/dup_dataset/alpha=0.2/3-6-15/00001.txt \
  --pause --greedy-high
```

## 4) Batch Evaluation (RL only or RL vs GLAH)

Use the evaluator script:

```bash
cd /data/liuw2/Platform_CRISP
python -m algorithms.CRP_D.rl.pg_fgb.evaluate_experiment1 \
  --model algorithms/CRP_D/rl/pg_fgb/trained_models/pg_fgb_crpd.pt \
  --dup-root benchmark/dup_dataset \
  --test-stacks 6,8,10 \
  --alphas 0.2,0.4,0.6,0.8 \
  --files-per-size 20 \
  --with-glah \
  --out-dir algorithms/CRP_D/rl/pg_fgb/exp1_results
```

Outputs:

- `exp1_results/exp1_detail.csv` (per-instance)
- `exp1_results/exp1_summary_s_alpha.csv` (aggregated by `S × alpha`)

For full paper-level testing, replace `--files-per-size 20` with `100`.

## 5) Recommended Reporting

Minimum tables:

1. `RL mean shifters` by `S × alpha`
2. `GLAH mean shifters` by `S × alpha`
3. `Delta (RL - GLAH)` by `S × alpha`

Interpretation:

- `Delta < 0`: RL better than GLAH
- `Delta > 0`: GLAH better

Also report:

- In-distribution (`S=6`) vs out-of-distribution (`S=8,10`)
- Performance trend as `alpha` increases (difficulty rises)

## 6) Suggested Runtime Plan

Practical staged plan:

1. **Smoke run**: `iterations=100`, `files-per-size=5`, with GLAH.
2. **Main run**: `iterations=1000`, `files-per-size=20`.
3. **Final run**: `iterations=1000~2000`, `files-per-size=100` for publishable tables.

## 7) Notes

- `dup_dataset` files are tiny; repeated runs are mostly CPU/env-step bound.
- `greedy_high=True` is recommended for stable CRP-D training.
- Keep the same test file set for RL and GLAH to ensure fair comparison.
