"""
Experiment-1 evaluator for CRP-D (dup_dataset).

What it does
------------
1) Load a trained PG-FGB-D checkpoint.
2) Evaluate RL greedy policy on selected dup_dataset files.
3) Optionally evaluate GLAH on the exact same files.
4) Save per-instance results and aggregated summaries as CSV.

Usage
-----
From Platform_CRISP root:
    python -m algorithms.CRP_D.rl.pg_fgb.evaluate_experiment1 \
        --model algorithms/CRP_D/rl/pg_fgb/trained_models/pg_fgb_crpd.pt \
        --dup-root benchmark/dup_dataset \
        --test-stacks 6,8,10 \
        --alphas 0.2,0.4,0.6,0.8 \
        --files-per-size 20 \
        --with-glah \
        --out-dir algorithms/CRP_D/rl/pg_fgb/exp1_results
"""

from __future__ import annotations

import argparse
import csv
import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

from core.base_problem import ProblemConfig
from core.zhu_dup_benchmark import parse_zhu_dup_file
from problems.CRP_D import CRP_D

from algorithms.CRP_D.rl.pg_fgb.network import CrpdPolicyNetwork


@dataclass(frozen=True)
class InstanceMeta:
    alpha: float
    g: int
    s: int
    n: int
    path: Path


def _parse_alphas(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _alpha_dir_name(alpha: float) -> str:
    return f"alpha={alpha:.1f}"


def _scan_instances(
    dup_root: Path,
    test_stacks: Iterable[int],
    alphas: Iterable[float],
    files_per_size: int,
    test_tiers: Iterable[int] | None = None,
) -> List[InstanceMeta]:
    from core.zhu_dup_benchmark import parse_zhu_dup_folder_name

    stack_set = set(test_stacks)
    tier_set = set(test_tiers) if test_tiers is not None else None
    metas: List[InstanceMeta] = []

    for alpha in alphas:
        alpha_dir = dup_root / _alpha_dir_name(alpha)
        if not alpha_dir.is_dir():
            continue
        for size_dir in sorted(p for p in alpha_dir.iterdir() if p.is_dir()):
            parsed = parse_zhu_dup_folder_name(size_dir.name)
            if parsed is None:
                continue
            g, s, n = parsed
            if s not in stack_set:
                continue
            # g here is max_tiers (H) in the ZhuDup naming convention G-S-N
            if tier_set is not None and g not in tier_set:
                continue

            files = sorted(size_dir.glob("*.txt"))
            if files_per_size > 0:
                files = files[: files_per_size]
            for f in files:
                metas.append(InstanceMeta(alpha=alpha, g=g, s=s, n=n, path=f))

    metas.sort(key=lambda m: (m.s, m.alpha, m.g, m.n, m.path.name))
    return metas


def _load_policy(model_path: Path) -> Tuple[CrpdPolicyNetwork, Dict]:
    ckpt = torch.load(model_path, map_location="cpu")
    feature_scale = torch.tensor(ckpt["feature_scale"], dtype=torch.float32)
    policy = CrpdPolicyNetwork(
        embed_dim=ckpt["embed_dim"],
        num_enc_layers=ckpt["num_enc_layers"],
        num_heads=ckpt["num_heads"],
        ffn_dim=ckpt["ffn_dim"],
        clip_constant=ckpt["clip_constant"],
        feature_scale=feature_scale,
    )
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    return policy, ckpt


def _make_env_for_file(path: Path, max_tiers_h: int | None = None) -> CRP_D:
    # parse_zhu_dup_file returns (S, N, stacks)
    stacks, n, stack_groups = parse_zhu_dup_file(path)
    # Use the folder-name H (= paper's T) when provided; fall back to
    # actual content height only if H is unknown (e.g. standalone use).
    if max_tiers_h is not None:
        max_tiers = max_tiers_h
    else:
        max_tiers = int(max((len(col) for col in stack_groups), default=5))
    groups = int(max((g for col in stack_groups for g in col if g > 0), default=3))
    cfg = ProblemConfig(
        num_bays=stacks,
        num_rows=1,
        max_tiers=max_tiers,
        num_containers=n,
        num_groups=groups,
        vessel_bays=1,
        vessel_rows=max(80, n + 10),
        vessel_tiers=1,
        seed=0,
    )
    cfg.extra["layout_file_path"] = str(path.resolve())
    return CRP_D(config=cfg)


def _run_rl_episode(env: CRP_D, policy: CrpdPolicyNetwork, greedy_high: bool) -> int:
    obs, info = env.reset()
    done = False
    while not done:
        mask_raw = info.get("action_mask", np.ones(env.action_space.n, dtype=bool))
        mask_bool = mask_raw.astype(bool)
        # Safety guard: avoid all-invalid masks causing NaN logits.
        if not np.any(mask_bool):
            break
        action = None
        if greedy_high and env._mode == "high" and env._current_slot is not None:
            target_grp = int(env._vessel_state[env._current_slot, 4])
            for i in range(env.action_space.n):
                if not mask_bool[i]:
                    continue
                key = env._action_to_stack(i)
                stk = env.yard.stacks.get(key)
                if stk and not stk.is_empty and stk.top.group == target_grp:
                    action = i
                    break
        if action is None:
            forbidden = torch.tensor(~mask_bool, dtype=torch.bool).unsqueeze(0)
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                enc = policy.encode(obs_t)
                act_t, _ = policy(obs_t, forbidden, greedy=True, encoder_output=enc)
            action = int(act_t.item())
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return int(env.get_metrics().get("shifters", 0.0))


def _run_glah_episode(env: CRP_D, depth: int = 3) -> int:
    from algorithms.CRP_U.heuristic.glah.layout import GlahLayout, GlahState, lower_bound
    from algorithms.CRP_U.heuristic.glah.evaluate import evaluation_heuristic
    from algorithms.CRP_U.heuristic.glah.lookahead import Lookahead

    env.reset()
    initial_yard = copy.deepcopy(env.yard)
    containers = list(env.containers)
    glah_layout = GlahLayout.build_from_yard(initial_yard, containers)

    state1 = GlahState(glah_layout.copy())
    evaluation_heuristic(state1)

    la = Lookahead(depth, 5, 5, 3, 3, 1, 1)
    state2 = GlahState(glah_layout.copy())
    state2.best_ops = state1.best_ops
    state2.best_reloc = state1.best_reloc
    state2.try_retrievals()

    while not state2.is_empty():
        lb_now = lower_bound(state2.inst)
        if lb_now + state2.reloc_count >= state2.best_reloc:
            break
        op = la.most_promising_relocation(state2)
        if op is None:
            break
        state2.go_one_step(op)
        state2.try_retrievals()

    if state2.best_ops is None:
        return int(state2.reloc_count)
    return int(state2.best_reloc)


def _write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment-1 evaluator for CRP-D")
    parser.add_argument("--model", required=True, help="Path to PG-FGB-D checkpoint")
    parser.add_argument("--dup-root", default="benchmark/dup_dataset")
    parser.add_argument("--test-stacks", default="6,8,10")
    parser.add_argument("--alphas", default="0.2,0.4,0.6,0.8")
    parser.add_argument("--files-per-size", type=int, default=20,
                        help="Number of files per (alpha, G-S-N) folder; <=0 means all")
    parser.add_argument("--test-tiers", default=None,
                        help="Comma-separated max-tier (H) values to include, e.g. '6' or '5,6'. "
                             "If omitted, all tier heights are included.")
    parser.add_argument("--with-glah", action="store_true")
    parser.add_argument("--glah-depth", type=int, default=3)
    parser.add_argument("--no-greedy-high", action="store_true")
    parser.add_argument("--out-dir", default="algorithms/CRP_D/rl/pg_fgb/exp1_results")
    args = parser.parse_args()

    model_path = Path(args.model)
    dup_root = Path(args.dup_root)
    out_dir = Path(args.out_dir)
    test_stacks = _parse_int_list(args.test_stacks)
    alphas = _parse_alphas(args.alphas)
    test_tiers = _parse_int_list(args.test_tiers) if args.test_tiers else None
    greedy_high = not args.no_greedy_high

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not dup_root.is_dir():
        raise FileNotFoundError(f"dup_dataset root not found: {dup_root}")

    policy, ckpt = _load_policy(model_path)
    metas = _scan_instances(
        dup_root=dup_root,
        test_stacks=test_stacks,
        alphas=alphas,
        files_per_size=args.files_per_size,
        test_tiers=test_tiers,
    )
    if not metas:
        raise RuntimeError("No test files found for the given filters.")

    print("=" * 72)
    print("Experiment-1 Evaluation (CRP-D, dup_dataset)")
    print("=" * 72)
    print(f"model           : {model_path}")
    print(f"best_shifters   : {ckpt.get('best_shifters', 'N/A')}")
    print(f"test_stacks     : {test_stacks}")
    print(f"test_tiers (H)  : {test_tiers if test_tiers is not None else 'all'}")
    print(f"alphas          : {alphas}")
    print(f"files_per_size  : {args.files_per_size}")
    print(f"with_glah       : {args.with_glah} (depth={args.glah_depth})")
    print(f"greedy_high     : {greedy_high}")
    print(f"total instances : {len(metas)}")
    print("=" * 72)

    detail_rows: List[Dict] = []
    for i, meta in enumerate(metas, start=1):
        env = _make_env_for_file(meta.path, max_tiers_h=meta.g)
        t0 = time.perf_counter()
        rl_shifters = _run_rl_episode(env, policy, greedy_high=greedy_high)
        rl_time_s = time.perf_counter() - t0
        row = {
            "alpha": meta.alpha,
            "G": meta.g,
            "S": meta.s,
            "N": meta.n,
            "file": str(meta.path),
            "rl_shifters": rl_shifters,
            "rl_time_s": round(rl_time_s, 6),
        }
        if args.with_glah:
            env2 = _make_env_for_file(meta.path, max_tiers_h=meta.g)
            t1 = time.perf_counter()
            glah_shifters = _run_glah_episode(env2, depth=args.glah_depth)
            glah_time_s = time.perf_counter() - t1
            row["glah_shifters"] = glah_shifters
            row["glah_time_s"] = round(glah_time_s, 6)
            row["delta_rl_minus_glah"] = rl_shifters - glah_shifters
        detail_rows.append(row)

        if i % 50 == 0 or i == len(metas):
            print(f"progress: {i}/{len(metas)}")

    summary_by_s_alpha: Dict[Tuple[int, float], List[Dict]] = {}
    for r in detail_rows:
        key = (int(r["S"]), float(r["alpha"]))
        summary_by_s_alpha.setdefault(key, []).append(r)

    summary_rows: List[Dict] = []
    for (s, alpha), rows in sorted(summary_by_s_alpha.items()):
        rl_mean = float(np.mean([x["rl_shifters"] for x in rows]))
        rl_time_mean = float(np.mean([x["rl_time_s"] for x in rows]))
        out = {
            "S": s,
            "alpha": alpha,
            "num_instances": len(rows),
            "rl_mean_shifters": round(rl_mean, 4),
            "rl_mean_time_s": round(rl_time_mean, 6),
        }
        if args.with_glah:
            glah_mean = float(np.mean([x["glah_shifters"] for x in rows]))
            glah_time_mean = float(np.mean([x["glah_time_s"] for x in rows]))
            out["glah_mean_shifters"] = round(glah_mean, 4)
            out["glah_mean_time_s"] = round(glah_time_mean, 6)
            out["delta_mean_rl_minus_glah"] = round(rl_mean - glah_mean, 4)
        summary_rows.append(out)

    # ── Overall aggregate across ALL tested instances ────────────────── #
    overall_rl_mean = float(np.mean([x["rl_shifters"] for x in detail_rows]))
    overall_rl_time = float(np.mean([x["rl_time_s"] for x in detail_rows]))

    detail_csv = out_dir / "exp1_detail.csv"
    summary_csv = out_dir / "exp1_summary_s_alpha.csv"
    detail_fields = ["alpha", "G", "S", "N", "file", "rl_shifters", "rl_time_s"]
    if args.with_glah:
        detail_fields += ["glah_shifters", "glah_time_s", "delta_rl_minus_glah"]
    summary_fields = ["S", "alpha", "num_instances", "rl_mean_shifters", "rl_mean_time_s"]
    if args.with_glah:
        summary_fields += ["glah_mean_shifters", "glah_mean_time_s", "delta_mean_rl_minus_glah"]

    _write_csv(detail_csv, detail_rows, detail_fields)
    _write_csv(summary_csv, summary_rows, summary_fields)

    print("-" * 72)
    print(f"Saved detail  : {detail_csv}")
    print(f"Saved summary : {summary_csv}")
    print("-" * 72)
    print(f"[OVERALL]  n={len(detail_rows)}")
    print(f"           rl_mean_shifters = {overall_rl_mean:.4f}")
    print(f"           rl_mean_time_s   = {overall_rl_time:.6f} s/instance")
    if args.with_glah:
        overall_glah_mean = float(np.mean([x["glah_shifters"] for x in detail_rows]))
        overall_glah_time = float(np.mean([x["glah_time_s"] for x in detail_rows]))
        print(f"           glah_mean_shifters = {overall_glah_mean:.4f}")
        print(f"           glah_mean_time_s   = {overall_glah_time:.6f} s/instance")
        print(f"           delta (rl-glah)    = {overall_rl_mean - overall_glah_mean:+.4f}")
    print("-" * 72)


if __name__ == "__main__":
    main()
