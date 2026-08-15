"""
hier_pg 训练脚本 — Hierarchical PG-FGB for CRP-D
==================================================
两层策略：
  high_decoder → 选择哪个含有当前组箱子的 stack 来处理
  low_decoder  → 选择把挡路箱搬到哪个 stack

运行（从 Platform_CRISP 根目录）：
    python -m algorithms.CRP_D.rl.hier_pg.run
    python -m algorithms.CRP_D.rl.hier_pg.run --quick
"""

import sys
import pathlib
import multiprocessing as mp
import argparse

_ALGO_DIR = pathlib.Path(__file__).parent
_ROOT     = _ALGO_DIR.parents[3]
sys.path.insert(0, str(_ROOT))

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--quick", action="store_true",
                     help="快速验证：S=6 only, 100 轮")
_args, _ = _parser.parse_known_args()
QUICK_TEST = _args.quick

# ════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════

DUP_ROOT = str(_ROOT / "benchmark" / "dup_dataset")

# quick 模式只用 S=6；正式训练混合 S=6~10
DUP_STACKS_LIST = [6] if QUICK_TEST else [6, 7, 8, 9, 10]
DUP_STACKS      = DUP_STACKS_LIST[0]

# 只用 T=6（和测试集一致）
DUP_TIERS                = 6 if not QUICK_TEST else None
DUP_MAX_FILES_PER_FOLDER = 40 if not QUICK_TEST else 0

YARD = dict(
    num_bays       = DUP_STACKS,
    num_rows       = 1,
    max_tiers      = 10,
    num_containers = 30,
    num_groups     = 3,
)
VESSEL = dict(
    vessel_bays  = 1,
    vessel_rows  = 80,
    vessel_tiers = 1,
)

# 5-dim feature scale (het-pg pads to 12 internally)
FEATURE_SCALE = "10,80,10,1,10"

if QUICK_TEST:
    TRAIN = dict(
        num_iterations    = 100,
        episodes_per_iter = 8,
        eval_episodes     = 8,
        learning_rate     = 3e-4,
        ent_coef          = 0.02,
        p_value_threshold = 0.05,
        gamma             = 1.0,
        anneal_lr         = True,
        greedy_high       = True,
        report_every      = 10,
    )
else:
    TRAIN = dict(
        num_iterations    = 1000,
        episodes_per_iter = 16,
        eval_episodes     = 16,
        learning_rate     = 2.5e-4,
        ent_coef          = 0.01,
        p_value_threshold = 0.05,
        gamma             = 1.0,
        anneal_lr         = True,
        greedy_high       = True,
        report_every      = 20,
    )

NETWORK = dict(
    embed_dim      = 64 if QUICK_TEST else 128,
    num_enc_layers = 2,
    num_heads      = 4,
    ffn_dim        = 128 if QUICK_TEST else 256,
    clip_constant  = 10.0,
)

FIXED_EVAL = dict(
    enabled         = not QUICK_TEST,
    dup_root        = DUP_ROOT,
    alpha           = 0.8,
    tiers           = 6,
    stacks          = [6, 7, 8, 9, 10],
    files_per_combo = 2,
)

SEED = 0


# ════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════

def main():
    from core.base_problem   import ProblemConfig
    from core.base_algorithm import AlgorithmConfig
    from problems.CRP_D      import CRP_D
    from algorithms.CRP_D.rl.hier_pg.algorithm import HierPgFgbD
    from algorithms.CRP_D.rl.pg_fgb.algorithm  import (
        _scan_dup_files, _build_fixed_eval_set
    )

    save_dir = _ALGO_DIR / "trained_models"
    save_dir.mkdir(exist_ok=True)
    model_name = "hier_pg_quick.pt" if QUICK_TEST else "hier_pg_crpd.pt"
    save_path  = str(save_dir / model_name)

    cfg_p = ProblemConfig(seed=SEED, **YARD, **VESSEL)

    n_iters = TRAIN["num_iterations"]
    cfg_a   = AlgorithmConfig(
        max_iterations  = n_iters,
        report_interval = max(1, n_iters // TRAIN["report_every"]),
        seed            = SEED,
    )
    cfg_a.extra.update({
        "num_iterations":           n_iters,
        "save_path":                save_path,
        "report_every":             TRAIN["report_every"],
        "dup_train_root":           DUP_ROOT,
        "dup_num_stacks_list":      DUP_STACKS_LIST,
        "dup_num_stacks":           DUP_STACKS,
        "dup_tiers":                DUP_TIERS,
        "dup_max_files_per_folder": DUP_MAX_FILES_PER_FOLDER,
        "feature_scale":            FEATURE_SCALE,
        **{k: v for k, v in TRAIN.items() if k not in ("num_iterations", "report_every")},
        **NETWORK,
    })

    # ── scan training files ─────────────────────────────────────── #
    dup_files = []
    for s_val in DUP_STACKS_LIST:
        dup_files.extend(_scan_dup_files(
            DUP_ROOT, s_val,
            target_H            = DUP_TIERS,
            max_files_per_folder= DUP_MAX_FILES_PER_FOLDER,
        ))

    # ── fixed eval set ──────────────────────────────────────────── #
    fixed_eval_set = []
    if FIXED_EVAL["enabled"]:
        fixed_eval_set = _build_fixed_eval_set(
            dup_root        = FIXED_EVAL["dup_root"],
            alpha           = FIXED_EVAL["alpha"],
            tiers           = FIXED_EVAL["tiers"],
            stacks          = FIXED_EVAL["stacks"],
            files_per_combo = FIXED_EVAL["files_per_combo"],
        )
        cfg_a.extra["fixed_eval_set"] = fixed_eval_set

    mode_tag = "【QUICK_TEST】" if QUICK_TEST else "【正式训练】"
    print("=" * 66)
    print(f"  CRP-D × Hier-PG-FGB 训练  {mode_tag}")
    print("=" * 66)
    print(f"  dup_dataset 根目录  : {DUP_ROOT}")
    print(f"  训练 stack 数 (S)   : {DUP_STACKS_LIST}")
    print(f"  训练 tier  数 (T)   : {DUP_TIERS if DUP_TIERS else '全部'}")
    print(f"  每文件夹最多文件数  : {DUP_MAX_FILES_PER_FOLDER if DUP_MAX_FILES_PER_FOLDER else '全取'}")
    print(f"  可用 dup 文件数      : {len(dup_files)}")
    if fixed_eval_set:
        s_vals = sorted(set(s for _, s, _, _ in fixed_eval_set))
        print(f"  固定评估集         : T=6, α=0.8, S={s_vals}, {len(fixed_eval_set)} 个实例")
    print(f"  迭代: {n_iters}  eps/iter={TRAIN['episodes_per_iter']}"
          f"  lr={TRAIN['learning_rate']}")
    print(f"  模型将保存到: {save_path}")
    print("=" * 66)

    if not dup_files:
        print("[ERROR] 未找到训练文件，请检查 DUP_ROOT 和参数。")
        return

    inst = HierPgFgbD(cfg_a)
    q    = mp.Queue()
    ev   = mp.Event()

    def factory():
        return CRP_D(config=cfg_p)

    proc = mp.Process(target=inst.train, args=(factory, q, ev), daemon=True)
    proc.start()

    while proc.is_alive():
        try:
            r = q.get(timeout=1.0)
            m = r.metrics
            s6  = m.get("shifters_s6",  float("nan"))
            ood = m.get("shifters_ood", float("nan"))
            line = (
                f"  step={r.step:5d}/{n_iters}"
                f"  shifters={m.get('shifters', r.metric):.2f}"
                f"  best={m.get('best_shifters', float('inf')):.2f}"
            )
            if fixed_eval_set:
                line += f"  S6={s6:.2f}" if s6 == s6 else ""
                line += f"  OOD={ood:.2f}" if ood == ood else ""
            line += (
                f"  H={int(m.get('high_steps', 0))}"
                f"  L={int(m.get('low_steps', 0))}"
                f"  loss={m.get('policy_loss', float('nan')):.4f}"
                f"  entropy={m.get('entropy', float('nan')):.4f}"
                f"  adv={m.get('mean_advantage', float('nan')):+.4f}"
                f"  bl_upd={int(m.get('baseline_updates', 0))}"
                f"  [{r.progress*100:.1f}%]"
            )
            print(line)
        except Exception:
            pass

    proc.join()
    ev.set()
    if pathlib.Path(save_path).exists():
        print(f"\n[✓] 模型已保存 → {save_path}")
    else:
        print("\n[!] 模型未能保存，请检查训练过程")
    print("Done.")


if __name__ == "__main__":
    main()
