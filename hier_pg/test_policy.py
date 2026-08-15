"""
PG-FGB × CRP-D 训练后模型的逐步测试脚本（实验一）

用法（在 Platform_CRISP 根目录下执行）：
    python -m algorithms.CRP_D.rl.pg_fgb.test_policy [选项]

常用选项：
    --model     trained_models/pg_fgb_crpd.pt   模型文件路径
    --file      PATH                              指定 dup_dataset 中的某个 .txt 文件
    --stacks    6                                 随机生成时的 stack 数（无 --file 时生效）
    --seed      42                                随机 seed（无 --file 时控制布局）
    --tiers     5                                 随机生成时的最大层高
    --groups    3                                 随机生成时的分组数
    --containers 18                               随机生成时的箱子总数
    --pause                                       每步按 Enter 继续（交互模式）
    --greedy-high                                 顶部即目标组时直接取走（不经过模型）

示例：
    # 加载一个具体 dup 文件，交互模式：
    python -m algorithms.CRP_D.rl.pg_fgb.test_policy \\
        --file benchmark/dup_dataset/alpha=0.2/3-6-15/00001.txt --pause

    # 随机生成 S=6 实例：
    python -m algorithms.CRP_D.rl.pg_fgb.test_policy --stacks 6 --seed 7 --pause
"""
from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

# ── 路径设置 ──────────────────────────────────────────────────── #
_ALGO_DIR = Path(__file__).parent                          # .../pg_fgb/
_ROOT     = _ALGO_DIR.parents[3]                           # Platform_CRISP/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── 默认模型路径 ─────────────────────────────────────────────── #
_DEFAULT_MODEL = str(_ALGO_DIR / "trained_models" / "pg_fgb_crpd.pt")


# ════════════════════════════════════════════════════════════════
#  打印辅助
# ════════════════════════════════════════════════════════════════

def _print_yard_1d(env, title: str = "", highlight: set | None = None) -> None:
    """
    CRP-D 单列堆场可视化（S stacks × 1 row）。

    以列为 stack，行为 tier，从上到下 T..1 显示。
    highlight: {(bay, 1), ...}  用 ★ 标记高亮 stack 的列头。
    """
    cfg  = env.config
    snap = env.yard.group_snapshot()   # {(bay, row): [group_id, ...] bottom→top}

    bays    = sorted(set(k[0] for k in snap))
    max_h   = cfg.max_tiers
    hl      = highlight or set()
    col_w   = 6   # 每列显示宽度

    if title:
        print(f"\n{'─'*60}")
        print(f"  {title}")
        print(f"{'─'*60}")

    # 列头：S1 S2 ... SN（高亮用 ★ 前缀）
    header_parts = ["  Tier |"]
    for b in bays:
        mark = "★" if (b, 1) in hl else " "
        header_parts.append(f"{mark}S{b:2d}  ")
    print("".join(header_parts))
    sep_len = 8 + len(bays) * (col_w + 1)
    print("  " + "-" * sep_len)

    # Tier 行：从高到低
    for tier in range(max_h, 0, -1):
        cells = [f"  T{tier:2d}  |"]
        for b in bays:
            groups = snap.get((b, 1), [])
            idx = tier - 1
            if idx < len(groups):
                g   = groups[idx]
                txt = f"[G{g}]" if idx == len(groups) - 1 else f" G{g} "
            else:
                txt = " --- "
            cells.append(f" {txt:<{col_w-1}}")
        print("".join(cells))

    # 高度行
    heights = "  ".join(f"S{b}:{len(snap.get((b, 1), []))}" for b in bays)
    print(f"  heights: {heights}")
    print()


def _print_target(env) -> None:
    """打印当前取箱目标（哪个 Group）以及当前阶段。"""
    if env._current_slot is None or env._vessel_state is None:
        print("  Target: all done")
        return
    slot      = env._vessel_state[env._current_slot]
    target_grp = int(slot[4])
    avail      = list(env._available_groups)
    phase_str  = f"Phase: {env._mode.upper()}"
    if env._mode == "low":
        src     = env._source_stack
        stk     = env.yard.stacks.get(src) if src else None
        top_txt = f"G{stk.top.group}" if (stk and not stk.is_empty) else "?"
        phase_str += f"  ← 源垛=S{src[0]}  顶部=【{top_txt}】阻挡目标 G{target_grp}"
    print(f"  Target Group = G{target_grp}  "
          f"剩余组: {[f'G{g}' for g in avail]}  |  {phase_str}")


def _get_top_group(env, stack_key) -> str:
    stk = env.yard.stacks.get(stack_key)
    if stk is None or stk.is_empty:
        return "?"
    return f"G{stk.top.group}"


# ════════════════════════════════════════════════════════════════
#  主逻辑
# ════════════════════════════════════════════════════════════════

def run_test(
    model_path:     str,
    dup_file:       str | None,
    stacks:         int,
    seed:           int,
    pause:          bool,
    greedy_high:    bool,
    max_tiers:      int | None,
    num_groups:     int | None,
    num_containers: int | None,
) -> None:
    import torch
    import numpy as np

    # ── 加载 checkpoint ─────────────────────────────────────── #
    if not os.path.isfile(model_path):
        print(f"[ERROR] 模型文件不存在: {model_path}")
        print("  请先训练：python -m algorithms.CRP_D.rl.pg_fgb.run")
        sys.exit(1)

    ckpt = torch.load(model_path, map_location="cpu")
    print(f"[INFO] 加载模型: {model_path}")
    print(f"       训练时最优 greedy shifters = {ckpt.get('best_shifters', '?'):.3f}")

    from algorithms.CRP_D.rl.pg_fgb.network import CrpdPolicyNetwork

    feature_scale = torch.tensor(ckpt["feature_scale"], dtype=torch.float32)
    policy = CrpdPolicyNetwork(
        embed_dim      = ckpt["embed_dim"],
        num_enc_layers = ckpt["num_enc_layers"],
        num_heads      = ckpt["num_heads"],
        ffn_dim        = ckpt["ffn_dim"],
        clip_constant  = ckpt["clip_constant"],
        feature_scale  = feature_scale,
    )
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()

    # ── 构建 CRP-D 环境 ─────────────────────────────────────── #
    from core.base_problem import ProblemConfig
    from problems.CRP_D    import CRP_D

    if dup_file:
        # 从 dup 文件解析 S 和 N
        from core.zhu_dup_benchmark import parse_zhu_dup_file
        _stacks, _N, _stack_groups = parse_zhu_dup_file(Path(dup_file))
        _tiers  = max((len(col) for col in _stack_groups), default=5)
        _groups = max(
            (g for col in _stack_groups for g in col if g > 0), default=3
        )
        print(f"[INFO] dup 文件: {dup_file}")
        print(f"       S={_stacks}  N={_N}  max_tier={_tiers}  G≈{_groups}")
    else:
        _stacks = stacks
        _tiers  = max_tiers or ckpt.get("max_tiers", 5)
        _groups = num_groups or ckpt.get("num_groups", 3)
        _N      = num_containers or ckpt.get("num_containers", _stacks * _groups)

    # vessel 足够大（顺序分组）
    _vessel_rows = max(80, _N + 10)

    cfg = ProblemConfig(
        num_bays       = _stacks,
        num_rows       = 1,
        max_tiers      = _tiers,
        num_containers = _N,
        num_groups     = _groups,
        vessel_bays    = 1,
        vessel_rows    = _vessel_rows,
        vessel_tiers   = 1,
        seed           = seed,
    )

    if dup_file:
        cfg.extra["layout_file_path"] = str(Path(dup_file).resolve())

    env = CRP_D(config=cfg)
    obs, info = env.reset(seed=seed)

    # ── 打印初始状态 ─────────────────────────────────────────── #
    from collections import Counter
    print(f"\n{'═'*66}")
    print(f"  CRP-D 测试  seed={seed}"
          f"  {'dup文件' if dup_file else '随机生成'}")
    print(f"  堆场: S={env._n_stacks} stacks × T={cfg.max_tiers} tiers")
    print(f"  箱子: {cfg.num_containers} 个")
    print(f"{'═'*66}")

    _print_yard_1d(env, "初始堆场状态")

    grp_cnt = Counter(c.group for stk in env.yard.stacks.values() for c in stk.containers)
    print("  初始各组箱子数: " + "  ".join(f"G{g}:{n}" for g, n in sorted(grp_cnt.items())))

    slot_cnt = Counter(int(s[4]) for s in env._vessel_state)
    if 0 in slot_cnt:
        del slot_cnt[0]
    print("  取箱顺序（组→槽位数）: " + "  ".join(
        f"G{g}:{n}" for g, n in sorted(slot_cnt.items())
    ))

    _print_target(env)
    if pause:
        input("\n  [Enter] 开始逐步执行...")

    # ── 逐步执行 ────────────────────────────────────────────── #
    step_num = 0
    done     = False

    while not done:
        mask_raw = info.get("action_mask", np.ones(env.action_space.n, dtype=bool))

        # ── Greedy HIGH override ─────────────────────────────── #
        action = None
        if greedy_high and env._mode == "high" and env._current_slot is not None:
            target_grp = int(env._vessel_state[env._current_slot, 4])
            for i in range(env.action_space.n):
                if not mask_raw[i]:
                    continue
                key = env._action_to_stack(i)
                stk = env.yard.stacks.get(key)
                if stk and not stk.is_empty and stk.top.group == target_grp:
                    action = i
                    break

        # ── 模型决策 ─────────────────────────────────────────── #
        if action is None:
            forbidden = torch.tensor(~mask_raw.astype(bool), dtype=torch.bool).unsqueeze(0)
            obs_t     = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                enc_out     = policy.encode(obs_t)
                action_t, _ = policy(obs_t, forbidden, greedy=True, encoder_output=enc_out)
            action = int(action_t.item())

        # 记录执行前信息
        phase        = env._mode
        target_key   = env._action_to_stack(action)
        target_top   = _get_top_group(env, target_key)
        source_stack = env._source_stack
        source_top   = _get_top_group(env, source_stack) if source_stack else None
        need_grp     = (
            f"G{env._vessel_state[env._current_slot, 4]}"
            if env._current_slot is not None else "?"
        )

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_num += 1

        # ── 动作描述 ─────────────────────────────────────────── #
        src_label = f"S{target_key[0]}"
        if phase == "high":
            if target_top == need_grp:
                action_desc = (
                    f"HIGH ✓ 取走  {src_label} 顶部【{target_top}】→ 装入船"
                )
            else:
                action_desc = (
                    f"HIGH→LOW  {src_label} 顶部=【{target_top}】≠{need_grp}，"
                    f"进入 LOW 搬走阻挡"
                )
        else:
            dst_label   = f"S{target_key[0]}"
            src_hl_label = f"S{source_stack[0]}" if source_stack else "?"
            action_desc = (
                f"LOW ★RELOCATE★  "
                f"把 {src_hl_label} 顶部【{source_top}】搬到 {dst_label}  "
                f"(目标仍是{need_grp})  +1 shifter"
            )

        hl_keys = (
            {target_key} if phase == "high"
            else ({source_stack, target_key} if source_stack else {target_key})
        )

        print(f"\n{'━'*60}")
        print(f"  Step {step_num:3d}  │  {action_desc}  reward={reward:+.0f}")
        print(f"{'━'*60}")
        _print_yard_1d(env, f"Step {step_num} 后的堆场", highlight=hl_keys)
        _print_target(env)

        metrics = env.get_metrics()
        print(f"  累计 shifters={metrics['shifters']:.0f}  "
              f"已装船={metrics.get('vessel_filled', '?')}  "
              f"装船率={metrics.get('vessel_utilisation', 0)*100:.1f}%")

        if pause and not done:
            cmd = input("\n  [Enter] 下一步 / [q] 退出: ").strip().lower()
            if cmd == "q":
                print("  用户中断。")
                break

    # ── 最终汇报 ────────────────────────────────────────────── #
    final = env.get_metrics()
    print(f"\n{'═'*66}")
    print(f"  测试完成  共 {step_num} 步")
    print(f"  总 shifters (rehandles) = {final['shifters']:.0f}")
    print(f"  装船容器数              = {final.get('vessel_filled', '?')} / {env._vessel_slots}")
    print(f"  装船利用率              = {final.get('vessel_utilisation', 0)*100:.1f}%")
    print(f"{'═'*66}")


# ════════════════════════════════════════════════════════════════
#  CLI 入口
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PG-FGB × CRP-D 逐步测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 加载 dup 文件，交互模式:
  python -m algorithms.CRP_D.rl.pg_fgb.test_policy \\
      --file benchmark/dup_dataset/alpha=0.2/3-6-15/00001.txt --pause

  # 随机生成 S=6 实例:
  python -m algorithms.CRP_D.rl.pg_fgb.test_policy --stacks 6 --seed 7 --pause

  # 指定模型:
  python -m algorithms.CRP_D.rl.pg_fgb.test_policy \\
      --model algorithms/CRP_D/rl/pg_fgb/trained_models/pg_fgb_crpd.pt \\
      --file benchmark/dup_dataset/alpha=0.2/3-6-15/00001.txt --pause
""")
    parser.add_argument("--model", default=_DEFAULT_MODEL,
                        help=f"模型文件路径（默认: {_DEFAULT_MODEL}）")
    parser.add_argument("--file", default=None, dest="dup_file",
                        help="指定 dup_dataset .txt 文件（优先于随机生成）")
    parser.add_argument("--stacks",     type=int, default=6,
                        help="随机生成时的 stack 数 S（默认: 6）")
    parser.add_argument("--seed",       type=int, default=42,
                        help="随机 seed（默认: 42）")
    parser.add_argument("--tiers",      type=int, default=None,
                        help="随机生成时的最大层高（默认从 checkpoint 读取）")
    parser.add_argument("--groups",     type=int, default=None,
                        help="随机生成时的分组数（默认从 checkpoint 读取）")
    parser.add_argument("--containers", type=int, default=None,
                        help="随机生成时的箱子总数（默认从 checkpoint 读取）")
    parser.add_argument("--pause", action="store_true",
                        help="每步按 Enter 继续（交互模式）")
    parser.add_argument("--greedy-high", action="store_true", dest="greedy_high",
                        help="顶部即目标组时直接取走（不经过模型）")
    args = parser.parse_args()

    run_test(
        model_path     = args.model,
        dup_file       = args.dup_file,
        stacks         = args.stacks,
        seed           = args.seed,
        pause          = args.pause,
        greedy_high    = args.greedy_high,
        max_tiers      = args.tiers,
        num_groups     = args.groups,
        num_containers = args.containers,
    )
