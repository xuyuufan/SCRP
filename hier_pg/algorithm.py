"""
Hierarchical PG-FGB for CRP-D  (hier_pg).

Difference from pg_fgb
-----------------------
Uses a HierPolicyNetwork with two separate pointer decoders:
  - high_decoder : decides WHICH target-group stack to process next.
  - low_decoder  : decides WHERE to relocate a blocking container.

Episode collection records mode ("high"/"low") alongside obs/action/mask.
Gradient update computes separate log-probs for each decoder, then combines:
    loss = -(lp_high * adv + lp_low * adv) / n_steps - ent_coef * (H_high + H_low)

Everything else (FGB baseline, paired t-test update, dup_dataset cycling,
fixed eval set, mixed-S training) is identical to pg_fgb.
"""

from __future__ import annotations

import copy
import multiprocessing as mp
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig

# ── re-use infrastructure from pg_fgb ─────────────────────────── #
from algorithms.CRP_D.rl.pg_fgb.algorithm import (
    _get_feature_scale,
    _scan_dup_files,
    _reconfigure_env_for_dup,
    _prepare_dup_episode,
    _build_fixed_eval_set,
    _make_env_for_fixed_eval,
    _compute_discounted_returns,
)


# ================================================================ #
#  Hierarchical episode runners                                      #
# ================================================================ #

def _greedy_high_override(env, mask_bool):
    """
    If in high-phase and the target group's container is on top of an
    allowed stack, return that action immediately (free retrieve).
    """
    if env._mode != "high" or env._current_slot is None:
        return None
    target_grp = int(env._vessel_state[env._current_slot, 4])
    for i in range(len(mask_bool)):
        if not mask_bool[i]:
            continue
        key = env._action_to_stack(i)
        stk = env.yard.stacks.get(key)
        if stk and not stk.is_empty and stk.top.group == target_grp:
            return i
    return None


def _run_episode_stochastic_hier(
    env, agent, device, greedy_high: bool = True
) -> Tuple[List, List, List, List[str], List[float], float]:
    """
    Run one episode with the hierarchical stochastic policy.

    Returns
    -------
    obs_list   : observations at each model-decision step
    act_list   : actions chosen
    mask_list  : boolean masks (False = valid)
    mode_list  : "high" or "low" for each step
    rew_list   : per-step reward
    ep_return  : -total_shifters
    """
    import torch
    obs_list, act_list, mask_list, mode_list, rew_list = [], [], [], [], []
    obs, info = env.reset()
    done = False

    while not done:
        mask_raw  = info.get("action_mask", np.ones(env.action_space.n, dtype=bool))
        mask_bool = mask_raw.astype(bool)

        if not np.any(mask_bool):
            break

        mode = getattr(env, "_mode", "high")

        # greedy_high override: if target-group container is on top, take it free
        if greedy_high and mode == "high":
            free_action = _greedy_high_override(env, mask_bool)
            if free_action is not None:
                obs, reward, terminated, truncated, info = env.step(free_action)
                done = terminated or truncated
                continue

        forbidden = torch.tensor(~mask_bool, dtype=torch.bool, device=device).unsqueeze(0)
        obs_t     = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            enc = agent.encode(obs_t)
            action_t, _ = agent.forward(obs_t, forbidden, greedy=False,
                                        enc_out=enc, mode=mode)
        action = int(action_t.item())

        obs_list.append(obs.copy())
        act_list.append(action)
        mask_list.append(mask_bool.copy())
        mode_list.append(mode)

        obs, reward, terminated, truncated, info = env.step(action)
        rew_list.append(float(reward))
        done = terminated or truncated

    ep_return = -float(env.get_metrics().get("shifters", 0.0))
    return obs_list, act_list, mask_list, mode_list, rew_list, ep_return


def _run_episode_greedy_hier(env, agent, device, greedy_high: bool = True) -> float:
    """Greedy episode. Returns -total_shifters."""
    import torch
    obs, info = env.reset()
    done = False

    while not done:
        mask_raw  = info.get("action_mask", np.ones(env.action_space.n, dtype=bool))
        mask_bool = mask_raw.astype(bool)

        if not np.any(mask_bool):
            break

        mode = getattr(env, "_mode", "high")

        if greedy_high and mode == "high":
            free_action = _greedy_high_override(env, mask_bool)
            if free_action is not None:
                obs, reward, terminated, truncated, info = env.step(free_action)
                done = terminated or truncated
                continue

        forbidden = torch.tensor(~mask_bool, dtype=torch.bool, device=device).unsqueeze(0)
        obs_t     = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            enc = agent.encode(obs_t)
            action_t, _ = agent.forward(obs_t, forbidden, greedy=True,
                                        enc_out=enc, mode=mode)
        obs, reward, terminated, truncated, info = env.step(int(action_t.item()))
        done = terminated or truncated

    return -float(env.get_metrics().get("shifters", 0.0))


# ================================================================ #
#  Algorithm class                                                   #
# ================================================================ #

class HierPgFgbD(BaseAlgorithm):
    """
    Hierarchical PG-FGB for CRP-D.

    Two-level policy:
      high_decoder → selects which target-group stack to process.
      low_decoder  → selects relocation destination for blocking containers.

    Both decoders are trained jointly with REINFORCE + Frozen Greedy Baseline.
    """

    name     = "Hier-PG-FGB-D"
    category = "RL"
    description = (
        "Hierarchical Policy Gradient with Frozen Greedy Baseline for CRP-D. "
        "Uses a shared Transformer encoder with two separate pointer decoders: "
        "one for source-stack selection (high-level) and one for relocation "
        "destination selection (low-level). "
        "Set extra['dup_train_root'] to train on dup_dataset."
    )
    compatible_problems = ["CRP-D"]
    step_label          = "Iteration"

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)
        self._policy = None

    def train(
        self,
        problem_factory: Callable,
        result_queue:    mp.Queue,
        stop_event:      mp.Event,
    ) -> None:
        try:
            import torch
            import torch.optim as optim
            import scipy.stats
        except ImportError as e:
            raise RuntimeError(f"Missing dependency: {e}.  pip install torch scipy")

        from .network import HierPolicyNetwork

        cfg    = self.config
        device = torch.device(
            "cuda" if torch.cuda.is_available() and cfg.extra.get("use_cuda", True)
            else "cpu"
        )
        torch.manual_seed(cfg.seed)
        rng = random.Random(cfg.seed)

        # ── Hyper-params ──────────────────────────────────────────── #
        num_iters      = cfg.extra.get("num_iterations",    500)
        eps_per_iter   = cfg.extra.get("episodes_per_iter",   8)
        eval_episodes  = cfg.extra.get("eval_episodes",        8)
        lr             = cfg.extra.get("learning_rate",    2.5e-4)
        ent_coef       = cfg.extra.get("ent_coef",          0.01)
        p_threshold    = cfg.extra.get("p_value_threshold", 0.05)
        gamma          = cfg.extra.get("gamma",              1.0)
        embed_dim      = cfg.extra.get("embed_dim",          128)
        num_enc_layers = cfg.extra.get("num_enc_layers",       2)
        num_heads      = cfg.extra.get("num_heads",             4)
        ffn_dim        = cfg.extra.get("ffn_dim",             256)
        clip_const     = cfg.extra.get("clip_constant",      10.0)
        report_every   = cfg.extra.get("report_every",         10)
        anneal_lr      = cfg.extra.get("anneal_lr",           True)
        greedy_high    = cfg.extra.get("greedy_high",          True)

        # ── Build env ──────────────────────────────────────────────── #
        env = problem_factory()
        env.config.seed = cfg.seed

        # ── dup_dataset cycling setup ──────────────────────────────── #
        dup_train_root = cfg.extra.get("dup_train_root", None)
        dup_files: List[Tuple[int, int, int, Path]] = []

        if dup_train_root:
            raw_stacks = cfg.extra.get("dup_num_stacks_list", None) or \
                         cfg.extra.get("dup_num_stacks", 0) or \
                         (env.config.num_bays * env.config.num_rows)
            stacks_list = [raw_stacks] if isinstance(raw_stacks, int) else list(raw_stacks)

            for s_val in stacks_list:
                dup_files.extend(_scan_dup_files(
                    dup_train_root, int(s_val),
                    target_H=cfg.extra.get("dup_tiers", None),
                    max_files_per_folder=cfg.extra.get("dup_max_files_per_folder", 0),
                ))
            if not dup_files:
                raise RuntimeError(
                    f"No dup files with S in {stacks_list} found under '{dup_train_root}'."
                )
            max_tiers_all = max(h for h, _, _, _ in dup_files)
            _reconfigure_env_for_dup(env, int(stacks_list[0]), max_tiers_all)

        current_env_S: int = env.config.num_bays
        dup_files_by_s: dict = {}
        for h, s, n, p in dup_files:
            dup_files_by_s.setdefault(s, []).append((h, s, n, p))
        available_s_values: list = sorted(dup_files_by_s.keys())

        # ── Build policy ───────────────────────────────────────────── #
        env.reset()
        feature_scale = _get_feature_scale(cfg.extra, device)
        # pad scale to INPUT_DIM=12 if only 5 values provided
        if feature_scale.shape[0] < HierPolicyNetwork.INPUT_DIM:
            pad = torch.ones(
                HierPolicyNetwork.INPUT_DIM - feature_scale.shape[0],
                dtype=torch.float32, device=device
            )
            feature_scale = torch.cat([feature_scale, pad])

        policy = HierPolicyNetwork(
            embed_dim      = embed_dim,
            num_enc_layers = num_enc_layers,
            num_heads      = num_heads,
            ffn_dim        = ffn_dim,
            clip_constant  = clip_const,
            feature_scale  = feature_scale,
        ).to(device)
        policy.train()

        baseline = copy.deepcopy(policy).to(device)
        baseline.eval()

        optimizer            = optim.Adam(policy.parameters(), lr=lr, eps=1e-5)
        best_greedy_shifters = float("inf")
        self._policy         = policy
        baseline_updates     = 0

        for iteration in range(1, num_iters + 1):
            if stop_event.is_set():
                break

            if anneal_lr:
                frac = 1.0 - (iteration - 1.0) / num_iters
                for pg in optimizer.param_groups:
                    pg["lr"] = frac * lr

            # ── Collect episodes (per-iteration S selection) ──────── #
            policy.train()
            all_obs_high:   List = []
            all_acts_high:  List = []
            all_masks_high: List = []
            adv_high:       List[float] = []

            all_obs_low:    List = []
            all_acts_low:   List = []
            all_masks_low:  List = []
            adv_low:        List[float] = []

            policy_returns:   List[float] = []
            baseline_returns: List[float] = []

            if available_s_values:
                iter_S     = rng.choice(available_s_values)
                iter_files = dup_files_by_s[iter_S]
                if iter_S != current_env_S:
                    _reconfigure_env_for_dup(env, iter_S,
                                             max(h for h, _, _, _ in dup_files))
                    current_env_S = iter_S
            else:
                iter_files = dup_files

            for ep in range(eps_per_iter):
                seed = cfg.seed * 100_000 + iteration * 1000 + ep

                if iter_files:
                    H, S, N, path = rng.choice(iter_files)
                    _prepare_dup_episode(env, H, S, N, path)
                else:
                    env.config.extra.pop("layout_file_path", None)

                env.config.seed = seed
                (obs_ep, act_ep, mask_ep,
                 mode_ep, rew_ep, ret_pol) = _run_episode_stochastic_hier(
                    env, policy, device, greedy_high=greedy_high
                )

                env.config.seed = seed
                ret_bl = _run_episode_greedy_hier(
                    env, baseline, device, greedy_high=greedy_high
                )

                policy_returns.append(ret_pol)
                baseline_returns.append(ret_bl)

                if rew_ep:
                    step_returns = _compute_discounted_returns(rew_ep, gamma)
                    per_step_bl  = ret_bl / max(len(rew_ep), 1)
                    adv_steps    = [g - per_step_bl for g in step_returns]
                else:
                    adv_steps = []

                # split by mode
                for i, mode in enumerate(mode_ep):
                    a = adv_steps[i] if i < len(adv_steps) else 0.0
                    if mode == "high":
                        all_obs_high.append(obs_ep[i])
                        all_acts_high.append(act_ep[i])
                        all_masks_high.append(mask_ep[i])
                        adv_high.append(a)
                    else:
                        all_obs_low.append(obs_ep[i])
                        all_acts_low.append(act_ep[i])
                        all_masks_low.append(mask_ep[i])
                        adv_low.append(a)

            if not all_obs_high and not all_obs_low:
                continue

            # ── REINFORCE update (both decoders jointly) ──────────── #
            def _normalise(adv_list: List[float]) -> "torch.Tensor":
                t = torch.tensor(adv_list, dtype=torch.float32, device=device)
                if t.std() > 1e-8:
                    t = (t - t.mean()) / (t.std() + 1e-8)
                return t

            pg_loss = torch.tensor(0.0, device=device)
            ent_acc = torch.tensor(0.0, device=device)
            n_steps = 0

            if all_obs_high:
                obs_h  = torch.tensor(np.array(all_obs_high),  dtype=torch.float32, device=device)
                acts_h = torch.tensor(all_acts_high,            dtype=torch.long,    device=device)
                mask_h = torch.tensor(
                    np.array([~m.astype(bool) for m in all_masks_high]),
                    dtype=torch.bool, device=device
                )
                adv_h = _normalise(adv_high).detach()
                lp_h, ent_h = policy.evaluate_actions(obs_h, mask_h, acts_h, mode="high")
                pg_loss = pg_loss - (lp_h * adv_h).mean()
                ent_acc = ent_acc + ent_h.mean()
                n_steps += len(all_obs_high)

            if all_obs_low:
                obs_l  = torch.tensor(np.array(all_obs_low),   dtype=torch.float32, device=device)
                acts_l = torch.tensor(all_acts_low,             dtype=torch.long,    device=device)
                mask_l = torch.tensor(
                    np.array([~m.astype(bool) for m in all_masks_low]),
                    dtype=torch.bool, device=device
                )
                adv_l = _normalise(adv_low).detach()
                lp_l, ent_l = policy.evaluate_actions(obs_l, mask_l, acts_l, mode="low")
                pg_loss = pg_loss - (lp_l * adv_l).mean()
                ent_acc = ent_acc + ent_l.mean()
                n_steps += len(all_obs_low)

            total_loss = pg_loss - ent_coef * ent_acc
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

            # ── Baseline update (paired t-test) ───────────────────── #
            if len(policy_returns) > 1:
                t_stat, p_val = scipy.stats.ttest_rel(policy_returns, baseline_returns)
                if t_stat > 0 and p_val / 2 < p_threshold:
                    baseline = copy.deepcopy(policy).to(device)
                    baseline.eval()
                    baseline_updates += 1

            # ── Greedy eval & reporting ───────────────────────────── #
            if iteration % report_every == 0 or iteration == num_iters:
                policy.eval()
                fixed_eval_set = cfg.extra.get("fixed_eval_set", None)

                if fixed_eval_set:
                    sh_s6, sh_ood = [], []
                    for H_e, S_e, N_e, path_e in fixed_eval_set:
                        ev_env = _make_env_for_fixed_eval(H_e, S_e, N_e, path_e)
                        ev_env.config.seed = cfg.seed
                        gs = -_run_episode_greedy_hier(ev_env, policy, device,
                                                       greedy_high=greedy_high)
                        (sh_s6 if S_e == int(stacks_list[0]) else sh_ood).append(gs)
                    greedy_shifters = sh_s6 + sh_ood
                    eval_s6  = float(np.mean(sh_s6))  if sh_s6  else float("nan")
                    eval_ood = float(np.mean(sh_ood)) if sh_ood else float("nan")
                else:
                    greedy_shifters = []
                    for ev in range(eval_episodes):
                        if iter_files:
                            H, S, N, path = rng.choice(iter_files)
                            _prepare_dup_episode(env, H, S, N, path)
                        env.config.seed = cfg.seed * 100_000 + iteration * 1000 + ev + 90000
                        gs = _run_episode_greedy_hier(env, policy, device, greedy_high=greedy_high)
                        greedy_shifters.append(-gs)
                    eval_s6 = eval_ood = float("nan")

                policy.train()

                mean_sh = float(np.mean(greedy_shifters)) if greedy_shifters else float("nan")
                if mean_sh < best_greedy_shifters:
                    best_greedy_shifters = mean_sh

                self._push(
                    result_queue,
                    step     = iteration,
                    metric   = mean_sh,
                    metrics  = {
                        "shifters":         mean_sh,
                        "best_shifters":    best_greedy_shifters,
                        "policy_loss":      float(total_loss.item()),
                        "entropy":          float(ent_acc.item()),
                        "baseline_updates": float(baseline_updates),
                        "mean_advantage":   float(np.mean([
                            r - b for r, b in zip(policy_returns, baseline_returns)
                        ])),
                        "dup_pool_size":    float(len(dup_files)),
                        "shifters_s6":      eval_s6,
                        "shifters_ood":     eval_ood,
                        "high_steps":       float(len(all_obs_high)),
                        "low_steps":        float(len(all_obs_low)),
                    },
                    progress = iteration / num_iters,
                    snapshot = env.get_state_snapshot(),
                )

        self._policy = policy

        # ── Save ──────────────────────────────────────────────────── #
        import torch as _torch
        save_path = cfg.extra.get("save_path", None)
        if save_path is None:
            _algo_dir = Path(__file__).parent / "trained_models"
            _algo_dir.mkdir(exist_ok=True)
            save_path = str(_algo_dir / "hier_pg_crpd.pt")

        _torch.save({
            "policy_state_dict": policy.state_dict(),
            "embed_dim":      embed_dim,
            "num_enc_layers": num_enc_layers,
            "num_heads":      num_heads,
            "ffn_dim":        ffn_dim,
            "clip_constant":  clip_const,
            "feature_scale":  policy.feature_scale.cpu().tolist(),
            "best_shifters":  best_greedy_shifters,
            "num_bays":       current_env_S,
            "num_rows":       env.config.num_rows,
            "max_tiers":      env.config.max_tiers,
        }, save_path)
        print(f"[Hier-PG-FGB-D] Policy saved → {save_path}  "
              f"(best greedy shifters={best_greedy_shifters:.3f})")

    def get_best_solution(self) -> Optional[List]:
        return None

    @classmethod
    def config_schema(cls) -> Dict:
        return {
            "num_iterations":    {"type": "int",   "default": 1000,   "label": "Training iterations"},
            "episodes_per_iter": {"type": "int",   "default": 16,     "label": "Episodes per iteration"},
            "eval_episodes":     {"type": "int",   "default": 16,     "label": "Greedy eval episodes"},
            "learning_rate":     {"type": "float", "default": 2.5e-4, "label": "Learning rate"},
            "ent_coef":          {"type": "float", "default": 0.01,   "label": "Entropy coefficient"},
            "embed_dim":         {"type": "int",   "default": 128,    "label": "Embedding dim"},
            "num_enc_layers":    {"type": "int",   "default": 2,      "label": "Encoder layers"},
        }
