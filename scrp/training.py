"""Phase 2.5 LOW-only SCRP runner and tiny REINFORCE sanity training.

This module is intentionally independent from the CRP-D training runner. It
reuses HierPolicyNetwork and its LOW action evaluation, but does not introduce
HIGH pseudo-actions, duplicate-dataset logic, or mutable environment seeds.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from hier_pg.network import HierPolicyNetwork

from .environment import SCRPEnvironment
from .models import Container, SCRPConfig, SCRPInstance
from .rl_adapter import SCRPRLAdapter
from .scenario import Scenario


SCRP_O1_FEATURE_SCALE: Tuple[float, ...] = (1.0,) * 12
SCRPEnvFactory = Callable[[], SCRPRLAdapter]


@dataclass
class SCRPLowTrajectory:
    episode_seed: int
    observations: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    action_masks: List[np.ndarray] = field(default_factory=list)
    log_probabilities: List[float] = field(default_factory=list)
    entropies: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    decision_modes: List[str] = field(default_factory=list)
    scenario_id: str = ""
    relocation_count: int = 0
    episode_return: float = 0.0
    terminated: bool = False
    truncated: bool = False
    invalid_action_count: int = 0

    @property
    def low_decisions(self) -> int:
        return len(self.actions)

    @property
    def high_decisions(self) -> int:
        return 0


@dataclass(frozen=True)
class PairedScenarioRecord:
    episode_seed: int
    policy_scenario_id: str
    baseline_scenario_id: str
    match: bool


@dataclass(frozen=True)
class SCRPSanityTrainingConfig:
    iterations: int = 5
    episodes_per_iteration: int = 4
    learning_rate: float = 2.5e-4
    entropy_coefficient: float = 0.01
    gamma: float = 1.0
    root_seed: int = 20260815
    embed_dim: int = 32
    num_encoder_layers: int = 1
    num_heads: int = 4
    ffn_dim: int = 64
    clip_constant: float = 10.0
    checkpoint_path: Optional[str] = None
    debug_assertions: bool = True

    def __post_init__(self) -> None:
        if self.iterations <= 0 or self.episodes_per_iteration <= 0:
            raise ValueError("iterations and episodes_per_iteration must be positive")
        if not 0.0 <= self.entropy_coefficient:
            raise ValueError("entropy_coefficient must be non-negative")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")


@dataclass(frozen=True)
class SCRPTrainingIterationMetrics:
    iteration: int
    episodes: int
    mean_relocations: float
    mean_reward: float
    mean_baseline_relocations: float
    mean_advantage: float
    policy_loss: float
    entropy: float
    low_decisions: int
    high_decisions: int
    invalid_action_count: int
    truncated_count: int
    scenario_mismatch_count: int


@dataclass
class SCRPSanityTrainingResult:
    policy: HierPolicyNetwork
    optimizer: torch.optim.Optimizer
    baseline_policy: HierPolicyNetwork
    iteration_metrics: List[SCRPTrainingIterationMetrics]
    paired_scenarios: List[PairedScenarioRecord]
    gradients_finite: bool
    optimizer_steps: int
    parameters_changed: bool
    checkpoint_path: Optional[str]


def make_scrp_o1_policy(
    *,
    embed_dim: int = 32,
    num_encoder_layers: int = 1,
    num_heads: int = 4,
    ffn_dim: int = 64,
    clip_constant: float = 10.0,
    device: torch.device | str = "cpu",
) -> HierPolicyNetwork:
    """Create a HierPolicyNetwork configured for normalized SCRP O1 input."""

    scale = torch.tensor(SCRP_O1_FEATURE_SCALE, dtype=torch.float32, device=device)
    return HierPolicyNetwork(
        embed_dim=embed_dim,
        num_enc_layers=num_encoder_layers,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        clip_constant=clip_constant,
        feature_scale=scale,
    ).to(device)


def run_scrp_low_episode(
    env: SCRPRLAdapter,
    policy: HierPolicyNetwork,
    episode_seed: int,
    *,
    greedy: bool,
    device: torch.device | str = "cpu",
) -> SCRPLowTrajectory:
    """Run one complete SCRP episode using LOW destination decisions only."""

    observation, info = env.reset(seed=episode_seed)
    trajectory = SCRPLowTrajectory(
        episode_seed=episode_seed,
        scenario_id=env.get_metrics()["scenario_id"],
    )

    if bool(info.get("terminated", False)):
        metrics = env.get_metrics()
        trajectory.terminated = True
        trajectory.relocation_count = metrics["relocation_count"]
        trajectory.episode_return = float(metrics["total_reward"])
        if trajectory.relocation_count != 0 or trajectory.episode_return != 0.0:
            raise AssertionError("direct-terminal reset must have zero relocations and reward")
        return trajectory

    while True:
        legal_mask = np.asarray(info["action_mask"])
        if legal_mask.dtype != np.bool_ or legal_mask.shape != (env.action_space.n,):
            raise AssertionError("SCRP action mask must be bool[S]")
        if not legal_mask.any():
            raise AssertionError("non-terminal SCRP decision has no legal action")

        observation_tensor = torch.tensor(
            observation, dtype=torch.float32, device=device
        ).unsqueeze(0)
        forbidden_mask = torch.tensor(
            ~legal_mask, dtype=torch.bool, device=device
        ).unsqueeze(0)
        with torch.no_grad():
            action_tensor, log_probability = policy.forward(
                observation_tensor,
                forbidden_mask,
                greedy=greedy,
                mode="low",
            )
            _, entropy = policy.evaluate_actions(
                observation_tensor,
                forbidden_mask,
                action_tensor,
                mode="low",
            )

        if not torch.isfinite(log_probability).all() or not torch.isfinite(entropy).all():
            raise FloatingPointError("LOW action log probability or entropy is non-finite")
        action = int(action_tensor.item())
        if not legal_mask[action]:
            trajectory.invalid_action_count += 1
            raise AssertionError(f"policy selected illegal SCRP destination {action}")

        trajectory.observations.append(observation.copy())
        trajectory.actions.append(action)
        trajectory.action_masks.append(legal_mask.copy())
        trajectory.log_probabilities.append(float(log_probability.item()))
        trajectory.entropies.append(float(entropy.item()))
        trajectory.decision_modes.append("low")

        observation, reward, terminated, truncated, info = env.step(action)
        trajectory.rewards.append(float(reward))
        if truncated:
            trajectory.truncated = True
            break
        if terminated:
            trajectory.terminated = True
            break

    metrics = env.get_metrics()
    trajectory.relocation_count = int(metrics["relocation_count"])
    trajectory.episode_return = float(sum(trajectory.rewards))
    if trajectory.terminated:
        if trajectory.episode_return != -float(metrics["shifters"]):
            raise AssertionError("step rewards do not equal -shifters")
        if trajectory.low_decisions != trajectory.relocation_count:
            raise AssertionError("LOW decision count does not equal relocation count")
    return trajectory


def _discounted_returns(rewards: Sequence[float], gamma: float) -> List[float]:
    returns: List[float] = []
    running = 0.0
    for reward in reversed(rewards):
        running = float(reward) + gamma * running
        returns.append(running)
    returns.reverse()
    return returns


def _finite_gradients(parameters) -> bool:
    return all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in parameters
    )


def run_scrp_sanity_training(
    env_factories: Sequence[SCRPEnvFactory],
    config: SCRPSanityTrainingConfig,
    *,
    device: torch.device | str = "cpu",
    policy: Optional[HierPolicyNetwork] = None,
) -> SCRPSanityTrainingResult:
    """Run a small LOW-only policy-gradient sanity training job."""

    if not env_factories:
        raise ValueError("at least one SCRP environment factory is required")
    torch.manual_seed(config.root_seed)

    policy = policy or make_scrp_o1_policy(
        embed_dim=config.embed_dim,
        num_encoder_layers=config.num_encoder_layers,
        num_heads=config.num_heads,
        ffn_dim=config.ffn_dim,
        clip_constant=config.clip_constant,
        device=device,
    )
    policy.train()
    baseline_policy = copy.deepcopy(policy).to(device)
    baseline_policy.eval()
    for parameter in baseline_policy.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate, eps=1e-5)
    initial_parameters = [parameter.detach().clone() for parameter in policy.parameters()]
    iteration_metrics: List[SCRPTrainingIterationMetrics] = []
    paired_records: List[PairedScenarioRecord] = []
    gradients_finite = True
    optimizer_steps = 0

    for iteration in range(1, config.iterations + 1):
        policy_trajectories: List[SCRPLowTrajectory] = []
        baseline_trajectories: List[SCRPLowTrajectory] = []
        raw_advantages: List[float] = []
        all_observations: List[np.ndarray] = []
        all_actions: List[int] = []
        all_masks: List[np.ndarray] = []
        all_advantages: List[float] = []
        scenario_mismatches = 0

        for episode in range(config.episodes_per_iteration):
            factory = env_factories[episode % len(env_factories)]
            episode_seed = config.root_seed * 100_000 + iteration * 1_000 + episode
            policy_env = factory()
            baseline_env = factory()
            policy_trajectory = run_scrp_low_episode(
                policy_env, policy, episode_seed, greedy=False, device=device
            )
            baseline_trajectory = run_scrp_low_episode(
                baseline_env,
                baseline_policy,
                episode_seed,
                greedy=True,
                device=device,
            )
            match = policy_trajectory.scenario_id == baseline_trajectory.scenario_id
            paired_records.append(
                PairedScenarioRecord(
                    episode_seed,
                    policy_trajectory.scenario_id,
                    baseline_trajectory.scenario_id,
                    match,
                )
            )
            if not match:
                scenario_mismatches += 1
                if config.debug_assertions:
                    raise AssertionError("policy and frozen baseline scenarios do not match")
            policy_trajectories.append(policy_trajectory)
            baseline_trajectories.append(baseline_trajectory)

            step_returns = _discounted_returns(policy_trajectory.rewards, config.gamma)
            baseline_per_step = baseline_trajectory.episode_return / max(
                policy_trajectory.low_decisions, 1
            )
            episode_advantages = [
                step_return - baseline_per_step for step_return in step_returns
            ]
            raw_advantages.extend(episode_advantages)
            all_observations.extend(policy_trajectory.observations)
            all_actions.extend(policy_trajectory.actions)
            all_masks.extend(policy_trajectory.action_masks)
            all_advantages.extend(episode_advantages)

        if not all_observations:
            raise AssertionError("sanity training iteration contains no LOW decisions")
        observation_widths = {observation.shape for observation in all_observations}
        if len(observation_widths) != 1:
            raise ValueError("one training batch must use a fixed O1 observation shape")

        advantage_tensor = torch.tensor(
            all_advantages, dtype=torch.float32, device=device
        )
        if advantage_tensor.numel() > 1 and advantage_tensor.std().item() > 1e-8:
            advantage_tensor = (
                advantage_tensor - advantage_tensor.mean()
            ) / (advantage_tensor.std() + 1e-8)
        observation_tensor = torch.tensor(
            np.asarray(all_observations), dtype=torch.float32, device=device
        )
        action_tensor = torch.tensor(all_actions, dtype=torch.long, device=device)
        forbidden_tensor = torch.tensor(
            np.asarray([~mask for mask in all_masks]),
            dtype=torch.bool,
            device=device,
        )
        log_probabilities, entropy = policy.evaluate_actions(
            observation_tensor,
            forbidden_tensor,
            action_tensor,
            mode="low",
        )
        if not torch.isfinite(log_probabilities).all() or not torch.isfinite(entropy).all():
            raise FloatingPointError("training LOW log probabilities or entropy are non-finite")
        loss = -(
            log_probabilities * advantage_tensor.detach()
        ).mean() - config.entropy_coefficient * entropy.mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("SCRP policy loss is non-finite")

        optimizer.zero_grad()
        loss.backward()
        current_gradients_finite = _finite_gradients(policy.parameters())
        gradients_finite = gradients_finite and current_gradients_finite
        if not current_gradients_finite:
            raise FloatingPointError("SCRP policy gradient contains NaN or infinity")
        optimizer.step()
        optimizer_steps += 1

        policy_relocations = [trajectory.relocation_count for trajectory in policy_trajectories]
        baseline_relocations = [
            trajectory.relocation_count for trajectory in baseline_trajectories
        ]
        iteration_metrics.append(
            SCRPTrainingIterationMetrics(
                iteration=iteration,
                episodes=len(policy_trajectories),
                mean_relocations=float(np.mean(policy_relocations)),
                mean_reward=float(np.mean([
                    trajectory.episode_return for trajectory in policy_trajectories
                ])),
                mean_baseline_relocations=float(np.mean(baseline_relocations)),
                mean_advantage=float(np.mean(raw_advantages)) if raw_advantages else 0.0,
                policy_loss=float(loss.item()),
                entropy=float(entropy.mean().item()),
                low_decisions=sum(
                    trajectory.low_decisions for trajectory in policy_trajectories
                ),
                high_decisions=0,
                invalid_action_count=sum(
                    trajectory.invalid_action_count for trajectory in policy_trajectories
                ),
                truncated_count=sum(
                    int(trajectory.truncated) for trajectory in policy_trajectories
                ),
                scenario_mismatch_count=scenario_mismatches,
            )
        )

    parameters_changed = any(
        not torch.equal(before, after.detach())
        for before, after in zip(initial_parameters, policy.parameters())
    )
    checkpoint_path = None
    if config.checkpoint_path:
        checkpoint_path = save_scrp_sanity_checkpoint(
            config.checkpoint_path,
            policy,
            optimizer,
            config,
            env_factories[0](),
        )

    return SCRPSanityTrainingResult(
        policy=policy,
        optimizer=optimizer,
        baseline_policy=baseline_policy,
        iteration_metrics=iteration_metrics,
        paired_scenarios=paired_records,
        gradients_finite=gradients_finite,
        optimizer_steps=optimizer_steps,
        parameters_changed=parameters_changed,
        checkpoint_path=checkpoint_path,
    )


def save_scrp_sanity_checkpoint(
    path: str,
    policy: HierPolicyNetwork,
    optimizer: torch.optim.Optimizer,
    config: SCRPSanityTrainingConfig,
    env: SCRPRLAdapter,
) -> str:
    """Save a Phase 2.5 checkpoint with SCRP-specific metadata."""

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "problem_type": "SCRP",
        "observation_version": "O1",
        "num_stacks": env.core_env.instance.num_stacks,
        "max_tiers": env.core_env.instance.max_tiers,
        "feature_dim": 12,
        "decision_mode": "low",
        "phase": "2.5_sanity",
        "feature_scale": list(SCRP_O1_FEATURE_SCALE),
    }
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_config": asdict(config),
            "metadata": metadata,
        },
        checkpoint_path,
    )
    return str(checkpoint_path)


class _TrainingTinyScenarioSampler:
    def sample(self, instance: SCRPInstance, root_seed: int) -> Scenario:
        return Scenario(
            root_seed=root_seed,
            order_seeds={1: 101, 2: 202},
            hidden_orders={1: (1,), 2: (3, 2)},
            scenario_id=f"scrp-training-tiny-{root_seed}",
        )


def make_scrp_training_tiny_env() -> SCRPRLAdapter:
    """Create a guaranteed-feasible hand-written environment for sanity training.

    Initial layout (bottom -> top): S0=[1,2], S1=[3], S2=[]. Moving
    blocker 2 to S2 costs one relocation overall; moving it to S1 blocks the
    revealed B2 target 3 and requires a second relocation. This gives the LOW
    policy a real, tiny action-quality distinction without using a generator.
    """

    instance = SCRPInstance(
        instance_id="phase-2.5-training-tiny",
        num_stacks=3,
        max_tiers=3,
        containers=(Container(1, 1), Container(2, 2), Container(3, 2)),
        initial_stacks=((1, 2), (3,), ()),
        batch_order=(1, 2),
        metadata={"purpose": "phase-2.5-sanity"},
    )
    core = SCRPEnvironment(
        SCRPConfig(
            num_stacks=3,
            max_tiers=3,
            root_seed=0,
            max_steps=20,
            validate_after_transition=True,
        ),
        instance,
        scenario_sampler=_TrainingTinyScenarioSampler(),
    )
    return SCRPRLAdapter(core)
