"""Phase 6 training-readiness implementation for LOW-only SCRP.

This module is deliberately separate from the Phase 2.5 sanity trainer.  It
implements the audited Frozen Greedy Baseline semantics, train-only sampling,
variable-S buckets, O1/O2 construction, and resumable checkpoint state.  It is
not a command to start a formal training campaign.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import re
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import ttest_rel

from experiments.protocol import (
    BaseInstanceRef,
    DEFAULT_DATASET_VERSION,
    SPLIT_PROTOCOL_VERSION,
    ScenarioSeedSchedule,
    SplitManifest,
)
from hier_pg.network import HierPolicyNetwork

from .datasets import merge_adjacent_batches, parse_ku_crptw
from .environment import SCRPEnvironment
from .models import SCRPConfig, SCRPInstance
from .observation import SCRP_O2_FEATURE_SCALE, SCRP_O2_MMAX
from .rl_adapter import SCRPRLAdapter
from .training import SCRP_O1_FEATURE_SCALE


TRAINING_PROTOCOL_VERSION = "scrp-training-protocol-v1"
_STACKS_PATTERN = re.compile(r"^S(?P<stacks>\d+)_")


@dataclass(frozen=True)
class FormalTrainingConfig:
    training_protocol_version: str = TRAINING_PROTOCOL_VERSION
    dataset_version: str = DEFAULT_DATASET_VERSION
    split_manifest_version: str = SPLIT_PROTOCOL_VERSION
    observation_version: str = "O2"
    Mmax: int | None = SCRP_O2_MMAX
    feature_dim: int = 12
    training_strategy: str = "mixed_ds1_ds2_base_balanced_bucket_by_S"
    optimizer: str = "Adam"
    learning_rate: float = 2.5e-4
    gamma: float = 1.0
    entropy_coeff: float = 0.01
    batch_size: int = 4
    gradient_clip: float = 0.5
    baseline_type: str = "frozen_greedy_policy"
    baseline_update_rule: str = "paired_one_sided_t_test_p_lt_0.05"
    seed: int = 20260816
    device: str = "cpu"
    checkpoint_interval: int = 10
    validation_interval: int = 10
    embed_dim: int = 32
    num_encoder_layers: int = 1
    num_heads: int = 4
    ffn_dim: int = 64
    clip_constant: float = 10.0
    max_steps: int = 10000
    hyperparameter_status: str = "NOT FINAL HYPERPARAMETERS"

    def __post_init__(self) -> None:
        if self.training_protocol_version != TRAINING_PROTOCOL_VERSION:
            raise ValueError("unsupported training protocol version")
        if self.observation_version not in {"O1", "O2"}:
            raise ValueError("observation_version must be O1 or O2")
        if self.observation_version == "O1" and self.Mmax is not None:
            raise ValueError("O1 config must use Mmax=null")
        if self.observation_version == "O2" and self.Mmax != SCRP_O2_MMAX:
            raise ValueError(f"O2 formal config requires audited Mmax={SCRP_O2_MMAX}")
        if self.feature_dim != 12 or self.batch_size <= 0:
            raise ValueError("feature_dim must be 12 and batch_size must be positive")
        if not 0.0 < self.gamma <= 1.0 or self.gradient_clip <= 0.0:
            raise ValueError("invalid gamma or gradient_clip")
        if self.hyperparameter_status not in {
            "NOT FINAL HYPERPARAMETERS",
            "CANDIDATE_FOR_REHEARSAL",
        }:
            raise ValueError("unsupported hyperparameter lifecycle status")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> "FormalTrainingConfig":
        expected = set(asdict(cls()))
        if set(record) != expected:
            raise ValueError("formal training config keys mismatch")
        return cls(**record)


def load_formal_training_config(path: str | Path) -> FormalTrainingConfig:
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read formal training config: {error}") from error
    return FormalTrainingConfig.from_record(record)


def save_formal_training_config(
    config: FormalTrainingConfig, path: str | Path
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")
    return destination


def make_scrp_policy(
    observation_version: str,
    num_stacks: int,
    max_tiers: int,
    *,
    Mmax: int = SCRP_O2_MMAX,
    embed_dim: int = 32,
    num_encoder_layers: int = 1,
    num_heads: int = 4,
    ffn_dim: int = 64,
    clip_constant: float = 10.0,
    device: torch.device | str = "cpu",
) -> HierPolicyNetwork:
    """Build an O1/O2 policy without requiring callers to calculate shapes."""

    if num_stacks <= 0 or max_tiers <= 0:
        raise ValueError("num_stacks and max_tiers must be positive")
    if observation_version == "O1":
        num_nodes = num_stacks + 1
        feature_scale = SCRP_O1_FEATURE_SCALE
        policy_mmax = None
    elif observation_version == "O2":
        if Mmax != SCRP_O2_MMAX:
            raise ValueError(f"formal O2 policy requires Mmax={SCRP_O2_MMAX}")
        num_nodes = num_stacks + Mmax + 1
        feature_scale = SCRP_O2_FEATURE_SCALE
        policy_mmax = Mmax
    else:
        raise ValueError("observation_version must be O1 or O2")
    policy = HierPolicyNetwork(
        embed_dim=embed_dim,
        num_enc_layers=num_encoder_layers,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        clip_constant=clip_constant,
        feature_scale=torch.tensor(feature_scale, dtype=torch.float32, device=device),
    ).to(device)
    policy.scrp_observation_version = observation_version
    policy.scrp_num_nodes = num_nodes
    policy.scrp_feature_dim = 12
    policy.scrp_candidate_count = num_stacks
    policy.scrp_num_stacks = num_stacks
    policy.scrp_max_tiers = max_tiers
    policy.scrp_mmax = policy_mmax
    return policy


def make_node_padding_mask(
    observations: torch.Tensor,
    observation_version: str,
    num_stacks: int,
    *,
    Mmax: int = SCRP_O2_MMAX,
) -> torch.Tensor | None:
    """Return True only for O2 order-padding nodes; O1 deliberately returns None."""

    if observation_version == "O1":
        return None
    if observation_version != "O2":
        raise ValueError("observation_version must be O1 or O2")
    nodes = observations.reshape(observations.shape[0], num_stacks + Mmax + 1, 12)
    mask = torch.zeros(nodes.shape[:2], dtype=torch.bool, device=nodes.device)
    mask[:, num_stacks : num_stacks + Mmax] = (
        nodes[:, num_stacks : num_stacks + Mmax, 11] > 0.5
    )
    return mask


@dataclass(frozen=True)
class TrainingSample:
    base_instance_id: str
    instance_id: str
    variant: str
    scenario_seed: int
    visit_index: int
    num_stacks: int


def _num_stacks(ref: BaseInstanceRef) -> int:
    match = _STACKS_PATTERN.match(ref.parameter_group)
    if not match:
        raise ValueError(f"cannot parse stack bucket from {ref.parameter_group!r}")
    return int(match.group("stacks"))


class BaseBalancedTrainingSampler:
    """Train-only sampler: base first, then DS1/DS2, then dynamic seed."""

    def __init__(
        self,
        manifest: SplitManifest,
        root_seed: int,
        *,
        allowed_base_ids: Sequence[str] | None = None,
    ) -> None:
        self.manifest = manifest
        allowed = None if allowed_base_ids is None else set(allowed_base_ids)
        self.refs = tuple(
            ref for ref in manifest.refs("train")
            if allowed is None or ref.base_instance_id in allowed
        )
        if not self.refs:
            raise ValueError("training sampler has no train-split base instances")
        self.root_seed = root_seed
        self.rng = random.Random(root_seed)
        self.schedule = ScenarioSeedSchedule(manifest)
        self.visit_counts = {ref.base_instance_id: 0 for ref in self.refs}
        self.by_stacks: dict[int, tuple[BaseInstanceRef, ...]] = {}
        for stacks in sorted({_num_stacks(ref) for ref in self.refs}):
            self.by_stacks[stacks] = tuple(
                ref for ref in self.refs if _num_stacks(ref) == stacks
            )

    def _materialize(self, ref: BaseInstanceRef) -> TrainingSample:
        variant = self.rng.choice(("DS1", "DS2"))
        visit = self.visit_counts[ref.base_instance_id]
        self.visit_counts[ref.base_instance_id] = visit + 1
        return TrainingSample(
            base_instance_id=ref.base_instance_id,
            instance_id=ref.ds1_instance_id if variant == "DS1" else ref.ds2_instance_id,
            variant=variant,
            scenario_seed=self.schedule.seed_for("train", ref.base_instance_id, visit),
            visit_index=visit,
            num_stacks=_num_stacks(ref),
        )

    def sample(self) -> TrainingSample:
        return self._materialize(self.rng.choice(self.refs))

    def sample_same_stack(self, num_stacks: int) -> TrainingSample:
        if num_stacks not in self.by_stacks:
            raise ValueError(f"no training bases for S={num_stacks}")
        return self._materialize(self.rng.choice(self.by_stacks[num_stacks]))

    def sample_bucket(self, batch_size: int) -> tuple[TrainingSample, ...]:
        first = self.sample()
        return (first,) + tuple(
            self.sample_same_stack(first.num_stacks) for _ in range(batch_size - 1)
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "root_seed": self.root_seed,
            "rng_state": self.rng.getstate(),
            "visit_counts": dict(self.visit_counts),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if state["root_seed"] != self.root_seed:
            raise ValueError("sampler root seed mismatch")
        counts = dict(state["visit_counts"])
        if set(counts) != set(self.visit_counts):
            raise ValueError("sampler visit-counter keys mismatch")
        self.visit_counts = {key: int(value) for key, value in counts.items()}
        self.rng.setstate(state["rng_state"])


class KuTrainingInstanceProvider:
    """Resolve manifest IDs to exact local Ku files without copying the corpus."""

    def __init__(self, source_root: str | Path) -> None:
        root = Path(source_root)
        self.paths = {path.stem: path for path in root.rglob("*.txt")}
        if not self.paths:
            raise ValueError(f"no Ku source instances under {root}")

    def __call__(self, sample: TrainingSample) -> SCRPInstance:
        try:
            ds1 = parse_ku_crptw(self.paths[sample.base_instance_id])
        except KeyError as error:
            raise ValueError(f"missing source instance {sample.base_instance_id}") from error
        return ds1 if sample.variant == "DS1" else merge_adjacent_batches(ds1)


@dataclass
class FormalTrajectory:
    sample: TrainingSample
    observations: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    legal_masks: list[np.ndarray] = field(default_factory=list)
    node_padding_masks: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    scenario_id: str = ""
    relocations: int = 0
    terminated: bool = False
    truncated: bool = False
    invalid_actions: int = 0

    @property
    def episode_return(self) -> float:
        return float(sum(self.rewards))


def run_formal_episode(
    instance: SCRPInstance,
    sample: TrainingSample,
    policy: HierPolicyNetwork,
    config: FormalTrainingConfig,
    *,
    greedy: bool,
    device: torch.device | str = "cpu",
) -> FormalTrajectory:
    core = SCRPEnvironment(
        SCRPConfig(
            instance.num_stacks,
            instance.max_tiers,
            root_seed=config.seed,
            max_steps=config.max_steps,
            validate_after_transition=True,
        ),
        instance,
    )
    env = SCRPRLAdapter(
        core,
        observation_version=config.observation_version,
        o2_mmax=config.Mmax or SCRP_O2_MMAX,
    )
    observation, info = env.reset(seed=sample.scenario_seed)
    trajectory = FormalTrajectory(sample=sample, scenario_id=core.scenario_id)
    if info.get("terminated", False):
        trajectory.terminated = True
        return trajectory
    while True:
        legal = np.asarray(info["action_mask"], dtype=bool)
        if not legal.any():
            raise AssertionError("non-terminal SCRP state has no legal action")
        obs_t = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        forbidden = torch.tensor(~legal, dtype=torch.bool, device=device).unsqueeze(0)
        node_mask = make_node_padding_mask(
            obs_t, config.observation_version, instance.num_stacks,
            Mmax=config.Mmax or SCRP_O2_MMAX,
        )
        with torch.no_grad():
            action_t, _ = policy.forward(
                obs_t, forbidden, greedy=greedy, mode="low",
                node_padding_mask=node_mask,
            )
        action = int(action_t.item())
        if not legal[action]:
            trajectory.invalid_actions += 1
            raise AssertionError("policy selected an illegal stack action")
        trajectory.observations.append(observation.copy())
        trajectory.actions.append(action)
        trajectory.legal_masks.append(legal.copy())
        if node_mask is not None:
            trajectory.node_padding_masks.append(node_mask[0].cpu().numpy().copy())
        observation, reward, terminated, truncated, info = env.step(action)
        trajectory.rewards.append(float(reward))
        if terminated or truncated:
            trajectory.terminated = bool(terminated)
            trajectory.truncated = bool(truncated)
            break
    trajectory.relocations = int(env.get_metrics()["relocation_count"])
    return trajectory


def discounted_returns(rewards: Sequence[float], gamma: float) -> list[float]:
    result: list[float] = []
    running = 0.0
    for reward in reversed(rewards):
        running = float(reward) + gamma * running
        result.append(running)
    return list(reversed(result))


def frozen_greedy_advantages(
    policy_rewards: Sequence[float], baseline_episode_return: float, gamma: float
) -> list[float]:
    """Exact original hier_pg formula: A_t=G_t-G_b/|policy decisions|."""

    if not policy_rewards:
        return []
    baseline_per_step = baseline_episode_return / len(policy_rewards)
    return [value - baseline_per_step for value in discounted_returns(policy_rewards, gamma)]


@dataclass(frozen=True)
class FormalIterationMetrics:
    iteration: int
    episodes: int
    mean_policy_relocations: float
    mean_baseline_relocations: float
    mean_return: float
    mean_advantage: float
    loss: float
    policy_loss: float
    entropy: float
    grad_norm: float
    invalid_actions: int
    truncations: int
    baseline_updates: int
    low_decisions: int
    empty_decision_episodes: int
    scenario_mismatches: int


@dataclass(frozen=True)
class BaselineRefreshRecord:
    iteration: int
    sample_size: int
    paired_mean_difference: float
    t_statistic: float
    p_value: float
    old_baseline_state_sha256: str
    new_baseline_state_sha256: str


def policy_state_sha256(policy: HierPolicyNetwork) -> str:
    """Hash tensor content without creating a temporary checkpoint file."""

    digest = hashlib.sha256()
    for name, tensor in sorted(policy.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class SCRPFormalTrainer:
    """Auditable trainer used only for Phase 6 bounded readiness checks."""

    def __init__(
        self,
        config: FormalTrainingConfig,
        manifest: SplitManifest,
        instance_provider: Callable[[TrainingSample], SCRPInstance],
        *,
        allowed_base_ids: Sequence[str] | None = None,
        policy: HierPolicyNetwork | None = None,
    ) -> None:
        self.config = config
        self.manifest = manifest
        self.instance_provider = instance_provider
        self.allowed_base_ids = None if allowed_base_ids is None else tuple(allowed_base_ids)
        self.device = torch.device(config.device)
        torch.manual_seed(config.seed)
        self.sampler = BaseBalancedTrainingSampler(
            manifest, config.seed, allowed_base_ids=allowed_base_ids
        )
        example_ref = self.sampler.refs[0]
        example_stacks = _num_stacks(example_ref)
        self.policy = policy or make_scrp_policy(
            config.observation_version, example_stacks, 1,
            Mmax=config.Mmax or SCRP_O2_MMAX,
            embed_dim=config.embed_dim,
            num_encoder_layers=config.num_encoder_layers,
            num_heads=config.num_heads,
            ffn_dim=config.ffn_dim,
            clip_constant=config.clip_constant,
            device=self.device,
        )
        self.policy.train()
        self.baseline_policy = self._frozen_copy(self.policy)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config.learning_rate, eps=1e-5
        )
        self.iteration = 0
        self.episodes_seen = 0
        self.baseline_updates = 0
        self.baseline_refresh_history: list[BaselineRefreshRecord] = []
        self.metrics: list[FormalIterationMetrics] = []
        self.sample_history: list[TrainingSample] = []

    def _frozen_copy(self, policy: HierPolicyNetwork) -> HierPolicyNetwork:
        baseline = copy.deepcopy(policy).to(self.device)
        baseline.eval()
        for parameter in baseline.parameters():
            parameter.requires_grad_(False)
        return baseline

    def train_iterations(self, count: int) -> list[FormalIterationMetrics]:
        if count <= 0:
            raise ValueError("iteration count must be positive")
        new_metrics = []
        for _ in range(count):
            self.iteration += 1
            samples = self.sampler.sample_bucket(self.config.batch_size)
            if len({sample.num_stacks for sample in samples}) != 1:
                raise AssertionError("variable-S batch was not bucketed")
            self.sample_history.extend(samples)
            policy_runs, baseline_runs = [], []
            observations, actions, masks, node_masks, advantages = [], [], [], [], []
            policy_returns, baseline_returns = [], []
            for sample in samples:
                instance = self.instance_provider(sample)
                policy_run = run_formal_episode(
                    instance, sample, self.policy, self.config,
                    greedy=False, device=self.device,
                )
                baseline_run = run_formal_episode(
                    instance, sample, self.baseline_policy, self.config,
                    greedy=True, device=self.device,
                )
                if policy_run.scenario_id != baseline_run.scenario_id:
                    raise AssertionError("policy/baseline scenario_id mismatch")
                episode_adv = frozen_greedy_advantages(
                    policy_run.rewards, baseline_run.episode_return, self.config.gamma
                )
                observations.extend(policy_run.observations)
                actions.extend(policy_run.actions)
                masks.extend(policy_run.legal_masks)
                node_masks.extend(policy_run.node_padding_masks)
                advantages.extend(episode_adv)
                policy_runs.append(policy_run)
                baseline_runs.append(baseline_run)
                policy_returns.append(policy_run.episode_return)
                baseline_returns.append(baseline_run.episode_return)
            self.episodes_seen += len(samples)

            loss_value = policy_loss_value = entropy_value = grad_norm_value = 0.0
            if observations:
                advantage_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
                if advantage_t.numel() > 1 and advantage_t.std().item() > 1e-8:
                    advantage_t = (advantage_t - advantage_t.mean()) / (
                        advantage_t.std() + 1e-8
                    )
                observation_t = torch.tensor(
                    np.asarray(observations), dtype=torch.float32, device=self.device
                )
                action_t = torch.tensor(actions, dtype=torch.long, device=self.device)
                forbidden_t = torch.tensor(
                    np.asarray([~mask for mask in masks]),
                    dtype=torch.bool, device=self.device,
                )
                node_mask_t = None
                if self.config.observation_version == "O2":
                    node_mask_t = torch.tensor(
                        np.asarray(node_masks), dtype=torch.bool, device=self.device
                    )
                log_prob, entropy = self.policy.evaluate_actions(
                    observation_t, forbidden_t, action_t, mode="low",
                    node_padding_mask=node_mask_t,
                )
                policy_loss = -(log_prob * advantage_t.detach()).mean()
                loss = policy_loss - self.config.entropy_coeff * entropy.mean()
                if not torch.isfinite(loss) or not torch.isfinite(entropy).all():
                    raise FloatingPointError("formal sanity loss/entropy is non-finite")
                self.optimizer.zero_grad()
                loss.backward()
                if not all(
                    parameter.grad is None or torch.isfinite(parameter.grad).all()
                    for parameter in self.policy.parameters()
                ):
                    raise FloatingPointError("formal sanity gradient is non-finite")
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.config.gradient_clip
                )
                self.optimizer.step()
                loss_value = float(loss.item())
                policy_loss_value = float(policy_loss.item())
                entropy_value = float(entropy.mean().item())
                grad_norm_value = float(grad_norm.item())

            if len(policy_returns) > 1:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    result = ttest_rel(policy_returns, baseline_returns)
                if (
                    np.isfinite(result.statistic)
                    and result.statistic > 0
                    and result.pvalue / 2 < 0.05
                ):
                    old_baseline_hash = policy_state_sha256(self.baseline_policy)
                    self.baseline_policy = self._frozen_copy(self.policy)
                    self.baseline_updates += 1
                    self.baseline_refresh_history.append(
                        BaselineRefreshRecord(
                            iteration=self.iteration,
                            sample_size=len(policy_returns),
                            paired_mean_difference=float(np.mean(
                                np.asarray(policy_returns) - np.asarray(baseline_returns)
                            )),
                            t_statistic=float(result.statistic),
                            p_value=float(result.pvalue / 2),
                            old_baseline_state_sha256=old_baseline_hash,
                            new_baseline_state_sha256=policy_state_sha256(
                                self.baseline_policy
                            ),
                        )
                    )

            metric = FormalIterationMetrics(
                iteration=self.iteration,
                episodes=len(samples),
                mean_policy_relocations=float(np.mean([r.relocations for r in policy_runs])),
                mean_baseline_relocations=float(np.mean([r.relocations for r in baseline_runs])),
                mean_return=float(np.mean(policy_returns)),
                mean_advantage=float(np.mean(advantages)) if advantages else 0.0,
                loss=loss_value,
                policy_loss=policy_loss_value,
                entropy=entropy_value,
                grad_norm=grad_norm_value,
                invalid_actions=sum(r.invalid_actions for r in policy_runs),
                truncations=sum(int(r.truncated) for r in policy_runs),
                baseline_updates=self.baseline_updates,
                low_decisions=sum(len(r.actions) for r in policy_runs),
                empty_decision_episodes=sum(not r.actions for r in policy_runs),
                scenario_mismatches=0,
            )
            self.metrics.append(metric)
            new_metrics.append(metric)
        return new_metrics

    def save_checkpoint(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "iteration": self.iteration,
            "episodes_seen": self.episodes_seen,
            "root_seed": self.config.seed,
            "torch_rng_state": torch.get_rng_state(),
            "observation_version": self.config.observation_version,
            "feature_dim": self.config.feature_dim,
            "Mmax": self.config.Mmax,
            "S_bucket_metadata": sorted(self.sampler.by_stacks),
            "dataset_version": self.config.dataset_version,
            "split_manifest_version": self.config.split_manifest_version,
            "training_protocol_version": self.config.training_protocol_version,
            "baseline_type": self.config.baseline_type,
            "baseline_state": self.baseline_policy.state_dict(),
            "baseline_updates": self.baseline_updates,
            "baseline_refresh_history": [
                asdict(record) for record in self.baseline_refresh_history
            ],
            "per_base_visit_counters": dict(self.sampler.visit_counts),
            "sampler_state": self.sampler.state_dict(),
            "config_snapshot": asdict(self.config),
        }
        torch.save(checkpoint, destination)
        return destination

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        manifest: SplitManifest,
        instance_provider: Callable[[TrainingSample], SCRPInstance],
        *,
        allowed_base_ids: Sequence[str] | None = None,
    ) -> "SCRPFormalTrainer":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = FormalTrainingConfig.from_record(checkpoint["config_snapshot"])
        trainer = cls(
            config, manifest, instance_provider, allowed_base_ids=allowed_base_ids
        )
        trainer.policy.load_state_dict(checkpoint["model_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.baseline_policy.load_state_dict(checkpoint["baseline_state"])
        trainer.sampler.load_state_dict(checkpoint["sampler_state"])
        if checkpoint["per_base_visit_counters"] != trainer.sampler.visit_counts:
            raise ValueError("checkpoint visit counters disagree with sampler state")
        trainer.iteration = int(checkpoint["iteration"])
        trainer.episodes_seen = int(checkpoint["episodes_seen"])
        trainer.baseline_updates = int(checkpoint["baseline_updates"])
        trainer.baseline_refresh_history = [
            BaselineRefreshRecord(**record)
            for record in checkpoint.get("baseline_refresh_history", [])
        ]
        torch.set_rng_state(checkpoint["torch_rng_state"])
        return trainer
