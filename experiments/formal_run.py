"""Phase 7B first formal SCRP training orchestration.

This module owns run identity validation, fixed validation evaluation,
checkpoint retention, and compact monitoring. It never evaluates the test
split and never invokes ERI.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from experiments.protocol import (
    DEFAULT_DATASET_VERSION,
    SPLIT_PROTOCOL_VERSION,
    ScenarioSeedSchedule,
    SplitManifest,
)
from scrp.formal_training import (
    FormalIterationMetrics,
    KuTrainingInstanceProvider,
    SCRPFormalTrainer,
    TrainingSample,
    load_formal_training_config,
    policy_state_sha256,
    run_formal_episode,
)


RUN_ID = "formal-o2-mixed-seed20260816-run1"
CANDIDATE_CONFIG_PATH = "experiments/configs/training_protocol_v1_candidate.json"
CANDIDATE_CONFIG_SHA256 = (
    "b284e0d1c9a8d750e4314b4bd7323af45f79d7cf2be3401ca22ec8b742793be3"
)
SELECTION_METRIC_NAME = "equal_variant_equal_instance_mean_relocations"


@dataclass(frozen=True)
class FormalRunIdentity:
    run_id: str
    code_sha: str
    config_path: str
    config_sha256: str
    split_manifest_version: str
    dataset_version: str
    observation_version: str
    Mmax: int
    root_seed: int
    planned_episodes: int
    validation_cadence_episodes: int
    validation_scenarios_per_static_variant: int
    checkpoint_selection_metric: str
    durable_milestone_episodes: tuple[int, ...]
    formal_test_episode_usage: int = 0

    def __post_init__(self) -> None:
        if self.run_id != RUN_ID:
            raise ValueError("unexpected formal run ID")
        if len(self.code_sha) != 40:
            raise ValueError("code_sha must be a full Git SHA")
        if self.config_path != CANDIDATE_CONFIG_PATH:
            raise ValueError("formal run must use the approved candidate path")
        if self.config_sha256 != CANDIDATE_CONFIG_SHA256:
            raise ValueError("candidate config hash mismatch")
        if self.split_manifest_version != SPLIT_PROTOCOL_VERSION:
            raise ValueError("split manifest version mismatch")
        if self.dataset_version != DEFAULT_DATASET_VERSION:
            raise ValueError("dataset version mismatch")
        if self.observation_version != "O2" or self.Mmax != 6:
            raise ValueError("first formal run is frozen to O2/Mmax=6")
        if self.root_seed != 20260816 or self.planned_episodes != 25_000:
            raise ValueError("formal seed/episode budget mismatch")
        if self.validation_cadence_episodes != 2_500:
            raise ValueError("validation cadence must be frozen at 2500 episodes")
        if self.validation_scenarios_per_static_variant != 20:
            raise ValueError("formal validation requires 20 scenarios/static variant")
        if self.checkpoint_selection_metric != SELECTION_METRIC_NAME:
            raise ValueError("checkpoint selection metric mismatch")
        if self.durable_milestone_episodes != (
            1_000, 2_500, 5_000, 10_000, 15_000, 20_000, 25_000
        ):
            raise ValueError("durable milestone schedule mismatch")
        if self.formal_test_episode_usage != 0:
            raise ValueError("formal test usage must remain zero")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> "FormalRunIdentity":
        values = dict(record)
        values["durable_milestone_episodes"] = tuple(
            int(value) for value in values["durable_milestone_episodes"]
        )
        return cls(**values)


def load_run_identity(path: str | Path) -> FormalRunIdentity:
    return FormalRunIdentity.from_record(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def atomic_write_json(path: str | Path, record: object) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return destination


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def committed_file_sha256(code_sha: str, path: str) -> str:
    content = subprocess.check_output(["git", "show", f"{code_sha}:{path}"])
    return hashlib.sha256(content).hexdigest()


def current_git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()


def validate_frozen_identity(identity: FormalRunIdentity, manifest: SplitManifest) -> None:
    if current_git_head() != identity.code_sha:
        raise RuntimeError("working code SHA differs from frozen formal run identity")
    if committed_file_sha256(identity.code_sha, identity.config_path) != identity.config_sha256:
        raise RuntimeError("committed candidate config bytes differ from frozen hash")
    config = load_formal_training_config(identity.config_path)
    if (
        config.hyperparameter_status != "CANDIDATE_FOR_REHEARSAL"
        or config.observation_version != identity.observation_version
        or config.Mmax != identity.Mmax
        or config.seed != identity.root_seed
        or config.dataset_version != identity.dataset_version
        or config.split_manifest_version != identity.split_manifest_version
    ):
        raise RuntimeError("candidate config semantics differ from run identity")
    if manifest.protocol_version != identity.split_manifest_version:
        raise RuntimeError("loaded split manifest version differs from run identity")


def selection_score(ds1_instance_means: Sequence[float], ds2_instance_means: Sequence[float]) -> float:
    """Frozen primary metric; every static instance has equal weight within variant."""

    if not ds1_instance_means or not ds2_instance_means:
        raise ValueError("selection score requires both DS1 and DS2 instance means")
    return (
        float(np.mean(ds1_instance_means)) + float(np.mean(ds2_instance_means))
    ) / 2.0


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    se = std / math.sqrt(len(array))
    mean = float(array.mean())
    return {
        "count": len(array),
        "mean": mean,
        "std": std,
        "standard_error": se,
        "ci95_low": mean - 1.96 * se,
        "ci95_high": mean + 1.96 * se,
    }


class CachedKuProvider(KuTrainingInstanceProvider):
    def __init__(self, source_root: str | Path) -> None:
        super().__init__(source_root)
        self._cache = {}

    def __call__(self, sample: TrainingSample):
        key = (sample.base_instance_id, sample.variant)
        if key not in self._cache:
            self._cache[key] = super().__call__(sample)
        return self._cache[key]


def evaluate_validation(
    trainer: SCRPFormalTrainer,
    manifest: SplitManifest,
    provider: CachedKuProvider,
    identity: FormalRunIdentity,
    episode: int,
) -> dict[str, object]:
    """Evaluate only the frozen validation split; ERI and test are absent."""

    schedule = ScenarioSeedSchedule(manifest)
    refs = manifest.refs("validation")
    if len(refs) != 240:
        raise RuntimeError("formal validation expects 240 base instances")
    instance_records = {"DS1": [], "DS2": []}
    variant_episode_relocations = {"DS1": [], "DS2": []}
    trainer.policy.eval()
    started = time.perf_counter()
    try:
        with torch.no_grad():
            for ref in refs:
                stacks = int(ref.parameter_group[1:3])
                for variant in ("DS1", "DS2"):
                    relocations = []
                    for scenario_index in range(
                        identity.validation_scenarios_per_static_variant
                    ):
                        sample = TrainingSample(
                            base_instance_id=ref.base_instance_id,
                            instance_id=(
                                ref.ds1_instance_id if variant == "DS1"
                                else ref.ds2_instance_id
                            ),
                            variant=variant,
                            scenario_seed=schedule.seed_for(
                                "validation", ref.base_instance_id, scenario_index
                            ),
                            visit_index=scenario_index,
                            num_stacks=stacks,
                        )
                        trajectory = run_formal_episode(
                            provider(sample), sample, trainer.policy, trainer.config,
                            greedy=True, device=trainer.device,
                        )
                        if (
                            not trajectory.terminated
                            or trajectory.truncated
                            or trajectory.invalid_actions
                        ):
                            raise RuntimeError("formal validation rollout failed")
                        relocations.append(trajectory.relocations)
                    variant_episode_relocations[variant].extend(relocations)
                    instance_records[variant].append({
                        "base_instance_id": ref.base_instance_id,
                        "instance_id": (
                            ref.ds1_instance_id if variant == "DS1"
                            else ref.ds2_instance_id
                        ),
                        "parameter_group": ref.parameter_group,
                        **_distribution(relocations),
                    })
    finally:
        trainer.policy.train()

    variant_records = {}
    for variant in ("DS1", "DS2"):
        per_instance_means = [
            record["mean"] for record in instance_records[variant]
        ]
        groups = {}
        for group in sorted({record["parameter_group"] for record in instance_records[variant]}):
            group_means = [
                record["mean"] for record in instance_records[variant]
                if record["parameter_group"] == group
            ]
            groups[group] = _distribution(group_means)
        variant_records[variant] = {
            "episode_distribution": _distribution(
                variant_episode_relocations[variant]
            ),
            "equal_instance_distribution": _distribution(per_instance_means),
            "per_instance": instance_records[variant],
            "parameter_group_equal_instance_means": groups,
        }

    score = selection_score(
        [record["mean"] for record in instance_records["DS1"]],
        [record["mean"] for record in instance_records["DS2"]],
    )
    return {
        "training_episode": episode,
        "validation_scenarios_per_static_variant": (
            identity.validation_scenarios_per_static_variant
        ),
        "validation_episode_count": 240 * 2 * 20,
        "selection_metric": identity.checkpoint_selection_metric,
        "selection_formula": "(mean(DS1 per-instance means) + mean(DS2 per-instance means)) / 2",
        "selection_score": score,
        "DS1": variant_records["DS1"],
        "DS2": variant_records["DS2"],
        "combined_equal_variant_mean": score,
        "baseline_state_version": trainer.baseline_updates,
        "model_state_sha256": policy_state_sha256(trainer.policy),
        "wall_seconds": time.perf_counter() - started,
        "ERI_used": False,
        "formal_test_used": False,
    }


def save_checkpoint_atomic(trainer: SCRPFormalTrainer, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    trainer.save_checkpoint(temporary)
    # Read the complete payload before publishing it as durable.
    torch.load(temporary, map_location="cpu", weights_only=False)
    os.replace(temporary, destination)
    return destination


def copy_checkpoint_verified(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    if file_sha256(source) != file_sha256(temporary):
        raise RuntimeError("checkpoint copy hash mismatch")
    os.replace(temporary, destination)
    return destination


def compact_window(
    metrics: Sequence[FormalIterationMetrics],
    samples: Sequence[TrainingSample],
    episode: int,
) -> dict[str, object]:
    if not metrics or not samples:
        raise ValueError("monitoring window cannot be empty")
    return {
        "episode": episode,
        "iteration": metrics[-1].iteration,
        "window_episodes": len(samples),
        "loss": float(np.mean([metric.loss for metric in metrics])),
        "policy_loss": float(np.mean([metric.policy_loss for metric in metrics])),
        "entropy": float(np.mean([metric.entropy for metric in metrics])),
        "grad_norm": float(np.mean([metric.grad_norm for metric in metrics])),
        "mean_policy_relocations": float(np.mean([
            metric.mean_policy_relocations for metric in metrics
        ])),
        "mean_baseline_relocations": float(np.mean([
            metric.mean_baseline_relocations for metric in metrics
        ])),
        "mean_advantage": float(np.mean([
            metric.mean_advantage for metric in metrics
        ])),
        "baseline_updates": metrics[-1].baseline_updates,
        "invalid_actions": sum(metric.invalid_actions for metric in metrics),
        "truncations": sum(metric.truncations for metric in metrics),
        "scenario_mismatches": sum(metric.scenario_mismatches for metric in metrics),
        "empty_decision_episodes": sum(
            metric.empty_decision_episodes for metric in metrics
        ),
        "S_bucket_counts": {
            str(stacks): sum(sample.num_stacks == stacks for sample in samples)
            for stacks in range(5, 11)
        },
        "variant_counts": {
            variant: sum(sample.variant == variant for sample in samples)
            for variant in ("DS1", "DS2")
        },
    }


def checkpoint_inventory(directory: Path) -> list[dict[str, object]]:
    return [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": file_sha256(path)}
        for path in sorted(directory.glob("*.pt"))
    ]
