"""Common-random-number evaluation harness for SCRP algorithms."""

from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol, Sequence, Tuple

from scrp.environment import SCRPEnvironment
from scrp.models import SCRPConfig, SCRPInstance
from scrp.rl_adapter import SCRPRLAdapter

from .baselines import SCRPBaseline, run_baseline_episode
from .protocol import ScenarioResult


@dataclass(frozen=True)
class EvaluationCase:
    """One static artifact and its explicit scenario-seed schedule."""

    instance: SCRPInstance
    dataset: str
    split: str
    base_instance_id: str
    parameter_group: str
    scenario_seeds: Tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.scenario_seeds:
            raise ValueError("evaluation case requires at least one scenario seed")


@dataclass(frozen=True)
class AlgorithmEpisode:
    scenario_id: str
    relocations: int
    terminated: bool
    truncated: bool
    invalid_action_count: int


class EvaluationAlgorithm(Protocol):
    name: str

    def run(self, env: SCRPEnvironment, scenario_seed: int) -> AlgorithmEpisode:
        """Run one episode on the supplied fresh environment."""


class BaselineAlgorithm:
    """Adapt a public-state baseline to the common evaluation interface."""

    def __init__(
        self,
        baseline_factory: Callable[[], SCRPBaseline],
        *,
        action_seed_root: int = 0,
    ) -> None:
        self._baseline_factory = baseline_factory
        self._action_seed_root = action_seed_root
        self.name = baseline_factory().name

    def action_seed_for(self, instance_id: str, scenario_seed: int) -> int:
        material = (
            f"scrp-baseline-action-v1|{self._action_seed_root}|{self.name}|"
            f"{instance_id}|{scenario_seed}"
        ).encode("utf-8")
        return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")

    def run(self, env: SCRPEnvironment, scenario_seed: int) -> AlgorithmEpisode:
        result = run_baseline_episode(
            env,
            self._baseline_factory(),
            scenario_seed,
            action_seed=self.action_seed_for(env.instance.instance_id, scenario_seed),
        )
        return AlgorithmEpisode(
            scenario_id=result.scenario_id,
            relocations=result.relocations,
            terminated=result.terminated,
            truncated=result.truncated,
            invalid_action_count=result.invalid_action_count,
        )


class LowPolicyAlgorithm:
    """Read-only adapter for evaluating the existing O1 LOW policy."""

    def __init__(self, policy, *, name: str = "current_o1_low", device: str = "cpu") -> None:
        self.policy = policy
        self.name = name
        self.device = device

    def run(self, env: SCRPEnvironment, scenario_seed: int) -> AlgorithmEpisode:
        # Kept here, rather than in training, so Phase 4 does not change policy code.
        from scrp.training import run_scrp_low_episode

        trajectory = run_scrp_low_episode(
            SCRPRLAdapter(env), self.policy, scenario_seed, greedy=True, device=self.device
        )
        return AlgorithmEpisode(
            scenario_id=trajectory.scenario_id,
            relocations=trajectory.relocation_count,
            terminated=trajectory.terminated,
            truncated=trajectory.truncated,
            invalid_action_count=trajectory.invalid_action_count,
        )


EnvironmentFactory = Callable[[SCRPInstance], SCRPEnvironment]


def default_environment_factory(instance: SCRPInstance) -> SCRPEnvironment:
    return SCRPEnvironment(
        SCRPConfig(
            num_stacks=instance.num_stacks,
            max_tiers=instance.max_tiers,
            max_steps=100_000,
        ),
        instance,
    )


def evaluate_algorithm_on_schedule(
    algorithm: EvaluationAlgorithm,
    cases: Sequence[EvaluationCase],
    *,
    environment_factory: EnvironmentFactory = default_environment_factory,
) -> Tuple[ScenarioResult, ...]:
    """Preserve one raw Phase 3.5 result for every scheduled scenario."""

    results = []
    for case in cases:
        for scenario_seed in case.scenario_seeds:
            episode = algorithm.run(environment_factory(case.instance), scenario_seed)
            if episode.invalid_action_count:
                raise AssertionError(
                    f"{algorithm.name} produced {episode.invalid_action_count} invalid actions"
                )
            results.append(
                ScenarioResult(
                    dataset=case.dataset,
                    split=case.split,
                    instance_id=case.instance.instance_id,
                    base_instance_id=case.base_instance_id,
                    parameter_group=case.parameter_group,
                    scenario_seed=scenario_seed,
                    scenario_id=episode.scenario_id,
                    algorithm=algorithm.name,
                    relocations=episode.relocations,
                    terminated=episode.terminated,
                    truncated=episode.truncated,
                )
            )
    return tuple(results)


def assert_paired_scenarios(*result_sets: Sequence[ScenarioResult]) -> None:
    """Assert identical scenario IDs for each static-artifact/seed coordinate."""

    if len(result_sets) < 2:
        raise ValueError("paired verification requires at least two algorithms")
    reference = {
        (result.dataset, result.instance_id, result.scenario_seed): result.scenario_id
        for result in result_sets[0]
    }
    for results in result_sets[1:]:
        observed = {
            (result.dataset, result.instance_id, result.scenario_seed): result.scenario_id
            for result in results
        }
        if observed != reference:
            raise AssertionError("paired algorithms did not receive identical scenarios")


@dataclass(frozen=True)
class RelocationSummary:
    algorithm: str
    count: int
    mean: float
    std: float
    minimum: int
    maximum: int


def aggregate_relocations(results: Iterable[ScenarioResult]) -> Tuple[RelocationSummary, ...]:
    grouped: dict[str, list[int]] = {}
    for result in results:
        grouped.setdefault(result.algorithm, []).append(result.relocations)
    return tuple(
        RelocationSummary(
            algorithm=algorithm,
            count=len(values),
            mean=float(statistics.fmean(values)),
            std=float(statistics.stdev(values)) if len(values) > 1 else 0.0,
            minimum=min(values),
            maximum=max(values),
        )
        for algorithm, values in sorted(grouped.items())
    )


def save_raw_results(results: Iterable[ScenarioResult], path: str | Path) -> Path:
    """Write JSONL raw results; callers should target the gitignored raw directory."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    records = tuple(results)
    destination.write_text(
        "".join(json.dumps(result.to_record(), sort_keys=True) + "\n" for result in records),
        encoding="utf-8",
    )
    return destination


def save_summaries(summaries: Iterable[RelocationSummary], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps([asdict(summary) for summary in summaries], indent=2) + "\n",
        encoding="utf-8",
    )
    return destination
