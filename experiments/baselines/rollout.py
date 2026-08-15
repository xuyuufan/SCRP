"""Unified full-episode rollout for public-state SCRP baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from scrp.environment import SCRPEnvironment

from .base import BaselineActionError, SCRPBaseline


@dataclass(frozen=True)
class BaselineEpisodeResult:
    algorithm: str
    instance_id: str
    scenario_seed: int
    scenario_id: str
    action_seed: int
    actions: Tuple[int, ...]
    relocations: int
    total_reward: int
    decision_count: int
    invalid_action_count: int
    terminated: bool
    truncated: bool


def run_baseline_episode(
    env: SCRPEnvironment,
    baseline: SCRPBaseline,
    scenario_seed: int,
    *,
    action_seed: int,
) -> BaselineEpisodeResult:
    """Run to normal termination without exposing the sampled Scenario."""

    state = env.reset(seed=scenario_seed)
    baseline.reset(action_seed)
    actions = []
    total_reward = 0
    invalid_action_count = 0

    while not state.terminated:
        legal = env.legal_destinations()
        action = baseline.select_destination(env.instance, env.state, legal)
        if action not in legal:
            invalid_action_count += 1
            raise BaselineActionError(
                f"{baseline.name} selected illegal destination {action}; legal={legal}"
            )
        actions.append(action)
        transition = env.step(action)
        total_reward += transition.reward
        state = transition.state

    if total_reward != -state.relocation_count:
        raise AssertionError("episode reward must equal negative relocations")
    if len(actions) != state.relocation_count:
        raise AssertionError("one baseline decision must equal one relocation")
    return BaselineEpisodeResult(
        algorithm=baseline.name,
        instance_id=env.instance.instance_id,
        scenario_seed=scenario_seed,
        scenario_id=env.scenario_id,
        action_seed=action_seed,
        actions=tuple(actions),
        relocations=state.relocation_count,
        total_reward=total_reward,
        decision_count=len(actions),
        invalid_action_count=invalid_action_count,
        terminated=state.terminated,
        truncated=False,
    )
