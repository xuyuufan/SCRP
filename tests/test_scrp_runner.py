import numpy as np
import torch

from scrp import Container, SCRPConfig, SCRPEnvironment, SCRPInstance, SCRPRLAdapter, Scenario
from scrp.training import (
    SCRP_O1_FEATURE_SCALE,
    make_scrp_o1_policy,
    make_scrp_training_tiny_env,
    run_scrp_low_episode,
)


def test_scrp_runner_uses_reset_seed_low_only_and_legal_actions():
    torch.manual_seed(7)
    policy = make_scrp_o1_policy()
    env = make_scrp_training_tiny_env()
    trajectory = run_scrp_low_episode(env, policy, 7654321, greedy=False)

    assert trajectory.episode_seed == 7654321
    assert trajectory.scenario_id == "scrp-training-tiny-7654321"
    assert trajectory.decision_modes == ["low"] * trajectory.low_decisions
    assert trajectory.high_decisions == 0
    assert trajectory.low_decisions == trajectory.relocation_count
    assert trajectory.invalid_action_count == 0
    assert trajectory.terminated and not trajectory.truncated
    for action, legal_mask in zip(trajectory.actions, trajectory.action_masks):
        assert legal_mask.dtype == np.bool_
        assert legal_mask[action]
    assert sum(trajectory.rewards) == -env.get_metrics()["shifters"]
    assert trajectory.episode_return == -trajectory.relocation_count


def test_policy_and_frozen_greedy_baseline_complete_identical_scenario():
    torch.manual_seed(19)
    policy = make_scrp_o1_policy()
    baseline = make_scrp_o1_policy()
    baseline.load_state_dict(policy.state_dict())
    baseline.eval()
    seed = 9090
    policy_trajectory = run_scrp_low_episode(
        make_scrp_training_tiny_env(), policy, seed, greedy=False
    )
    baseline_trajectory = run_scrp_low_episode(
        make_scrp_training_tiny_env(), baseline, seed, greedy=True
    )
    assert policy_trajectory.scenario_id == baseline_trajectory.scenario_id
    assert policy_trajectory.terminated
    assert baseline_trajectory.terminated
    assert np.isfinite(policy_trajectory.episode_return)
    assert np.isfinite(baseline_trajectory.episode_return)


def test_scrp_policy_uses_all_ones_o1_feature_scale():
    policy = make_scrp_o1_policy()
    np.testing.assert_array_equal(
        policy.feature_scale.detach().cpu().numpy(),
        np.asarray(SCRP_O1_FEATURE_SCALE, dtype=np.float32),
    )


class DirectTerminalSampler:
    def sample(self, instance, root_seed):
        return Scenario(root_seed, {1: 1}, {1: (1, 2)}, f"direct-{root_seed}")


def make_direct_terminal_env():
    instance = SCRPInstance(
        "runner-direct-terminal",
        2,
        2,
        (Container(1, 1), Container(2, 1)),
        ((2, 1), ()),
        (1,),
    )
    return SCRPRLAdapter(
        SCRPEnvironment(SCRPConfig(2, 2), instance, DirectTerminalSampler())
    )


def test_direct_terminal_reset_returns_empty_low_and_high_trajectories():
    trajectory = run_scrp_low_episode(
        make_direct_terminal_env(), make_scrp_o1_policy(), 44, greedy=False
    )
    assert trajectory.scenario_id == "direct-44"
    assert trajectory.observations == []
    assert trajectory.actions == []
    assert trajectory.rewards == []
    assert trajectory.low_decisions == 0
    assert trajectory.high_decisions == 0
    assert trajectory.episode_return == 0.0
    assert trajectory.relocation_count == 0
    assert trajectory.terminated
