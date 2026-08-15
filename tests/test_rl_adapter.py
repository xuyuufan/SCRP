import json

import numpy as np
import pytest

from scrp import (
    Container,
    EpisodeTerminatedError,
    SCRPConfig,
    SCRPEnvironment,
    SCRPInstance,
    SCRPRLAdapter,
    Scenario,
    StepLimitError,
)


class FixedSampler:
    def sample(self, instance, root_seed):
        return Scenario(
            root_seed,
            {1: 101, 2: 202},
            {1: (1, 2, 3), 2: (4, 5, 6, 7)},
            f"tiny-{root_seed}",
        )


def make_adapter(max_steps=10_000):
    instance = SCRPInstance(
        "adapter-tiny",
        3,
        3,
        tuple(
            [Container(i, 1) for i in (1, 2, 3)]
            + [Container(i, 2) for i in (4, 5, 6, 7)]
        ),
        ((1, 6, 7), (2, 5), (4, 3)),
        (1, 2),
    )
    core = SCRPEnvironment(
        SCRPConfig(3, 3, root_seed=7, max_steps=max_steps), instance, FixedSampler()
    )
    return SCRPRLAdapter(core)


def test_reset_contract_shape_dtype_and_true_means_legal():
    adapter = make_adapter()
    obs, info = adapter.reset()
    assert obs.dtype == np.float32
    assert obs.shape == ((3 + 1) * 12,)
    assert adapter.action_space.n == 3
    assert info["action_mask"].dtype == np.bool_
    assert info["action_mask"].shape == (3,)
    expected = np.zeros(3, dtype=bool)
    expected[list(adapter.core_env.legal_destinations())] = True
    np.testing.assert_array_equal(info["action_mask"], expected)
    assert info["action_mask"].tolist() == [False, True, True]


def test_step_returns_five_tuple_and_core_reward_terminal_semantics():
    adapter = make_adapter()
    adapter.reset()
    transition = adapter.step(2)
    assert len(transition) == 5
    obs, reward, terminated, truncated, info = transition
    assert obs.shape == (48,)
    assert reward == -1
    assert terminated is False
    assert truncated is False
    assert info["action_mask"].tolist() == [False, True, False]
    assert adapter.core_env.state.total_reward == reward


def test_metrics_and_snapshot_are_serializable_and_hide_future_orders():
    adapter = make_adapter()
    adapter.reset()
    metrics = adapter.get_metrics()
    assert metrics["shifters"] == metrics["relocation_count"] == 0
    assert metrics["retrieval_count"] == 0
    assert metrics["scenario_id"] == "tiny-7"

    snapshot = adapter.get_state_snapshot()
    json.dumps(snapshot)
    assert "hidden_orders" not in snapshot
    assert snapshot["revealed_orders"] == {"1": [1, 2, 3]}
    assert "2" not in snapshot["revealed_orders"]


def test_full_tiny_episode_adapter_metrics_match_core():
    adapter = make_adapter()
    adapter.reset()
    rewards = []
    for action in (2, 1, 0, 0, 0, 2):
        _, reward, terminated, truncated, _ = adapter.step(action)
        rewards.append(reward)
        assert not truncated
    assert terminated
    metrics = adapter.get_metrics()
    assert sum(rewards) == metrics["total_reward"] == -6
    assert metrics["shifters"] == metrics["relocation_count"] == 6
    assert metrics["retrieval_count"] == 7


def test_reset_direct_terminal_returns_shaped_obs_and_all_false_mask():
    instance = SCRPInstance(
        "direct-terminal",
        2,
        2,
        (Container(1, 1), Container(2, 1)),
        ((2, 1), ()),
        (1,),
    )

    class DirectSampler:
        def sample(self, instance, root_seed):
            return Scenario(root_seed, {1: 1}, {1: (1, 2)}, "direct")

    adapter = SCRPRLAdapter(
        SCRPEnvironment(SCRPConfig(2, 2), instance, DirectSampler())
    )
    obs, info = adapter.reset()
    assert obs.shape == (36,)
    assert info["terminated"] is True
    assert not info["action_mask"].any()
    assert adapter.get_metrics()["terminated"] is True
    with pytest.raises(EpisodeTerminatedError):
        adapter.step(0)


def test_step_limit_error_propagates_instead_of_becoming_normal_truncation():
    adapter = make_adapter(max_steps=1)
    adapter.reset()
    _, _, terminated, truncated, _ = adapter.step(2)
    assert not terminated and not truncated
    with pytest.raises(StepLimitError):
        adapter.step(1)
