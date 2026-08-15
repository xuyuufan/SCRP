import numpy as np

from scrp import (
    Container,
    SCRPConfig,
    SCRPEnvironment,
    SCRPInstance,
    SCRPRLAdapter,
    Scenario,
)


class OrdersSampler:
    def __init__(self, second_order=(4, 5, 6, 7)):
        self.second_order = second_order

    def sample(self, instance, root_seed):
        return Scenario(
            root_seed,
            {1: 1, 2: 2},
            {1: (1, 2, 3), 2: self.second_order},
            "orders",
        )


def make_adapter(second_order=(4, 5, 6, 7)):
    instance = SCRPInstance(
        "observation-tiny",
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
        SCRPConfig(3, 3, root_seed=4), instance, OrdersSampler(second_order)
    )
    return SCRPRLAdapter(core)


def test_stack_nodes_align_with_action_indices_and_context_is_last():
    adapter = make_adapter()
    obs, _ = adapter.reset()
    nodes = obs.reshape(4, 12)
    np.testing.assert_allclose(nodes[:3, 0], [0.0, 0.5, 1.0])
    assert nodes[3, 0] == 2.0
    assert nodes[3, 11] == 1.0


def test_observation_is_finite_float32_and_normalized_except_context_marker():
    adapter = make_adapter()
    obs, _ = adapter.reset()
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()
    nodes = obs.reshape(4, 12)
    assert np.all((nodes[:3] >= 0.0) & (nodes[:3] <= 1.0))
    assert np.all((nodes[3, 1:] >= 0.0) & (nodes[3, 1:] <= 1.0))


def test_different_future_hidden_orders_produce_identical_current_observation():
    first = make_adapter((4, 5, 6, 7))
    second = make_adapter((7, 6, 5, 4))
    obs_a, info_a = first.reset()
    obs_b, info_b = second.reset()
    np.testing.assert_array_equal(obs_a, obs_b)
    np.testing.assert_array_equal(info_a["action_mask"], info_b["action_mask"])


def test_visible_relocation_and_target_advance_change_observation():
    adapter = make_adapter()
    reset_obs, _ = adapter.reset()
    first_obs, *_ = adapter.step(2)
    second_obs, *_ = adapter.step(1)
    assert not np.array_equal(reset_obs, first_obs)
    assert not np.array_equal(first_obs, second_obs)
    assert adapter.core_env.state.current_target_id == 2
    # Context target stack changes from stack 0 to stack 1.
    assert reset_obs.reshape(4, 12)[-1, 4] == 0.0
    assert second_obs.reshape(4, 12)[-1, 4] == 0.5


def test_empty_stack_rules_are_explicit_in_terminal_observation():
    adapter = make_adapter()
    adapter.reset()
    for action in (2, 1, 0, 0, 0, 2):
        obs, _, terminated, _, _ = adapter.step(action)
    assert terminated
    nodes = obs.reshape(4, 12)
    np.testing.assert_array_equal(nodes[:3, 1], np.zeros(3))
    np.testing.assert_array_equal(nodes[:3, 2], np.ones(3))
    np.testing.assert_array_equal(nodes[:3, 4:9], np.zeros((3, 5)))
    np.testing.assert_array_equal(nodes[:3, 9], np.ones(3))
