from dataclasses import asdict

from scrp import (
    Container,
    EventKind,
    SCRPConfig,
    SCRPEnvironment,
    SCRPInstance,
    Scenario,
)


class FixedScenarioSampler:
    def sample(self, instance, root_seed):
        return Scenario(
            root_seed=root_seed,
            order_seeds={1: 101, 2: 202},
            hidden_orders={1: (1, 2, 3), 2: (4, 5, 6, 7)},
            scenario_id=f"fixed-{root_seed}",
        )


def make_env():
    instance = SCRPInstance(
        instance_id="tiny-revelation",
        num_stacks=3,
        max_tiers=3,
        containers=tuple(
            [Container(i, 1) for i in (1, 2, 3)]
            + [Container(i, 2) for i in (4, 5, 6, 7)]
        ),
        initial_stacks=((1, 6, 7), (2, 5), (4, 3)),
        batch_order=(1, 2),
    )
    return SCRPEnvironment(
        SCRPConfig(3, 3, root_seed=9),
        instance,
        scenario_sampler=FixedScenarioSampler(),
    )


def test_reset_reveals_complete_b1_only():
    state = make_env().reset()
    assert state.revealed_orders == {1: (1, 2, 3)}
    assert state.current_target_id == 1
    assert 2 not in state.revealed_orders


def test_future_hidden_order_is_absent_from_public_state_and_result():
    env = make_env()
    state = env.reset()
    public_reset = repr(asdict(state))
    assert "(4, 5, 6, 7)" not in public_reset

    result = env.step(2)
    public_result = repr(asdict(result))
    assert "(4, 5, 6, 7)" not in public_result


def test_b2_is_not_visible_before_b1_finishes():
    env = make_env()
    env.reset()
    for action in (2, 1, 0, 0):
        result = env.step(action)
        assert 2 not in result.state.revealed_orders
    assert result.state.current_target_id == 3


def test_revealed_order_is_not_resampled_within_batch():
    env = make_env()
    state = env.reset()
    original = state.revealed_orders[1]
    for action in (2, 1, 0, 0):
        result = env.step(action)
        assert result.state.revealed_orders[1] == original


def test_next_batch_reveal_is_atomic_with_previous_batch_completion():
    env = make_env()
    env.reset()
    for action in (2, 1, 0, 0):
        env.step(action)
    result = env.step(0)

    kinds = [event.kind for event in result.events]
    assert kinds == [
        EventKind.RELOCATE,
        EventKind.RETRIEVE,
        EventKind.FINISH_BATCH,
        EventKind.REVEAL_BATCH,
        EventKind.RETRIEVE,
    ]
    assert result.events[2].batch_id == 1
    assert result.events[3].batch_id == 2
    assert result.state.revealed_orders[2] == (4, 5, 6, 7)
    assert result.state.current_target_id == 5


def test_completed_batch_orders_remain_visible_but_future_orders_do_not():
    env = make_env()
    env.reset()
    for action in (2, 1, 0, 0, 0):
        state = env.step(action).state
    assert state.revealed_orders == {1: (1, 2, 3), 2: (4, 5, 6, 7)}
