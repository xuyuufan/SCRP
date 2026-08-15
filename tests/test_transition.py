from copy import deepcopy

import pytest

from scrp import (
    Container,
    EpisodeTerminatedError,
    EventKind,
    InvalidActionError,
    NoLegalRelocationError,
    Phase,
    SCRPConfig,
    SCRPEnvironment,
    SCRPInstance,
    Scenario,
    StateInvariantError,
)


class FixedScenarioSampler:
    def __init__(self, orders=None):
        self.orders = orders or {1: (1, 2, 3), 2: (4, 5, 6, 7)}

    def sample(self, instance, root_seed):
        return Scenario(
            root_seed=root_seed,
            order_seeds={batch_id: batch_id * 100 for batch_id in self.orders},
            hidden_orders=dict(self.orders),
            scenario_id=f"fixed-{root_seed}",
        )


def tiny_env(max_steps=100):
    instance = SCRPInstance(
        instance_id="tiny-acceptance",
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
        SCRPConfig(3, 3, root_seed=11, max_steps=max_steps),
        instance,
        FixedScenarioSampler(),
    )


def stacks(state):
    return [stack.containers for stack in state.stacks]


def test_only_current_target_top_blocker_is_relocated():
    env = tiny_env()
    before = env.reset()
    result = env.step(2)
    assert before.current_target_id == 1
    assert result.events[0].container_id == 7
    assert result.events[0].source_stack_id == 0
    assert stacks(result.state) == [[1, 6], [2, 5], [4, 3, 7]]


@pytest.mark.parametrize("destination", [-1, 3, True, 1.5])
def test_out_of_range_or_non_integer_destination_is_rejected_transactionally(destination):
    env = tiny_env()
    before = env.reset()
    with pytest.raises(InvalidActionError):
        env.step(destination)
    assert env.state == before


def test_destination_cannot_equal_source_and_state_is_unchanged():
    env = tiny_env()
    before = env.reset()
    with pytest.raises(InvalidActionError, match="equal source"):
        env.step(0)
    assert env.state == before


def test_full_destination_is_rejected_before_source_pop():
    instance = SCRPInstance(
        "full-destination",
        3,
        2,
        tuple([Container(1, 1)] + [Container(i, 2) for i in (2, 3, 4)]),
        ((1, 2), (3, 4), ()),
        (1, 2),
    )
    env = SCRPEnvironment(
        SCRPConfig(3, 2),
        instance,
        FixedScenarioSampler({1: (1,), 2: (2, 3, 4)}),
    )
    before = env.reset()
    with pytest.raises(InvalidActionError, match="full"):
        env.step(1)
    assert env.state == before


def test_nonterminal_state_without_legal_destination_is_rejected():
    instance = SCRPInstance(
        "no-destination",
        2,
        2,
        (Container(1, 1), Container(2, 2), Container(3, 2), Container(4, 2)),
        ((1, 2), (3, 4)),
        (1, 2),
    )
    env = SCRPEnvironment(
        SCRPConfig(2, 2),
        instance,
        FixedScenarioSampler({1: (1,), 2: (2, 3, 4)}),
    )
    with pytest.raises(NoLegalRelocationError):
        env.reset()


def test_tiny_acceptance_walkthrough_exact_states_events_and_rewards():
    env = tiny_env()
    state = env.reset()
    assert stacks(state) == [[1, 6, 7], [2, 5], [4, 3]]
    assert state.current_target_id == 1
    assert state.revealed_orders == {1: (1, 2, 3)}

    expected = [
        (2, 7, [[1, 6], [2, 5], [4, 3, 7]], 1, [EventKind.RELOCATE]),
        (1, 6, [[], [2, 5, 6], [4, 3, 7]], 2,
         [EventKind.RELOCATE, EventKind.RETRIEVE]),
        (0, 6, [[6], [2, 5], [4, 3, 7]], 2, [EventKind.RELOCATE]),
        (0, 5, [[6, 5], [], [4, 3, 7]], 3,
         [EventKind.RELOCATE, EventKind.RETRIEVE]),
        (0, 7, [[6, 5, 7], [], []], 5,
         [EventKind.RELOCATE, EventKind.RETRIEVE, EventKind.FINISH_BATCH,
          EventKind.REVEAL_BATCH, EventKind.RETRIEVE]),
        (2, 7, [[], [], []], None,
         [EventKind.RELOCATE, EventKind.RETRIEVE, EventKind.RETRIEVE,
          EventKind.RETRIEVE, EventKind.FINISH_BATCH, EventKind.TERMINATE]),
    ]

    reward_sum = 0
    for relocation_number, (action, blocker, expected_stacks, target, kinds) in enumerate(expected, 1):
        result = env.step(action)
        reward_sum += result.reward
        assert result.reward == -1
        assert result.events[0].container_id == blocker
        assert [event.kind for event in result.events] == kinds
        assert stacks(result.state) == expected_stacks
        assert result.state.current_target_id == target
        assert result.state.relocation_count == relocation_number
        assert result.state.total_reward == -relocation_number

    final = result.state
    assert reward_sum == -6
    assert final.retrieved_order == [1, 2, 3, 4, 5, 6, 7]
    assert final.relocation_count == 6
    assert final.retrieval_count == 7
    assert final.total_reward == -6
    assert final.terminated is True
    assert final.phase is Phase.TERMINATED
    assert all(stack.is_empty for stack in final.stacks)
    assert all(location is None for location in final.locations.values())


def test_retrievals_do_not_add_relocations_and_reward_identity_always_holds():
    env = tiny_env()
    env.reset()
    for action in (2, 1, 0, 0, 0, 2):
        before = env.state
        result = env.step(action)
        retrievals = sum(event.kind is EventKind.RETRIEVE for event in result.events)
        assert result.state.relocation_count == before.relocation_count + 1
        assert result.state.total_reward == -result.state.relocation_count
        if retrievals:
            assert result.state.retrieval_count == before.retrieval_count + retrievals


def test_step_after_termination_is_rejected():
    env = tiny_env()
    env.reset()
    for action in (2, 1, 0, 0, 0, 2):
        env.step(action)
    with pytest.raises(EpisodeTerminatedError):
        env.step(0)


def test_public_state_is_detached_from_environment_state():
    env = tiny_env()
    public_state = env.reset()
    public_state.stacks[0].containers.clear()
    assert env.state.stacks[0].containers == [1, 6, 7]


def test_state_conservation_and_location_mapping_after_every_transition():
    env = tiny_env()
    state = env.reset()
    all_ids = set(range(1, 8))
    for action in (2, 1, 0, 0, 0, 2):
        result = env.step(action)
        state = result.state
        remaining = [cid for stack in state.stacks for cid in stack.containers]
        assert set(remaining) | set(state.retrieved_order) == all_ids
        assert not (set(remaining) & set(state.retrieved_order))
        assert len(remaining) + state.retrieval_count == 7
        for stack in state.stacks:
            for tier, container_id in enumerate(stack.containers):
                location = state.locations[container_id]
                assert (location.stack_id, location.tier) == (stack.stack_id, tier)


def test_broken_internal_state_is_detected():
    env = tiny_env()
    env.reset()
    env._state.locations[7] = None
    with pytest.raises(StateInvariantError, match="conservation|location"):
        env._validate_state()


def test_automatic_retrieval_can_terminate_during_reset_without_actions():
    instance = SCRPInstance(
        "direct-only",
        2,
        2,
        (Container(1, 1), Container(2, 1)),
        ((2, 1), ()),
        (1,),
    )
    env = SCRPEnvironment(
        SCRPConfig(2, 2),
        instance,
        FixedScenarioSampler({1: (1, 2)}),
    )
    state = env.reset()
    assert state.terminated
    assert state.retrieved_order == [1, 2]
    assert state.total_reward == 0
    assert state.relocation_count == 0
