import random

import pytest

from scrp import (
    Container,
    SCRPConfig,
    SCRPEnvironment,
    SCRPInstance,
    ScenarioSampler,
)


def make_instance():
    return SCRPInstance(
        instance_id="seed-instance",
        num_stacks=3,
        max_tiers=3,
        containers=tuple(
            [Container(i, 1) for i in (1, 2, 3)]
            + [Container(i, 2) for i in (4, 5, 6, 7)]
        ),
        initial_stacks=((1, 6, 7), (2, 5), (4, 3)),
        batch_order=(1, 2),
    )


def test_same_seed_produces_identical_complete_scenario():
    sampler = ScenarioSampler()
    instance = make_instance()
    first = sampler.sample(instance, 12345)
    second = sampler.sample(instance, 12345)
    assert first == second
    assert first.order_seeds.keys() == {1, 2}
    assert first.hidden_orders.keys() == {1, 2}


def test_different_seeds_can_produce_multiple_scenarios_without_flaky_pair_assertion():
    sampler = ScenarioSampler()
    instance = make_instance()
    scenario_ids = {sampler.sample(instance, seed).scenario_id for seed in range(20)}
    hidden_orders = {
        tuple(sampler.sample(instance, seed).hidden_orders.items())
        for seed in range(20)
    }
    assert len(scenario_ids) > 1
    assert len(hidden_orders) > 1


def test_sampler_does_not_depend_on_global_random_state():
    sampler = ScenarioSampler()
    instance = make_instance()
    random.seed(1)
    first = sampler.sample(instance, 77)
    for _ in range(100):
        random.random()
    random.seed(999999)
    second = sampler.sample(instance, 77)
    assert first == second


def test_order_seed_is_stable_and_batch_specific():
    assert ScenarioSampler.derive_order_seed(9, 1) == ScenarioSampler.derive_order_seed(9, 1)
    assert ScenarioSampler.derive_order_seed(9, 1) != ScenarioSampler.derive_order_seed(9, 2)
    assert ScenarioSampler.derive_order_seed(9, 1) != ScenarioSampler.derive_order_seed(10, 1)


def test_reset_with_same_seed_reconstructs_same_scenario_after_actions():
    instance = make_instance()
    env = SCRPEnvironment(SCRPConfig(3, 3, root_seed=88), instance)
    first_state = env.reset()
    first_id = env.scenario_id
    env.step(env.legal_destinations()[0])
    second_state = env.reset()
    assert env.scenario_id == first_id
    assert second_state == first_state


def test_hidden_orders_are_independent_of_different_legal_action_paths():
    base = make_instance()
    # One extra tier ensures that both non-source stacks are legal destinations,
    # regardless of which B1 member is the sampled current target.
    instance = SCRPInstance(
        instance_id="two-legal-paths",
        num_stacks=3,
        max_tiers=4,
        containers=base.containers,
        initial_stacks=base.initial_stacks,
        batch_order=base.batch_order,
    )
    config = SCRPConfig(3, 4, root_seed=2026)
    env_a = SCRPEnvironment(config, instance)
    env_b = SCRPEnvironment(config, instance)
    env_a.reset()
    env_b.reset()
    scenario_id = env_a.scenario_id
    assert env_b.scenario_id == scenario_id

    legal = env_a.legal_destinations()
    assert len(legal) == 2
    env_a.step(legal[0])
    env_b.step(legal[1])

    assert env_a.scenario_id == scenario_id
    assert env_b.scenario_id == scenario_id
    expected = ScenarioSampler().sample(instance, 2026)
    assert env_a.scenario_id == expected.scenario_id


def test_future_batch_is_pre_sampled_but_not_revealed_on_reset():
    instance = make_instance()
    expected = ScenarioSampler().sample(instance, 42)
    assert 2 in expected.hidden_orders

    env = SCRPEnvironment(SCRPConfig(3, 3, root_seed=42), instance)
    state = env.reset()
    assert 1 in state.revealed_orders
    assert 2 not in state.revealed_orders


def test_scenario_mappings_and_orders_are_immutable_copies():
    source_seeds = {1: 11}
    source_orders = {1: [1, 2, 3]}
    from scrp import Scenario

    scenario = Scenario(5, source_seeds, source_orders, "immutable")
    source_seeds[1] = 99
    source_orders[1].reverse()
    assert scenario.order_seeds[1] == 11
    assert scenario.hidden_orders[1] == (1, 2, 3)
    with pytest.raises(TypeError):
        scenario.order_seeds[1] = 12
    with pytest.raises(TypeError):
        scenario.hidden_orders[1] = (3, 2, 1)
