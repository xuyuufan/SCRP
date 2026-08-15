from __future__ import annotations

from fractions import Fraction

import pytest

from experiments.baselines import ERIBaseline, run_baseline_episode
from scrp import (
    Container,
    SCRPConfig,
    SCRPEnvironment,
    SCRPInstance,
    Scenario,
    load_instance_json,
)


class FixedSampler:
    def __init__(self, hidden_orders, suffix="reference"):
        self.hidden_orders = hidden_orders
        self.suffix = suffix

    def sample(self, instance, root_seed):
        return Scenario(
            root_seed=root_seed,
            order_seeds={batch: root_seed + batch for batch in instance.batch_order},
            hidden_orders=self.hidden_orders,
            scenario_id=f"{instance.instance_id}-{root_seed}-{self.suffix}",
        )


def make_env(
    stacks,
    container_batches,
    batch_order,
    hidden_orders,
    *,
    max_tiers=4,
    instance_id="eri-golden",
    suffix="reference",
):
    instance = SCRPInstance(
        instance_id=instance_id,
        num_stacks=len(stacks),
        max_tiers=max_tiers,
        containers=tuple(
            Container(container_id, batch_id)
            for container_id, batch_id in sorted(container_batches.items())
        ),
        initial_stacks=tuple(tuple(stack) for stack in stacks),
        batch_order=tuple(batch_order),
    )
    return SCRPEnvironment(
        SCRPConfig(instance.num_stacks, instance.max_tiers, max_steps=1_000),
        instance,
        FixedSampler(hidden_orders, suffix),
    )


def select(env, seed=17):
    state = env.reset(seed=seed)
    baseline = ERIBaseline()
    baseline.reset(999)
    return state, baseline.select_destination(
        env.instance, state, env.legal_destinations()
    )


def test_eri_obvious_minimum_score_is_selected_and_legal():
    # Stack 1 contains an earlier-batch container (score 1), while stack 2
    # contains a later-batch container (score 0), so published ERI selects 2.
    env = make_env(
        ((1, 2), (3,), (4,), ()),
        {1: 1, 2: 2, 3: 1, 4: 3},
        (1, 2, 3),
        {1: (1, 3), 2: (2,), 3: (4,)},
    )
    _, action = select(env)
    assert action == 2
    assert action in env.legal_destinations()
    assert len(env.legal_destinations()) == 3


def test_eri_strict_precedence_score_branch_is_additive():
    # Two earlier-batch containers below a batch-2 blocker each contribute 1.
    env = make_env(
        ((1, 2), (3, 4), (5,), ()),
        {1: 1, 2: 2, 3: 1, 4: 1, 5: 3},
        (1, 2, 3),
        {1: (1, 3, 4), 2: (2,), 3: (5,)},
    )
    state = env.reset(seed=1)
    assert ERIBaseline.destination_score(env.instance, state, 2, 1) == 2
    assert ERIBaseline.destination_score(env.instance, state, 2, 2) == 0


def test_eri_unrevealed_same_batch_contributes_one_half():
    # Blocker 2 and lower container 3 share future batch 2; their internal
    # order is unknown, so the paper equation contributes exactly one-half.
    env = make_env(
        ((1, 2), (3,), (4,), ()),
        {1: 1, 2: 2, 3: 2, 4: 3},
        (1, 2, 3),
        {1: (1,), 2: (2, 3), 3: (4,)},
    )
    state = env.reset(seed=2)
    assert ERIBaseline.destination_score(env.instance, state, 2, 1) == Fraction(1, 2)
    assert ERIBaseline.destination_score(env.instance, state, 2, 2) == 0


def test_eri_mixed_score_combines_one_and_one_half():
    # An earlier-batch container contributes 1 and an unresolved batch peer
    # contributes 1/2, giving the golden score 3/2 for the whole destination.
    env = make_env(
        ((1, 2), (3, 4), (5,), ()),
        {1: 1, 2: 2, 3: 1, 4: 2, 5: 3},
        (1, 2, 3),
        {1: (1, 3), 2: (2, 4), 3: (5,)},
    )
    state = env.reset(seed=3)
    assert ERIBaseline.destination_score(env.instance, state, 2, 1) == Fraction(3, 2)


def test_eri_empty_destination_has_zero_score():
    # The sum over an empty stack is empty, so its ERI is exactly zero.
    env = make_env(
        ((1, 2), (3,), ()),
        {1: 1, 2: 2, 3: 1},
        (1, 2),
        {1: (1, 3), 2: (2,)},
    )
    state = env.reset(seed=4)
    assert ERIBaseline.destination_score(env.instance, state, 2, 2) == 0


def test_eri_nearly_full_zero_score_stack_wins_height_tie_break():
    # Both candidates score zero; the legal height-3 stack is preferred to the
    # empty stack by the published taller-column tie-break.
    env = make_env(
        ((1, 2), (3, 4, 5), ()),
        {1: 1, 2: 2, 3: 3, 4: 3, 5: 3},
        (1, 2, 3),
        {1: (1,), 2: (2,), 3: (3, 4, 5)},
        max_tiers=4,
    )
    _, action = select(env)
    assert action == 1


def test_eri_primary_score_overrides_destination_height():
    # A tall stack with three earlier containers scores 3; a short later-batch
    # stack scores 0, so height cannot override the primary ERI criterion.
    env = make_env(
        ((1, 2), (3, 4, 5), (6,)),
        {1: 1, 2: 2, 3: 1, 4: 1, 5: 1, 6: 3},
        (1, 2, 3),
        {1: (1, 3, 4, 5), 2: (2,), 3: (6,)},
    )
    _, action = select(env)
    assert action == 2


def test_eri_equal_scores_prefer_tallest_stack():
    # All destination containers are in a later batch and score zero; the
    # height-2 candidate therefore beats the height-1 candidate.
    env = make_env(
        ((1, 2), (3, 4), (5,)),
        {1: 1, 2: 2, 3: 3, 4: 3, 5: 3},
        (1, 2, 3),
        {1: (1,), 2: (2,), 3: (3, 4, 5)},
    )
    _, action = select(env)
    assert action == 1


def test_eri_equal_score_and_height_prefer_leftmost_stack_deterministically():
    # Equal zero scores and equal heights reach Galle's final leftmost tie-break.
    env = make_env(
        ((1, 2), (3,), (4,)),
        {1: 1, 2: 2, 3: 3, 4: 3},
        (1, 2, 3),
        {1: (1,), 2: (2,), 3: (3, 4)},
    )
    state = env.reset(seed=5)
    baseline = ERIBaseline()
    baseline.reset(1)
    first = baseline.select_destination(env.instance, state, env.legal_destinations())
    baseline.reset(999_999)
    second = baseline.select_destination(env.instance, state, env.legal_destinations())
    assert first == second == 1


def test_eri_uses_revealed_order_when_blocker_is_in_current_batch():
    # With target 1 fixed first, swapping whether 3 or 4 precedes blocker 2
    # swaps the zero-score destination selected by ERI.
    common = (((1, 2), (3,), (4,)), {1: 1, 2: 1, 3: 1, 4: 1}, (1,))
    env_a = make_env(*common, {1: (1, 3, 2, 4)}, instance_id="revealed-a")
    env_b = make_env(*common, {1: (1, 4, 2, 3)}, instance_id="revealed-b")
    _, action_a = select(env_a)
    _, action_b = select(env_b)
    assert (action_a, action_b) == (2, 1)


def test_eri_supports_blocker_from_a_later_unrevealed_batch():
    # Restricted CRP can place a future-batch container above today's target;
    # a later-than-blocker destination scores zero and is preferred.
    env = make_env(
        ((1, 2), (3,), (4,)),
        {1: 1, 2: 3, 3: 2, 4: 4},
        (1, 2, 3, 4),
        {1: (1,), 2: (3,), 3: (2,), 4: (4,)},
    )
    _, action = select(env)
    assert action == 2


def test_eri_future_hidden_order_change_keeps_visible_state_and_action_same():
    # Reversing the hidden order of future batch 2 cannot alter its public
    # one-half contribution before reveal, hence cannot alter the action.
    common = (
        ((1, 2), (3,), (4,)),
        {1: 1, 2: 2, 3: 2, 4: 3},
        (1, 2, 3),
    )
    env_a = make_env(
        *common, {1: (1,), 2: (2, 3), 3: (4,)}, suffix="future-a"
    )
    env_b = make_env(
        *common, {1: (1,), 2: (3, 2), 3: (4,)}, suffix="future-b"
    )
    state_a = env_a.reset(seed=6)
    state_b = env_b.reset(seed=6)
    assert state_a == state_b
    baseline = ERIBaseline()
    baseline.reset(0)
    action_a = baseline.select_destination(env_a.instance, state_a, env_a.legal_destinations())
    action_b = baseline.select_destination(env_b.instance, state_b, env_b.legal_destinations())
    assert action_a == action_b


def test_eri_rollout_terminates_with_reward_and_action_invariants():
    # The unified runner must preserve one decision per relocation and the
    # project's reward identity while never emitting an invalid action.
    env = make_env(
        ((1, 2), (3,), (4,), ()),
        {1: 1, 2: 2, 3: 1, 4: 3},
        (1, 2, 3),
        {1: (1, 3), 2: (2,), 3: (4,)},
    )
    result = run_baseline_episode(env, ERIBaseline(), 7, action_seed=123)
    assert result.terminated and not result.truncated
    assert result.total_reward == -result.relocations
    assert result.decision_count == result.relocations
    assert result.invalid_action_count == 0


@pytest.mark.parametrize(
    "dataset,path",
    [
        ("DS1", "data/phase3_sanity/S05_T03_mu050/ds1_001.json"),
        ("DS2", "data/phase3_sanity/S05_T03_mu050/ds2_001.json"),
    ],
)
def test_eri_runs_on_reproduced_ds1_and_ds2_artifacts(dataset, path):
    # The same public ERI rule must complete both published static variants;
    # this is an integration check, not a cross-dataset scenario pairing.
    instance = load_instance_json(path)
    env = SCRPEnvironment(SCRPConfig(instance.num_stacks, instance.max_tiers), instance)
    result = run_baseline_episode(env, ERIBaseline(), 8, action_seed=0)
    assert dataset in {"DS1", "DS2"}
    assert result.terminated and result.invalid_action_count == 0
