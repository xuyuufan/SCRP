from __future__ import annotations

import inspect
import json

import pytest
import torch

from experiments.baselines import (
    BaselineActionError,
    MinBlockingGreedyBaseline,
    RandomLegalBaseline,
    run_baseline_episode,
)
from experiments.evaluation import (
    BaselineAlgorithm,
    EvaluationCase,
    LowPolicyAlgorithm,
    aggregate_relocations,
    assert_paired_scenarios,
    evaluate_algorithm_on_schedule,
    save_raw_results,
)
from experiments.protocol import load_split_manifest
from scrp.environment import SCRPEnvironment
from scrp.models import Container, SCRPConfig, SCRPInstance
from scrp.scenario import Scenario
from scrp.training import make_scrp_o1_policy


class FixedSampler:
    def __init__(self, hidden_orders):
        self.hidden_orders = hidden_orders

    def sample(self, instance, root_seed):
        return Scenario(
            root_seed=root_seed,
            order_seeds={batch: root_seed + batch for batch in instance.batch_order},
            hidden_orders=self.hidden_orders,
            scenario_id=f"fixed-{instance.instance_id}-{root_seed}",
        )


def decision_instance() -> SCRPInstance:
    return SCRPInstance(
        instance_id="baseline-tiny",
        num_stacks=4,
        max_tiers=4,
        containers=tuple(Container(i, 1) for i in range(1, 5)),
        initial_stacks=((1, 2), (3,), (4,), ()),
        batch_order=(1,),
        metadata={"source": "hand-test"},
    )


def make_env(order=(1, 3, 2, 4)) -> SCRPEnvironment:
    instance = decision_instance()
    return SCRPEnvironment(
        SCRPConfig(instance.num_stacks, instance.max_tiers, max_steps=100),
        instance,
        FixedSampler({1: order}),
    )


def test_baseline_api_has_no_scenario_parameter():
    for baseline_type in (RandomLegalBaseline, MinBlockingGreedyBaseline):
        parameters = inspect.signature(baseline_type.select_destination).parameters
        assert tuple(parameters) == (
            "self", "instance", "state", "legal_destinations"
        )


def test_random_baseline_selects_only_legal_destinations():
    env = make_env()
    state = env.reset(seed=7)
    legal = env.legal_destinations()
    baseline = RandomLegalBaseline()
    baseline.reset(99)
    assert {baseline.select_destination(env.instance, state, legal) for _ in range(100)} <= set(legal)


def test_random_action_seed_reproduces_action_sequence():
    first = run_baseline_episode(make_env(), RandomLegalBaseline(), 4, action_seed=88)
    second = run_baseline_episode(make_env(), RandomLegalBaseline(), 4, action_seed=88)
    assert first.actions == second.actions
    assert first.relocations == second.relocations


def test_action_seed_does_not_change_scenario_id():
    first = run_baseline_episode(make_env(), RandomLegalBaseline(), 4, action_seed=1)
    second = run_baseline_episode(make_env(), RandomLegalBaseline(), 4, action_seed=2)
    assert first.scenario_id == second.scenario_id


def test_greedy_is_deterministic_and_uses_stable_stack_tie_break():
    env = make_env()
    state = env.reset(seed=0)
    baseline = MinBlockingGreedyBaseline()
    baseline.reset(1)
    first = baseline.select_destination(env.instance, state, env.legal_destinations())
    baseline.reset(999)
    second = baseline.select_destination(env.instance, state, env.legal_destinations())
    assert first == second == 2


def test_greedy_responds_to_current_revealed_order():
    baseline = MinBlockingGreedyBaseline()
    env_a = make_env((1, 3, 2, 4))
    env_b = make_env((1, 4, 2, 3))
    state_a = env_a.reset(seed=0)
    state_b = env_b.reset(seed=0)
    baseline.reset(0)
    action_a = baseline.select_destination(env_a.instance, state_a, env_a.legal_destinations())
    action_b = baseline.select_destination(env_b.instance, state_b, env_b.legal_destinations())
    assert (action_a, action_b) == (2, 1)


def test_greedy_action_unchanged_when_only_hidden_future_order_changes():
    instance = SCRPInstance(
        "future-hidden", 3, 3,
        (Container(1, 1), Container(2, 2), Container(3, 2)),
        ((1, 2), (3,), ()), (1, 2),
    )
    config = SCRPConfig(3, 3)
    env_a = SCRPEnvironment(config, instance, FixedSampler({1: (1,), 2: (2, 3)}))
    env_b = SCRPEnvironment(config, instance, FixedSampler({1: (1,), 2: (3, 2)}))
    state_a = env_a.reset(seed=5)
    state_b = env_b.reset(seed=5)
    assert state_a == state_b
    baseline = MinBlockingGreedyBaseline()
    baseline.reset(0)
    assert (
        baseline.select_destination(instance, state_a, env_a.legal_destinations())
        == baseline.select_destination(instance, state_b, env_b.legal_destinations())
    )


def test_public_state_mutation_cannot_modify_environment():
    env = make_env()
    state = env.reset(seed=0)
    state.stacks[0].containers.clear()
    assert env.state.stacks[0].containers == [1, 2]


@pytest.mark.parametrize("baseline", [RandomLegalBaseline(), MinBlockingGreedyBaseline()])
def test_rollout_terminates_with_reward_and_decision_invariants(baseline):
    result = run_baseline_episode(make_env(), baseline, 3, action_seed=10)
    assert result.terminated and not result.truncated
    assert result.total_reward == -result.relocations
    assert result.decision_count == result.relocations
    assert result.invalid_action_count == 0


def test_rollout_rejects_illegal_baseline_action():
    class BadBaseline:
        name = "bad"
        def reset(self, action_seed):
            pass
        def select_destination(self, instance, state, legal_destinations):
            return 999

    with pytest.raises(BaselineActionError, match="illegal destination"):
        run_baseline_episode(make_env(), BadBaseline(), 1, action_seed=2)


def evaluation_case() -> EvaluationCase:
    return EvaluationCase(
        instance=decision_instance(), dataset="DS1", split="train",
        base_instance_id="baseline-tiny", parameter_group="hand",
        scenario_seeds=(10, 11, 12),
    )


def test_evaluation_emits_one_phase35_result_per_seed_with_provenance():
    results = evaluate_algorithm_on_schedule(
        BaselineAlgorithm(MinBlockingGreedyBaseline), [evaluation_case()]
    )
    assert len(results) == 3
    assert [result.scenario_seed for result in results] == [10, 11, 12]
    assert {(r.dataset, r.split, r.base_instance_id, r.parameter_group) for r in results} == {
        ("DS1", "train", "baseline-tiny", "hand")
    }


def test_aggregate_reports_count_mean_std_min_max():
    results = evaluate_algorithm_on_schedule(
        BaselineAlgorithm(MinBlockingGreedyBaseline), [evaluation_case()]
    )
    summary = aggregate_relocations(results)[0]
    assert summary.count == 3
    assert summary.minimum <= summary.mean <= summary.maximum
    assert summary.std >= 0


def test_raw_result_jsonl_round_trip(tmp_path):
    results = evaluate_algorithm_on_schedule(
        BaselineAlgorithm(RandomLegalBaseline), [evaluation_case()]
    )
    path = save_raw_results(results, tmp_path / "raw.jsonl")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records == [result.to_record() for result in results]


def test_random_greedy_and_current_low_receive_paired_scenario_ids():
    torch.manual_seed(0)
    policy = make_scrp_o1_policy(embed_dim=16, num_heads=4, ffn_dim=32)
    algorithms = (
        BaselineAlgorithm(RandomLegalBaseline),
        BaselineAlgorithm(MinBlockingGreedyBaseline),
        LowPolicyAlgorithm(policy),
    )
    result_sets = [evaluate_algorithm_on_schedule(a, [evaluation_case()]) for a in algorithms]
    assert_paired_scenarios(*result_sets)
    assert all(result.terminated and not result.truncated for results in result_sets for result in results)


def test_evaluation_does_not_mutate_frozen_split_manifest():
    path = "experiments/splits/scrp_split_v1.json"
    before = open(path, "rb").read()
    manifest = load_split_manifest(path)
    evaluate_algorithm_on_schedule(
        BaselineAlgorithm(MinBlockingGreedyBaseline), [evaluation_case()]
    )
    assert manifest.num_base_instances == 1440
    assert open(path, "rb").read() == before
