import ast
import inspect

import pytest
import torch

from experiments.phase13_longrun_stability import (
    EPISODES_PER_ARM,
    FROZEN_SEEDS,
    TOTAL_TRAINING_EPISODES,
    VALIDATION_CHECKPOINTS,
    _scenario_fingerprint,
    aggregate_training_window,
    classify_delta_trajectory,
    load_phase13_protocol,
    longrun_success_gate,
    optimization_warning_gate,
)
from scrp.formal_training import FormalIterationMetrics, TrainingSample, resolve_training_device


def _metric(index, *, entropy=0.8, gradient=1.0, clipped=True, updates=0):
    return FormalIterationMetrics(
        iteration=index, episodes=4, mean_policy_relocations=1.0,
        mean_baseline_relocations=1.0, mean_return=-1.0, mean_advantage=0.0,
        loss=0.1, policy_loss=0.05, eri_aux_loss=0.2, entropy=entropy,
        grad_norm=gradient, gradient_clipped=clipped, rl_gradient_norm=2.0,
        weighted_eri_gradient_norm=0.2, invalid_actions=0, truncations=0,
        baseline_updates=updates, low_decisions=1, empty_decision_episodes=0,
        scenario_mismatches=0,
    )


def test_protocol_freezes_seeds_budget_schedule_and_cuda():
    seeds, checkpoints, config = load_phase13_protocol(
        "experiments/configs/phase13_longrun_stability_v1.json"
    )
    assert seeds == FROZEN_SEEDS == (20260816, 20260818, 20260819)
    assert checkpoints == VALIDATION_CHECKPOINTS
    assert EPISODES_PER_ARM == 15_000
    assert TOTAL_TRAINING_EPISODES == 90_000
    assert config.device == "cuda:0" and config.eri_aux_coefficient == 0.10


def test_phase13_cuda_has_no_cpu_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="cannot access CUDA"):
        resolve_training_device("cuda:0")


def test_paired_scenario_fingerprint_is_compact_and_sensitive():
    sample = TrainingSample("base", "instance", "DS1", 17, 0, 5)
    same = TrainingSample("base", "instance", "DS1", 17, 0, 5)
    changed = TrainingSample("base", "instance", "DS2", 17, 0, 5)
    assert _scenario_fingerprint([sample]) == _scenario_fingerprint([same])
    assert _scenario_fingerprint([sample]) != _scenario_fingerprint([changed])


def test_entropy_gradient_clipping_and_loss_window_aggregation():
    metrics = [
        _metric(1, entropy=0.9, gradient=1.0, clipped=False),
        _metric(2, entropy=0.7, gradient=3.0, clipped=True, updates=1),
    ]
    result = aggregate_training_window(
        metrics, start_episode=0, end_episode=8, accepted_refreshes=1
    )
    assert result["entropy_mean"] == pytest.approx(0.8)
    assert result["entropy_median"] == pytest.approx(0.8)
    assert result["pre_clip_gradient_norm_mean"] == pytest.approx(2.0)
    assert result["pre_clip_gradient_norm_p95"] == pytest.approx(2.9)
    assert result["gradient_clipping_frequency"] == pytest.approx(0.5)
    assert result["weighted_ERI_to_RL_gradient_ratio_mean"] == pytest.approx(0.1)
    assert result["FGB_accepted_refreshes_in_window"] == 1
    assert result["FGB_rejected_refreshes_in_window"] == 1


@pytest.mark.parametrize(("deltas", "expected"), [
    ([-0.1, -0.2, -0.3, -0.3, -0.4, -0.4], "sustained_improvement"),
    ([-0.5, -0.4, -0.3, -0.2, -0.2, -0.2], "early_improvement_then_deterioration"),
    ([-0.3, -0.3, -0.2, -0.2, -0.2, -0.2], "early_improvement_then_plateau"),
    ([-0.6, 0.6, -0.6, 0.6, -0.6, 0.6], "high_variance"),
])
def test_trajectory_classification_is_predefined(deltas, expected):
    assert classify_delta_trajectory(deltas)["classification"] == expected


def _aggregate(delta=-0.2, clip=0.995, entropy=0.7, gradient=2.0):
    return {
        "pooled_paired_bootstrap": {
            "overall": {"delta": delta, "ci95_high": 0.1},
            "DS1": {"delta": delta}, "DS2": {"delta": delta},
        },
        "favorable_seed_counts": {"overall": 3, "DS1": 3, "DS2": 3},
        "ERI_mechanism": {
            "strictly_worse_ERI_score_action_rate": {"control": 0.1, "treatment": 0.05},
            "mean_ERI_score_penalty": {"control": 0.1, "treatment": 0.05},
        },
        "optimization_window": {
            "gradient_clipping_frequency": {"treatment": clip},
            "entropy_mean": {"treatment": entropy},
            "pre_clip_gradient_norm_mean": {"treatment": gradient},
        },
        "integrity": {
            "invalid_actions": 0, "truncations": 0,
            "numerical_failures": 0, "scenario_mismatches": 0,
        },
    }


def _endpoint_record(delta=-0.2, entropy=0.7):
    integrity = {
        "invalid_actions": 0, "truncations": 0,
        "numerical_failures": 0, "scenario_mismatches": 0,
        "entropy_mean": entropy,
    }
    return {
        "overall_delta": delta,
        "control": {"optimization_window": dict(integrity)},
        "treatment": {"optimization_window": dict(integrity)},
    }


def test_success_and_optimization_warning_gates_are_frozen():
    aggregates = {str(cp): _aggregate() for cp in VALIDATION_CHECKPOINTS}
    records = [_endpoint_record() for _ in FROZEN_SEEDS]
    assert longrun_success_gate(aggregates, records, cuda_integrity=True)["passed"]
    warning = optimization_warning_gate(aggregates, records)
    assert warning["warning"]
    assert warning["triggers"]["consecutive_late_treatment_clipping_at_least_99_percent"]


def test_phase13_source_only_requests_train_and_validation_splits():
    import experiments.phase13_longrun_stability as module
    tree = ast.parse(inspect.getsource(module))
    values = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "fixed_development_refs":
            values.append(ast.literal_eval(node.args[1]))
    assert values and set(values) <= {"train", "validation"}
