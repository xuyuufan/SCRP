import ast
import inspect
from dataclasses import replace

import numpy as np
import pytest
import torch

from experiments.phase14_rl_vs_eri_development import (
    FROZEN_SEEDS,
    Phase14PreflightError,
    checkpoint_preflight,
    development_success_gate,
    eri_mechanism_diagnostics,
    greedy_public_action,
    hierarchical_paired_bootstrap,
    load_phase14_protocol,
    paired_rollout,
    phase14_development_holdout,
    relocation_metrics,
    secondary_statistics,
)
from experiments.protocol import load_split_manifest
from scrp import Container, SCRPInstance
from scrp.formal_training import FormalTrainingConfig, TrainingSample, make_scrp_policy


PROTOCOL = "experiments/configs/phase14_rl_vs_eri_development_v1.json"
MANIFEST = "experiments/splits/scrp_split_v1.json"


def _tiny_instance():
    return SCRPInstance(
        instance_id="phase14-tiny",
        num_stacks=5,
        max_tiers=3,
        containers=(Container(1, 1), Container(2, 2), Container(3, 2)),
        initial_stacks=((1, 2), (3,), (), (), ()),
        batch_order=(1, 2),
    )


def _synthetic_rows(delta=-1, scenarios=2):
    rows = []
    for seed in FROZEN_SEEDS:
        for dataset in ("DS1", "DS2"):
            for base in ("base-a", "base-b"):
                for scenario in range(scenarios):
                    eri = 10
                    rows.append({
                        "seed": seed, "dataset": dataset,
                        "base_instance_id": base, "scenario_seed": scenario,
                        "rl_relocations": eri + delta, "eri_relocations": eri,
                        "delta": delta, "invalid_actions": 0, "truncations": 0,
                        "numerical_failures": 0, "scenario_mismatches": 0,
                        "S": 5, "fill": 0.5,
                    })
    return rows


def test_checkpoint_selection_was_frozen_before_eri_evaluation():
    protocol = load_phase14_protocol(PROTOCOL)
    selection = protocol["checkpoint_selection"]
    assert selection["frozen_before_eri_evaluation"] is True
    assert [item["episode"] for item in selection["selections"]] == [15_000] * 3
    assert "Phase 14 ERI results are prohibited" in selection["selection_inputs"]


def test_missing_frozen_checkpoints_fail_closed_without_training(tmp_path):
    protocol = load_phase14_protocol(PROTOCOL)
    result = checkpoint_preflight(protocol, tmp_path)
    assert result["ready"] is False
    assert len(result["missing"]) == 3
    assert result["new_training_performed"] is False
    assert result["eri_evaluation_started"] is False


def test_holdout_is_validation_only_disjoint_and_large():
    manifest = load_split_manifest(MANIFEST)
    holdout = phase14_development_holdout(manifest)
    assert len(holdout) == len({ref.base_instance_id for ref in holdout}) == 192
    assert all(manifest.split_for_base(ref.base_instance_id) == "validation" for ref in holdout)
    assert all(ref not in manifest.refs("test") for ref in holdout)


def test_phase14_source_never_requests_test_split():
    import experiments.phase14_rl_vs_eri_development as module

    tree = ast.parse(inspect.getsource(module))
    requested = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"refs", "split_for_base", "seed_for"}:
                for argument in node.args:
                    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                        requested.append(argument.value)
    assert "test" not in requested


def test_hidden_future_invariance_and_deterministic_low_index_tie():
    class TiedPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def action_log_probabilities(self, observation, forbidden, **_):
            scores = torch.zeros((observation.shape[0], forbidden.shape[1]), device=observation.device)
            return torch.log_softmax(scores.masked_fill(forbidden, float("-inf")), dim=-1)

    policy = TiedPolicy()
    observation = np.zeros((12, 12), dtype=np.float32)
    action_a, _ = greedy_public_action(policy, observation, [False, True, True, False, True], 5)
    action_b, _ = greedy_public_action(policy, observation.copy(), [False, True, True, False, True], 5)
    assert action_a == action_b == 1
    assert list(inspect.signature(greedy_public_action).parameters) == [
        "policy", "observation", "legal_mask", "num_stacks"
    ]


def test_non_finite_policy_output_is_detected():
    class InvalidPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def action_log_probabilities(self, observation, forbidden, **_):
            return torch.full(
                (observation.shape[0], forbidden.shape[1]), float("nan"),
                device=observation.device,
            )

    with pytest.raises(FloatingPointError, match="non-finite"):
        greedy_public_action(
            InvalidPolicy(), np.zeros((12, 12), dtype=np.float32),
            [False, True, True, False, True], 5,
        )


def test_paired_rollout_uses_same_initial_state_and_scenario_and_keeps_policy_frozen():
    torch.manual_seed(14)
    policy = make_scrp_policy("O2", 5, 3, Mmax=6, device="cpu")
    config = replace(FormalTrainingConfig(observation_version="O2", Mmax=6), seed=20260816)
    sample = TrainingSample("base", "phase14-tiny", "DS1", 2_000_000_000_000, 0, 5)
    before = {name: tensor.detach().clone() for name, tensor in policy.state_dict().items()}
    row, actions = paired_rollout(
        _tiny_instance(), sample, policy, config, "S05_T03_mu0.50"
    )
    assert row["scenario_mismatches"] == 0
    assert row["invalid_actions"] == row["truncations"] == 0
    assert row["delta"] == row["rl_relocations"] - row["eri_relocations"]
    assert actions
    assert all(torch.equal(before[name], tensor) for name, tensor in policy.state_dict().items())
    assert all(parameter.grad is None for parameter in policy.parameters())


def test_phase14_module_has_no_training_or_optimizer_step():
    import experiments.phase14_rl_vs_eri_development as module

    source = inspect.getsource(module)
    assert "SCRPFormalTrainer" not in source
    assert ".optimizer" not in source
    assert ".backward(" not in source
    assert "train_iterations" not in source


def test_relocation_metrics_aggregate_ds1_ds2_and_win_tie_loss():
    rows = _synthetic_rows(delta=-1)
    rows[0] = {**rows[0], "rl_relocations": 10, "delta": 0}
    rows[1] = {**rows[1], "rl_relocations": 11, "delta": 1}
    result = relocation_metrics(rows)
    assert result["coordinates"] == len(rows)
    assert result["RL_wins"] == len(rows) - 2
    assert result["ties"] == result["ERI_wins"] == 1


def test_hierarchical_bootstrap_is_paired_and_reproducible():
    rows = _synthetic_rows(delta=-1)
    first = hierarchical_paired_bootstrap(rows, repetitions=200, bootstrap_seed=14)
    second = hierarchical_paired_bootstrap(rows, repetitions=200, bootstrap_seed=14)
    assert first == second
    assert first["delta"] == first["ci95_low"] == first["ci95_high"] == -1
    assert "seed -> dataset -> base layout -> scenario" in first["method"]


@pytest.mark.parametrize(
    ("delta", "ci_high", "expected"),
    [(-0.2, -0.01, "YES"), (-0.2, 0.01, "INCONCLUSIVE"), (0.2, 0.4, "NO")],
)
def test_rl_beats_eri_development_gate(delta, ci_high, expected):
    overall = {"delta": delta, "RL_wins": 8 if delta < 0 else 2,
               "ERI_wins": 2 if delta < 0 else 8}
    bootstrap = {"ci95_high": ci_high}
    dataset = {"delta": min(delta, 0)}
    seeds = [{"delta": delta}, {"delta": delta}, {"delta": -delta}]
    integrity = {
        "invalid_actions": 0, "truncations": 0, "numerical_failures": 0,
        "scenario_mismatches": 0, "hidden_information_leaks": 0,
        "checkpoint_selection_frozen": True, "test_split_usage": 0,
    }
    result = development_success_gate(
        overall, bootstrap, dataset, dataset, seeds, integrity
    )
    assert result["result"] == expected


def test_gate_rejects_each_integrity_failure():
    overall = {"delta": -1.0, "RL_wins": 10, "ERI_wins": 1}
    bootstrap = {"ci95_high": -0.1}
    dataset = {"delta": -1.0}
    seeds = [{"delta": -1.0}] * 3
    baseline = {
        "invalid_actions": 0, "truncations": 0, "numerical_failures": 0,
        "scenario_mismatches": 0, "hidden_information_leaks": 0,
        "checkpoint_selection_frozen": True, "test_split_usage": 0,
    }
    for field in ("invalid_actions", "truncations", "numerical_failures",
                  "scenario_mismatches", "hidden_information_leaks", "test_split_usage"):
        integrity = {**baseline, field: 1}
        assert development_success_gate(
            overall, bootstrap, dataset, dataset, seeds, integrity
        )["result"] == "NO"


def test_secondary_statistics_and_eri_action_diagnostics():
    rows = _synthetic_rows(delta=-1)
    statistics = secondary_statistics(rows)
    assert statistics["paired_wilcoxon"]["p_value"] < 0.05
    assert statistics["paired_t_test"]["p_value"] == 0.0
    actions = [
        {"exact": 1, "equivalent": 1, "strictly_worse": 0, "penalty": 0,
         "non_eri_outcome": None},
        {"exact": 0, "equivalent": 0, "strictly_worse": 1, "penalty": 1,
         "non_eri_outcome": "downstream_better"},
    ]
    diagnostic = eri_mechanism_diagnostics(actions)
    assert diagnostic["exact_action_agreement_rate"] == 0.5
    assert diagnostic["strictly_worse_ERI_score_rate"] == 0.5
    assert diagnostic["non_ERI_minimum_actions"]["downstream_better"] == 1


def test_checkpoint_paths_cannot_escape_repository(tmp_path):
    protocol = load_phase14_protocol(PROTOCOL)
    protocol["checkpoint_selection"]["selections"][0]["relative_path"] = "../outside.pt"
    with pytest.raises(Phase14PreflightError, match="repository-relative"):
        checkpoint_preflight(protocol, tmp_path)
