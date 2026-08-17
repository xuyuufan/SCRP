import pytest
import torch

from experiments.phase12_multiseed_replication import (
    FROZEN_SEEDS,
    _train_pair,
    hierarchical_seed_base_bootstrap,
    load_phase12_protocol,
    multiseed_success_gate,
)
from experiments.protocol import load_split_manifest
from scrp import Container, SCRPInstance
from scrp.formal_training import FormalTrainingConfig, resolve_training_device


def _rows(delta):
    control, treatment = [], []
    for base_id in ("a", "b", "c"):
        for dataset in ("DS1", "DS2"):
            control.append({"base_instance_id": base_id, "dataset": dataset, "relocations": 10})
            treatment.append({"base_instance_id": base_id, "dataset": dataset, "relocations": 10 + delta})
    return control, treatment


def _tiny_instance(stacks=5):
    return SCRPInstance(
        "phase12-tiny", stacks, 3,
        (Container(1, 1), Container(2, 2), Container(3, 2)),
        ((1, 2), (3,)) + ((),) * (stacks - 2),
        (1, 2),
    )


def _seed_result(delta=-0.2, strictly_worse_delta=-0.01, penalty_delta=-0.01):
    optimization = {
        "invalid_actions": 0, "truncations": 0,
        "numerical_failures": 0, "scenario_mismatches": 0,
    }
    return {
        "overall_delta": delta, "DS1_delta": delta, "DS2_delta": delta,
        "control_ERI_diagnostic": {
            "strictly_worse_ERI_score_action_rate": 0.10,
            "mean_ERI_score_penalty": 0.08,
        },
        "treatment_ERI_diagnostic": {
            "strictly_worse_ERI_score_action_rate": 0.10 + strictly_worse_delta,
            "mean_ERI_score_penalty": 0.08 + penalty_delta,
        },
        "control_optimization": dict(optimization),
        "treatment_optimization": dict(optimization),
    }


def test_phase12_protocol_freezes_exact_seeds_cuda_and_lambda():
    seeds, config = load_phase12_protocol("experiments/configs/phase12_multiseed_v1.json")
    assert seeds == FROZEN_SEEDS
    assert config.device == "cuda:0"
    assert config.eri_aux_coefficient == 0.10


def test_cuda_request_fails_explicitly_when_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="cannot access CUDA"):
        resolve_training_device("cuda:0")


def test_cpu_legacy_device_remains_supported():
    assert resolve_training_device(FormalTrainingConfig().device) == torch.device("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA regression requires NVIDIA GPU")
def test_cuda_auxiliary_pairing_and_checkpoint_map_location(tmp_path):
    manifest = load_split_manifest("experiments/splits/scrp_split_v1.json")
    _, template = load_phase12_protocol("experiments/configs/phase12_multiseed_v1.json")
    control, treatment, control_metrics, treatment_metrics, _, fingerprint = _train_pair(
        manifest, lambda sample: _tiny_instance(sample.num_stacks),
        template, FROZEN_SEEDS[0], 4,
    )
    assert fingerprint
    assert next(control.policy.parameters()).device.type == "cuda"
    assert next(treatment.policy.parameters()).device.type == "cuda"
    assert treatment_metrics[0].eri_aux_loss > 0.0
    assert treatment_metrics[0].weighted_eri_gradient_norm > 0.0
    assert control_metrics[0].invalid_actions == treatment_metrics[0].invalid_actions == 0
    checkpoint = treatment.save_checkpoint(tmp_path / "cuda-map-location.pt")
    allowed = treatment.allowed_base_ids
    from scrp.formal_training import SCRPFormalTrainer
    resumed = SCRPFormalTrainer.from_checkpoint(
        checkpoint, manifest, lambda sample: _tiny_instance(sample.num_stacks),
        allowed_base_ids=allowed,
    )
    assert next(resumed.policy.parameters()).device.type == "cuda"
    assert all(
        state["exp_avg"].device.type == state["exp_avg_sq"].device.type == "cuda"
        for state in resumed.optimizer.state.values()
    )


def test_hierarchical_bootstrap_preserves_seed_and_base_pairing():
    paired = {seed: _rows(-1 if index < 4 else 1) for index, seed in enumerate(FROZEN_SEEDS)}
    result = hierarchical_seed_base_bootstrap(paired, repetitions=1000, bootstrap_seed=17)
    assert result["overall"]["delta"] == pytest.approx(-0.6)
    assert result["DS1"]["delta"] == pytest.approx(-0.6)
    assert result["DS2"]["delta"] == pytest.approx(-0.6)
    assert "seed -> base layout" in result["method"]


def test_bootstrap_rejects_unpaired_coordinates():
    control, treatment = _rows(-1)
    treatment.pop()
    with pytest.raises(AssertionError, match="coordinates differ"):
        hierarchical_seed_base_bootstrap({FROZEN_SEEDS[0]: (control, treatment)}, repetitions=10)


def test_success_gate_is_frozen_and_requires_four_favorable_seeds():
    seed_results = [_seed_result() for _ in range(4)] + [_seed_result(delta=0.1)]
    pooled = {
        "overall": {"delta": -0.1, "ci95_high": 0.2},
        "DS1": {"delta": -0.1}, "DS2": {"delta": -0.1},
    }
    gate = multiseed_success_gate(seed_results, pooled, cuda_integrity=True)
    assert gate["passed"]
    assert gate["counts"]["overall_favorable"] == 4
    failed = multiseed_success_gate(
        [replace_result(result, overall_delta=0.1) if index == 3 else result for index, result in enumerate(seed_results)],
        pooled, cuda_integrity=True,
    )
    assert not failed["passed"]


def replace_result(result, **changes):
    return {**result, **changes}
