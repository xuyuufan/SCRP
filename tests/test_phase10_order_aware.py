import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.order_aware_prototype import (
    controlled_untrained_probe,
    parameter_count,
)
from experiments.posttest_analysis import fixed_development_refs
from hier_pg.network import (
    O2_ORDER_XATTN_V1,
    O2_SHARED_ENCODER_V1,
    OrderAwareHierPolicyNetwork,
)
from scrp.formal_training import (
    FormalTrainingConfig,
    SCRPFormalTrainer,
    make_node_padding_mask,
    make_scrp_policy,
)
from tests.test_formal_training import tiny_provider
from tests.test_phase9_posttest_analysis import _manifest


SUMMARY = Path("experiments/summaries/phase10_order_aware_prototype.json")


def _policies(S=5):
    torch.manual_seed(7)
    control = make_scrp_policy("O2", S, 3)
    torch.manual_seed(7)
    treatment = make_scrp_policy(
        "O2", S, 3, architecture_version=O2_ORDER_XATTN_V1
    )
    return control.eval(), treatment.eval()


def _observation(S=5, real_orders=3):
    nodes = torch.zeros(1, S + 7, 12)
    nodes[:, :S, 0] = 0.0
    nodes[:, :S, 1] = torch.arange(S) / max(S - 1, 1)
    nodes[:, S:S + 6, 0] = 0.5
    nodes[:, S:S + real_orders, 1] = torch.linspace(0, 1, real_orders)
    nodes[:, S + real_orders:S + 6, 11] = 1.0
    nodes[:, -1, 0] = 1.0
    flat = nodes.reshape(1, -1)
    return flat, make_node_padding_mask(flat, "O2", S)


def test_architecture_versions_are_explicit_and_distinct():
    control, treatment = _policies()
    assert control.scrp_architecture_version == O2_SHARED_ENCODER_V1
    assert treatment.scrp_architecture_version == O2_ORDER_XATTN_V1
    assert isinstance(treatment, OrderAwareHierPolicyNetwork)
    with pytest.raises(ValueError, match="only for O2"):
        make_scrp_policy("O1", 5, 3, architecture_version=O2_ORDER_XATTN_V1)


def test_old_o2_state_dict_loads_strictly_into_unchanged_default_factory():
    old, _ = _policies()
    clone = make_scrp_policy("O2", 5, 3)
    clone.load_state_dict(old.state_dict(), strict=True)
    assert set(clone.state_dict()) == set(old.state_dict())
    assert not any("stack_to_order" in key for key in old.state_dict())


@pytest.mark.parametrize("S", [5, 7, 10])
def test_stack_order_separation_and_cross_attention_output_shape(S):
    _, treatment = _policies(S)
    observation, padding = _observation(S)
    stacks, orders, context = treatment.encode_partitions(observation, padding)
    assert stacks.shape == (1, S, 32)
    assert orders.shape == (1, 6, 32)
    assert context.shape == (1, 1, 32)
    assert treatment.encode(observation, padding).shape == (1, S + 7, 32)


def test_padding_is_masked_from_order_cross_attention():
    _, treatment = _policies()
    observation, padding = _observation()
    changed = observation.clone().reshape(1, 12, 12)
    changed[:, 5:11, 1:11][padding[:, 5:11]] = 0.731
    changed = changed.reshape(1, -1)
    legal = torch.zeros(1, 5, dtype=torch.bool)
    action = torch.tensor([0])
    first, _ = treatment.evaluate_actions(
        observation, legal, action, mode="low", node_padding_mask=padding
    )
    second, _ = treatment.evaluate_actions(
        changed, legal, action, mode="low", node_padding_mask=padding
    )
    assert torch.equal(first, second)


def test_hidden_future_order_invariant_and_revealed_order_sensitive():
    _, treatment = _policies(4)
    probe = controlled_untrained_probe(treatment)
    assert probe["hidden_future_order_invariant"] is True
    assert probe["padding_invariant"] is True
    assert probe["revealed_order_nonzero"] is True
    assert probe["passed"] is True


def test_candidate_width_and_greedy_stochastic_actions_are_legal():
    _, treatment = _policies()
    observation, padding = _observation()
    forbidden = torch.tensor([[True, False, False, True, False]])
    greedy, _ = treatment(
        observation, forbidden, greedy=True, mode="low", node_padding_mask=padding
    )
    sampled = [
        int(treatment(
            observation, forbidden, greedy=False, mode="low",
            node_padding_mask=padding,
        )[0].item())
        for _ in range(30)
    ]
    assert treatment.scrp_candidate_count == 5
    assert int(greedy.item()) in {1, 2, 4}
    assert set(sampled) <= {1, 2, 4}


def test_treatment_adds_parameters_without_changing_pointer_width():
    control, treatment = _policies()
    assert parameter_count(treatment) > parameter_count(control)
    assert treatment.scrp_candidate_count == control.scrp_candidate_count == 5


def test_same_seed_control_treatment_sampler_schedule_and_stability():
    manifest = _manifest()
    allowed = [ref.base_instance_id for ref in fixed_development_refs(manifest, "train")]
    config = FormalTrainingConfig(batch_size=2)
    control, treatment = _policies()
    control_trainer = SCRPFormalTrainer(
        config, manifest, tiny_provider, allowed_base_ids=allowed, policy=control
    )
    treatment_trainer = SCRPFormalTrainer(
        config, manifest, tiny_provider, allowed_base_ids=allowed, policy=treatment
    )
    torch.manual_seed(11)
    common = torch.get_rng_state()
    torch.set_rng_state(common.clone())
    control_metrics = control_trainer.train_iterations(2)
    torch.set_rng_state(common.clone())
    treatment_metrics = treatment_trainer.train_iterations(2)
    control_schedule = [
        (row.base_instance_id, row.variant, row.scenario_seed)
        for row in control_trainer.sample_history
    ]
    treatment_schedule = [
        (row.base_instance_id, row.variant, row.scenario_seed)
        for row in treatment_trainer.sample_history
    ]
    assert control_schedule == treatment_schedule
    assert all(np.isfinite(metric.loss) for metric in control_metrics + treatment_metrics)


def test_phase10_guard_rejects_test_split():
    with pytest.raises(ValueError, match="train/validation"):
        fixed_development_refs(_manifest(), "test")


def test_recorded_1k_smoke_and_compact_artifact_schema():
    record = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert record["status"] == "DEVELOPMENT_ONLY"
    assert record["splits_used"] == ["train", "validation"]
    assert record["formal_test_raw_rows_accessed"] is False
    assert record["formal_test_evaluation_performed"] is False
    assert record["formal_test_split_used"] is False
    assert record["ERI_auxiliary_objective_used"] is False
    assert record["FGB_semantics_changed"] is False
    assert record["smoke_1k"]["episodes_per_model"] == 1_000
    assert record["smoke_1k"]["passed"] is True
    assert record["development_checkpoints_saved"] is False
    encoded = json.dumps(record).lower()
    for forbidden in ("scenario_id", "scenario_seed", "raw_results"):
        assert forbidden not in encoded
