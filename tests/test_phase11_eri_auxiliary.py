import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from experiments.baselines import ERIBaseline
from experiments.phase11_eri_auxiliary import _fingerprint
from experiments.protocol import load_split_manifest
from scrp import Container, SCRPInstance
from scrp.environment import SCRPEnvironment
from scrp.formal_training import (
    ERI_AUXILIARY_VERSION,
    FormalTrainingConfig,
    SCRPFormalTrainer,
    eri_optimal_action_mask,
    eri_set_probability_loss,
    load_formal_training_config,
)
from scrp.models import SCRPConfig
from scrp.scenario import Scenario
from scrp.rl_adapter import SCRPRLAdapter


MANIFEST = "experiments/splits/scrp_split_v1.json"


class FixedSampler:
    def __init__(self, orders):
        self.orders = orders

    def sample(self, instance, root_seed):
        return Scenario(
            root_seed=root_seed,
            order_seeds={batch: root_seed + batch for batch in self.orders},
            hidden_orders={batch: tuple(order) for batch, order in self.orders.items()},
            scenario_id=f"phase11-fixed-{root_seed}",
        )


def decision_state(*, future_order=(2, 4)):
    instance = SCRPInstance(
        "phase11-label",
        4,
        3,
        (
            Container(1, 1), Container(3, 1),
            Container(2, 2), Container(4, 2),
        ),
        ((1, 2), (3,), (4,), ()),
        (1, 2),
    )
    core = SCRPEnvironment(
        SCRPConfig(4, 3), instance,
        FixedSampler({1: (1, 3), 2: future_order}),
    )
    env = SCRPRLAdapter(core, observation_version="O2")
    observation, info = env.reset(seed=17)
    return instance, core, observation, np.asarray(info["action_mask"], dtype=bool)


def tiny_instance(stacks=5):
    return SCRPInstance(
        "phase11-tiny", stacks, 3,
        (Container(1, 1), Container(2, 2), Container(3, 2)),
        ((1, 2), (3,)) + ((),) * (stacks - 2),
        (1, 2),
    )


def small_train_id(manifest):
    return next(
        ref.base_instance_id for ref in manifest.refs("train")
        if ref.parameter_group.startswith("S05_")
    )


def phase11_config(coefficient=0.1):
    return FormalTrainingConfig(
        batch_size=2,
        seed=991,
        eri_aux_coefficient=coefficient,
        eri_auxiliary_version=ERI_AUXILIARY_VERSION,
    )


def test_eri_positive_set_reuses_exact_scores_and_includes_all_tied_minima():
    instance, core, _, legal_mask = decision_state()
    legal = tuple(np.flatnonzero(legal_mask))
    positive = eri_optimal_action_mask(instance, core.state, legal)
    location = core.state.locations[core.state.current_target_id]
    blocker = core.state.stacks[location.stack_id].top_id
    scores = {
        action: ERIBaseline.destination_score(instance, core.state, blocker, action)
        for action in legal
    }
    minimum = min(scores.values())
    assert set(np.flatnonzero(positive)) == {
        action for action, score in scores.items() if score == minimum
    }
    assert set(np.flatnonzero(positive)) == {3}
    assert not positive[1] and not positive[2]


def test_multiple_tied_eri_optima_and_worse_action_exclusion():
    instance = SCRPInstance(
        "phase11-ties", 4, 3,
        (Container(1, 1), Container(2, 2), Container(3, 1)),
        ((1, 2), (3,), (), ()), (1, 2),
    )
    core = SCRPEnvironment(
        SCRPConfig(4, 3), instance, FixedSampler({1: (1, 3), 2: (2,)})
    )
    _, info = SCRPRLAdapter(core, observation_version="O2").reset(seed=4)
    positive = eri_optimal_action_mask(
        instance, core.state, np.flatnonzero(info["action_mask"])
    )
    assert set(np.flatnonzero(positive)) == {2, 3}
    assert not positive[1]


def test_all_legal_actions_tied():
    instance = SCRPInstance(
        "phase11-all-tied", 3, 3,
        (Container(1, 1), Container(2, 2)),
        ((1, 2), (), ()), (1, 2),
    )
    core = SCRPEnvironment(
        SCRPConfig(3, 3), instance, FixedSampler({1: (1,), 2: (2,)})
    )
    _, info = SCRPRLAdapter(core, observation_version="O2").reset(seed=4)
    legal = np.asarray(info["action_mask"], dtype=bool)
    assert np.array_equal(eri_optimal_action_mask(instance, core.state, np.flatnonzero(legal)), legal)


def test_set_probability_loss_is_stable_masked_and_differentiable():
    logits = torch.tensor([[1000.0, 999.0, -1000.0, 998.0]], requires_grad=True)
    legal = torch.tensor([[True, True, False, True]])
    log_probabilities = torch.log_softmax(logits.masked_fill(~legal, float("-inf")), dim=-1)
    positive = torch.tensor([[True, True, False, False]])
    loss = eri_set_probability_loss(log_probabilities, positive, legal)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert logits.grad[0, 2] == 0


def test_eri_target_is_invariant_to_hidden_future_permutation():
    first = decision_state(future_order=(2, 4))
    second = decision_state(future_order=(4, 2))
    assert np.array_equal(first[2], second[2])
    assert np.array_equal(
        eri_optimal_action_mask(first[0], first[1].state, np.flatnonzero(first[3])),
        eri_optimal_action_mask(second[0], second[1].state, np.flatnonzero(second[3])),
    )


def test_zero_lambda_exactly_reproduces_control_update_and_seed_pairing():
    manifest = load_split_manifest(MANIFEST)
    allowed = [small_train_id(manifest)]
    legacy = SCRPFormalTrainer(
        FormalTrainingConfig(batch_size=2, seed=991), manifest,
        lambda sample: tiny_instance(sample.num_stacks), allowed_base_ids=allowed,
    )
    auxiliary_zero = SCRPFormalTrainer(
        phase11_config(0.0), manifest,
        lambda sample: tiny_instance(sample.num_stacks), allowed_base_ids=allowed,
    )
    common_rng = torch.Generator().manual_seed(12345).get_state()
    torch.set_rng_state(common_rng.clone())
    legacy.train_iterations(1)
    torch.set_rng_state(common_rng.clone())
    auxiliary_zero.train_iterations(1)
    assert _fingerprint(legacy.sample_history) == _fingerprint(auxiliary_zero.sample_history)
    for expected, actual in zip(legacy.policy.parameters(), auxiliary_zero.policy.parameters()):
        assert torch.equal(expected, actual)


def test_phase11_smoke_training_checkpoint_and_config_round_trip(tmp_path):
    config = load_formal_training_config("experiments/configs/phase11_eri_aux_v1.json")
    config = replace(config, batch_size=2)
    manifest = load_split_manifest(MANIFEST)
    allowed = [small_train_id(manifest)]
    trainer = SCRPFormalTrainer(
        config, manifest, lambda sample: tiny_instance(sample.num_stacks),
        allowed_base_ids=allowed,
    )
    before = copy.deepcopy(trainer.policy.state_dict())
    metric = trainer.train_iterations(1)[0]
    assert metric.eri_aux_loss > 0.0
    assert metric.weighted_eri_gradient_norm > 0.0
    assert metric.invalid_actions == metric.truncations == 0
    assert any(
        not torch.equal(before[name], value)
        for name, value in trainer.policy.state_dict().items()
    )
    resumed = SCRPFormalTrainer.from_checkpoint(
        trainer.save_checkpoint(tmp_path / "phase11.pt"), manifest,
        lambda sample: tiny_instance(sample.num_stacks), allowed_base_ids=allowed,
    )
    assert resumed.config == config
    assert resumed.episodes_seen == trainer.episodes_seen


def test_illegal_positive_is_rejected():
    with pytest.raises(ValueError, match="illegal"):
        eri_set_probability_loss(
            torch.log_softmax(torch.tensor([[1.0, 2.0]]), dim=-1),
            torch.tensor([[False, True]]),
            torch.tensor([[True, False]]),
        )
