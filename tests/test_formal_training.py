import copy
import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from experiments.protocol import ScenarioSeedSchedule, load_split_manifest
from scrp import Container, SCRPInstance
from scrp.formal_training import (
    BaseBalancedTrainingSampler,
    FormalTrainingConfig,
    KuTrainingInstanceProvider,
    SCRPFormalTrainer,
    TrainingSample,
    frozen_greedy_advantages,
    load_formal_training_config,
    make_node_padding_mask,
    make_scrp_policy,
    run_formal_episode,
)
from scrp.rl_adapter import SCRPRLAdapter
from scrp.environment import SCRPEnvironment
from scrp.models import SCRPConfig


MANIFEST = "experiments/splits/scrp_split_v1.json"
SOURCE_ROOT = "tmp/StochasticCRP/crptw_instance"


def tiny_instance(stacks=5, *, terminal=False):
    if terminal:
        containers = (Container(1, 1),)
        layout = ((1,),) + ((),) * (stacks - 1)
        batches = (1,)
    else:
        containers = (Container(1, 1), Container(2, 2), Container(3, 2))
        layout = ((1, 2), (3,)) + ((),) * (stacks - 2)
        batches = (1, 2)
    return SCRPInstance(
        instance_id=f"tiny-S{stacks}",
        num_stacks=stacks,
        max_tiers=3,
        containers=containers,
        initial_stacks=layout,
        batch_order=batches,
    )


def tiny_provider(sample):
    return tiny_instance(sample.num_stacks)


def config(version="O2", **updates):
    base = FormalTrainingConfig(
        observation_version=version,
        Mmax=None if version == "O1" else 6,
        batch_size=2,
        seed=711,
    )
    return replace(base, **updates)


def small_train_ids(manifest, stacks=(5, 7, 10)):
    result = []
    for stack_count in stacks:
        result.append(next(
            ref.base_instance_id for ref in manifest.refs("train")
            if ref.parameter_group.startswith(f"S{stack_count:02d}_")
        ))
    return result


def test_original_fgb_formula_matches_phase6_function():
    rewards = [-1.0, -1.0, -1.0]
    assert frozen_greedy_advantages(rewards, -6.0, 1.0) == [-1.0, 0.0, 1.0]
    source = open("hier_pg/algorithm.py", encoding="utf-8").read()
    assert "per_step_bl  = ret_bl / max(len(rew_ep), 1)" in source
    assert "adv_steps    = [g - per_step_bl for g in step_returns]" in source


def test_advantage_sign_is_positive_when_policy_is_better_and_negative_when_worse():
    assert frozen_greedy_advantages([-1.0], -2.0, 1.0) == [1.0]
    assert frozen_greedy_advantages([-2.0], -1.0, 1.0) == [-1.0]


def test_baseline_parameters_are_frozen():
    manifest = load_split_manifest(MANIFEST)
    trainer = SCRPFormalTrainer(
        config(), manifest, tiny_provider,
        allowed_base_ids=small_train_ids(manifest, (5,)),
    )
    assert not trainer.baseline_policy.training
    assert all(not parameter.requires_grad for parameter in trainer.baseline_policy.parameters())


def test_policy_and_baseline_use_identical_scenario_ids():
    manifest = load_split_manifest(MANIFEST)
    trainer = SCRPFormalTrainer(
        config(), manifest, tiny_provider,
        allowed_base_ids=small_train_ids(manifest, (5,)),
    )
    trainer.train_iterations(1)
    assert trainer.metrics[0].invalid_actions == 0


def test_o1_policy_factory_shape_and_scale():
    policy = make_scrp_policy("O1", 7, 4)
    assert policy.scrp_num_nodes == 8
    assert policy.scrp_feature_dim == 12
    assert policy.scrp_candidate_count == 7
    assert torch.equal(policy.feature_scale, torch.ones(12))


def test_o2_policy_factory_shape_and_scale():
    policy = make_scrp_policy("O2", 7, 4)
    assert policy.scrp_num_nodes == 14
    assert policy.scrp_mmax == 6
    assert torch.equal(policy.feature_scale, torch.ones(12))


def test_o2_candidate_count_is_S_not_node_count():
    policy = make_scrp_policy("O2", 5, 3)
    obs = torch.zeros(2, (5 + 7) * 12)
    forbidden = torch.zeros(2, 5, dtype=torch.bool)
    actions, _ = policy(obs, forbidden, greedy=True, mode="low")
    assert actions.shape == (2,)
    assert (actions < 5).all()


def test_padding_mask_makes_invalid_padding_content_semantically_inert():
    instance = tiny_instance(5)
    env = SCRPRLAdapter(
        SCRPEnvironment(SCRPConfig(5, 3), instance), observation_version="O2"
    )
    observation, info = env.reset(seed=9)
    original = torch.tensor(observation).unsqueeze(0)
    changed = original.clone().reshape(1, 12, 12)
    padding = changed[:, 5:11, 11] > 0.5
    changed[:, 5:11, 1:11][padding] = 0.73
    changed = changed.reshape(1, -1)
    forbidden = torch.tensor(~info["action_mask"]).unsqueeze(0)
    policy = make_scrp_policy("O2", 5, 3)
    mask_a = make_node_padding_mask(original, "O2", 5)
    mask_b = make_node_padding_mask(changed, "O2", 5)
    action = torch.tensor([int(np.flatnonzero(info["action_mask"])[0])])
    masked_a, _ = policy.evaluate_actions(
        original, forbidden, action, mode="low", node_padding_mask=mask_a
    )
    masked_b, _ = policy.evaluate_actions(
        changed, forbidden, action, mode="low", node_padding_mask=mask_b
    )
    marker_a, _ = policy.evaluate_actions(original, forbidden, action, mode="low")
    marker_b, _ = policy.evaluate_actions(changed, forbidden, action, mode="low")
    assert torch.equal(masked_a, masked_b)
    assert not torch.equal(marker_a, marker_b)


def test_o1_has_no_node_padding_mask_regression():
    assert make_node_padding_mask(torch.zeros(2, 72), "O1", 5) is None


def test_bucket_by_S_keeps_one_action_dimension_per_batch():
    manifest = load_split_manifest(MANIFEST)
    sampler = BaseBalancedTrainingSampler(
        manifest, 8, allowed_base_ids=small_train_ids(manifest)
    )
    for _ in range(10):
        batch = sampler.sample_bucket(4)
        assert len({item.num_stacks for item in batch}) == 1


def test_T_does_not_change_observation_node_count():
    assert make_scrp_policy("O2", 7, 3).scrp_num_nodes == 14
    assert make_scrp_policy("O2", 7, 6).scrp_num_nodes == 14


def test_mixed_ds1_ds2_sampling_is_base_first_and_not_variant_flattened():
    manifest = load_split_manifest(MANIFEST)
    allowed = small_train_ids(manifest, (5,))
    sampler = BaseBalancedTrainingSampler(manifest, 33, allowed_base_ids=allowed)
    samples = [sampler.sample() for _ in range(20)]
    assert {sample.base_instance_id for sample in samples} == set(allowed)
    assert {sample.variant for sample in samples} == {"DS1", "DS2"}
    assert [sample.visit_index for sample in samples] == list(range(20))


def test_training_sampler_and_seed_sequence_are_deterministic():
    manifest = load_split_manifest(MANIFEST)
    allowed = small_train_ids(manifest)
    first = BaseBalancedTrainingSampler(manifest, 99, allowed_base_ids=allowed)
    second = BaseBalancedTrainingSampler(manifest, 99, allowed_base_ids=allowed)
    assert [first.sample() for _ in range(12)] == [second.sample() for _ in range(12)]


def test_train_validation_test_seed_streams_remain_separated():
    manifest = load_split_manifest(MANIFEST)
    schedule = ScenarioSeedSchedule(manifest)
    pools = {
        split: set(schedule.seeds(split, manifest.refs(split)[0].base_instance_id, 5))
        for split in ("train", "validation", "test")
    }
    assert pools["train"].isdisjoint(pools["validation"])
    assert pools["train"].isdisjoint(pools["test"])
    assert pools["validation"].isdisjoint(pools["test"])


def test_checkpoint_metadata_is_complete(tmp_path):
    manifest = load_split_manifest(MANIFEST)
    trainer = SCRPFormalTrainer(
        config(), manifest, tiny_provider,
        allowed_base_ids=small_train_ids(manifest, (5,)),
    )
    trainer.train_iterations(1)
    checkpoint = torch.load(
        trainer.save_checkpoint(tmp_path / "formal.pt"),
        map_location="cpu", weights_only=False,
    )
    required = {
        "model_state_dict", "optimizer_state_dict", "iteration", "episodes_seen",
        "root_seed", "torch_rng_state", "observation_version", "feature_dim",
        "Mmax", "S_bucket_metadata", "dataset_version", "split_manifest_version",
        "training_protocol_version", "baseline_type", "baseline_state",
        "per_base_visit_counters", "config_snapshot",
    }
    assert required <= set(checkpoint)


def test_checkpoint_resume_is_deterministic(tmp_path):
    manifest = load_split_manifest(MANIFEST)
    allowed = small_train_ids(manifest, (5, 7))
    cfg = config()
    continuous = SCRPFormalTrainer(cfg, manifest, tiny_provider, allowed_base_ids=allowed)
    continuous.train_iterations(2)
    continuous_tail = continuous.metrics[-1]
    continuous_samples = continuous.sample_history

    split = SCRPFormalTrainer(cfg, manifest, tiny_provider, allowed_base_ids=allowed)
    split.train_iterations(1)
    path = split.save_checkpoint(tmp_path / "resume.pt")
    resumed = SCRPFormalTrainer.from_checkpoint(
        path, manifest, tiny_provider, allowed_base_ids=allowed
    )
    resumed_tail = resumed.train_iterations(1)[0]
    split_samples = split.sample_history + resumed.sample_history
    assert continuous_samples == split_samples
    assert continuous_tail.loss == pytest.approx(resumed_tail.loss, abs=0.0)
    for expected, actual in zip(continuous.policy.parameters(), resumed.policy.parameters()):
        assert torch.equal(expected, actual)


def test_zero_decision_episode_is_safe():
    manifest = load_split_manifest(MANIFEST)
    allowed = small_train_ids(manifest, (5,))
    trainer = SCRPFormalTrainer(
        config(), manifest, lambda sample: tiny_instance(sample.num_stacks, terminal=True),
        allowed_base_ids=allowed,
    )
    metric = trainer.train_iterations(1)[0]
    assert metric.loss == metric.policy_loss == metric.entropy == metric.grad_norm == 0.0


def test_formal_loss_is_finite():
    manifest = load_split_manifest(MANIFEST)
    trainer = SCRPFormalTrainer(
        config(), manifest, tiny_provider,
        allowed_base_ids=small_train_ids(manifest, (5,)),
    )
    assert math.isfinite(trainer.train_iterations(1)[0].loss)


def test_formal_gradients_are_finite():
    manifest = load_split_manifest(MANIFEST)
    trainer = SCRPFormalTrainer(
        config(), manifest, tiny_provider,
        allowed_base_ids=small_train_ids(manifest, (5,)),
    )
    trainer.train_iterations(1)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in trainer.policy.parameters()
    )


def test_policy_parameters_update():
    manifest = load_split_manifest(MANIFEST)
    trainer = SCRPFormalTrainer(
        config(), manifest, tiny_provider,
        allowed_base_ids=small_train_ids(manifest, (5,)),
    )
    before = [parameter.detach().clone() for parameter in trainer.policy.parameters()]
    trainer.train_iterations(1)
    assert any(not torch.equal(a, b) for a, b in zip(before, trainer.policy.parameters()))


def test_baseline_parameters_do_not_receive_gradients():
    manifest = load_split_manifest(MANIFEST)
    trainer = SCRPFormalTrainer(
        config(), manifest, tiny_provider,
        allowed_base_ids=small_train_ids(manifest, (5,)),
    )
    trainer.train_iterations(1)
    assert all(parameter.grad is None for parameter in trainer.baseline_policy.parameters())


@pytest.mark.parametrize("version", ["O1", "O2"])
def test_o1_o2_sanity_training_is_finite_and_updates_policy(version):
    manifest = load_split_manifest(MANIFEST)
    allowed = small_train_ids(manifest, (5,))
    trainer = SCRPFormalTrainer(
        config(version), manifest, tiny_provider, allowed_base_ids=allowed
    )
    before = [parameter.detach().clone() for parameter in trainer.policy.parameters()]
    baseline_before = copy.deepcopy(trainer.baseline_policy.state_dict())
    metric = trainer.train_iterations(1)[0]
    assert all(math.isfinite(value) for value in (
        metric.loss, metric.policy_loss, metric.entropy, metric.grad_norm
    ))
    assert any(not torch.equal(a, b) for a, b in zip(before, trainer.policy.parameters()))
    assert all(parameter.grad is None for parameter in trainer.baseline_policy.parameters())
    if trainer.baseline_updates == 0:
        assert all(
            torch.equal(value, trainer.baseline_policy.state_dict()[key])
            for key, value in baseline_before.items()
        )


def test_real_ds1_training_rollout_uses_train_split_only():
    manifest = load_split_manifest(MANIFEST)
    ref = next(ref for ref in manifest.refs("train") if ref.parameter_group.startswith("S05_"))
    sample = TrainingSample(ref.base_instance_id, ref.ds1_instance_id, "DS1", 1_000_000_000_000, 0, 5)
    instance = KuTrainingInstanceProvider(SOURCE_ROOT)(sample)
    trajectory = run_formal_episode(instance, sample, make_scrp_policy("O1", 5, 3), config("O1"), greedy=False)
    assert trajectory.terminated and not trajectory.truncated


def test_real_ds2_training_rollout_uses_train_split_only():
    manifest = load_split_manifest(MANIFEST)
    ref = next(ref for ref in manifest.refs("train") if ref.parameter_group.startswith("S05_"))
    sample = TrainingSample(ref.base_instance_id, ref.ds2_instance_id, "DS2", 1_000_000_000_000, 0, 5)
    instance = KuTrainingInstanceProvider(SOURCE_ROOT)(sample)
    trajectory = run_formal_episode(instance, sample, make_scrp_policy("O2", 5, 3), config(), greedy=False)
    assert trajectory.terminated and not trajectory.truncated


def test_training_protocol_config_is_versioned_and_not_final():
    cfg = load_formal_training_config("experiments/configs/training_protocol_v1.json")
    assert cfg.training_protocol_version == "scrp-training-protocol-v1"
    assert cfg.hyperparameter_status == "NOT FINAL HYPERPARAMETERS"
    assert cfg.training_strategy == "mixed_ds1_ds2_base_balanced_bucket_by_S"
