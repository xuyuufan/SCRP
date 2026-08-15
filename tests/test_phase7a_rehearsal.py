from types import SimpleNamespace
from unittest.mock import patch

import torch

from experiments.protocol import load_split_manifest
from experiments.run_phase7a_rehearsal import _coverage, _coverage_stable, _peak_ram_bytes
from scrp.formal_training import (
    FormalTrainingConfig,
    SCRPFormalTrainer,
    TrainingSample,
    load_formal_training_config,
    policy_state_sha256,
)
from tests.test_formal_training import small_train_ids, tiny_provider


MANIFEST = "experiments/splits/scrp_split_v1.json"
CANDIDATE = "experiments/configs/training_protocol_v1_candidate.json"


def test_candidate_config_is_the_single_frozen_primary_candidate():
    candidate = load_formal_training_config(CANDIDATE)
    assert candidate.hyperparameter_status == "CANDIDATE_FOR_REHEARSAL"
    assert candidate.observation_version == "O2"
    assert candidate.Mmax == 6
    assert candidate.training_strategy == "mixed_ds1_ds2_base_balanced_bucket_by_S"


def test_phase6_sanity_lifecycle_status_remains_accepted():
    assert FormalTrainingConfig().hyperparameter_status == "NOT FINAL HYPERPARAMETERS"


def test_policy_state_hash_changes_after_an_optimizer_update():
    manifest = load_split_manifest(MANIFEST)
    trainer = SCRPFormalTrainer(
        FormalTrainingConfig(batch_size=2), manifest, tiny_provider,
        allowed_base_ids=small_train_ids(manifest, (5,)),
    )
    before = policy_state_sha256(trainer.policy)
    trainer.train_iterations(1)
    assert policy_state_sha256(trainer.policy) != before


def test_baseline_refresh_audit_records_and_checkpoint_resume(tmp_path):
    manifest = load_split_manifest(MANIFEST)
    trainer = SCRPFormalTrainer(
        FormalTrainingConfig(batch_size=2), manifest, tiny_provider,
        allowed_base_ids=small_train_ids(manifest, (5,)),
    )
    forced_significance = SimpleNamespace(statistic=3.0, pvalue=0.02)
    with patch("scrp.formal_training.ttest_rel", return_value=forced_significance):
        trainer.train_iterations(1)
    assert len(trainer.baseline_refresh_history) == 1
    record = trainer.baseline_refresh_history[0]
    assert record.iteration == 1
    assert record.sample_size == 2
    assert record.p_value == 0.01
    assert record.old_baseline_state_sha256 != record.new_baseline_state_sha256

    checkpoint = trainer.save_checkpoint(tmp_path / "rehearsal.pt")
    restored = SCRPFormalTrainer.from_checkpoint(
        checkpoint, manifest, tiny_provider,
        allowed_base_ids=small_train_ids(manifest, (5,)),
    )
    assert restored.baseline_refresh_history == trainer.baseline_refresh_history
    assert policy_state_sha256(restored.baseline_policy) == record.new_baseline_state_sha256


def test_balanced_coverage_audit_reports_all_buckets_variants_and_unique_seeds():
    manifest = load_split_manifest(MANIFEST)
    samples = []
    seed = 1_000_000_000_000
    for repetition in range(20):
        for stacks in range(5, 11):
            for variant in ("DS1", "DS2"):
                samples.append(TrainingSample(
                    base_instance_id=f"S{stacks}-base-{repetition % 4}",
                    instance_id=f"S{stacks}-{variant}-{repetition}",
                    variant=variant,
                    scenario_seed=seed,
                    visit_index=repetition,
                    num_stacks=stacks,
                ))
                seed += 1
    coverage = _coverage(samples)
    assert coverage["unique_scenario_seeds"] == len(samples)
    assert coverage["variant_episode_counts"] == {"DS1": 120, "DS2": 120}
    assert all(value == 40 for value in coverage["S_bucket_episode_counts"].values())
    assert _coverage_stable(samples, manifest)


def test_peak_ram_measurement_is_available_and_positive():
    peak = _peak_ram_bytes()
    assert peak is not None and peak > 0


def test_candidate_gradient_clip_and_baseline_rule_are_frozen():
    candidate = load_formal_training_config(CANDIDATE)
    assert candidate.gradient_clip == 0.5
    assert candidate.baseline_update_rule == "paired_one_sided_t_test_p_lt_0.05"
    assert candidate.checkpoint_interval == 10
