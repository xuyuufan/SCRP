import json
from pathlib import Path

import pytest
import torch

from experiments.formal_run import (
    CANDIDATE_CONFIG_SHA256,
    CANDIDATE_CONFIG_PATH,
    RUN_ID,
    SELECTION_METRIC_NAME,
    FormalRunIdentity,
    atomic_write_json,
    committed_file_sha256,
    compact_window,
    copy_checkpoint_verified,
    file_sha256,
    save_checkpoint_atomic,
    selection_score,
)
from experiments.protocol import load_split_manifest
from scrp.formal_training import FormalTrainingConfig, SCRPFormalTrainer
from tests.test_formal_training import small_train_ids, tiny_provider


MANIFEST = "experiments/splits/scrp_split_v1.json"
RUN_SUMMARY_ROOT = Path("experiments/summaries")


def identity(**updates):
    values = {
        "run_id": RUN_ID,
        "code_sha": "a" * 40,
        "config_path": CANDIDATE_CONFIG_PATH,
        "config_sha256": CANDIDATE_CONFIG_SHA256,
        "split_manifest_version": "scrp-static-split-v1",
        "dataset_version": "ku-galle-bacci-ds1-ds2-ec672df",
        "observation_version": "O2",
        "Mmax": 6,
        "root_seed": 20260816,
        "planned_episodes": 25_000,
        "validation_cadence_episodes": 2_500,
        "validation_scenarios_per_static_variant": 20,
        "checkpoint_selection_metric": SELECTION_METRIC_NAME,
        "durable_milestone_episodes": (
            1_000, 2_500, 5_000, 10_000, 15_000, 20_000, 25_000
        ),
        "formal_test_episode_usage": 0,
    }
    values.update(updates)
    return FormalRunIdentity(**values)


def test_formal_run_identity_freezes_25k_o2_and_validation_protocol():
    run = identity()
    assert run.planned_episodes == 25_000
    assert run.observation_version == "O2" and run.Mmax == 6
    assert run.validation_cadence_episodes == 2_500
    assert run.validation_scenarios_per_static_variant == 20
    assert run.formal_test_episode_usage == 0


@pytest.mark.parametrize(
    "updates",
    [
        {"planned_episodes": 25_004},
        {"validation_cadence_episodes": 1_000},
        {"validation_scenarios_per_static_variant": 19},
        {"formal_test_episode_usage": 1},
        {"config_sha256": "b" * 64},
    ],
)
def test_formal_run_identity_rejects_protocol_drift(updates):
    with pytest.raises(ValueError):
        identity(**updates)


def test_candidate_hash_uses_canonical_committed_bytes():
    assert committed_file_sha256("HEAD", CANDIDATE_CONFIG_PATH) == CANDIDATE_CONFIG_SHA256


def test_checkpoint_selection_metric_equal_weights_variants_and_instances():
    assert selection_score([1.0, 3.0], [5.0, 9.0]) == 4.5
    with pytest.raises(ValueError):
        selection_score([], [1.0])


def test_atomic_json_write_has_no_temporary_residue(tmp_path):
    destination = atomic_write_json(tmp_path / "result.json", {"ok": True})
    assert json.loads(destination.read_text()) == {"ok": True}
    assert not (tmp_path / "result.json.tmp").exists()


def test_atomic_checkpoint_and_verified_copy(tmp_path):
    manifest = load_split_manifest(MANIFEST)
    allowed = small_train_ids(manifest, (5,))
    trainer = SCRPFormalTrainer(
        FormalTrainingConfig(batch_size=2), manifest, tiny_provider,
        allowed_base_ids=allowed,
    )
    trainer.train_iterations(1)
    latest = save_checkpoint_atomic(trainer, tmp_path / "latest.pt")
    milestone = copy_checkpoint_verified(latest, tmp_path / "milestone.pt")
    assert file_sha256(latest) == file_sha256(milestone)
    assert torch.load(milestone, map_location="cpu", weights_only=False)["iteration"] == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_compact_window_records_required_monitoring_fields():
    manifest = load_split_manifest(MANIFEST)
    allowed = small_train_ids(manifest, (5,))
    trainer = SCRPFormalTrainer(
        FormalTrainingConfig(batch_size=2), manifest, tiny_provider,
        allowed_base_ids=allowed,
    )
    metrics = trainer.train_iterations(1)
    window = compact_window(metrics, trainer.sample_history, 2)
    assert window["episode"] == 2
    assert window["invalid_actions"] == 0
    assert window["truncations"] == 0
    assert window["scenario_mismatches"] == 0
    assert sum(window["variant_counts"].values()) == 2
    assert sum(window["S_bucket_counts"].values()) == 2


def test_committed_phase7b_validation_is_checkpoint_level_and_frozen():
    path = RUN_SUMMARY_ROOT / f"{RUN_ID}-validation.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))

    assert artifact["artifact_class"] == "compact_checkpoint_validation_summary"
    assert artifact["formal_test_episode_usage"] == 0
    assert len(artifact["history"]) == 10
    assert [row["training_episode"] for row in artifact["history"]] == list(
        range(2_500, 25_001, 2_500)
    )
    for row in artifact["history"]:
        assert row["validation_episode_count"] == 9_600
        assert row["formal_test_used"] is False
        assert row["ERI_used"] is False
        for variant in ("DS1", "DS2"):
            assert "per_instance" not in row[variant]
            assert "parameter_group_equal_instance_means" not in row[variant]

    best = min(artifact["history"], key=lambda row: row["selection_score"])
    assert best["training_episode"] == 15_000
    assert best["selection_score"] == 10.292083333333334
    raw = artifact["local_full_derived_artifact"]
    assert raw["sha256"] == (
        "be02cff39c29ce4d36cf9c843791c522385f9926b0790f1cfb514ef3709c5bca"
    )
