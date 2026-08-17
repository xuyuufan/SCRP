import pytest

from experiments.posttest_analysis import (
    audit_training_history,
    fixed_development_refs,
    parse_parameter_group,
)
from experiments.protocol import BaseInstanceRef, SplitCounts, SplitManifest


def _manifest():
    groups = {}
    for group in ("S05_T03_mu0.50", "S09_T06_mu0.67"):
        assignments = {}
        for split, count in (("train", 2), ("validation", 1), ("test", 1)):
            refs = []
            for index in range(count):
                base = f"{group}-{split}-{index}"
                refs.append(BaseInstanceRef(
                    base_instance_id=base,
                    original_instance_id=base,
                    ds1_instance_id=f"ku2016-{base}",
                    ds2_instance_id=f"ku2016-{base}-merge2",
                    parameter_group=group,
                ))
            assignments[split] = tuple(refs)
        groups[group] = assignments
    return SplitManifest(
        protocol_version="scrp-static-split-v1",
        dataset_version="synthetic",
        split_seed=1,
        split_counts=SplitCounts(train=2, validation=1, test=1),
        groups=groups,
    )


def _curve_artifacts():
    history = []
    windows = []
    for index, episode in enumerate(range(2_500, 25_001, 2_500), start=1):
        score = 10.0 + abs(6 - index) / 10
        history.append({
            "training_episode": episode,
            "selection_score": score,
            "baseline_state_version": index,
            "DS1": {"equal_instance_distribution": {"mean": score + 0.1}},
            "DS2": {"equal_instance_distribution": {"mean": score - 0.1}},
        })
        windows.append({
            "episode": episode,
            "iteration": episode // 4,
            "baseline_updates": index,
            "mean_policy_relocations": score,
            "mean_baseline_relocations": score - 1,
            "mean_advantage": -2.0,
            "entropy": 1.0,
            "grad_norm": 0.75,
        })
    training = {"formal_test_episode_usage": 0, "windows": windows}
    validation = {"formal_test_episode_usage": 0, "history": history}
    completion = {
        "formal_test_episode_usage": 0,
        "best_validation_checkpoint": {"checkpoint_episode": 15_000},
        "baseline_refresh_history": [
            {"iteration": 25, "sample_size": 4},
            {"iteration": 50, "sample_size": 4},
        ],
        "S_bucket_episode_counts": {str(S): 100 for S in range(5, 11)},
        "variant_episode_counts": {"DS1": 300, "DS2": 300},
    }
    return training, validation, completion


def test_parameter_group_parser():
    assert parse_parameter_group("S09_T06_mu0.67") == {
        "S": 9, "T": 6, "fill": 0.67,
    }


def test_fixed_refs_are_deterministic_and_development_only():
    manifest = _manifest()
    first = fixed_development_refs(manifest, "validation")
    second = fixed_development_refs(manifest, "validation")
    assert first == second
    assert len(first) == manifest.num_groups
    with pytest.raises(ValueError, match="train/validation"):
        fixed_development_refs(manifest, "test")


def test_training_curve_audit_selects_15k_and_rejects_test_usage():
    artifacts = _curve_artifacts()
    audit = audit_training_history(*artifacts)
    assert audit["best_episode"] == 15_000
    assert audit["formal_test_used"] is False
    assert audit["optimization"]["advantage_variance_available_in_compact_trace"] is False
    artifacts[0]["formal_test_episode_usage"] = 1
    with pytest.raises(ValueError, match="formal-test usage"):
        audit_training_history(*artifacts)
