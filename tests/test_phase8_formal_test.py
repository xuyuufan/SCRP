from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from unittest.mock import patch

import pytest

from experiments.formal_test import (
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_SEED,
    ERI_ALGORITHM,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_CONFIG_SHA256,
    FORMAL_TEST_RUN_ID,
    PRIMARY_ALGORITHMS,
    RL_ALGORITHM,
    FormalTestIdentity,
    aggregate_per_instance,
    assert_compact_artifact_schema,
    build_formal_test_coordinates,
    create_formal_test_identity,
    dataset_summary,
    hierarchical_paired_bootstrap,
    pair_and_validate_primary_results,
    parameter_group_summary,
    robustness_statistics,
    verify_file_sha256,
)
from experiments.protocol import (
    DEFAULT_DATASET_VERSION,
    SPLIT_PROTOCOL_VERSION,
    BaseInstanceRef,
    ScenarioResult,
    SplitCounts,
    SplitManifest,
)


def _synthetic_manifest() -> SplitManifest:
    groups = {}
    counter = 0
    for stacks in range(5, 11):
        for tiers in range(3, 7):
            for fill in (0.50, 0.67):
                group = f"S{stacks:02d}_T{tiers:02d}_mu{fill:.2f}"
                refs = []
                for index in range(7):
                    base = f"synthetic-{counter:02d}-{index:02d}"
                    refs.append(BaseInstanceRef(
                        base_instance_id=base,
                        original_instance_id=base,
                        ds1_instance_id=f"ku2016-{base}",
                        ds2_instance_id=f"ku2016-{base}-merge2",
                        parameter_group=group,
                    ))
                groups[group] = {
                    "train": tuple(refs[:1]),
                    "validation": tuple(refs[1:2]),
                    "test": tuple(refs[2:]),
                }
                counter += 1
    return SplitManifest(
        protocol_version=SPLIT_PROTOCOL_VERSION,
        dataset_version=DEFAULT_DATASET_VERSION,
        split_seed=20260816,
        split_counts=SplitCounts(train=1, validation=1, test=5),
        groups=groups,
    )


@pytest.fixture(scope="module")
def synthetic_coordinates():
    return build_formal_test_coordinates(_synthetic_manifest())


@pytest.fixture(scope="module")
def synthetic_results(synthetic_coordinates):
    base_rank = {
        base: rank
        for rank, base in enumerate(sorted({row.base_instance_id for row in synthetic_coordinates}))
    }
    rows = []
    for coordinate in synthetic_coordinates:
        scenario_index = coordinate.scenario_seed % 1_000_000
        eri = 10 + scenario_index % 3
        delta = (base_rank[coordinate.base_instance_id] % 3) - 1
        scenario_id = (
            f"synthetic-{coordinate.dataset}-{coordinate.base_instance_id}-"
            f"{coordinate.scenario_seed}"
        )
        for algorithm, relocations in (
            (RL_ALGORITHM, eri + delta),
            (ERI_ALGORITHM, eri),
        ):
            rows.append(ScenarioResult(
                dataset=coordinate.dataset,
                split="test",
                instance_id=coordinate.instance_id,
                base_instance_id=coordinate.base_instance_id,
                parameter_group=coordinate.parameter_group,
                scenario_seed=coordinate.scenario_seed,
                scenario_id=scenario_id,
                algorithm=algorithm,
                relocations=relocations,
                terminated=True,
                truncated=False,
            ))
    return tuple(rows)


@pytest.fixture(scope="module")
def synthetic_pairs(synthetic_results, synthetic_coordinates):
    return pair_and_validate_primary_results(synthetic_results, synthetic_coordinates)


@pytest.fixture(scope="module")
def synthetic_instances(synthetic_pairs):
    return aggregate_per_instance(synthetic_pairs)


def _identity(**updates):
    values = {
        "run_id": FORMAL_TEST_RUN_ID,
        "code_sha": "a" * 40,
        "branch": "phase/scrp-phase-8-formal-test",
        "git_worktree_clean": True,
        "git_tracked_clean": True,
        "git_status_short": (),
        "frozen_at_utc": "2026-08-16T00:00:00Z",
        "runtime_versions": {
            "python": "3", "platform": "test", "numpy": "test",
            "scipy": "test", "torch": "test",
        },
        "training_run_id": "formal-o2-mixed-seed20260816-run1",
        "checkpoint_episode": 15_000,
        "checkpoint_path": "checkpoints/best-validation.pt",
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "split_manifest_version": SPLIT_PROTOCOL_VERSION,
        "dataset_version": DEFAULT_DATASET_VERSION,
        "observation_version": "O2",
        "Mmax": 6,
        "formal_test_protocol_version": "scrp-formal-protocol-v1",
        "root_seed": 20260816,
        "scenarios_per_static_variant": 50,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "primary_algorithms": PRIMARY_ALGORITHMS,
        "expected_static_variants_per_dataset": 240,
        "expected_rollouts_per_algorithm": 24_000,
        "expected_total_rows": 48_000,
    }
    values.update(updates)
    return FormalTestIdentity(**values)


def test_identity_freezes_checkpoint_seed_methods_and_budget():
    identity = _identity()
    assert identity.checkpoint_episode == 15_000
    assert identity.checkpoint_sha256 == EXPECTED_CHECKPOINT_SHA256
    assert identity.root_seed == 20260816
    assert identity.primary_algorithms == PRIMARY_ALGORITHMS
    assert identity.expected_total_rows == 48_000


@pytest.mark.parametrize(
    "updates",
    [
        {"checkpoint_episode": 15_001},
        {"checkpoint_sha256": "0" * 64},
        {"scenarios_per_static_variant": 49},
        {"bootstrap_repetitions": 9_999},
        {"primary_algorithms": tuple(reversed(PRIMARY_ALGORITHMS))},
    ],
)
def test_identity_rejects_semantic_drift(updates):
    with pytest.raises(ValueError):
        _identity(**updates)


def test_identity_is_written_with_git_and_runtime_fields_before_results(tmp_path):
    path = tmp_path / "run_identity.json"
    identity = create_formal_test_identity(
        path, checkpoint_path="checkpoints/frozen.pt"
    )
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record == identity.to_record()
    assert len(record["code_sha"]) == 40
    assert record["branch"]
    assert record["frozen_at_utc"].endswith("Z")
    assert set(record["runtime_versions"]) == {
        "python", "platform", "numpy", "scipy", "torch"
    }
    with pytest.raises(FileExistsError):
        create_formal_test_identity(path, checkpoint_path="checkpoints/frozen.pt")


def test_checkpoint_hash_guard_rejects_wrong_bytes(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"frozen-checkpoint")
    import hashlib
    expected = hashlib.sha256(b"frozen-checkpoint").hexdigest()
    assert verify_file_sha256(checkpoint, expected) == expected
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        verify_file_sha256(checkpoint, "0" * 64)


def test_synthetic_frozen_schedule_has_exact_counts(synthetic_coordinates):
    assert len(synthetic_coordinates) == 24_000
    assert len(set(synthetic_coordinates)) == 24_000
    assert sum(row.dataset == "DS1" for row in synthetic_coordinates) == 12_000
    assert sum(row.dataset == "DS2" for row in synthetic_coordinates) == 12_000


def test_primary_integrity_rejects_non_test_rows(
    synthetic_results, synthetic_coordinates,
):
    changed = (replace(synthetic_results[0], split="train"),) + synthetic_results[1:]
    with pytest.raises(AssertionError, match="non-test"):
        pair_and_validate_primary_results(changed, synthetic_coordinates)


def test_primary_integrity_rejects_crn_mismatch(
    synthetic_results, synthetic_coordinates,
):
    changed = list(synthetic_results)
    changed[1] = replace(changed[1], scenario_id="mismatch")
    with pytest.raises(AssertionError, match="scenario_id mismatch"):
        pair_and_validate_primary_results(changed, synthetic_coordinates)


def test_primary_integrity_rejects_duplicate_rows(
    synthetic_results, synthetic_coordinates,
):
    changed = list(synthetic_results)
    changed[2] = changed[0]
    with pytest.raises(AssertionError, match="duplicate"):
        pair_and_validate_primary_results(changed, synthetic_coordinates)


def test_primary_integrity_rejects_missing_algorithm_pair(
    synthetic_results, synthetic_coordinates,
):
    with pytest.raises(AssertionError, match="48,000"):
        pair_and_validate_primary_results(synthetic_results[:-1], synthetic_coordinates)


def test_per_instance_aggregation_and_delta_sign(
    synthetic_pairs, synthetic_instances,
):
    assert len(synthetic_instances) == 480
    assert all(row["scenario_count"] == 50 for row in synthetic_instances)
    first_pair = synthetic_pairs[0]
    assert first_pair.delta == first_pair.rl_relocations - first_pair.eri_relocations


def test_hierarchical_bootstrap_is_deterministic(synthetic_pairs):
    ds1 = tuple(pair for pair in synthetic_pairs if pair.dataset == "DS1")
    first = hierarchical_paired_bootstrap(ds1, repetitions=40, seed=1234)
    second = hierarchical_paired_bootstrap(ds1, repetitions=40, seed=1234)
    assert first == second
    assert first["strata"] == 48
    assert first["base_instances_per_stratum"] == 5
    assert first["paired_scenarios_per_instance"] == 50


def test_wilcoxon_receives_only_per_instance_means(synthetic_instances):
    ds1 = tuple(row for row in synthetic_instances if row["dataset"] == "DS1")
    with patch("experiments.formal_test.wilcoxon") as signed_rank, patch(
        "experiments.formal_test.ttest_rel"
    ) as paired_t:
        signed_rank.return_value.statistic = 1.0
        signed_rank.return_value.pvalue = 0.5
        paired_t.return_value.statistic = 1.0
        paired_t.return_value.pvalue = 0.5
        robustness_statistics(ds1)
    assert len(signed_rank.call_args.args[0]) == 240


def test_paired_t_receives_only_per_instance_means(synthetic_instances):
    ds1 = tuple(row for row in synthetic_instances if row["dataset"] == "DS1")
    with patch("experiments.formal_test.wilcoxon") as signed_rank, patch(
        "experiments.formal_test.ttest_rel"
    ) as paired_t:
        signed_rank.return_value.statistic = 1.0
        signed_rank.return_value.pvalue = 0.5
        paired_t.return_value.statistic = 1.0
        paired_t.return_value.pvalue = 0.5
        robustness_statistics(ds1)
    assert len(paired_t.call_args.args[0]) == 240
    assert len(paired_t.call_args.args[1]) == 240


def test_parameter_group_summary_is_descriptive_n5(synthetic_instances):
    groups = parameter_group_summary(synthetic_instances)
    assert len(groups) == 96
    assert all(row["n_test_instances"] == 5 for row in groups)
    assert all(row["interpretation"] == "descriptive_heterogeneity_only_n5" for row in groups)


def test_compact_artifact_schema_uses_per_instance_not_raw_rows(
    synthetic_pairs, synthetic_instances,
):
    ds1_pairs = tuple(pair for pair in synthetic_pairs if pair.dataset == "DS1")
    ds1_instances = tuple(
        row for row in synthetic_instances if row["dataset"] == "DS1"
    )
    bootstrap = hierarchical_paired_bootstrap(ds1_pairs, repetitions=20, seed=9)
    robustness = robustness_statistics(ds1_instances)
    summary = dataset_summary(
        "DS1", ds1_pairs, ds1_instances, bootstrap, robustness
    )
    assert_compact_artifact_schema(summary)
    assert len(summary["per_instance"]) == 240
    assert "scenario_rows" not in summary


def test_formal_raw_output_location_is_gitignored():
    probe = (
        "experiments/raw_results/formal-test-o2-vs-eri-seed20260816-v1/"
        "primary-results.jsonl"
    )
    completed = subprocess.run(
        ["git", "check-ignore", probe], capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == probe
