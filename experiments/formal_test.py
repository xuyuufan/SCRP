"""Frozen Phase 8 formal-test protocol and statistical analysis helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import scipy
import torch
from scipy.stats import ttest_rel, wilcoxon

from .protocol import (
    DEFAULT_DATASET_VERSION,
    SPLIT_PROTOCOL_VERSION,
    ScenarioResult,
    ScenarioSeedSchedule,
    SplitManifest,
)


FORMAL_TEST_RUN_ID = "formal-test-o2-vs-eri-seed20260816-v1"
RL_ALGORITHM = "rl_o2_episode15000"
ERI_ALGORITHM = "eri_reproduction_v1"
PRIMARY_ALGORITHMS = (RL_ALGORITHM, ERI_ALGORITHM)
FORMAL_TEST_SCENARIOS_PER_STATIC_VARIANT = 50
BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_SEED = 20260816
EXPECTED_CHECKPOINT_SHA256 = (
    "1dbcb20686840df3d392a89a66cb28b79a3a7531d4669300bf09818a714ed255"
)
EXPECTED_CONFIG_SHA256 = (
    "b284e0d1c9a8d750e4314b4bd7323af45f79d7cf2be3401ca22ec8b742793be3"
)


@dataclass(frozen=True)
class FormalTestIdentity:
    run_id: str
    code_sha: str
    branch: str
    git_worktree_clean: bool
    git_tracked_clean: bool
    git_status_short: tuple[str, ...]
    frozen_at_utc: str
    runtime_versions: Mapping[str, str]
    training_run_id: str
    checkpoint_episode: int
    checkpoint_path: str
    checkpoint_sha256: str
    config_sha256: str
    split_manifest_version: str
    dataset_version: str
    observation_version: str
    Mmax: int
    formal_test_protocol_version: str
    root_seed: int
    scenarios_per_static_variant: int
    bootstrap_seed: int
    bootstrap_repetitions: int
    primary_algorithms: tuple[str, str]
    expected_static_variants_per_dataset: int
    expected_rollouts_per_algorithm: int
    expected_total_rows: int

    def __post_init__(self) -> None:
        if self.run_id != FORMAL_TEST_RUN_ID:
            raise ValueError("formal-test run ID is frozen")
        if len(self.code_sha) != 40:
            raise ValueError("code SHA must be a full Git SHA")
        if not self.branch or not self.frozen_at_utc.endswith("Z"):
            raise ValueError("branch and UTC freeze timestamp are required")
        if set(self.runtime_versions) != {"python", "platform", "numpy", "scipy", "torch"}:
            raise ValueError("formal-test runtime version keys mismatch")
        if self.git_worktree_clean != (not self.git_status_short):
            raise ValueError("Git clean flag/status mismatch")
        if self.checkpoint_episode != 15_000:
            raise ValueError("formal test must use the episode-15000 checkpoint")
        if self.checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
            raise ValueError("checkpoint SHA drift")
        if self.config_sha256 != EXPECTED_CONFIG_SHA256:
            raise ValueError("candidate config SHA drift")
        if self.observation_version != "O2" or self.Mmax != 6:
            raise ValueError("formal test requires O2/Mmax=6")
        if self.scenarios_per_static_variant != 50:
            raise ValueError("formal test requires K=50")
        if self.bootstrap_repetitions < 10_000:
            raise ValueError("formal test requires at least 10,000 bootstrap repetitions")
        if tuple(self.primary_algorithms) != PRIMARY_ALGORITHMS:
            raise ValueError("primary algorithms are frozen")
        if self.expected_static_variants_per_dataset != 240:
            raise ValueError("formal test requires 240 static variants per dataset")
        if self.expected_rollouts_per_algorithm != 24_000:
            raise ValueError("formal test requires 24,000 rollouts per algorithm")
        if self.expected_total_rows != 48_000:
            raise ValueError("formal test requires 48,000 total rows")

    def to_record(self) -> dict[str, object]:
        record = asdict(self)
        record["git_status_short"] = list(self.git_status_short)
        record["runtime_versions"] = dict(self.runtime_versions)
        record["primary_algorithms"] = list(self.primary_algorithms)
        return record

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> "FormalTestIdentity":
        expected = {field.name for field in fields(cls)}
        if set(record) != expected:
            raise ValueError("formal-test identity keys mismatch")
        values = dict(record)
        values["git_status_short"] = tuple(values["git_status_short"])
        values["runtime_versions"] = dict(values["runtime_versions"])
        values["primary_algorithms"] = tuple(values["primary_algorithms"])
        return cls(**values)


@dataclass(frozen=True, order=True)
class TestCoordinate:
    __test__ = False

    dataset: str
    instance_id: str
    base_instance_id: str
    parameter_group: str
    scenario_seed: int


@dataclass(frozen=True)
class PairedScenario:
    dataset: str
    instance_id: str
    base_instance_id: str
    parameter_group: str
    scenario_seed: int
    scenario_id: str
    rl_relocations: int
    eri_relocations: int

    @property
    def delta(self) -> int:
        """RL minus ERI; negative values favor RL."""

        return self.rl_relocations - self.eri_relocations


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file_sha256(path: str | Path, expected: str) -> str:
    observed = file_sha256(path)
    if observed != expected:
        raise RuntimeError(
            f"SHA-256 mismatch for {path}: expected {expected}, observed {observed}"
        )
    return observed


def atomic_write_json(path: str | Path, payload: object) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return destination


def load_formal_test_identity(path: str | Path) -> FormalTestIdentity:
    return FormalTestIdentity.from_record(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def _git_output(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=repo_root, check=True,
        capture_output=True, text=True, encoding="utf-8",
    )
    return completed.stdout.strip()


def create_formal_test_identity(
    path: str | Path,
    *,
    checkpoint_path: str | Path,
    repo_root: str | Path = ".",
) -> FormalTestIdentity:
    """Freeze Git/runtime/input identity before any formal-test result exists."""

    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite formal-test identity {destination}")
    root = Path(repo_root).resolve()
    status = tuple(
        line for line in _git_output(root, "status", "--short").splitlines() if line
    )
    tracked_status = tuple(
        line for line in _git_output(root, "status", "--short", "--untracked-files=no").splitlines()
        if line
    )
    identity = FormalTestIdentity(
        run_id=FORMAL_TEST_RUN_ID,
        code_sha=_git_output(root, "rev-parse", "HEAD"),
        branch=_git_output(root, "branch", "--show-current"),
        git_worktree_clean=not status,
        git_tracked_clean=not tracked_status,
        git_status_short=status,
        frozen_at_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        runtime_versions={
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
        },
        training_run_id="formal-o2-mixed-seed20260816-run1",
        checkpoint_episode=15_000,
        checkpoint_path=Path(checkpoint_path).as_posix(),
        checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256,
        config_sha256=EXPECTED_CONFIG_SHA256,
        split_manifest_version=SPLIT_PROTOCOL_VERSION,
        dataset_version=DEFAULT_DATASET_VERSION,
        observation_version="O2",
        Mmax=6,
        formal_test_protocol_version="scrp-formal-protocol-v1",
        root_seed=20260816,
        scenarios_per_static_variant=FORMAL_TEST_SCENARIOS_PER_STATIC_VARIANT,
        bootstrap_seed=BOOTSTRAP_SEED,
        bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
        primary_algorithms=PRIMARY_ALGORITHMS,
        expected_static_variants_per_dataset=240,
        expected_rollouts_per_algorithm=24_000,
        expected_total_rows=48_000,
    )
    atomic_write_json(destination, identity.to_record())
    return identity


def build_formal_test_coordinates(
    manifest: SplitManifest,
    *,
    scenarios_per_static_variant: int = FORMAL_TEST_SCENARIOS_PER_STATIC_VARIANT,
) -> tuple[TestCoordinate, ...]:
    if scenarios_per_static_variant != FORMAL_TEST_SCENARIOS_PER_STATIC_VARIANT:
        raise ValueError("formal-test schedule is frozen at 50 scenarios")
    schedule = ScenarioSeedSchedule(manifest)
    coordinates: list[TestCoordinate] = []
    refs = sorted(manifest.refs("test"), key=lambda ref: ref.base_instance_id)
    if len(refs) != 240:
        raise ValueError(f"expected 240 test base layouts, observed {len(refs)}")
    for dataset in ("DS1", "DS2"):
        for ref in refs:
            instance_id = ref.ds1_instance_id if dataset == "DS1" else ref.ds2_instance_id
            for seed in schedule.seeds(
                "test", ref.base_instance_id, scenarios_per_static_variant
            ):
                coordinates.append(TestCoordinate(
                    dataset=dataset,
                    instance_id=instance_id,
                    base_instance_id=ref.base_instance_id,
                    parameter_group=ref.parameter_group,
                    scenario_seed=seed,
                ))
    if len(coordinates) != 24_000 or len(set(coordinates)) != 24_000:
        raise AssertionError("formal-test coordinate schedule is incomplete or duplicated")
    return tuple(coordinates)


def pair_and_validate_primary_results(
    results: Sequence[ScenarioResult],
    coordinates: Sequence[TestCoordinate],
) -> tuple[PairedScenario, ...]:
    expected = {
        (item.dataset, item.instance_id, item.scenario_seed): item
        for item in coordinates
    }
    if len(expected) != 24_000:
        raise AssertionError("expected schedule must contain exactly 24,000 coordinates")
    if len(results) != 48_000:
        raise AssertionError(f"expected 48,000 primary rows, observed {len(results)}")

    observed: dict[tuple[str, str, int], dict[str, ScenarioResult]] = {}
    for result in results:
        if result.split != "test":
            raise AssertionError("formal-test raw results contain a non-test row")
        if result.algorithm not in PRIMARY_ALGORITHMS:
            raise AssertionError("formal-test raw results contain an unexpected algorithm")
        if not result.terminated or result.truncated:
            raise AssertionError("formal-test episode did not terminate cleanly")
        key = (result.dataset, result.instance_id, result.scenario_seed)
        coordinate = expected.get(key)
        if coordinate is None:
            raise AssertionError("formal-test raw result is not on the frozen schedule")
        if (
            result.base_instance_id != coordinate.base_instance_id
            or result.parameter_group != coordinate.parameter_group
        ):
            raise AssertionError("formal-test result metadata does not match its coordinate")
        algorithms = observed.setdefault(key, {})
        if result.algorithm in algorithms:
            raise AssertionError("duplicate formal-test algorithm row")
        algorithms[result.algorithm] = result

    if set(observed) != set(expected):
        raise AssertionError("formal-test coordinates are missing")
    pairs: list[PairedScenario] = []
    for key in sorted(observed):
        algorithms = observed[key]
        if set(algorithms) != set(PRIMARY_ALGORITHMS):
            raise AssertionError("formal-test coordinate is missing an algorithm pair")
        rl = algorithms[RL_ALGORITHM]
        eri = algorithms[ERI_ALGORITHM]
        if rl.scenario_id != eri.scenario_id:
            raise AssertionError("formal-test CRN scenario_id mismatch")
        pairs.append(PairedScenario(
            dataset=rl.dataset,
            instance_id=rl.instance_id,
            base_instance_id=rl.base_instance_id,
            parameter_group=rl.parameter_group,
            scenario_seed=rl.scenario_seed,
            scenario_id=rl.scenario_id,
            rl_relocations=rl.relocations,
            eri_relocations=eri.relocations,
        ))
    return tuple(pairs)


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("distribution requires values")
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "mean": float(statistics.fmean(numeric)),
        "median": float(statistics.median(numeric)),
        "std": float(statistics.stdev(numeric)) if len(numeric) > 1 else 0.0,
        "minimum": min(numeric),
        "maximum": max(numeric),
    }


def aggregate_per_instance(
    pairs: Sequence[PairedScenario],
) -> tuple[dict[str, object], ...]:
    grouped: dict[tuple[str, str, str, str], list[PairedScenario]] = {}
    for pair in pairs:
        key = (
            pair.dataset, pair.instance_id, pair.base_instance_id,
            pair.parameter_group,
        )
        grouped.setdefault(key, []).append(pair)
    records: list[dict[str, object]] = []
    for key, rows in sorted(grouped.items()):
        if len(rows) != 50 or len({row.scenario_seed for row in rows}) != 50:
            raise AssertionError("each static variant requires 50 unique paired scenarios")
        deltas = [row.delta for row in rows]
        records.append({
            "dataset": key[0],
            "instance_id": key[1],
            "base_instance_id": key[2],
            "parameter_group": key[3],
            "scenario_count": len(rows),
            "RL_mean": float(statistics.fmean(row.rl_relocations for row in rows)),
            "ERI_mean": float(statistics.fmean(row.eri_relocations for row in rows)),
            "mean_delta_RL_minus_ERI": float(statistics.fmean(deltas)),
            "std_delta_RL_minus_ERI": (
                float(statistics.stdev(deltas)) if len(deltas) > 1 else 0.0
            ),
        })
    return tuple(records)


def hierarchical_paired_bootstrap(
    pairs: Sequence[PairedScenario],
    *,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    seed: int = BOOTSTRAP_SEED,
    chunk_size: int = 100,
) -> dict[str, object]:
    if repetitions < 1 or chunk_size < 1:
        raise ValueError("bootstrap repetitions and chunk size must be positive")
    grouped: dict[str, dict[str, list[int]]] = {}
    for pair in pairs:
        grouped.setdefault(pair.parameter_group, {}).setdefault(
            pair.base_instance_id, []
        ).append(pair.delta)
    if len(grouped) != 48:
        raise AssertionError(f"expected 48 parameter groups, observed {len(grouped)}")
    arrays = []
    for group in sorted(grouped):
        bases = grouped[group]
        if len(bases) != 5:
            raise AssertionError("each test parameter group requires five base layouts")
        values = []
        for base in sorted(bases):
            if len(bases[base]) != 50:
                raise AssertionError("each test base requires 50 paired scenarios")
            values.append(bases[base])
        arrays.append(values)
    data = np.asarray(arrays, dtype=np.float64)
    if data.shape != (48, 5, 50):
        raise AssertionError(f"unexpected bootstrap input shape {data.shape}")

    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions, dtype=np.float64)
    offset = 0
    while offset < repetitions:
        count = min(chunk_size, repetitions - offset)
        base_indices = rng.integers(0, 5, size=(count, 48, 5))
        selected = np.take_along_axis(
            np.broadcast_to(data, (count,) + data.shape),
            base_indices[..., None],
            axis=2,
        )
        scenario_indices = rng.integers(0, 50, size=selected.shape)
        resampled = np.take_along_axis(selected, scenario_indices, axis=3)
        samples[offset:offset + count] = resampled.mean(axis=(1, 2, 3))
        offset += count
    low, high = np.quantile(samples, (0.025, 0.975))
    return {
        "method": "stratified_hierarchical_paired_bootstrap",
        "strata": 48,
        "base_instances_per_stratum": 5,
        "paired_scenarios_per_instance": 50,
        "repetitions": repetitions,
        "seed": seed,
        "mean_delta_RL_minus_ERI": float(data.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def robustness_statistics(
    per_instance: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    rl = np.asarray([float(row["RL_mean"]) for row in per_instance])
    eri = np.asarray([float(row["ERI_mean"]) for row in per_instance])
    if len(rl) != 240:
        raise AssertionError("robustness tests require 240 per-instance pairs")
    deltas = rl - eri
    if np.all(deltas == 0.0):
        wilcoxon_statistic, wilcoxon_p = 0.0, 1.0
        t_statistic, t_p = 0.0, 1.0
        effect_size = 0.0
    else:
        signed_rank = wilcoxon(deltas, alternative="two-sided")
        paired_t = ttest_rel(rl, eri)
        wilcoxon_statistic = float(signed_rank.statistic)
        wilcoxon_p = float(signed_rank.pvalue)
        t_statistic = float(paired_t.statistic)
        t_p = float(paired_t.pvalue)
        delta_std = float(np.std(deltas, ddof=1))
        effect_size = float(np.mean(deltas) / delta_std) if delta_std else math.inf
    return {
        "input_unit": "static_instance_mean_over_50_paired_scenarios",
        "n": len(deltas),
        "mean_delta_RL_minus_ERI": float(np.mean(deltas)),
        "median_delta_RL_minus_ERI": float(np.median(deltas)),
        "wilcoxon_signed_rank": {
            "statistic": wilcoxon_statistic,
            "p_value_two_sided": wilcoxon_p,
        },
        "paired_t_test": {
            "statistic": t_statistic,
            "p_value_two_sided": t_p,
        },
        "cohen_dz": effect_size,
    }


def parameter_group_summary(
    per_instance: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in per_instance:
        grouped.setdefault(
            (str(row["dataset"]), str(row["parameter_group"])), []
        ).append(row)
    records = []
    for (dataset, group), rows in sorted(grouped.items()):
        if len(rows) != 5:
            raise AssertionError("parameter-group summary requires five test instances")
        parts = group.split("_")
        records.append({
            "dataset": dataset,
            "parameter_group": group,
            "S": int(parts[0][1:]),
            "T": int(parts[1][1:]),
            "fill": float(parts[2][2:]),
            "n_test_instances": 5,
            "RL_mean": float(statistics.fmean(float(row["RL_mean"]) for row in rows)),
            "ERI_mean": float(statistics.fmean(float(row["ERI_mean"]) for row in rows)),
            "mean_delta_RL_minus_ERI": float(statistics.fmean(
                float(row["mean_delta_RL_minus_ERI"]) for row in rows
            )),
            "interpretation": "descriptive_heterogeneity_only_n5",
        })
    if len(records) != 96:
        raise AssertionError("expected 48 parameter groups for each of DS1 and DS2")
    return tuple(records)


def dataset_summary(
    dataset: str,
    pairs: Sequence[PairedScenario],
    per_instance: Sequence[Mapping[str, object]],
    bootstrap: Mapping[str, object],
    robustness: Mapping[str, object],
) -> dict[str, object]:
    dataset_pairs = [pair for pair in pairs if pair.dataset == dataset]
    dataset_instances = [row for row in per_instance if row["dataset"] == dataset]
    if len(dataset_pairs) != 12_000 or len(dataset_instances) != 240:
        raise AssertionError("dataset summary input count mismatch")
    rl_values = [pair.rl_relocations for pair in dataset_pairs]
    eri_values = [pair.eri_relocations for pair in dataset_pairs]
    instance_deltas = [float(row["mean_delta_RL_minus_ERI"]) for row in dataset_instances]
    scenario_deltas = [pair.delta for pair in dataset_pairs]
    return {
        "dataset": dataset,
        "static_instances": 240,
        "scenarios_per_instance": 50,
        "rows_per_algorithm": 12_000,
        "algorithms": {
            RL_ALGORITHM: {
                "scenario_distribution": _distribution(rl_values),
                "per_instance_mean_distribution": _distribution(
                    [float(row["RL_mean"]) for row in dataset_instances]
                ),
            },
            ERI_ALGORITHM: {
                "scenario_distribution": _distribution(eri_values),
                "per_instance_mean_distribution": _distribution(
                    [float(row["ERI_mean"]) for row in dataset_instances]
                ),
            },
        },
        "paired_delta_sign_convention": "RL_minus_ERI; negative_favors_RL",
        "paired_delta": {
            "scenario_distribution_supplementary": _distribution(scenario_deltas),
            "per_instance_distribution": _distribution(instance_deltas),
            "hierarchical_bootstrap": dict(bootstrap),
            "robustness": dict(robustness),
        },
        "instance_win_tie_loss": {
            "RL_wins": sum(value < 0 for value in instance_deltas),
            "ties": sum(value == 0 for value in instance_deltas),
            "ERI_wins": sum(value > 0 for value in instance_deltas),
        },
        "scenario_win_tie_loss_supplementary": {
            "RL_wins": sum(value < 0 for value in scenario_deltas),
            "ties": sum(value == 0 for value in scenario_deltas),
            "ERI_wins": sum(value > 0 for value in scenario_deltas),
        },
        "per_instance": list(dataset_instances),
    }


def assert_compact_artifact_schema(summary: Mapping[str, object]) -> None:
    required = {
        "dataset", "static_instances", "scenarios_per_instance",
        "rows_per_algorithm", "algorithms", "paired_delta_sign_convention",
        "paired_delta", "instance_win_tie_loss",
        "scenario_win_tie_loss_supplementary", "per_instance",
    }
    if set(summary) != required:
        raise AssertionError("compact formal-test dataset artifact schema mismatch")
    if summary["static_instances"] != 240 or summary["scenarios_per_instance"] != 50:
        raise AssertionError("compact formal-test artifact count mismatch")
    if len(summary["per_instance"]) != 240:
        raise AssertionError("compact artifact requires 240 per-instance aggregates")


def read_raw_results(path: str | Path) -> tuple[ScenarioResult, ...]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            rows.append(ScenarioResult.from_record(json.loads(line)))
    return tuple(rows)
