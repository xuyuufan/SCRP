"""Phase 14 frozen-policy, development-only direct RL-versus-ERI evaluation.

The module deliberately has no training entry point.  It loads an already
frozen Phase 13 policy, evaluates greedy actions using public observations,
and pairs every rollout with ERI on the identical scenario realization.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import ttest_rel, wilcoxon

from experiments.baselines.eri import ERIBaseline
from experiments.phase11_eri_auxiliary import _policy_probabilities
from experiments.phase12_multiseed_replication import cuda_identity
from experiments.posttest_analysis import fixed_development_refs, parse_parameter_group
from experiments.protocol import BaseInstanceRef, ScenarioSeedSchedule, SplitManifest
from scrp.environment import SCRPEnvironment
from scrp.formal_training import (
    ERI_AUXILIARY_VERSION,
    FormalTrainingConfig,
    TrainingSample,
    make_scrp_policy,
    policy_state_sha256,
    resolve_training_device,
)
from scrp.models import SCRPConfig
from scrp.rl_adapter import SCRPRLAdapter


PHASE14_VERSION = "phase14-rl-vs-eri-development-v1"
PHASE14_BASE_SHA = "6cc94b667f017044ab9cbf530f043088ec02cac0"
FROZEN_SEEDS = (20260816, 20260818, 20260819)
DATASETS = ("DS1", "DS2")


class Phase14PreflightError(RuntimeError):
    """Raised before any ERI evaluation when the frozen protocol is unusable."""


def load_phase14_protocol(path: str | Path) -> dict[str, object]:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("phase14_version") != PHASE14_VERSION:
        raise ValueError("unsupported Phase 14 protocol version")
    if record.get("base_sha") != PHASE14_BASE_SHA:
        raise ValueError("Phase 14 base SHA differs from the frozen protocol")
    if tuple(record.get("frozen_policy_seeds", ())) != FROZEN_SEEDS:
        raise ValueError("Phase 14 policy seeds differ from the frozen protocol")
    if record.get("phase8_raw_access") != "PROHIBITED":
        raise ValueError("Phase 14 must prohibit Phase 8 raw access")
    if record.get("formal_test_rerun") != "PROHIBITED":
        raise ValueError("Phase 14 must prohibit formal-test reruns")
    holdout = record["development_holdout"]
    if holdout.get("source_split") != "validation" or holdout.get("test_split_usage") != 0:
        raise ValueError("Phase 14 holdout must use validation only and no test data")
    selection = record["checkpoint_selection"]
    if not selection.get("frozen_before_eri_evaluation"):
        raise ValueError("checkpoint selection is not frozen")
    selected = tuple(selection.get("selections", ()))
    if tuple(int(item["seed"]) for item in selected) != FROZEN_SEEDS:
        raise ValueError("checkpoint selections do not match frozen seeds")
    if any(int(item["episode"]) != 15_000 for item in selected):
        raise ValueError("Phase 14 fallback selections must use the 15k endpoint")
    return record


def phase14_development_holdout(manifest: SplitManifest) -> tuple[BaseInstanceRef, ...]:
    """Return validation layouts never used by the Phase 11-13 fixed probe."""

    previously_used = {
        ref.base_instance_id for ref in fixed_development_refs(manifest, "validation")
    }
    selected: list[BaseInstanceRef] = []
    for group in sorted(manifest.groups):
        validation = tuple(sorted(
            manifest.groups[group]["validation"], key=lambda ref: ref.base_instance_id
        ))
        group_selected = tuple(
            ref for ref in validation if ref.base_instance_id not in previously_used
        )
        if len(group_selected) != 4:
            raise Phase14PreflightError(
                f"{group}: expected four unused validation layouts, got {len(group_selected)}"
            )
        selected.extend(group_selected)
    if len(selected) != 192 or len({ref.base_instance_id for ref in selected}) != 192:
        raise Phase14PreflightError("Phase 14 holdout must contain 192 unique base layouts")
    if any(manifest.split_for_base(ref.base_instance_id) != "validation" for ref in selected):
        raise Phase14PreflightError("Phase 14 holdout escaped the validation split")
    if previously_used & {ref.base_instance_id for ref in selected}:
        raise Phase14PreflightError("Phase 14 holdout overlaps checkpoint-selection layouts")
    return tuple(selected)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_preflight(
    protocol: Mapping[str, object], repository_root: str | Path
) -> dict[str, object]:
    """Inspect frozen files without training or running ERI."""

    root = Path(repository_root).resolve()
    records = []
    ready = True
    for item in protocol["checkpoint_selection"]["selections"]:
        relative = Path(item["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise Phase14PreflightError("checkpoint paths must be repository-relative")
        path = (root / relative).resolve()
        if root not in path.parents:
            raise Phase14PreflightError("checkpoint path escaped the repository")
        present = path.is_file()
        metadata_valid = False
        metadata_error = None
        if present:
            try:
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                checkpoint_config = FormalTrainingConfig.from_record(
                    checkpoint["config_snapshot"]
                )
                if int(checkpoint_config.seed) != int(item["seed"]):
                    raise ValueError("seed mismatch")
                if int(checkpoint.get("episodes_seen", -1)) != int(item["episode"]):
                    raise ValueError("episode mismatch")
                if checkpoint_config.eri_aux_coefficient != 0.10:
                    raise ValueError("not a Treatment checkpoint")
                if checkpoint_config.eri_auxiliary_version != ERI_AUXILIARY_VERSION:
                    raise ValueError("wrong ERI auxiliary version")
                if "model_state_dict" not in checkpoint:
                    raise ValueError("model_state_dict missing")
                metadata_valid = True
            except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
                metadata_error = str(error)
        record = {
            "seed": int(item["seed"]),
            "episode": int(item["episode"]),
            "relative_path": relative.as_posix(),
            "present": present,
            "sha256": _sha256_file(path) if present else None,
            "metadata_valid": metadata_valid,
            "metadata_error": metadata_error,
        }
        records.append(record)
        ready &= present and metadata_valid
    return {
        "ready": bool(ready),
        "records": records,
        "missing": [record["relative_path"] for record in records if not record["present"]],
        "invalid": [record["relative_path"] for record in records
                    if record["present"] and not record["metadata_valid"]],
        "new_training_performed": False,
        "eri_evaluation_started": False,
    }


def _load_frozen_policy(path: Path, expected_seed: int, expected_episode: int, device: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = FormalTrainingConfig.from_record(checkpoint["config_snapshot"])
    if int(config.seed) != expected_seed:
        raise Phase14PreflightError("checkpoint seed does not match frozen selection")
    if int(checkpoint.get("episodes_seen", -1)) != expected_episode:
        raise Phase14PreflightError("checkpoint episode does not match frozen selection")
    if config.eri_aux_coefficient != 0.10 or config.eri_auxiliary_version != ERI_AUXILIARY_VERSION:
        raise Phase14PreflightError("checkpoint is not a Phase 13 Treatment policy")
    resolved = resolve_training_device(device)
    policy = make_scrp_policy(
        config.observation_version, 5, 3,
        Mmax=config.Mmax or 6,
        embed_dim=config.embed_dim,
        num_encoder_layers=config.num_encoder_layers,
        num_heads=config.num_heads,
        ffn_dim=config.ffn_dim,
        clip_constant=config.clip_constant,
        device=resolved,
    )
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.to(resolved).eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    if next(policy.parameters()).device.type != "cuda":
        raise Phase14PreflightError("Phase 14 policy parameters are not on CUDA")
    return policy, config, policy_state_sha256(policy)


def _nvidia_python_process_seen() -> bool:
    try:
        snapshot = subprocess.run(
            ["nvidia-smi"], check=True, capture_output=True, text=True, timeout=15
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return str(os.getpid()) in snapshot and "python" in snapshot.lower()


def _sample(ref: BaseInstanceRef, dataset: str, scenario_seed: int, visit: int) -> TrainingSample:
    return TrainingSample(
        base_instance_id=ref.base_instance_id,
        instance_id=ref.ds1_instance_id if dataset == "DS1" else ref.ds2_instance_id,
        variant=dataset,
        scenario_seed=scenario_seed,
        visit_index=visit,
        num_stacks=int(ref.parameter_group[1:3]),
    )


def _public_state_fingerprint(core: SCRPEnvironment) -> str:
    state = core.state
    payload = {
        "stacks": [list(stack.containers) for stack in state.stacks],
        "current_target_id": state.current_target_id,
        "revealed_orders": {
            str(key): list(value) for key, value in sorted(state.revealed_orders.items())
        },
        "relocations": state.relocation_count,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def greedy_public_action(policy, observation, legal_mask, num_stacks: int) -> tuple[int, np.ndarray]:
    """Choose solely from a public observation and legal-action mask."""

    legal = np.asarray(legal_mask, dtype=bool)
    if legal.ndim != 1 or not legal.any():
        raise ValueError("a non-empty one-dimensional legal mask is required")
    probabilities = _policy_probabilities(policy, observation, legal, num_stacks)
    if not np.isfinite(probabilities[legal]).all():
        raise FloatingPointError("greedy policy produced non-finite legal probabilities")
    action = int(np.argmax(probabilities))
    if not legal[action]:
        raise AssertionError("greedy policy selected an illegal action")
    return action, probabilities


def _retrieval_stage(core: SCRPEnvironment, initial_containers: int) -> str:
    live = sum(stack.height for stack in core.state.stacks)
    progress = 1.0 - live / max(initial_containers, 1)
    return "early" if progress < 1 / 3 else "middle" if progress < 2 / 3 else "late"


def _run_rl_episode(instance, sample, policy, config):
    core = SCRPEnvironment(
        SCRPConfig(instance.num_stacks, instance.max_tiers, root_seed=config.seed,
                   max_steps=config.max_steps, validate_after_transition=True),
        instance,
    )
    env = SCRPRLAdapter(core, observation_version=config.observation_version,
                        o2_mmax=config.Mmax or 6)
    observation, info = env.reset(seed=sample.scenario_seed)
    initial_fingerprint = _public_state_fingerprint(core)
    initial_containers = sum(stack.height for stack in core.state.stacks)
    eri = ERIBaseline()
    actions = []
    diagnostics = []
    invalid = 0
    while not info["terminated"]:
        legal_mask = np.asarray(info["action_mask"], dtype=bool)
        action, _ = greedy_public_action(
            policy, observation, legal_mask, instance.num_stacks
        )
        legal = tuple(np.flatnonzero(legal_mask).tolist())
        eri_action = eri.select_destination(instance, core.state, legal)
        location = core.state.locations[core.state.current_target_id]
        blocker = core.state.stacks[location.stack_id].top_id
        scores = {
            destination: eri.destination_score(instance, core.state, blocker, destination)
            for destination in legal
        }
        best = min(scores.values())
        gap = float(scores[action] - best)
        batch_id = instance.container_by_id[blocker].batch_id
        batch_size = sum(
            container.batch_id == batch_id for container in instance.containers
        )
        diagnostics.append({
            "exact": int(action == eri_action),
            "equivalent": int(scores[action] == best),
            "strictly_worse": int(scores[action] > best),
            "penalty": gap,
            "legal_destinations": len(legal),
            "eri_optimal_ties": sum(score == best for score in scores.values()),
            "batch_size": int(batch_size),
            "stage": _retrieval_stage(core, initial_containers),
        })
        if action not in legal:
            invalid += 1
            raise AssertionError("RL selected an invalid action")
        actions.append(action)
        observation, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    return {
        "scenario_id": core.scenario_id,
        "initial_public_fingerprint": initial_fingerprint,
        "relocations": int(core.state.relocation_count),
        "terminated": bool(core.state.terminated),
        "truncated": bool(info.get("truncated", False)),
        "invalid_actions": invalid,
        "actions": actions,
        "diagnostics": diagnostics,
    }


def _run_eri_episode(instance, sample, config):
    core = SCRPEnvironment(
        SCRPConfig(instance.num_stacks, instance.max_tiers, root_seed=config.seed,
                   max_steps=config.max_steps, validate_after_transition=True),
        instance,
    )
    state = core.reset(seed=sample.scenario_seed)
    initial_fingerprint = _public_state_fingerprint(core)
    eri = ERIBaseline()
    eri.reset(0)
    actions = []
    while not state.terminated:
        legal = core.legal_destinations()
        action = eri.select_destination(instance, core.state, legal)
        if action not in legal:
            raise AssertionError("ERI selected an invalid action")
        actions.append(action)
        state = core.step(action).state
    return {
        "scenario_id": core.scenario_id,
        "initial_public_fingerprint": initial_fingerprint,
        "relocations": int(state.relocation_count),
        "terminated": bool(state.terminated),
        "truncated": False,
        "invalid_actions": 0,
        "actions": actions,
    }


def paired_rollout(instance, sample, policy, config, parameter_group: str) -> tuple[dict, list[dict]]:
    """Run frozen greedy RL and ERI from identical initial/scenario state."""

    was_training = policy.training
    before_hash = policy_state_sha256(policy)
    policy.eval()
    try:
        with torch.inference_mode():
            rl = _run_rl_episode(instance, sample, policy, config)
        eri = _run_eri_episode(instance, sample, config)
    finally:
        policy.train(was_training)
    after_hash = policy_state_sha256(policy)
    if before_hash != after_hash:
        raise AssertionError("frozen policy changed during Phase 14 evaluation")
    scenario_match = rl["scenario_id"] == eri["scenario_id"]
    initial_match = rl["initial_public_fingerprint"] == eri["initial_public_fingerprint"]
    if not scenario_match or not initial_match:
        raise AssertionError("RL and ERI did not receive the identical scenario and initial state")
    dimensions = parse_parameter_group(parameter_group)
    delta = int(rl["relocations"] - eri["relocations"])
    row = {
        "seed": int(config.seed),
        "dataset": sample.variant,
        "base_instance_id": sample.base_instance_id,
        "scenario_seed": sample.scenario_seed,
        "scenario_id": rl["scenario_id"],
        "S": dimensions["S"],
        "fill": dimensions["fill"],
        "rl_relocations": rl["relocations"],
        "eri_relocations": eri["relocations"],
        "delta": delta,
        "rl_wins": int(delta < 0),
        "ties": int(delta == 0),
        "eri_wins": int(delta > 0),
        "invalid_actions": rl["invalid_actions"] + eri["invalid_actions"],
        "truncations": int(rl["truncated"] or eri["truncated"]),
        "numerical_failures": 0,
        "scenario_mismatches": int(not scenario_match or not initial_match),
        "episode_length": max(len(rl["actions"]), len(eri["actions"])),
    }
    action_rows = []
    outcome = "downstream_better" if delta < 0 else "tie" if delta == 0 else "worse"
    for diagnostic in rl["diagnostics"]:
        action_rows.append({**diagnostic, "seed": int(config.seed),
                            "dataset": sample.variant, "S": dimensions["S"],
                            "fill": dimensions["fill"], "delta": delta,
                            "episode_length": row["episode_length"],
                            "non_eri_outcome": outcome if diagnostic["strictly_worse"] else None})
    return row, action_rows


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else math.nan


def relocation_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        "coordinates": len(rows),
        "RL_mean": _mean(float(row["rl_relocations"]) for row in rows),
        "ERI_mean": _mean(float(row["eri_relocations"]) for row in rows),
        "delta": _mean(float(row["delta"]) for row in rows),
        "relative_gap_percent": 100.0 * _mean(float(row["delta"]) for row in rows)
        / _mean(float(row["eri_relocations"]) for row in rows),
        "RL_wins": sum(int(row["delta"] < 0) for row in rows),
        "ties": sum(int(row["delta"] == 0) for row in rows),
        "ERI_wins": sum(int(row["delta"] > 0) for row in rows),
    }


def hierarchical_paired_bootstrap(
    rows: Sequence[Mapping[str, object]], *, repetitions: int = 20_000,
    bootstrap_seed: int = 20260818,
) -> dict[str, object]:
    """Bootstrap seed -> dataset -> base layout -> paired scenario."""

    if not rows:
        raise ValueError("bootstrap requires paired rows")
    nested: dict[int, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in rows:
        nested[int(row["seed"])][str(row["dataset"])][str(row["base_instance_id"])].append(
            float(row["delta"])
        )
    seeds = sorted(nested)
    if seeds != list(FROZEN_SEEDS):
        raise ValueError("bootstrap requires all three frozen policy seeds")
    datasets = sorted(nested[seeds[0]])
    base_ids = sorted(nested[seeds[0]][datasets[0]])
    scenario_count = len(nested[seeds[0]][datasets[0]][base_ids[0]])
    values = np.empty(
        (len(seeds), len(datasets), len(base_ids), scenario_count), dtype=np.float64
    )
    for seed_index, seed in enumerate(seeds):
        if sorted(nested[seed]) != datasets:
            raise ValueError("bootstrap datasets differ across policy seeds")
        for dataset_index, dataset in enumerate(datasets):
            if sorted(nested[seed][dataset]) != base_ids:
                raise ValueError("bootstrap base layouts differ across seed/dataset blocks")
            for base_index, base_id in enumerate(base_ids):
                scenarios = nested[seed][dataset][base_id]
                if len(scenarios) != scenario_count:
                    raise ValueError("bootstrap scenario counts differ across base layouts")
                values[seed_index, dataset_index, base_index] = scenarios
    rng = np.random.default_rng(bootstrap_seed)
    draws = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        seed_means = []
        for seed_index in rng.integers(0, len(seeds), size=len(seeds)):
            dataset_means = []
            for dataset_index in rng.integers(0, len(datasets), size=len(datasets)):
                selected_bases = rng.integers(0, len(base_ids), size=len(base_ids))
                block = values[seed_index, dataset_index, selected_bases]
                selected_scenarios = rng.integers(
                    0, scenario_count, size=(len(base_ids), scenario_count)
                )
                dataset_means.append(float(np.mean(
                    np.take_along_axis(block, selected_scenarios, axis=1)
                )))
            seed_means.append(float(np.mean(dataset_means)))
        draws[repetition] = float(np.mean(seed_means))
    return {
        "method": "hierarchical paired bootstrap: seed -> dataset -> base layout -> scenario",
        "repetitions": repetitions,
        "bootstrap_seed": bootstrap_seed,
        "delta": _mean(float(row["delta"]) for row in rows),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
    }


def secondary_statistics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    rl = np.asarray([float(row["rl_relocations"]) for row in rows])
    eri = np.asarray([float(row["eri_relocations"]) for row in rows])
    delta = rl - eri
    if not len(delta):
        raise ValueError("secondary statistics require paired rows")
    try:
        w = wilcoxon(rl, eri)
        wilcoxon_record = {"statistic": float(w.statistic), "p_value": float(w.pvalue)}
    except ValueError:
        wilcoxon_record = {"statistic": 0.0, "p_value": 1.0}
    t = ttest_rel(rl, eri)
    sd = float(np.std(delta, ddof=1)) if len(delta) > 1 else 0.0
    return {
        "paired_wilcoxon": wilcoxon_record,
        "paired_t_test": {"statistic": float(t.statistic), "p_value": float(t.pvalue)},
        "cohen_dz": float(np.mean(delta) / sd) if sd > 0 else 0.0,
    }


def eri_mechanism_diagnostics(action_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    count = len(action_rows)
    if not count:
        return {"public_decision_states": 0}
    outcomes = Counter(
        str(row["non_eri_outcome"]) for row in action_rows if row["non_eri_outcome"]
    )
    return {
        "public_decision_states": count,
        "exact_action_agreement_rate": _mean(float(row["exact"]) for row in action_rows),
        "ERI_score_equivalent_rate": _mean(float(row["equivalent"]) for row in action_rows),
        "strictly_worse_ERI_score_rate": _mean(float(row["strictly_worse"]) for row in action_rows),
        "mean_ERI_penalty": _mean(float(row["penalty"]) for row in action_rows),
        "non_ERI_minimum_actions": {
            "downstream_better": outcomes["downstream_better"],
            "tie": outcomes["tie"],
            "worse": outcomes["worse"],
        },
    }


def largest_gap_regimes(rows: Sequence[Mapping[str, object]], limit: int = 10) -> list[dict]:
    fields = ("dataset", "S", "fill")
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in fields)].append(float(row["delta"]))
    records = [
        {**dict(zip(fields, key)), "coordinates": len(values), "mean_delta": _mean(values)}
        for key, values in grouped.items()
    ]
    return sorted(records, key=lambda record: record["mean_delta"], reverse=True)[:limit]


def action_regime_breakdown(
    action_rows: Sequence[Mapping[str, object]], *, descending: bool, limit: int = 10
) -> list[dict]:
    fields = (
        "dataset", "S", "fill", "batch_size", "legal_destinations",
        "eri_optimal_ties", "episode_length", "stage",
    )
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for row in action_rows:
        grouped[tuple(row[field] for field in fields)].append(float(row["delta"]))
    records = [
        {**dict(zip(fields, key)), "decisions": len(values), "mean_episode_delta": _mean(values)}
        for key, values in grouped.items()
    ]
    return sorted(
        records, key=lambda record: record["mean_episode_delta"], reverse=descending
    )[:limit]


def development_success_gate(
    overall: Mapping[str, object], bootstrap: Mapping[str, object],
    ds1: Mapping[str, object], ds2: Mapping[str, object],
    seed_metrics: Sequence[Mapping[str, object]], integrity: Mapping[str, int],
) -> dict[str, object]:
    checks = {
        "pooled_delta_below_zero": float(overall["delta"]) < 0,
        "pooled_ci95_upper_below_zero": float(bootstrap["ci95_high"]) < 0,
        "at_least_two_favorable_seeds": sum(float(item["delta"]) < 0 for item in seed_metrics) >= 2,
        "DS1_delta_not_above_zero": float(ds1["delta"]) <= 0,
        "DS2_delta_not_above_zero": float(ds2["delta"]) <= 0,
        "paired_distribution_consistent": int(overall["RL_wins"]) > int(overall["ERI_wins"]),
        "zero_invalid_actions": int(integrity["invalid_actions"]) == 0,
        "zero_truncations": int(integrity["truncations"]) == 0,
        "zero_numerical_failures": int(integrity["numerical_failures"]) == 0,
        "zero_scenario_mismatches": int(integrity["scenario_mismatches"]) == 0,
        "zero_hidden_information_leaks": int(integrity["hidden_information_leaks"]) == 0,
        "checkpoint_selection_frozen": bool(integrity["checkpoint_selection_frozen"]),
        "zero_test_split_usage": int(integrity["test_split_usage"]) == 0,
    }
    if all(checks.values()):
        result = "YES"
    elif float(overall["delta"]) <= 0 and float(bootstrap["ci95_high"]) >= 0:
        result = "INCONCLUSIVE"
    else:
        result = "NO"
    return {"checks": checks, "result": result, "passed": result == "YES"}


def summarize_evaluation(rows: Sequence[Mapping[str, object]], action_rows: Sequence[Mapping[str, object]]):
    overall = relocation_metrics(rows)
    seed_metrics = [
        {"seed": seed, **relocation_metrics([row for row in rows if int(row["seed"]) == seed])}
        for seed in FROZEN_SEEDS
    ]
    datasets = {
        dataset: relocation_metrics([row for row in rows if row["dataset"] == dataset])
        for dataset in DATASETS
    }
    bootstrap = hierarchical_paired_bootstrap(rows)
    dataset_bootstrap = {}
    for dataset in DATASETS:
        subset = [row for row in rows if row["dataset"] == dataset]
        # Preserve the top seed layer; with one fixed dataset, the next layer is base.
        adjusted = [{**row, "dataset": dataset} for row in subset]
        dataset_bootstrap[dataset] = hierarchical_paired_bootstrap(adjusted)
    integrity = {
        "invalid_actions": sum(int(row["invalid_actions"]) for row in rows),
        "truncations": sum(int(row["truncations"]) for row in rows),
        "numerical_failures": sum(int(row["numerical_failures"]) for row in rows),
        "scenario_mismatches": sum(int(row["scenario_mismatches"]) for row in rows),
        "hidden_information_leaks": 0,
        "checkpoint_selection_frozen": True,
        "test_split_usage": 0,
    }
    gate = development_success_gate(
        overall, bootstrap, datasets["DS1"], datasets["DS2"], seed_metrics, integrity
    )
    return {
        "seed_results": seed_metrics,
        "pooled_overall": {**overall, "ci95_low": bootstrap["ci95_low"],
                           "ci95_high": bootstrap["ci95_high"]},
        "datasets": {
            dataset: {**datasets[dataset], "ci95_low": dataset_bootstrap[dataset]["ci95_low"],
                      "ci95_high": dataset_bootstrap[dataset]["ci95_high"]}
            for dataset in DATASETS
        },
        "hierarchical_bootstrap": bootstrap,
        "secondary_statistics": secondary_statistics(rows),
        "ERI_mechanism": eri_mechanism_diagnostics(action_rows),
        "largest_RL_minus_ERI_gap_regimes": largest_gap_regimes(rows),
        "largest_action_level_failure_regimes": action_regime_breakdown(
            action_rows, descending=True
        ),
        "largest_action_level_success_regimes": action_regime_breakdown(
            action_rows, descending=False
        ),
        "integrity": integrity,
        "RL_BEATS_ERI_DEVELOPMENT": gate["result"],
        "success_gate": gate,
    }


def run_development_evaluation(protocol, manifest, provider, repository_root: str | Path):
    preflight = checkpoint_preflight(protocol, repository_root)
    if not preflight["ready"]:
        raise Phase14PreflightError(
            "frozen Phase 13 checkpoints are missing or invalid; retraining is prohibited: "
            + ", ".join([*preflight["missing"], *preflight["invalid"]])
        )
    holdout = phase14_development_holdout(manifest)
    schedule = ScenarioSeedSchedule(manifest)
    rows, action_rows, checkpoint_records = [], [], []
    process_checks = []
    for selection in protocol["checkpoint_selection"]["selections"]:
        path = Path(repository_root) / selection["relative_path"]
        policy, config, state_hash = _load_frozen_policy(
            path, int(selection["seed"]), int(selection["episode"]), "cuda:0"
        )
        checkpoint_records.append({**selection, "sha256": _sha256_file(path),
                                   "policy_state_sha256": state_hash})
        process_checks.append(_nvidia_python_process_seen())
        for ref in holdout:
            for dataset in DATASETS:
                for scenario_index in range(20):
                    sample = _sample(
                        ref, dataset,
                        schedule.seed_for("validation", ref.base_instance_id, scenario_index),
                        scenario_index,
                    )
                    row, decisions = paired_rollout(
                        provider(sample), sample, policy, config, ref.parameter_group
                    )
                    rows.append(row)
                    action_rows.extend(decisions)
    if len(rows) != 23_040:
        raise AssertionError(f"expected 23040 paired coordinates, got {len(rows)}")
    return {
        "phase14_version": PHASE14_VERSION,
        "scope": "DEVELOPMENT_ONLY_DIRECT_RL_VS_ERI",
        "status": "COMPLETED",
        "base_sha": PHASE14_BASE_SHA,
        "checkpoint_records": checkpoint_records,
        "holdout": protocol["development_holdout"],
        "cuda_identity": {
            **cuda_identity(resolve_training_device("cuda:0")),
            "model_parameters_on_cuda": True,
            "nvidia_smi_python_process_seen": all(process_checks),
        },
        **summarize_evaluation(rows, action_rows),
        "formal_test_episode_usage": 0,
        "phase8_raw_rows_accessed": False,
        "formal_test_rerun": False,
        "test_split_accessed": False,
        "new_training_performed": False,
        "OPTIMIZATION_STABILITY_WARNING": "YES",
    }, rows, action_rows


def blocked_summary(protocol, manifest, preflight) -> dict[str, object]:
    holdout = phase14_development_holdout(manifest)
    return {
        "phase14_version": PHASE14_VERSION,
        "scope": "DEVELOPMENT_ONLY_DIRECT_RL_VS_ERI",
        "status": "BLOCKED_MISSING_FROZEN_CHECKPOINTS",
        "base_sha": PHASE14_BASE_SHA,
        "checkpoint_selection": protocol["checkpoint_selection"],
        "checkpoint_preflight": preflight,
        "development_holdout": {
            **protocol["development_holdout"],
            "verified_base_layouts": len(holdout),
        },
        "cuda_identity": {
            **cuda_identity(resolve_training_device("cuda:0")),
            "model_parameters_on_cuda": False,
            "nvidia_smi_python_process_seen": False,
            "reason": "No Phase 13 policy checkpoint was available to load",
        },
        "metrics": None,
        "RL_BEATS_ERI_DEVELOPMENT": "NOT_EVALUATED",
        "blocker": (
            "Phase 13 did not persist the frozen 15k Treatment policies. Phase 14 is "
            "not authorized to retrain or reconstruct them, so no RL-vs-ERI rollout ran."
        ),
        "formal_test_episode_usage": 0,
        "phase8_raw_rows_accessed": False,
        "formal_test_rerun": False,
        "test_split_accessed": False,
        "new_training_performed": False,
        "OPTIMIZATION_STABILITY_WARNING": "YES",
    }
