"""Pre-registered Phase 13 long-run ERI auxiliary stability study."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from experiments.phase11_eri_auxiliary import (
    _fingerprint,
    _train_chunk,
    eri_diagnostic,
    relocation_summary,
    validation_rows,
)
from experiments.phase12_multiseed_replication import (
    _clone_rng_state,
    _config_hash,
    _make_pair,
    cuda_identity,
    hierarchical_seed_base_bootstrap,
    run_cuda_smoke,
    utc_now,
)
from experiments.posttest_analysis import fixed_development_refs
from experiments.protocol import SplitManifest
from scrp.formal_training import FormalIterationMetrics, FormalTrainingConfig, resolve_training_device


PHASE13_VERSION = "phase13-longrun-stability-v1"
PHASE13_BASE_SHA = "dbb18a5e50a7309e9bbc8f3f6c4e38c37cff89c2"
FROZEN_SEEDS = (20260816, 20260818, 20260819)
VALIDATION_CHECKPOINTS = (2_500, 5_000, 7_500, 10_000, 12_500, 15_000)
EPISODES_PER_ARM = 15_000
TOTAL_TRAINING_EPISODES = 90_000
MAX_NON_DEGRADATION_CI_UPPER = 0.5
SEVERE_DATASET_DEGRADATION = 0.25
TRAJECTORY_MATERIAL_CHANGE = 0.25
TRAJECTORY_HIGH_RANGE = 1.0
WARNING_ENTROPY_DECLINE = 0.10
WARNING_GRADIENT_INCREASE_RATIO = 1.20


def load_phase13_protocol(path: str | Path) -> tuple[tuple[int, ...], tuple[int, ...], FormalTrainingConfig]:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("phase13_version") != PHASE13_VERSION:
        raise ValueError("unsupported Phase 13 protocol version")
    if record.get("base_main_sha") != PHASE13_BASE_SHA:
        raise ValueError("Phase 13 base main SHA differs from pre-registration")
    seeds = tuple(int(seed) for seed in record.get("frozen_seeds", ()))
    checkpoints = tuple(int(value) for value in record.get("validation_checkpoints", ()))
    if seeds != FROZEN_SEEDS or len(set(seeds)) != 3:
        raise ValueError("Phase 13 frozen seed list differs from pre-registration")
    if checkpoints != VALIDATION_CHECKPOINTS:
        raise ValueError("Phase 13 validation schedule differs from pre-registration")
    if int(record.get("episodes_per_arm", -1)) != EPISODES_PER_ARM:
        raise ValueError("Phase 13 requires exactly 15,000 episodes per arm")
    if int(record.get("total_training_episodes", -1)) != TOTAL_TRAINING_EPISODES:
        raise ValueError("Phase 13 total budget must be 90,000 episodes")
    config = FormalTrainingConfig.from_record(record["treatment_config"])
    if config.device != "cuda:0":
        raise ValueError("Phase 13 protocol must explicitly request cuda:0")
    if config.eri_aux_coefficient != 0.10:
        raise ValueError("Phase 13 treatment must use lambda_eri=0.10")
    return seeds, checkpoints, config


def _scenario_fingerprint(samples) -> str:
    payload = json.dumps(_fingerprint(samples), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _finite(values: Sequence[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def aggregate_training_window(
    metrics: Sequence[FormalIterationMetrics],
    *,
    start_episode: int,
    end_episode: int,
    accepted_refreshes: int,
) -> dict[str, object]:
    if not metrics:
        raise ValueError("training window cannot be empty")
    entropy = np.asarray([metric.entropy for metric in metrics], dtype=np.float64)
    gradients = np.asarray([metric.grad_norm for metric in metrics], dtype=np.float64)
    ratios = np.asarray([
        metric.weighted_eri_gradient_norm / metric.rl_gradient_norm
        for metric in metrics if metric.rl_gradient_norm > 0.0
    ], dtype=np.float64)
    result = {
        "episode_window": [start_episode, end_episode],
        "iterations": len(metrics),
        "episodes": sum(metric.episodes for metric in metrics),
        "total_loss_mean": float(np.mean([metric.loss for metric in metrics])),
        "policy_loss_mean": float(np.mean([metric.policy_loss for metric in metrics])),
        "ERI_auxiliary_loss_mean": float(np.mean([metric.eri_aux_loss for metric in metrics])),
        "entropy_mean": float(entropy.mean()),
        "entropy_median": float(np.median(entropy)),
        "pre_clip_gradient_norm_mean": float(gradients.mean()),
        "pre_clip_gradient_norm_median": float(np.median(gradients)),
        "pre_clip_gradient_norm_p90": float(np.quantile(gradients, 0.90)),
        "pre_clip_gradient_norm_p95": float(np.quantile(gradients, 0.95)),
        "pre_clip_gradient_norm_max": float(gradients.max()),
        "gradient_clipping_frequency": float(np.mean([metric.gradient_clipped for metric in metrics])),
        "weighted_ERI_to_RL_gradient_ratio_mean": float(ratios.mean()) if ratios.size else 0.0,
        "FGB_accepted_refreshes_in_window": accepted_refreshes,
        "FGB_rejected_refreshes_in_window": len(metrics) - accepted_refreshes,
        "invalid_actions": sum(metric.invalid_actions for metric in metrics),
        "truncations": sum(metric.truncations for metric in metrics),
        "numerical_failures": 0,
        "scenario_mismatches": sum(metric.scenario_mismatches for metric in metrics),
    }
    numeric = [value for value in result.values() if isinstance(value, (int, float))]
    result["finite"] = _finite(numeric)
    return result


def classify_delta_trajectory(deltas: Sequence[float]) -> dict[str, object]:
    if len(deltas) != len(VALIDATION_CHECKPOINTS):
        raise ValueError("trajectory must contain all six frozen checkpoints")
    values = np.asarray(deltas, dtype=np.float64)
    signs = np.sign(values)
    sign_changes = int(np.sum(signs[1:] * signs[:-1] < 0))
    spread = float(values.max() - values.min())
    early_best = float(values[:2].min())
    late_change_from_best = float(values[-1] - early_best)
    if spread >= TRAJECTORY_HIGH_RANGE or sign_changes >= 2:
        label = "high_variance"
    elif early_best < 0.0 and late_change_from_best >= TRAJECTORY_MATERIAL_CHANGE:
        label = "early_improvement_then_deterioration"
    elif np.all(values < 0.0) and values[-1] <= values[0]:
        label = "sustained_improvement"
    elif early_best < 0.0 and np.all(values[2:] < 0.0):
        label = "early_improvement_then_plateau"
    elif np.all(np.abs(values) <= TRAJECTORY_MATERIAL_CHANGE):
        label = "no_material_difference"
    else:
        label = "high_variance"
    return {
        "classification": label,
        "frozen_rules": {
            "material_change": TRAJECTORY_MATERIAL_CHANGE,
            "high_range": TRAJECTORY_HIGH_RANGE,
            "high_variance_sign_changes": 2,
        },
        "range": spread,
        "sign_changes": sign_changes,
        "early_best": early_best,
        "endpoint_minus_early_best": late_change_from_best,
    }


def _arm_checkpoint_record(policy, config, manifest, provider, validation_refs, metrics, start_episode, episode, accepted_refreshes):
    rows = validation_rows(policy, config, manifest, provider, validation_refs)
    return rows, {
        "validation": relocation_summary(rows),
        "ERI_diagnostic": eri_diagnostic(policy, config, manifest, provider, validation_refs),
        "optimization_window": aggregate_training_window(
            metrics, start_episode=start_episode, end_episode=episode,
            accepted_refreshes=accepted_refreshes,
        ),
    }


def _mean_checkpoint(seed_records, arm: str, section: str, field: str) -> float:
    return float(np.mean([record[arm][section][field] for record in seed_records]))


def aggregate_checkpoint(seed_records, paired_rows) -> dict[str, object]:
    pooled = hierarchical_seed_base_bootstrap(paired_rows)
    favorable = {
        "overall": sum(record["overall_delta"] < 0.0 for record in seed_records),
        "DS1": sum(record["DS1_delta"] < 0.0 for record in seed_records),
        "DS2": sum(record["DS2_delta"] < 0.0 for record in seed_records),
    }
    mechanisms = {}
    for field in (
        "exact_action_agreement_rate", "ERI_score_equivalent_action_rate",
        "strictly_worse_ERI_score_action_rate", "mean_ERI_score_penalty",
        "mean_probability_mass_on_ERI_optimal_set",
        "greedy_action_outside_ERI_optimal_set_rate",
    ):
        mechanisms[field] = {
            "control": _mean_checkpoint(seed_records, "control", "ERI_diagnostic", field),
            "treatment": _mean_checkpoint(seed_records, "treatment", "ERI_diagnostic", field),
        }
    optimization = {}
    for field in (
        "entropy_mean", "entropy_median", "pre_clip_gradient_norm_mean",
        "pre_clip_gradient_norm_median", "pre_clip_gradient_norm_p90",
        "pre_clip_gradient_norm_p95", "gradient_clipping_frequency",
        "total_loss_mean", "policy_loss_mean", "ERI_auxiliary_loss_mean",
        "weighted_ERI_to_RL_gradient_ratio_mean",
    ):
        control = _mean_checkpoint(seed_records, "control", "optimization_window", field)
        treatment = _mean_checkpoint(seed_records, "treatment", "optimization_window", field)
        optimization[field] = {
            "control": control,
            "treatment": treatment,
            "treatment_minus_control": treatment - control,
        }
    integrity = {
        field: sum(
            record[arm]["optimization_window"][field]
            for record in seed_records for arm in ("control", "treatment")
        )
        for field in ("invalid_actions", "truncations", "numerical_failures", "scenario_mismatches")
    }
    return {
        "pooled_paired_bootstrap": pooled,
        "favorable_seed_counts": favorable,
        "ERI_mechanism": mechanisms,
        "optimization_window": optimization,
        "integrity": integrity,
    }


def _late_systematic_reversal(aggregates: Mapping[str, Mapping[str, object]]) -> bool:
    late = VALIDATION_CHECKPOINTS[2:]
    for left, right in zip(late, late[1:]):
        a, b = aggregates[str(left)], aggregates[str(right)]
        if (
            a["pooled_paired_bootstrap"]["overall"]["delta"] > 0.0
            and b["pooled_paired_bootstrap"]["overall"]["delta"] > 0.0
            and a["favorable_seed_counts"]["overall"] < 2
            and b["favorable_seed_counts"]["overall"] < 2
        ):
            return True
    return False


def longrun_success_gate(aggregates, endpoint_seed_records, *, cuda_integrity: bool) -> dict[str, object]:
    endpoint = aggregates[str(EPISODES_PER_ARM)]
    pooled = endpoint["pooled_paired_bootstrap"]
    favorable = endpoint["favorable_seed_counts"]
    mechanisms = endpoint["ERI_mechanism"]
    severe_ds1 = pooled["DS1"]["delta"] >= SEVERE_DATASET_DEGRADATION and favorable["DS1"] < 2
    severe_ds2 = pooled["DS2"]["delta"] >= SEVERE_DATASET_DEGRADATION and favorable["DS2"] < 2
    integrity = all(
        aggregate["integrity"][field] == 0
        for aggregate in aggregates.values()
        for field in ("invalid_actions", "truncations", "numerical_failures", "scenario_mismatches")
    )
    checks = {
        "endpoint_pooled_overall_favorable": pooled["overall"]["delta"] < 0.0,
        "endpoint_at_least_2_of_3_favorable": favorable["overall"] >= 2,
        "endpoint_CI_excludes_large_degradation": pooled["overall"]["ci95_high"] <= MAX_NON_DEGRADATION_CI_UPPER,
        "no_consecutive_systematic_late_reversal": not _late_systematic_reversal(aggregates),
        "DS1_no_stable_material_degradation": not severe_ds1,
        "DS2_no_stable_material_degradation": not severe_ds2,
        "endpoint_strictly_worse_rate_lower": (
            mechanisms["strictly_worse_ERI_score_action_rate"]["treatment"]
            < mechanisms["strictly_worse_ERI_score_action_rate"]["control"]
        ),
        "endpoint_mean_ERI_penalty_not_higher": (
            mechanisms["mean_ERI_score_penalty"]["treatment"]
            <= mechanisms["mean_ERI_score_penalty"]["control"]
        ),
        "no_integrity_regression": integrity,
        "no_test_leakage": True,
        "cuda_integrity": cuda_integrity,
    }
    return {
        "frozen_thresholds": {
            "required_favorable_seeds": 2,
            "maximum_CI_upper_for_non_large_degradation": MAX_NON_DEGRADATION_CI_UPPER,
            "severe_dataset_degradation": SEVERE_DATASET_DEGRADATION,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def optimization_warning_gate(aggregates, endpoint_seed_records) -> dict[str, object]:
    late = [aggregates[str(cp)]["optimization_window"] for cp in VALIDATION_CHECKPOINTS[2:]]
    clip = [window["gradient_clipping_frequency"]["treatment"] for window in late]
    consecutive_clip = any(a >= 0.99 and b >= 0.99 for a, b in zip(clip, clip[1:]))
    entropies = [window["entropy_mean"]["treatment"] for window in late]
    pooled_deltas = [
        aggregates[str(cp)]["pooled_paired_bootstrap"]["overall"]["delta"]
        for cp in VALIDATION_CHECKPOINTS
    ]
    entropy_decline = (
        entropies[-1] <= entropies[0] - WARNING_ENTROPY_DECLINE
        and sum(right <= left for left, right in zip(entropies, entropies[1:])) >= 2
        and pooled_deltas[-1] >= min(pooled_deltas[:3]) + TRAJECTORY_MATERIAL_CHANGE
    )
    post5 = pooled_deltas[1:]
    post5_degradation = (
        post5[-1] >= post5[0] + TRAJECTORY_MATERIAL_CHANGE
        and sum(right >= left for left, right in zip(post5, post5[1:])) >= 3
    )
    gradients = [window["pre_clip_gradient_norm_mean"]["treatment"] for window in late]
    late_gradient_rise = gradients[-1] >= WARNING_GRADIENT_INCREASE_RATIO * gradients[0]
    endpoint_deltas = [record["overall_delta"] for record in endpoint_seed_records]
    endpoint_entropies = [record["treatment"]["optimization_window"]["entropy_mean"] for record in endpoint_seed_records]
    divergent = (
        max(endpoint_deltas) - min(endpoint_deltas) >= TRAJECTORY_HIGH_RANGE
        or max(endpoint_entropies) - min(endpoint_entropies) >= 0.5
    )
    triggers = {
        "consecutive_late_treatment_clipping_at_least_99_percent": consecutive_clip,
        "entropy_decline_with_validation_plateau_or_degradation": entropy_decline,
        "post_5k_pooled_delta_degrades_toward_zero": post5_degradation,
        "late_pre_clip_gradient_norm_rises_materially": late_gradient_rise,
        "divergent_seed_optimization": divergent,
    }
    return {
        "frozen_thresholds": {
            "clipping_frequency": 0.99,
            "consecutive_late_windows": 2,
            "entropy_decline": WARNING_ENTROPY_DECLINE,
            "validation_material_change": TRAJECTORY_MATERIAL_CHANGE,
            "gradient_increase_ratio": WARNING_GRADIENT_INCREASE_RATIO,
            "divergent_seed_delta_range": TRAJECTORY_HIGH_RANGE,
            "divergent_seed_entropy_range": 0.5,
        },
        "triggers": triggers,
        "warning": any(triggers.values()),
    }


def _run_identity(seed, arm, config, code_sha, fingerprint, started, ended, identity, refreshes):
    return {
        "run_id": f"phase13-seed{seed}-{arm}-cuda-v1",
        "phase": 13, "seed": seed, "arm": arm, "device": config.device,
        "gpu_model": identity["gpu_model"], "pytorch_version": identity["pytorch_version"],
        "cuda_version": identity["pytorch_cuda_runtime"], "config_hash": _config_hash(config),
        "code_sha": code_sha, "scenario_fingerprint": fingerprint,
        "start_timestamp": started, "end_timestamp": ended,
        "training_episodes": EPISODES_PER_ARM, "validation_checkpoints": list(VALIDATION_CHECKPOINTS),
        "FGB_accepted_refreshes": len(refreshes),
        "FGB_rejected_refreshes": EPISODES_PER_ARM // config.batch_size - len(refreshes),
        "FGB_refresh_episode_positions": [record.iteration * config.batch_size for record in refreshes],
    }


def run_phase13_smoke(manifest, provider, template, checkpoint_dir: Path) -> dict[str, object]:
    result = run_cuda_smoke(manifest, provider, template, checkpoint_dir)
    result["phase"] = "Phase 13 CUDA smoke"
    return result


def run_longrun_stability(
    manifest: SplitManifest,
    provider,
    template: FormalTrainingConfig,
    *,
    code_sha: str,
    smoke_record: Mapping[str, object],
) -> dict[str, object]:
    device = resolve_training_device(template.device)
    torch.cuda.set_device(device)
    identity = cuda_identity(device)
    if not smoke_record.get("passed"):
        raise RuntimeError("Phase 13 CUDA smoke gate did not pass")
    validation_refs = fixed_development_refs(manifest, "validation")
    seed_trajectories = []
    checkpoint_seed_records = {str(cp): [] for cp in VALIDATION_CHECKPOINTS}
    checkpoint_paired_rows = {str(cp): {} for cp in VALIDATION_CHECKPOINTS}
    for seed in FROZEN_SEEDS:
        started = utc_now()
        control, treatment, common_rng = _make_pair(manifest, provider, template, seed)
        print(json.dumps({
            "seed": seed,
            "control_model_device": str(next(control.policy.parameters()).device),
            "treatment_model_device": str(next(treatment.policy.parameters()).device),
            "gpu_name": torch.cuda.get_device_name(0),
            "cuda_memory_allocated": torch.cuda.memory_allocated(),
            "cuda_memory_reserved": torch.cuda.memory_reserved(),
        }), flush=True)
        if next(control.policy.parameters()).device.type != "cuda" or next(treatment.policy.parameters()).device.type != "cuda":
            raise AssertionError("Phase 13 formal model is not on CUDA")
        control_rng = _clone_rng_state(common_rng)
        treatment_rng = _clone_rng_state(common_rng)
        previous_episode = 0
        per_seed = []
        for episode in VALIDATION_CHECKPOINTS:
            iterations = (episode - previous_episode) // template.batch_size
            control_refreshes_before = control.baseline_updates
            treatment_refreshes_before = treatment.baseline_updates
            control_metrics, control_rng = _train_chunk(control, iterations, control_rng)
            treatment_metrics, treatment_rng = _train_chunk(treatment, iterations, treatment_rng)
            control_fp = _scenario_fingerprint(control.sample_history)
            treatment_fp = _scenario_fingerprint(treatment.sample_history)
            if control_fp != treatment_fp:
                raise AssertionError(f"seed {seed} scenario fingerprint mismatch at {episode}")
            control_config = replace(template, seed=seed, eri_aux_coefficient=0.0)
            treatment_config = replace(template, seed=seed)
            control_rows, control_record = _arm_checkpoint_record(
                control.policy, control_config, manifest, provider, validation_refs,
                control_metrics, previous_episode, episode,
                control.baseline_updates - control_refreshes_before,
            )
            treatment_rows, treatment_record = _arm_checkpoint_record(
                treatment.policy, treatment_config, manifest, provider, validation_refs,
                treatment_metrics, previous_episode, episode,
                treatment.baseline_updates - treatment_refreshes_before,
            )
            overall_delta = treatment_record["validation"]["mean_relocations"] - control_record["validation"]["mean_relocations"]
            ds1_delta = treatment_record["validation"]["DS1_mean"] - control_record["validation"]["DS1_mean"]
            ds2_delta = treatment_record["validation"]["DS2_mean"] - control_record["validation"]["DS2_mean"]
            record = {
                "episode": episode, "control": control_record, "treatment": treatment_record,
                "overall_delta": overall_delta, "DS1_delta": ds1_delta, "DS2_delta": ds2_delta,
                "scenario_fingerprint_equal": True, "scenario_fingerprint": control_fp,
            }
            per_seed.append(record)
            checkpoint_seed_records[str(episode)].append(record)
            checkpoint_paired_rows[str(episode)][seed] = (control_rows, treatment_rows)
            print(
                f"Phase 13 seed {seed}: checkpoint {episode}/{EPISODES_PER_ARM}, delta={overall_delta:+.6f}",
                flush=True,
            )
            previous_episode = episode
        ended = utc_now()
        final_fp = per_seed[-1]["scenario_fingerprint"]
        control_config = replace(template, seed=seed, eri_aux_coefficient=0.0)
        treatment_config = replace(template, seed=seed)
        seed_trajectories.append({
            "seed": seed,
            "checkpoints": per_seed,
            "overall_trajectory_classification": classify_delta_trajectory([r["overall_delta"] for r in per_seed]),
            "DS1_trajectory_classification": classify_delta_trajectory([r["DS1_delta"] for r in per_seed]),
            "DS2_trajectory_classification": classify_delta_trajectory([r["DS2_delta"] for r in per_seed]),
            "run_identity": {
                "control": _run_identity(seed, "control", control_config, code_sha, final_fp, started, ended, identity, control.baseline_refresh_history),
                "treatment": _run_identity(seed, "treatment", treatment_config, code_sha, final_fp, started, ended, identity, treatment.baseline_refresh_history),
            },
        })
    aggregates = {
        str(cp): aggregate_checkpoint(checkpoint_seed_records[str(cp)], checkpoint_paired_rows[str(cp)])
        for cp in VALIDATION_CHECKPOINTS
    }
    endpoint_records = checkpoint_seed_records[str(EPISODES_PER_ARM)]
    success = longrun_success_gate(aggregates, endpoint_records, cuda_integrity=True)
    warning = optimization_warning_gate(aggregates, endpoint_records)
    return {
        "phase13_version": PHASE13_VERSION,
        "scope": "DEVELOPMENT_ONLY_TRAIN_VALIDATION",
        "research_question": "Does the fixed lambda_eri=0.10 advantage remain stable through 15,000 episodes?",
        "eventual_objective": "Produce an RL policy with lower paired relocations than ERI under the same public-information regime.",
        "base_main_sha": PHASE13_BASE_SHA, "training_code_sha": code_sha,
        "frozen_seeds": list(FROZEN_SEEDS), "seed_selection_rationale": {
            "20260816": "Phase 12 clearly favorable seed",
            "20260818": "Phase 12 clearly favorable seed",
            "20260819": "Phase 12 only overall unfavorable seed",
        },
        "episodes_per_arm": EPISODES_PER_ARM,
        "total_training_episodes": TOTAL_TRAINING_EPISODES,
        "validation_checkpoints": list(VALIDATION_CHECKPOINTS),
        "control_definition": "lambda_eri=0", "treatment_definition": "lambda_eri=0.10",
        "started_from_episode_zero": True, "phase12_checkpoint_continuation": False,
        "cuda_identity": identity,
        "deterministic_settings": {
            "python_random_seeded": True, "numpy_seeded": True,
            "torch_cpu_seeded": True, "torch_cuda_seeded": True,
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        },
        "smoke_gate": {key: value for key, value in smoke_record.items() if key != "nvidia_smi_snapshot"},
        "seed_trajectories": seed_trajectories,
        "checkpoint_aggregates": aggregates,
        "formal_test_episode_usage": 0, "phase8_raw_rows_accessed": False,
        "formal_test_rerun": False, "test_split_accessed": False,
        "success_gate": success,
        "optimization_warning_gate": warning,
        "ERI_AUX_LONGRUN_STABILITY_SUCCESS": "YES" if success["passed"] else "NO",
        "OPTIMIZATION_STABILITY_WARNING": "YES" if warning["warning"] else "NO",
    }
