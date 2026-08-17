"""Pre-registered Phase 12 CUDA multi-seed ERI auxiliary replication."""

from __future__ import annotations

import copy
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from experiments.phase11_eri_auxiliary import (
    _fingerprint,
    _train_chunk,
    eri_diagnostic,
    optimization_summary,
    relocation_summary,
    validation_rows,
)
from experiments.posttest_analysis import fixed_development_refs
from experiments.protocol import SplitManifest
from scrp.formal_training import (
    ERI_AUXILIARY_VERSION,
    FormalIterationMetrics,
    FormalTrainingConfig,
    SCRPFormalTrainer,
    make_scrp_policy,
    policy_state_sha256,
    resolve_training_device,
    seed_reproducibly,
)


PHASE12_VERSION = "phase12-multiseed-replication-v1"
PHASE12_PARENT_SHA = "adcffa7375cb37cdb5b8d652e003a894cd5af126"
FROZEN_SEEDS = (20260816, 20260817, 20260818, 20260819, 20260820)
EPISODES_PER_ARM = 5_000
SMOKE_EPISODES_PER_ARM = 16
TIMING_EPISODES_PER_ARM = 100
BOOTSTRAP_REPETITIONS = 20_000
BOOTSTRAP_SEED = 20260817
MAX_ACCEPTABLE_POOLED_CI_UPPER = 0.5
SEVERE_DATASET_DEGRADATION = 0.25


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _config_hash(config: FormalTrainingConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _schedule_hash(samples) -> str:
    payload = json.dumps(_fingerprint(samples), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_phase12_protocol(path: str | Path) -> tuple[tuple[int, ...], FormalTrainingConfig]:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("phase12_version") != PHASE12_VERSION:
        raise ValueError("unsupported Phase 12 protocol version")
    if record.get("parent_phase11_sha") != PHASE12_PARENT_SHA:
        raise ValueError("Phase 12 parent SHA differs from pre-registration")
    seeds = tuple(int(seed) for seed in record.get("frozen_seeds", ()))
    if seeds != FROZEN_SEEDS or len(set(seeds)) != 5:
        raise ValueError("Phase 12 frozen seed list differs from pre-registration")
    config = FormalTrainingConfig.from_record(record["treatment_config"])
    if config.device != "cuda:0":
        raise ValueError("Phase 12 protocol must explicitly request cuda:0")
    if config.eri_aux_coefficient != 0.10:
        raise ValueError("Phase 12 treatment must use lambda_eri=0.10")
    if config.eri_auxiliary_version != ERI_AUXILIARY_VERSION:
        raise ValueError("Phase 12 must use the audited ERI auxiliary objective")
    return seeds, config


def cuda_identity(device: torch.device) -> dict[str, object]:
    if device.type != "cuda":
        raise RuntimeError("Phase 12 identity requires CUDA")
    return {
        "device": str(device),
        "gpu_model": torch.cuda.get_device_name(device),
        "pytorch_version": torch.__version__,
        "pytorch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "driver_version": _nvidia_query("driver_version"),
    }


def _nvidia_query(field: str) -> str | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip().splitlines()[0].strip()


def _nvidia_process_snapshot() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi"], check=True, capture_output=True, text=True, timeout=15
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        return f"nvidia-smi unavailable: {error}"


def _initial_rng_state(seed: int, device: torch.device):
    torch.manual_seed(seed + 11)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed + 11)
    return (
        torch.get_rng_state().clone(),
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if device.type == "cuda" else None,
    )


def _make_pair(
    manifest: SplitManifest,
    provider,
    treatment_template: FormalTrainingConfig,
    seed: int,
):
    seed_reproducibly(seed)
    treatment_config = replace(treatment_template, seed=seed)
    control_config = replace(treatment_config, eri_aux_coefficient=0.0)
    device = resolve_training_device(treatment_config.device)
    train_refs = fixed_development_refs(manifest, "train")
    allowed = tuple(ref.base_instance_id for ref in train_refs)
    initial = make_scrp_policy(
        treatment_config.observation_version, 5, 3,
        Mmax=treatment_config.Mmax or 6,
        embed_dim=treatment_config.embed_dim,
        num_encoder_layers=treatment_config.num_encoder_layers,
        num_heads=treatment_config.num_heads,
        ffn_dim=treatment_config.ffn_dim,
        clip_constant=treatment_config.clip_constant,
        device=device,
    )
    control = SCRPFormalTrainer(
        control_config, manifest, provider, allowed_base_ids=allowed,
        policy=copy.deepcopy(initial),
    )
    treatment = SCRPFormalTrainer(
        treatment_config, manifest, provider, allowed_base_ids=allowed,
        policy=copy.deepcopy(initial),
    )
    if next(control.policy.parameters()).device != device:
        raise AssertionError("control policy is not on the requested device")
    if next(treatment.policy.parameters()).device != device:
        raise AssertionError("treatment policy is not on the requested device")
    if policy_state_sha256(control.policy) != policy_state_sha256(treatment.policy):
        raise AssertionError("paired arms do not share the same initialization")
    return control, treatment, _initial_rng_state(seed, device)


def _clone_rng_state(state):
    return (
        state[0].clone(),
        None if state[1] is None else [value.clone() for value in state[1]],
    )


def _train_pair(
    manifest: SplitManifest,
    provider,
    template: FormalTrainingConfig,
    seed: int,
    episodes: int,
    *,
    progress_label: str | None = None,
):
    if episodes <= 0 or episodes % template.batch_size:
        raise ValueError("episode budget must be a positive multiple of batch_size")
    control, treatment, common_rng = _make_pair(manifest, provider, template, seed)
    iterations = episodes // template.batch_size
    started = time.perf_counter()
    def train_arm(trainer, arm):
        state = _clone_rng_state(common_rng)
        metrics: list[FormalIterationMetrics] = []
        remaining = iterations
        while remaining:
            count = min(250, remaining)
            chunk, state = _train_chunk(trainer, count, state)
            metrics.extend(chunk)
            remaining -= count
            if progress_label is not None:
                print(
                    f"{progress_label} {arm}: {trainer.episodes_seen}/{episodes} episodes",
                    flush=True,
                )
        return metrics
    control_metrics = train_arm(control, "control")
    treatment_metrics = train_arm(treatment, "treatment")
    elapsed = time.perf_counter() - started
    control_fp = _schedule_hash(control.sample_history)
    treatment_fp = _schedule_hash(treatment.sample_history)
    if control_fp != treatment_fp:
        raise AssertionError(f"seed {seed} control/treatment schedules differ")
    return control, treatment, control_metrics, treatment_metrics, elapsed, control_fp


def run_cuda_smoke(
    manifest: SplitManifest,
    provider,
    template: FormalTrainingConfig,
    checkpoint_dir: Path,
) -> dict[str, object]:
    device = resolve_training_device(template.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats()
    control, treatment, control_metrics, treatment_metrics, elapsed, fingerprint = _train_pair(
        manifest, provider, template, FROZEN_SEEDS[0], SMOKE_EPISODES_PER_ARM
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = treatment.save_checkpoint(checkpoint_dir / "phase12-smoke.pt")
    allowed = tuple(
        ref.base_instance_id for ref in fixed_development_refs(manifest, "train")
    )
    resumed = SCRPFormalTrainer.from_checkpoint(
        checkpoint, manifest, provider, allowed_base_ids=allowed
    )
    checkpoint_round_trip = (
        resumed.episodes_seen == treatment.episodes_seen
        and next(resumed.policy.parameters()).device == device
    )
    checkpoint.unlink()
    checkpoint_dir.rmdir()
    control_summary = optimization_summary(control_metrics)
    treatment_summary = optimization_summary(treatment_metrics)
    process_snapshot = _nvidia_process_snapshot()
    python_seen = "python" in process_snapshot.lower()
    passed = bool(all((
        control_summary["finite"], treatment_summary["finite"],
        control_summary["invalid_actions"] == 0,
        treatment_summary["invalid_actions"] == 0,
        control_summary["truncations"] == 0,
        treatment_summary["truncations"] == 0,
        treatment_summary["ERI_auxiliary_loss_mean"] > 0.0,
        treatment_summary["weighted_ERI_gradient_norm_mean"] > 0.0,
        checkpoint_round_trip,
    )))
    return {
        "phase": "Phase 12 CUDA smoke",
        "episodes_per_arm": SMOKE_EPISODES_PER_ARM,
        "model_device": str(next(treatment.policy.parameters()).device),
        "elapsed_seconds": elapsed,
        "scenario_fingerprint": fingerprint,
        "control": control_summary,
        "treatment": treatment_summary,
        "checkpoint_round_trip": checkpoint_round_trip,
        "temporary_artifacts_removed": True,
        "cuda_memory_allocated": torch.cuda.memory_allocated(),
        "cuda_memory_reserved": torch.cuda.memory_reserved(),
        "cuda_max_memory_allocated": torch.cuda.max_memory_allocated(),
        "nvidia_smi_python_process_seen": python_seen,
        "nvidia_smi_snapshot": process_snapshot,
        "passed": passed,
    }


def run_timing_probe(
    manifest: SplitManifest,
    provider,
    template: FormalTrainingConfig,
) -> dict[str, object]:
    device = resolve_training_device(template.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats()
    started_cpu = time.process_time()
    _, _, control_metrics, treatment_metrics, elapsed, fingerprint = _train_pair(
        manifest, provider, template, FROZEN_SEEDS[0], TIMING_EPISODES_PER_ARM
    )
    cpu_seconds = time.process_time() - started_cpu
    total_episodes = TIMING_EPISODES_PER_ARM * 2
    return {
        "phase": "Phase 12 CUDA timing probe",
        "episodes_per_arm": TIMING_EPISODES_PER_ARM,
        "elapsed_seconds": elapsed,
        "episodes_per_second": total_episodes / elapsed,
        "process_cpu_seconds": cpu_seconds,
        "approximate_process_cpu_utilization_percent": 100.0 * cpu_seconds / elapsed,
        "gpu_utilization_percent_snapshot": _nvidia_query("utilization.gpu"),
        "cuda_max_memory_allocated": torch.cuda.max_memory_allocated(),
        "cuda_memory_reserved": torch.cuda.memory_reserved(),
        "scenario_fingerprint": fingerprint,
        "control": optimization_summary(control_metrics),
        "treatment": optimization_summary(treatment_metrics),
    }


def _run_identity(seed: int, arm: str, config: FormalTrainingConfig, code_sha: str, fingerprint: str, started: str, ended: str, identity: Mapping[str, object]):
    return {
        "run_id": f"phase12-seed{seed}-{arm}-cuda-v1",
        "phase": 12,
        "seed": seed,
        "arm": arm,
        "device": config.device,
        "gpu_model": identity["gpu_model"],
        "pytorch_version": identity["pytorch_version"],
        "cuda_version": identity["pytorch_cuda_runtime"],
        "config_hash": _config_hash(config),
        "code_sha": code_sha,
        "scenario_fingerprint": fingerprint,
        "start_timestamp": started,
        "end_timestamp": ended,
        "training_episodes": EPISODES_PER_ARM,
        "validation_budget": 96,
    }


def _seed_result(
    seed: int,
    control_config: FormalTrainingConfig,
    treatment_config: FormalTrainingConfig,
    control_rows,
    treatment_rows,
    control_diag,
    treatment_diag,
    control_metrics,
    treatment_metrics,
    fingerprint: str,
    code_sha: str,
    started: str,
    ended: str,
    elapsed: float,
    identity: Mapping[str, object],
):
    control_validation = relocation_summary(control_rows)
    treatment_validation = relocation_summary(treatment_rows)
    overall_delta = treatment_validation["mean_relocations"] - control_validation["mean_relocations"]
    ds1_delta = treatment_validation["DS1_mean"] - control_validation["DS1_mean"]
    ds2_delta = treatment_validation["DS2_mean"] - control_validation["DS2_mean"]
    return {
        "seed": seed,
        "control_validation": control_validation,
        "treatment_validation": treatment_validation,
        "overall_delta": overall_delta,
        "DS1_delta": ds1_delta,
        "DS2_delta": ds2_delta,
        "control_ERI_diagnostic": control_diag,
        "treatment_ERI_diagnostic": treatment_diag,
        "control_optimization": optimization_summary(control_metrics),
        "treatment_optimization": optimization_summary(treatment_metrics),
        "scenario_fingerprint_equal": True,
        "scenario_fingerprint": fingerprint,
        "elapsed_seconds": elapsed,
        "run_identity": {
            "control": _run_identity(seed, "control", control_config, code_sha, fingerprint, started, ended, identity),
            "treatment": _run_identity(seed, "treatment", treatment_config, code_sha, fingerprint, started, ended, identity),
        },
    }


def hierarchical_seed_base_bootstrap(
    paired_rows: Mapping[int, tuple[Sequence[Mapping[str, object]], Sequence[Mapping[str, object]]]],
    *,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    seeds = sorted(paired_rows)
    if not seeds:
        raise ValueError("bootstrap requires at least one seed")
    matrices: dict[int, np.ndarray] = {}
    for seed in seeds:
        control_rows, treatment_rows = paired_rows[seed]
        control = {(row["base_instance_id"], row["dataset"]): row for row in control_rows}
        treatment = {(row["base_instance_id"], row["dataset"]): row for row in treatment_rows}
        if control.keys() != treatment.keys():
            raise AssertionError(f"seed {seed} paired validation coordinates differ")
        base_ids = sorted({key[0] for key in control})
        matrices[seed] = np.asarray([
            [
                float(treatment[(base_id, dataset)]["relocations"])
                - float(control[(base_id, dataset)]["relocations"])
                for dataset in ("DS1", "DS2")
            ]
            for base_id in base_ids
        ])
    observed = np.concatenate(list(matrices.values()), axis=0)
    rng = np.random.default_rng(bootstrap_seed)
    draws = np.empty((repetitions, 3), dtype=np.float64)
    for repetition in range(repetitions):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        blocks = []
        for sampled_seed in sampled_seeds:
            matrix = matrices[int(sampled_seed)]
            blocks.append(matrix[rng.integers(0, len(matrix), size=len(matrix))])
        sample = np.concatenate(blocks, axis=0)
        draws[repetition] = (sample.mean(), sample[:, 0].mean(), sample[:, 1].mean())
    def estimate(column: int, values: np.ndarray):
        return {
            "delta": float(values.mean()),
            "ci95_low": float(np.quantile(draws[:, column], 0.025)),
            "ci95_high": float(np.quantile(draws[:, column], 0.975)),
        }
    return {
        "method": "hierarchical paired bootstrap: seed -> base layout, preserving DS1/DS2 blocks",
        "repetitions": repetitions,
        "bootstrap_seed": bootstrap_seed,
        "overall": estimate(0, observed),
        "DS1": estimate(1, observed[:, 0]),
        "DS2": estimate(2, observed[:, 1]),
    }


def multiseed_success_gate(seed_results, pooled, *, cuda_integrity: bool) -> dict[str, object]:
    overall_favorable = sum(result["overall_delta"] < 0.0 for result in seed_results)
    strictly_worse_lower = sum(
        result["treatment_ERI_diagnostic"]["strictly_worse_ERI_score_action_rate"]
        < result["control_ERI_diagnostic"]["strictly_worse_ERI_score_action_rate"]
        for result in seed_results
    )
    penalty_lower = sum(
        result["treatment_ERI_diagnostic"]["mean_ERI_score_penalty"]
        < result["control_ERI_diagnostic"]["mean_ERI_score_penalty"]
        for result in seed_results
    )
    ds1_unfavorable = sum(result["DS1_delta"] > 0.0 for result in seed_results)
    ds2_unfavorable = sum(result["DS2_delta"] > 0.0 for result in seed_results)
    severe_opposition = bool(
        (ds1_unfavorable >= 4 and pooled["DS1"]["delta"] >= SEVERE_DATASET_DEGRADATION)
        or (ds2_unfavorable >= 4 and pooled["DS2"]["delta"] >= SEVERE_DATASET_DEGRADATION)
    )
    integrity = all(
        result[arm][field] == 0
        for result in seed_results
        for arm in ("control_optimization", "treatment_optimization")
        for field in ("invalid_actions", "truncations", "numerical_failures", "scenario_mismatches")
    )
    checks = {
        "at_least_4_of_5_overall_favorable": overall_favorable >= 4,
        "pooled_overall_delta_favorable": pooled["overall"]["delta"] < 0.0,
        "pooled_CI_excludes_large_positive_degradation": pooled["overall"]["ci95_high"] <= MAX_ACCEPTABLE_POOLED_CI_UPPER,
        "strictly_worse_rate_lower_at_least_4_of_5": strictly_worse_lower >= 4,
        "mean_ERI_penalty_lower_in_majority": penalty_lower >= 3,
        "no_severe_stable_dataset_opposition": not severe_opposition,
        "no_integrity_regression": integrity,
        "cuda_execution_integrity": cuda_integrity,
        "no_leakage": True,
    }
    return {
        "frozen_thresholds": {
            "required_overall_favorable_seeds": 4,
            "maximum_pooled_CI_upper_for_non_large_degradation": MAX_ACCEPTABLE_POOLED_CI_UPPER,
            "severe_dataset_degradation": SEVERE_DATASET_DEGRADATION,
        },
        "counts": {
            "overall_favorable": overall_favorable,
            "strictly_worse_rate_lower": strictly_worse_lower,
            "mean_ERI_penalty_lower": penalty_lower,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def aggregate_scalar(seed_results, section: str, field: str) -> float:
    return float(np.mean([result[section][field] for result in seed_results]))


def run_multiseed_replication(
    manifest: SplitManifest,
    provider,
    template: FormalTrainingConfig,
    *,
    code_sha: str,
    smoke_record: Mapping[str, object],
    timing_record: Mapping[str, object],
) -> dict[str, object]:
    device = resolve_training_device(template.device)
    identity = cuda_identity(device)
    if not smoke_record.get("passed"):
        raise RuntimeError("Phase 12 CUDA smoke gate did not pass")
    seed_results = []
    paired_rows = {}
    validation_refs = fixed_development_refs(manifest, "validation")
    for seed in FROZEN_SEEDS:
        started = utc_now()
        control, treatment, control_metrics, treatment_metrics, elapsed, fingerprint = _train_pair(
            manifest, provider, template, seed, EPISODES_PER_ARM,
            progress_label=f"Phase 12 seed {seed}",
        )
        control_config = replace(template, seed=seed, eri_aux_coefficient=0.0)
        treatment_config = replace(template, seed=seed)
        control_rows = validation_rows(control.policy, control_config, manifest, provider, validation_refs)
        treatment_rows = validation_rows(treatment.policy, treatment_config, manifest, provider, validation_refs)
        paired_rows[seed] = (control_rows, treatment_rows)
        control_diag = eri_diagnostic(control.policy, control_config, manifest, provider, validation_refs)
        treatment_diag = eri_diagnostic(treatment.policy, treatment_config, manifest, provider, validation_refs)
        ended = utc_now()
        seed_results.append(_seed_result(
            seed, control_config, treatment_config, control_rows, treatment_rows,
            control_diag, treatment_diag, control_metrics, treatment_metrics,
            fingerprint, code_sha, started, ended, elapsed, identity,
        ))
    pooled = hierarchical_seed_base_bootstrap(paired_rows)
    favorable = {
        "overall": sum(result["overall_delta"] < 0.0 for result in seed_results),
        "DS1": sum(result["DS1_delta"] < 0.0 for result in seed_results),
        "DS2": sum(result["DS2_delta"] < 0.0 for result in seed_results),
    }
    diagnostics = {}
    for field in (
        "exact_action_agreement_rate", "ERI_score_equivalent_action_rate",
        "strictly_worse_ERI_score_action_rate", "mean_ERI_score_penalty",
        "mean_probability_mass_on_ERI_optimal_set",
        "greedy_action_outside_ERI_optimal_set_rate",
    ):
        diagnostics[field] = {
            "control_mean_across_seeds": aggregate_scalar(seed_results, "control_ERI_diagnostic", field),
            "treatment_mean_across_seeds": aggregate_scalar(seed_results, "treatment_ERI_diagnostic", field),
        }
    optimization = {}
    for field in (
        "policy_loss_mean", "ERI_auxiliary_loss_mean", "entropy_mean",
        "pre_clip_gradient_norm_mean", "gradient_clipping_frequency",
        "weighted_ERI_to_RL_gradient_ratio_mean", "FGB_refreshes",
    ):
        optimization[field] = {
            "control_mean_across_seeds": aggregate_scalar(seed_results, "control_optimization", field),
            "treatment_mean_across_seeds": aggregate_scalar(seed_results, "treatment_optimization", field),
        }
    gate = multiseed_success_gate(seed_results, pooled, cuda_integrity=True)
    return {
        "phase12_version": PHASE12_VERSION,
        "scope": "DEVELOPMENT_ONLY_TRAIN_VALIDATION",
        "research_question": "Does lambda_eri=0.10 replicate across five paired random seeds?",
        "base_sha": PHASE12_PARENT_SHA,
        "training_code_sha": code_sha,
        "frozen_seeds": list(FROZEN_SEEDS),
        "episodes_per_arm": EPISODES_PER_ARM,
        "total_training_episodes": 2 * len(FROZEN_SEEDS) * EPISODES_PER_ARM,
        "control_definition": "lambda_eri=0",
        "treatment_definition": "lambda_eri=0.10",
        "cuda_identity": identity,
        "deterministic_settings": {
            "python_random_seeded": True, "numpy_seeded": True,
            "torch_cpu_seeded": True, "torch_cuda_seeded": True,
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        },
        "smoke_gate": {
            key: value for key, value in smoke_record.items()
            if key != "nvidia_smi_snapshot"
        },
        "timing_probe": dict(timing_record),
        "seed_results": seed_results,
        "favorable_seed_counts": favorable,
        "pooled_paired_bootstrap": pooled,
        "pooled_ERI_diagnostics": diagnostics,
        "pooled_optimization": optimization,
        "formal_test_episode_usage": 0,
        "phase8_raw_rows_accessed": False,
        "formal_test_rerun": False,
        "success_gate": gate,
        "ERI_AUX_MULTISEED_REPLICATION_SUCCESS": "YES" if gate["passed"] else "NO",
        "host": {"platform": platform.platform(), "python": platform.python_version()},
    }
