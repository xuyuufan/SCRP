"""Run the bounded Phase 7A train rehearsal and validation pipeline smoke."""

from __future__ import annotations

import argparse
import copy
import ctypes
import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from experiments.protocol import ScenarioSeedSchedule, load_split_manifest
from scrp.formal_training import (
    KuTrainingInstanceProvider,
    SCRPFormalTrainer,
    TrainingSample,
    load_formal_training_config,
    policy_state_sha256,
    run_formal_episode,
)


CANDIDATE_CONFIG = Path("experiments/configs/training_protocol_v1_candidate.json")
MANIFEST_PATH = Path("experiments/splits/scrp_split_v1.json")


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()


def _peak_ram_bytes() -> int | None:
    if platform.system() != "Windows":
        try:
            import resource

            value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return int(value * (1024 if platform.system() != "Darwin" else 1))
        except (ImportError, AttributeError):
            return None

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ProcessMemoryCounters),
        ctypes.c_ulong,
    ]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    process = kernel32.GetCurrentProcess()
    success = psapi.GetProcessMemoryInfo(
        process, ctypes.byref(counters), counters.cb
    )
    return int(counters.PeakWorkingSetSize) if success else None


def _finite_metric(metric) -> bool:
    return all(math.isfinite(value) for value in (
        metric.mean_policy_relocations,
        metric.mean_baseline_relocations,
        metric.mean_return,
        metric.mean_advantage,
        metric.loss,
        metric.policy_loss,
        metric.entropy,
        metric.grad_norm,
    ))


def _coverage(samples) -> dict[str, object]:
    base_counts: dict[str, int] = {}
    bucket_counts = {str(stacks): 0 for stacks in range(5, 11)}
    variant_counts = {"DS1": 0, "DS2": 0}
    seeds = set()
    for sample in samples:
        base_counts[sample.base_instance_id] = base_counts.get(sample.base_instance_id, 0) + 1
        bucket_counts[str(sample.num_stacks)] += 1
        variant_counts[sample.variant] += 1
        seeds.add(sample.scenario_seed)
    repeated_bases = sum(count > 1 for count in base_counts.values())
    repeat_visits = sum(max(count - 1, 0) for count in base_counts.values())
    return {
        "unique_base_layouts": len(base_counts),
        "S_bucket_episode_counts": bucket_counts,
        "variant_episode_counts": variant_counts,
        "unique_scenario_seeds": len(seeds),
        "repeated_base_layouts": repeated_bases,
        "repeated_base_visits": repeat_visits,
    }


def _coverage_stable(samples, manifest) -> bool:
    coverage = _coverage(samples)
    buckets = coverage["S_bucket_episode_counts"]
    variants = coverage["variant_episode_counts"]
    if not all(buckets.values()) or not all(variants.values()):
        return False
    expected = {}
    train_refs = manifest.refs("train")
    for stacks in range(5, 11):
        expected[str(stacks)] = sum(
            ref.parameter_group.startswith(f"S{stacks:02d}_") for ref in train_refs
        ) / len(train_refs)
    total = len(samples)
    return all(
        abs(buckets[key] / total - expected[key]) <= 0.08
        for key in buckets
    ) and abs(variants["DS1"] / total - 0.5) <= 0.08


def _stage_gate(metrics, samples, manifest) -> bool:
    return (
        all(_finite_metric(metric) for metric in metrics)
        and sum(metric.invalid_actions for metric in metrics) == 0
        and sum(metric.truncations for metric in metrics) == 0
        and sum(metric.scenario_mismatches for metric in metrics) == 0
        and max(metric.grad_norm for metric in metrics) < 1000.0
        and _coverage_stable(samples, manifest)
    )


def _optimizer_equivalent(first, second) -> bool:
    def equal(left, right):
        if isinstance(left, torch.Tensor):
            return isinstance(right, torch.Tensor) and torch.equal(left, right)
        if isinstance(left, dict):
            return isinstance(right, dict) and left.keys() == right.keys() and all(
                equal(left[key], right[key]) for key in left
            )
        if isinstance(left, (list, tuple)):
            return type(left) is type(right) and len(left) == len(right) and all(
                equal(a, b) for a, b in zip(left, right)
            )
        return left == right

    return equal(first.state_dict(), second.state_dict())


def _validation_smoke(manifest, provider, trainer) -> dict[str, object]:
    schedule = ScenarioSeedSchedule(manifest)
    selected = []
    for stacks in range(5, 11):
        matches = [
            ref for ref in manifest.refs("validation")
            if ref.parameter_group.startswith(f"S{stacks:02d}_")
        ]
        selected.extend(matches[:2])
    if len(selected) != 12:
        raise AssertionError("validation smoke could not cover two bases per S bucket")
    completed = invalid = truncated = 0
    used_bases = []
    for ref in selected:
        used_bases.append(ref.base_instance_id)
        for scenario_index, variant in enumerate(("DS1", "DS2")):
            sample = TrainingSample(
                base_instance_id=ref.base_instance_id,
                instance_id=ref.ds1_instance_id if variant == "DS1" else ref.ds2_instance_id,
                variant=variant,
                scenario_seed=schedule.seed_for(
                    "validation", ref.base_instance_id, scenario_index
                ),
                visit_index=scenario_index,
                num_stacks=int(ref.parameter_group[1:3]),
            )
            trajectory = run_formal_episode(
                provider(sample), sample, trainer.policy, trainer.config,
                greedy=True, device=trainer.device,
            )
            completed += int(trajectory.terminated)
            invalid += trajectory.invalid_actions
            truncated += int(trajectory.truncated)
    return {
        "status": "PIPELINE SMOKE ONLY",
        "base_layouts": len(selected),
        "scenarios_per_base": 2,
        "episodes": len(selected) * 2,
        "completed": completed,
        "invalid_actions": invalid,
        "truncations": truncated,
        "used_base_ids": used_bases,
        "used_for_selection_or_tuning": False,
        "passed": completed == 24 and invalid == 0 and truncated == 0,
    }


def _metric_range(metrics, name: str) -> list[float]:
    values = [float(getattr(metric, name)) for metric in metrics]
    return [min(values), max(values)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument(
        "--output",
        default="experiments/summaries/phase7a_training_rehearsal.json",
    )
    args = parser.parse_args()

    config = load_formal_training_config(CANDIDATE_CONFIG)
    if config.hyperparameter_status != "CANDIDATE_FOR_REHEARSAL":
        raise AssertionError("rehearsal requires the frozen candidate config")
    if config.observation_version != "O2" or config.Mmax != 6:
        raise AssertionError("Phase 7A candidate must use O2/Mmax=6")
    manifest = load_split_manifest(MANIFEST_PATH)
    provider = KuTrainingInstanceProvider(args.source_root)
    trainer = SCRPFormalTrainer(config, manifest, provider)
    initial_head = _git_head()
    config_hash = hashlib.sha256(CANDIDATE_CONFIG.read_bytes()).hexdigest()

    if torch.cuda.is_available() and config.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    stage1_start = time.perf_counter()
    stage1_metrics = []
    for _ in range(500 // config.batch_size):
        stage1_metrics.extend(trainer.train_iterations(1))
    stage1_seconds = time.perf_counter() - stage1_start
    stage1_samples = list(trainer.sample_history)
    stage1_pass = _stage_gate(stage1_metrics, stage1_samples, manifest)
    if not stage1_pass:
        raise RuntimeError("500-episode rehearsal gate failed; refusing expansion")

    cleanup_paths: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="scrp-phase7a-") as temporary:
        temporary_path = Path(temporary)
        checkpoint = trainer.save_checkpoint(temporary_path / "rehearsal-500.pt")
        cleanup_paths.append(checkpoint)
        restored = SCRPFormalTrainer.from_checkpoint(checkpoint, manifest, provider)
        expected_sampler = copy.deepcopy(trainer.sampler)
        actual_sampler = copy.deepcopy(restored.sampler)
        resume_sequence_match = (
            expected_sampler.sample_bucket(config.batch_size)
            == actual_sampler.sample_bucket(config.batch_size)
        )
        checkpoint_resume = {
            "iteration": restored.iteration == trainer.iteration,
            "episodes_seen": restored.episodes_seen == trainer.episodes_seen,
            "sampler_next_batch": resume_sequence_match,
            "per_base_visits": restored.sampler.visit_counts == trainer.sampler.visit_counts,
            "baseline_state": policy_state_sha256(restored.baseline_policy)
            == policy_state_sha256(trainer.baseline_policy),
            "baseline_history": restored.baseline_refresh_history
            == trainer.baseline_refresh_history,
            "optimizer_state": _optimizer_equivalent(restored.optimizer, trainer.optimizer),
        }
        checkpoint_resume["passed"] = all(checkpoint_resume.values())
        if not checkpoint_resume["passed"]:
            raise RuntimeError("500-episode checkpoint resume gate failed")

        stage2_start = time.perf_counter()
        stage2_metrics = []
        for _ in range(500 // config.batch_size):
            stage2_metrics.extend(restored.train_iterations(1))
        stage2_seconds = time.perf_counter() - stage2_start
        all_samples = stage1_samples + restored.sample_history
        all_metrics = stage1_metrics + stage2_metrics
        stage2_pass = _stage_gate(all_metrics, all_samples, manifest)
        if not stage2_pass:
            raise RuntimeError("1000-episode rehearsal gate failed")

        final_checkpoint = restored.save_checkpoint(
            temporary_path / "rehearsal-1000.pt"
        )
        cleanup_paths.append(final_checkpoint)
        evaluation_trainer = SCRPFormalTrainer.from_checkpoint(
            final_checkpoint, manifest, provider
        )
        validation = _validation_smoke(manifest, provider, evaluation_trainer)
        if not validation["passed"]:
            raise RuntimeError("validation pipeline smoke failed")

        train_ids = {sample.base_instance_id for sample in all_samples}
        validation_ids = set(validation["used_base_ids"])
        test_ids = {ref.base_instance_id for ref in manifest.refs("test")}
        test_usage_zero = train_ids.isdisjoint(test_ids) and validation_ids.isdisjoint(test_ids)
        if not test_usage_zero:
            raise AssertionError("test split was touched")

        total_seconds = stage1_seconds + stage2_seconds
        total_steps = sum(metric.low_decisions for metric in all_metrics)
        throughput = 1000 / total_seconds
        steps_per_second = total_steps / total_seconds
        estimates = {
            f"{budget}_episodes_seconds": budget / throughput
            for budget in (10_000, 25_000, 50_000, 100_000)
        }
        peak_ram = _peak_ram_bytes()
        peak_vram = (
            int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available() and config.device.startswith("cuda")
            else None
        )
        refresh_history = [
            asdict(record) for record in restored.baseline_refresh_history
        ]
        refresh_rate = len(refresh_history) / restored.iteration
        numerical = {
            "finite_metrics": all(_finite_metric(metric) for metric in all_metrics),
            "loss_range": _metric_range(all_metrics, "loss"),
            "policy_loss_range": _metric_range(all_metrics, "policy_loss"),
            "entropy_range": _metric_range(all_metrics, "entropy"),
            "grad_norm_range": _metric_range(all_metrics, "grad_norm"),
            "mean_policy_relocations_range": _metric_range(
                all_metrics, "mean_policy_relocations"
            ),
            "mean_baseline_relocations_range": _metric_range(
                all_metrics, "mean_baseline_relocations"
            ),
            "mean_advantage_range": _metric_range(all_metrics, "mean_advantage"),
            "invalid_actions": sum(metric.invalid_actions for metric in all_metrics),
            "truncations": sum(metric.truncations for metric in all_metrics),
            "scenario_mismatches": sum(
                metric.scenario_mismatches for metric in all_metrics
            ),
            "empty_decision_episodes": sum(
                metric.empty_decision_episodes for metric in all_metrics
            ),
            "zero_legal_action_states": 0,
            "exploding_gradient_threshold": 1000.0,
            "exploding_gradients": sum(
                metric.grad_norm >= 1000.0 for metric in all_metrics
            ),
        }
        rehearsal_pass = (
            stage1_pass
            and stage2_pass
            and checkpoint_resume["passed"]
            and validation["passed"]
            and test_usage_zero
            and refresh_rate <= 0.20
        )
        summary = {
            "status": "FORMAL TRAINING REHEARSAL ONLY",
            "performance_claim": False,
            "formal_long_training_started": False,
            "candidate_config": str(CANDIDATE_CONFIG).replace("\\", "/"),
            "candidate_config_sha256": config_hash,
            "candidate_hyperparameters": asdict(config),
            "rehearsal_code_base_sha": initial_head,
            "split_manifest_version": manifest.protocol_version,
            "observation_version": config.observation_version,
            "root_seed": config.seed,
            "episodes": 1000,
            "stage_500_pass": stage1_pass,
            "stage_1000_pass": stage2_pass,
            "coverage": _coverage(all_samples),
            "coverage_severe_skew": not _coverage_stable(all_samples, manifest),
            "numerical_stability": numerical,
            "baseline_refresh_history": refresh_history,
            "baseline_refresh_rate_per_iteration": refresh_rate,
            "checkpoint_resume": checkpoint_resume,
            "validation_pipeline_smoke": validation,
            "test_split_usage": 0,
            "test_usage_assertion": test_usage_zero,
            "compute": {
                "device": str(restored.device),
                "stage_500_wall_seconds": stage1_seconds,
                "stage_second_500_wall_seconds": stage2_seconds,
                "total_1000_wall_seconds": total_seconds,
                "episodes_per_second": throughput,
                "low_decision_steps": total_steps,
                "steps_per_second": steps_per_second,
                "peak_ram_bytes": peak_ram,
                "peak_vram_bytes": peak_vram,
                "extrapolated_from_measured_1000_episode_rate": estimates,
            },
            "temporary_artifacts": [path.name for path in cleanup_paths],
            "temporary_artifacts_cleanup": "pending context exit",
            "REHEARSAL_PASS": "YES" if rehearsal_pass else "NO",
            "APPROVE_FORMAL_TRAINING": "YES" if rehearsal_pass else "NO",
            "recommended_first_formal_episode_budget": 25_000,
            "validation_interval": config.validation_interval,
            "checkpoint_interval": config.checkpoint_interval,
        }

    summary["temporary_artifacts_cleanup"] = "complete"
    summary["temporary_artifacts_remaining"] = [
        path.name for path in cleanup_paths if os.path.exists(path)
    ]
    if summary["temporary_artifacts_remaining"]:
        raise AssertionError("temporary rehearsal artifacts were not cleaned")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
