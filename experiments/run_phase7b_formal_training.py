"""Execute the frozen 25,000-episode Phase 7B formal training run."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from experiments.formal_run import (
    CANDIDATE_CONFIG_SHA256,
    CachedKuProvider,
    atomic_write_json,
    checkpoint_inventory,
    compact_window,
    copy_checkpoint_verified,
    evaluate_validation,
    file_sha256,
    load_run_identity,
    policy_state_sha256,
    save_checkpoint_atomic,
    validate_frozen_identity,
)
from experiments.protocol import load_split_manifest
from scrp.formal_training import SCRPFormalTrainer, load_formal_training_config


def _assert_finite(metric) -> None:
    values = (
        metric.loss, metric.policy_loss, metric.entropy, metric.grad_norm,
        metric.mean_policy_relocations, metric.mean_baseline_relocations,
        metric.mean_advantage,
    )
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("non-finite formal training metric")
    if metric.invalid_actions or metric.truncations or metric.scenario_mismatches:
        raise RuntimeError("formal training rollout safety failure")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--checkpoint-root", default="checkpoints")
    parser.add_argument("--summary-root", default="experiments/summaries")
    args = parser.parse_args()

    identity_path = Path(args.identity)
    identity = load_run_identity(identity_path)
    manifest = load_split_manifest("experiments/splits/scrp_split_v1.json")
    validate_frozen_identity(identity, manifest)
    config = load_formal_training_config(identity.config_path)
    provider = CachedKuProvider(args.source_root)
    trainer = SCRPFormalTrainer(config, manifest, provider)

    run_checkpoint_dir = Path(args.checkpoint_root) / identity.run_id
    summary_root = Path(args.summary_root)
    training_summary_path = summary_root / f"{identity.run_id}-training.json"
    validation_history_path = summary_root / f"{identity.run_id}-validation.json"
    completion_path = summary_root / f"{identity.run_id}-completion.json"
    config_snapshot_path = summary_root / f"{identity.run_id}-config-snapshot.json"
    atomic_write_json(config_snapshot_path, asdict(config))

    training_windows = []
    validation_history = []
    best = None
    all_samples = []
    all_metrics = []
    window_samples = []
    window_metrics = []
    train_seconds = 0.0
    validation_seconds = 0.0
    resume_audit = None
    started = time.perf_counter()
    internal_checkpoint_iterations = config.checkpoint_interval
    milestones = set(identity.durable_milestone_episodes)
    test_ids = {ref.base_instance_id for ref in manifest.refs("test")}
    train_ids = {ref.base_instance_id for ref in manifest.refs("train")}
    validation_ids = {ref.base_instance_id for ref in manifest.refs("validation")}
    if not train_ids.isdisjoint(test_ids) or not validation_ids.isdisjoint(test_ids):
        raise RuntimeError("split-overlap audit failed before formal training")

    try:
        while trainer.episodes_seen < identity.planned_episodes:
            if trainer.episodes_seen + config.batch_size > identity.planned_episodes:
                raise RuntimeError("batch size would exceed frozen episode budget")
            before_sample_count = len(trainer.sample_history)
            tick = time.perf_counter()
            metrics = trainer.train_iterations(1)
            train_seconds += time.perf_counter() - tick
            new_samples = trainer.sample_history[before_sample_count:]
            metric = metrics[0]
            _assert_finite(metric)
            if any(sample.base_instance_id in test_ids for sample in new_samples):
                raise RuntimeError("formal training sampled the test split")
            all_samples.extend(new_samples)
            all_metrics.extend(metrics)
            window_samples.extend(new_samples)
            window_metrics.extend(metrics)
            episode = trainer.episodes_seen

            if trainer.iteration % internal_checkpoint_iterations == 0:
                save_checkpoint_atomic(trainer, run_checkpoint_dir / "latest.pt")

            if episode % 100 == 0:
                training_windows.append(
                    compact_window(window_metrics, window_samples, episode)
                )
                window_metrics.clear()
                window_samples.clear()
                atomic_write_json(training_summary_path, {
                    "run_id": identity.run_id,
                    "completed_episodes": episode,
                    "planned_episodes": identity.planned_episodes,
                    "windows": training_windows,
                    "formal_test_episode_usage": 0,
                })

            if episode in milestones:
                latest = save_checkpoint_atomic(
                    trainer, run_checkpoint_dir / "latest.pt"
                )
                if episode != identity.planned_episodes:
                    copy_checkpoint_verified(
                        latest, run_checkpoint_dir / f"milestone-{episode}.pt"
                    )

            # Mandatory active stop/save/load/resume at the first durable milestone.
            if episode == 1_000 and resume_audit is None:
                latest = save_checkpoint_atomic(
                    trainer, run_checkpoint_dir / "latest.pt"
                )
                before_hash = policy_state_sha256(trainer.policy)
                before_baseline_hash = policy_state_sha256(trainer.baseline_policy)
                before_visits = dict(trainer.sampler.visit_counts)
                restored = SCRPFormalTrainer.from_checkpoint(latest, manifest, provider)
                resume_audit = {
                    "episode": episode,
                    "policy_state": policy_state_sha256(restored.policy) == before_hash,
                    "baseline_state": (
                        policy_state_sha256(restored.baseline_policy)
                        == before_baseline_hash
                    ),
                    "per_base_visits": restored.sampler.visit_counts == before_visits,
                    "iteration": restored.iteration == trainer.iteration,
                    "episodes_seen": restored.episodes_seen == trainer.episodes_seen,
                    "optimizer_state_nonempty": bool(restored.optimizer.state_dict()["state"]),
                }
                resume_audit["passed"] = all(resume_audit.values())
                if not resume_audit["passed"]:
                    raise RuntimeError("mandatory formal resume audit failed")
                trainer = restored

            if episode % identity.validation_cadence_episodes == 0:
                latest = save_checkpoint_atomic(
                    trainer, run_checkpoint_dir / "latest.pt"
                )
                validation = evaluate_validation(
                    trainer, manifest, provider, identity, episode
                )
                validation_seconds += float(validation["wall_seconds"])
                validation_history.append(validation)
                atomic_write_json(validation_history_path, {
                    "run_id": identity.run_id,
                    "selection_metric": identity.checkpoint_selection_metric,
                    "validation_cadence_episodes": identity.validation_cadence_episodes,
                    "history": validation_history,
                    "formal_test_episode_usage": 0,
                })
                if best is None or validation["selection_score"] < best["validation_score"]:
                    copy_checkpoint_verified(
                        latest, run_checkpoint_dir / "best-validation.pt"
                    )
                    best = {
                        "checkpoint_episode": episode,
                        "validation_score": validation["selection_score"],
                        "baseline_state_version": validation["baseline_state_version"],
                        "model_state_sha256": validation["model_state_sha256"],
                        "checkpoint_sha256": file_sha256(
                            run_checkpoint_dir / "best-validation.pt"
                        ),
                    }

        final_checkpoint = save_checkpoint_atomic(
            trainer, run_checkpoint_dir / "final.pt"
        )
        latest_checkpoint = save_checkpoint_atomic(
            trainer, run_checkpoint_dir / "latest.pt"
        )
        # milestone-25000 is redundant with final/latest and is not retained.
        redundant = run_checkpoint_dir / "milestone-25000.pt"
        if redundant.exists():
            redundant.unlink()

        wall_seconds = time.perf_counter() - started
        bucket_counts = {
            str(stacks): sum(sample.num_stacks == stacks for sample in all_samples)
            for stacks in range(5, 11)
        }
        variant_counts = {
            variant: sum(sample.variant == variant for sample in all_samples)
            for variant in ("DS1", "DS2")
        }
        scenario_seeds = {sample.scenario_seed for sample in all_samples}
        if (
            len(all_samples) != identity.planned_episodes
            or len(scenario_seeds) != identity.planned_episodes
            or not all(bucket_counts.values())
            or not all(variant_counts.values())
        ):
            raise RuntimeError("formal training coverage gate failed")
        if len(validation_history) != (
            identity.planned_episodes // identity.validation_cadence_episodes
        ):
            raise RuntimeError("formal validation cadence gate failed")
        if resume_audit is None or not resume_audit["passed"] or best is None:
            raise RuntimeError("formal completion artifact gate failed")

        completion = {
            "run_id": identity.run_id,
            "code_sha": identity.code_sha,
            "config_sha256": CANDIDATE_CONFIG_SHA256,
            "split_manifest_version": identity.split_manifest_version,
            "dataset_version": identity.dataset_version,
            "observation_version": identity.observation_version,
            "Mmax": identity.Mmax,
            "root_seed": identity.root_seed,
            "planned_episodes": identity.planned_episodes,
            "completed_episodes": trainer.episodes_seen,
            "wall_seconds": wall_seconds,
            "training_wall_seconds": train_seconds,
            "validation_wall_seconds": validation_seconds,
            "training_episodes_per_second": trainer.episodes_seen / train_seconds,
            "S_bucket_episode_counts": bucket_counts,
            "variant_episode_counts": variant_counts,
            "unique_train_base_layouts": len({
                sample.base_instance_id for sample in all_samples
            }),
            "unique_training_scenario_seeds": len(scenario_seeds),
            "numerical_stability": {
                "finite": all(math.isfinite(value) for metric in all_metrics for value in (
                    metric.loss, metric.policy_loss, metric.entropy, metric.grad_norm
                )),
                "loss_range": [
                    min(metric.loss for metric in all_metrics),
                    max(metric.loss for metric in all_metrics),
                ],
                "policy_loss_range": [
                    min(metric.policy_loss for metric in all_metrics),
                    max(metric.policy_loss for metric in all_metrics),
                ],
                "entropy_range": [
                    min(metric.entropy for metric in all_metrics),
                    max(metric.entropy for metric in all_metrics),
                ],
                "grad_norm_range": [
                    min(metric.grad_norm for metric in all_metrics),
                    max(metric.grad_norm for metric in all_metrics),
                ],
                "invalid_actions": sum(metric.invalid_actions for metric in all_metrics),
                "truncations": sum(metric.truncations for metric in all_metrics),
                "scenario_mismatches": sum(
                    metric.scenario_mismatches for metric in all_metrics
                ),
                "empty_decision_episodes": sum(
                    metric.empty_decision_episodes for metric in all_metrics
                ),
            },
            "baseline_refresh_count": trainer.baseline_updates,
            "baseline_refresh_history": [
                asdict(record) for record in trainer.baseline_refresh_history
            ],
            "resume_audit": resume_audit,
            "validation_cadence_episodes": identity.validation_cadence_episodes,
            "validation_scenarios_per_static_variant": (
                identity.validation_scenarios_per_static_variant
            ),
            "checkpoint_selection_metric": identity.checkpoint_selection_metric,
            "best_validation_checkpoint": best,
            "final_checkpoint": {
                "path": str(final_checkpoint).replace("\\", "/"),
                "sha256": file_sha256(final_checkpoint),
                "model_state_sha256": policy_state_sha256(trainer.policy),
            },
            "latest_checkpoint": {
                "path": str(latest_checkpoint).replace("\\", "/"),
                "sha256": file_sha256(latest_checkpoint),
            },
            "checkpoint_inventory": checkpoint_inventory(run_checkpoint_dir),
            "formal_test_episode_usage": 0,
            "ERI_training_or_selection_usage": 0,
            "cleanup": {
                "temporary_checkpoint_files": len(list(run_checkpoint_dir.glob("*.tmp"))),
                "retained_checkpoint_count": len(list(run_checkpoint_dir.glob("*.pt"))),
            },
            "performance_claim": False,
            "FORMAL_TRAINING_COMPLETE": "YES",
        }
        atomic_write_json(completion_path, completion)
        print(json.dumps({
            "status": "complete",
            "run_id": identity.run_id,
            "completed_episodes": trainer.episodes_seen,
            "best_validation_checkpoint": best,
            "final_checkpoint": completion["final_checkpoint"],
            "formal_test_episode_usage": 0,
            "FORMAL_TRAINING_COMPLETE": "YES",
        }, indent=2))
    except Exception as error:
        run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        debug_path = run_checkpoint_dir / "failure-debug-latest.pt"
        try:
            save_checkpoint_atomic(trainer, debug_path)
        except Exception:
            pass
        atomic_write_json(completion_path, {
            "run_id": identity.run_id,
            "status": "FAILED_STOPPED",
            "completed_episodes": trainer.episodes_seen,
            "error_type": type(error).__name__,
            "error": str(error),
            "formal_test_episode_usage": 0,
            "automatic_restart": False,
            "FORMAL_TRAINING_COMPLETE": "NO",
        })
        raise


if __name__ == "__main__":
    main()
