"""Run the bounded, train-split-only Phase 6 readiness sanity check."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from experiments.protocol import load_split_manifest
from scrp.formal_training import (
    FormalTrainingConfig,
    KuTrainingInstanceProvider,
    SCRPFormalTrainer,
    load_formal_training_config,
)


def _select_train_bases(manifest) -> tuple[str, ...]:
    selected = []
    for stacks in (5, 7, 10):
        matches = [
            ref.base_instance_id for ref in manifest.refs("train")
            if ref.parameter_group.startswith(f"S{stacks:02d}_")
        ]
        selected.extend(matches[:4])
    if len(selected) != 12:
        raise RuntimeError("could not select 12 train bases covering S=5,7,10")
    return tuple(selected)


def _finite_metrics(metrics) -> bool:
    return all(
        np.isfinite(value)
        for metric in metrics
        for value in (
            metric.mean_policy_relocations,
            metric.mean_baseline_relocations,
            metric.mean_return,
            metric.mean_advantage,
            metric.loss,
            metric.policy_loss,
            metric.entropy,
            metric.grad_norm,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument(
        "--output",
        default="experiments/summaries/phase6_training_readiness_summary.json",
    )
    args = parser.parse_args()

    manifest = load_split_manifest("experiments/splits/scrp_split_v1.json")
    base_config = load_formal_training_config(
        "experiments/configs/training_protocol_v1.json"
    )
    provider = KuTrainingInstanceProvider(args.source_root)
    selected = _select_train_bases(manifest)

    o2_trainer = SCRPFormalTrainer(
        base_config, manifest, provider, allowed_base_ids=selected
    )
    before = [parameter.detach().clone() for parameter in o2_trainer.policy.parameters()]
    o2_metrics = o2_trainer.train_iterations(25)
    o2_changed = any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, o2_trainer.policy.parameters())
    )
    with tempfile.TemporaryDirectory() as directory:
        checkpoint_path = o2_trainer.save_checkpoint(Path(directory) / "phase6.pt")
        restored = SCRPFormalTrainer.from_checkpoint(
            checkpoint_path, manifest, provider, allowed_base_ids=selected
        )
        checkpoint_round_trip = (
            restored.iteration == o2_trainer.iteration
            and restored.episodes_seen == o2_trainer.episodes_seen
            and restored.sampler.visit_counts == o2_trainer.sampler.visit_counts
            and all(
                torch.equal(expected, actual)
                for expected, actual in zip(
                    o2_trainer.policy.parameters(), restored.policy.parameters()
                )
            )
        )

    o1_record = {**base_config.__dict__, "observation_version": "O1", "Mmax": None}
    o1_config = FormalTrainingConfig(**o1_record)
    o1_trainer = SCRPFormalTrainer(
        o1_config, manifest, provider, allowed_base_ids=selected
    )
    o1_metrics = o1_trainer.train_iterations(1)

    variants = {sample.variant for sample in o2_trainer.sample_history}
    stack_buckets = {sample.num_stacks for sample in o2_trainer.sample_history}
    all_metrics = o2_metrics + o1_metrics
    summary = {
        "status": "training_readiness_sanity_only",
        "formal_training_started": False,
        "performance_claim": False,
        "split_used": "train",
        "validation_used": False,
        "test_used": False,
        "base_layouts": len(selected),
        "stack_buckets": sorted(stack_buckets),
        "variants": sorted(variants),
        "O2_episodes": o2_trainer.episodes_seen,
        "O1_episodes": o1_trainer.episodes_seen,
        "total_episodes": o2_trainer.episodes_seen + o1_trainer.episodes_seen,
        "metrics_finite": _finite_metrics(all_metrics),
        "gradients_finite": all(np.isfinite(item.grad_norm) for item in all_metrics),
        "policy_parameters_changed": o2_changed,
        "invalid_actions": sum(item.invalid_actions for item in all_metrics),
        "truncations": sum(item.truncations for item in all_metrics),
        "baseline_updates": o2_trainer.baseline_updates,
        "baseline_parameters_have_gradients": any(
            parameter.grad is not None for parameter in o2_trainer.baseline_policy.parameters()
        ),
        "paired_scenario_ids": "enforced_by_assertion",
        "paired_rollouts": o2_trainer.episodes_seen + o1_trainer.episodes_seen,
        "checkpoint_round_trip": checkpoint_round_trip,
        "O2_loss_range": [
            min(item.loss for item in o2_metrics),
            max(item.loss for item in o2_metrics),
        ],
        "O2_entropy_range": [
            min(item.entropy for item in o2_metrics),
            max(item.entropy for item in o2_metrics),
        ],
        "O2_grad_norm_range": [
            min(item.grad_norm for item in o2_metrics),
            max(item.grad_norm for item in o2_metrics),
        ],
        "O2_last_iteration": {
            "mean_policy_relocations": o2_metrics[-1].mean_policy_relocations,
            "mean_baseline_relocations": o2_metrics[-1].mean_baseline_relocations,
            "mean_return": o2_metrics[-1].mean_return,
            "mean_advantage": o2_metrics[-1].mean_advantage,
            "loss": o2_metrics[-1].loss,
            "policy_loss": o2_metrics[-1].policy_loss,
            "entropy": o2_metrics[-1].entropy,
            "grad_norm": o2_metrics[-1].grad_norm,
        },
        "hyperparameter_status": base_config.hyperparameter_status,
    }
    if variants != {"DS1", "DS2"} or stack_buckets != {5, 7, 10}:
        raise AssertionError("sanity coverage is incomplete")
    if not summary["metrics_finite"] or not summary["policy_parameters_changed"]:
        raise AssertionError("sanity numerical gate failed")
    if not checkpoint_round_trip or summary["baseline_parameters_have_gradients"]:
        raise AssertionError("checkpoint or frozen-baseline sanity gate failed")
    if summary["invalid_actions"] or summary["truncations"]:
        raise AssertionError("sanity rollout safety gate failed")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
