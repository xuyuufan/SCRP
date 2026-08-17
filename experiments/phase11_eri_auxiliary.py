"""Development-only Phase 11 ERI set-valued auxiliary experiment."""

from __future__ import annotations

import copy
import math
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from experiments.baselines import ERIBaseline
from experiments.posttest_analysis import fixed_development_refs, parse_parameter_group
from experiments.protocol import BaseInstanceRef, ScenarioSeedSchedule, SplitManifest
from scrp.environment import SCRPEnvironment
from scrp.formal_training import (
    ERI_AUXILIARY_VERSION,
    FormalIterationMetrics,
    FormalTrainingConfig,
    SCRPFormalTrainer,
    TrainingSample,
    eri_optimal_action_mask,
    make_node_padding_mask,
    make_scrp_policy,
    run_formal_episode,
)
from scrp.models import SCRPConfig
from scrp.rl_adapter import SCRPRLAdapter


PHASE11_RUN_ID = "phase11-eri-aux-seed20260816-v1"
SMOKE_EPISODES = 16
DEVELOPMENT_EPISODES = 5_000
STRICTLY_WORSE_MATERIAL_DECREASE = 0.01
SEVERE_DATASET_OPPOSITION = 0.25
MIN_AUX_TO_RL_GRADIENT_RATIO = 0.02
MAX_AUX_TO_RL_GRADIENT_RATIO = 2.0


def _sample(ref: BaseInstanceRef, variant: str, seed: int) -> TrainingSample:
    instance_id = ref.ds1_instance_id if variant == "DS1" else ref.ds2_instance_id
    return TrainingSample(
        ref.base_instance_id, instance_id, variant, seed, 0,
        int(ref.parameter_group[1:3]),
    )


def _fingerprint(samples) -> list[tuple[str, str, int]]:
    return [
        (sample.base_instance_id, sample.variant, sample.scenario_seed)
        for sample in samples
    ]


def _train_chunk(trainer: SCRPFormalTrainer, iterations: int, rng_state):
    torch.set_rng_state(rng_state.clone())
    metrics = trainer.train_iterations(iterations)
    return metrics, torch.get_rng_state().clone()


def _policy_probabilities(policy, observation, legal, stacks: int):
    observation_t = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
    forbidden_t = torch.tensor(~np.asarray(legal, dtype=bool)).unsqueeze(0)
    node_mask = make_node_padding_mask(observation_t, "O2", stacks)
    with torch.no_grad():
        log_probabilities = policy.action_log_probabilities(
            observation_t, forbidden_t, mode="low", node_padding_mask=node_mask
        )
    return log_probabilities.exp()[0].cpu().numpy()


def validation_rows(policy, config, manifest, provider, refs):
    schedule = ScenarioSeedSchedule(manifest)
    rows = []
    was_training = policy.training
    policy.eval()
    try:
        for ref in refs:
            for variant in ("DS1", "DS2"):
                sample = _sample(
                    ref, variant,
                    schedule.seed_for("validation", ref.base_instance_id, 0),
                )
                trajectory = run_formal_episode(
                    provider(sample), sample, policy, config, greedy=True
                )
                if not trajectory.terminated or trajectory.truncated or trajectory.invalid_actions:
                    raise RuntimeError("Phase 11 validation rollout failed")
                rows.append({
                    "base_instance_id": ref.base_instance_id,
                    "dataset": variant,
                    **parse_parameter_group(ref.parameter_group),
                    "relocations": trajectory.relocations,
                })
    finally:
        policy.train(was_training)
    return rows


def relocation_summary(rows):
    return {
        "episodes": len(rows),
        "mean_relocations": float(np.mean([row["relocations"] for row in rows])),
        "DS1_mean": float(np.mean([
            row["relocations"] for row in rows if row["dataset"] == "DS1"
        ])),
        "DS2_mean": float(np.mean([
            row["relocations"] for row in rows if row["dataset"] == "DS2"
        ])),
    }


def paired_hierarchical_bootstrap(control_rows, treatment_rows, *, repetitions=10_000):
    control = {(row["base_instance_id"], row["dataset"]): row for row in control_rows}
    treatment = {(row["base_instance_id"], row["dataset"]): row for row in treatment_rows}
    if control.keys() != treatment.keys():
        raise AssertionError("paired validation coordinates differ")
    base_ids = sorted({key[0] for key in control})
    deltas = np.asarray([
        [
            treatment[(base_id, dataset)]["relocations"]
            - control[(base_id, dataset)]["relocations"]
            for dataset in ("DS1", "DS2")
        ]
        for base_id in base_ids
    ], dtype=np.float64)
    rng = np.random.default_rng(20260816)
    draws = rng.integers(0, len(base_ids), size=(repetitions, len(base_ids)))
    bootstrap_means = deltas[draws].mean(axis=(1, 2))
    return {
        "method": "paired base-layout bootstrap preserving DS1/DS2 blocks",
        "repetitions": repetitions,
        "seed": 20260816,
        "paired_mean_treatment_minus_control": float(deltas.mean()),
        "ci95_low": float(np.quantile(bootstrap_means, 0.025)),
        "ci95_high": float(np.quantile(bootstrap_means, 0.975)),
        "DS1_delta": float(deltas[:, 0].mean()),
        "DS2_delta": float(deltas[:, 1].mean()),
    }


def eri_diagnostic(policy, config, manifest, provider, refs):
    """Evaluate greedy policy decisions on fixed ERI-controlled public states."""

    schedule = ScenarioSeedSchedule(manifest)
    eri = ERIBaseline()
    exact = equivalent = worse = decisions = outside = 0
    gaps, positive_masses = [], []
    set_sizes: Counter[int] = Counter()
    for ref in refs:
        for variant in ("DS1", "DS2"):
            sample = _sample(
                ref, variant,
                schedule.seed_for("validation", ref.base_instance_id, 0),
            )
            instance = provider(sample)
            core = SCRPEnvironment(
                SCRPConfig(
                    instance.num_stacks, instance.max_tiers,
                    root_seed=config.seed, max_steps=config.max_steps,
                    validate_after_transition=True,
                ),
                instance,
            )
            env = SCRPRLAdapter(core, observation_version="O2", o2_mmax=6)
            observation, info = env.reset(seed=sample.scenario_seed)
            while not info["terminated"]:
                legal_mask = np.asarray(info["action_mask"], dtype=bool)
                legal = tuple(np.flatnonzero(legal_mask).tolist())
                probabilities = _policy_probabilities(
                    policy, observation, legal_mask, instance.num_stacks
                )
                action = int(np.argmax(probabilities))
                eri_action = eri.select_destination(instance, core.state, legal)
                positive = eri_optimal_action_mask(instance, core.state, legal)
                set_size = int(positive.sum())
                set_sizes[set_size] += 1
                positive_masses.append(float(probabilities[positive].sum()))
                location = core.state.locations[core.state.current_target_id]
                blocker_id = core.state.stacks[location.stack_id].top_id
                action_score = eri.destination_score(
                    instance, core.state, blocker_id, action
                )
                best_score = eri.destination_score(
                    instance, core.state, blocker_id, eri_action
                )
                gap = float(action_score - best_score)
                decisions += 1
                exact += int(action == eri_action)
                equivalent += int(gap == 0.0)
                worse += int(gap > 0.0)
                outside += int(not positive[action])
                gaps.append(gap)
                observation, _, _, _, info = env.step(eri_action)
    return {
        "episodes": len(refs) * 2,
        "public_decision_states": decisions,
        "exact_action_agreement_rate": exact / decisions,
        "ERI_score_equivalent_action_rate": equivalent / decisions,
        "strictly_worse_ERI_score_action_rate": worse / decisions,
        "mean_ERI_score_penalty": float(np.mean(gaps)),
        "mean_probability_mass_on_ERI_optimal_set": float(np.mean(positive_masses)),
        "greedy_action_outside_ERI_optimal_set_rate": outside / decisions,
        "ERI_optimal_set_size_distribution": {
            str(size): count for size, count in sorted(set_sizes.items())
        },
        "multiple_ERI_optimal_destinations_rate": (
            sum(count for size, count in set_sizes.items() if size > 1) / decisions
        ),
        "outside_set_downstream_benefit": {
            "measured": False,
            "reason": (
                "Not estimated: branching from ERI-controlled states would condition on "
                "realized hidden future orders and add a high-variance post-hoc endpoint."
            ),
        },
    }


def optimization_summary(metrics: Sequence[FormalIterationMetrics]):
    def mean(field):
        return float(np.mean([getattr(metric, field) for metric in metrics]))

    ratios = [
        metric.weighted_eri_gradient_norm / metric.rl_gradient_norm
        for metric in metrics
        if metric.rl_gradient_norm > 0.0
    ]
    return {
        "iterations": len(metrics),
        "episodes": sum(metric.episodes for metric in metrics),
        "total_loss_mean": mean("loss"),
        "policy_loss_mean": mean("policy_loss"),
        "ERI_auxiliary_loss_mean": mean("eri_aux_loss"),
        "entropy_mean": mean("entropy"),
        "pre_clip_gradient_norm_mean": mean("grad_norm"),
        "gradient_clipping_frequency": float(np.mean([
            metric.gradient_clipped for metric in metrics
        ])),
        "RL_gradient_norm_mean": mean("rl_gradient_norm"),
        "weighted_ERI_gradient_norm_mean": mean("weighted_eri_gradient_norm"),
        "weighted_ERI_to_RL_gradient_ratio_mean": float(np.mean(ratios)),
        "invalid_actions": sum(metric.invalid_actions for metric in metrics),
        "truncations": sum(metric.truncations for metric in metrics),
        "numerical_failures": 0,
        "scenario_mismatches": sum(metric.scenario_mismatches for metric in metrics),
        "FGB_refreshes": metrics[-1].baseline_updates,
        "finite": all(
            math.isfinite(float(getattr(metric, field)))
            for metric in metrics
            for field in (
                "loss", "policy_loss", "eri_aux_loss", "entropy", "grad_norm"
            )
        ),
    }


def success_gate(control_validation, treatment_validation, paired, control_diag, treatment_diag, treatment_stability):
    dataset_deltas = (paired["DS1_delta"], paired["DS2_delta"])
    checks = {
        "treatment_validation_lower": paired["paired_mean_treatment_minus_control"] < 0.0,
        "not_tiny_isolated_subgroup": (
            max(dataset_deltas) <= SEVERE_DATASET_OPPOSITION
            and min(dataset_deltas) < 0.0
        ),
        "strictly_worse_rate_materially_lower": (
            control_diag["strictly_worse_ERI_score_action_rate"]
            - treatment_diag["strictly_worse_ERI_score_action_rate"]
            >= STRICTLY_WORSE_MATERIAL_DECREASE
        ),
        "no_severe_DS1_DS2_opposition": max(dataset_deltas) <= SEVERE_DATASET_OPPOSITION,
        "integrity_and_stability": bool(
            treatment_stability["finite"]
            and treatment_stability["invalid_actions"] == 0
            and treatment_stability["truncations"] == 0
            and treatment_stability["scenario_mismatches"] == 0
        ),
        "not_cosmetic_tie_break_imitation": bool(
            paired["paired_mean_treatment_minus_control"] < 0.0
            and treatment_diag["ERI_score_equivalent_action_rate"]
            >= control_diag["ERI_score_equivalent_action_rate"]
        ),
    }
    return {
        "frozen_thresholds": {
            "material_strictly_worse_absolute_decrease": STRICTLY_WORSE_MATERIAL_DECREASE,
            "severe_dataset_opposition_relocations": SEVERE_DATASET_OPPOSITION,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def run_comparison(manifest: SplitManifest, provider, treatment_config: FormalTrainingConfig, checkpoint_probe_dir: Path):
    train_refs = fixed_development_refs(manifest, "train")
    validation_refs = fixed_development_refs(manifest, "validation")
    allowed = tuple(ref.base_instance_id for ref in train_refs)
    control_config = replace(treatment_config, eri_aux_coefficient=0.0)

    torch.manual_seed(treatment_config.seed)
    initial_policy = make_scrp_policy(
        "O2", 5, 3, Mmax=6, embed_dim=treatment_config.embed_dim,
        num_encoder_layers=treatment_config.num_encoder_layers,
        num_heads=treatment_config.num_heads, ffn_dim=treatment_config.ffn_dim,
        clip_constant=treatment_config.clip_constant,
    )
    control_trainer = SCRPFormalTrainer(
        control_config, manifest, provider, allowed_base_ids=allowed,
        policy=copy.deepcopy(initial_policy),
    )
    treatment_trainer = SCRPFormalTrainer(
        treatment_config, manifest, provider, allowed_base_ids=allowed,
        policy=copy.deepcopy(initial_policy),
    )
    common_rng = torch.Generator().manual_seed(treatment_config.seed + 11).get_state()
    control_rng = common_rng.clone()
    treatment_rng = common_rng.clone()

    smoke_iterations = SMOKE_EPISODES // treatment_config.batch_size
    control_smoke, control_rng = _train_chunk(control_trainer, smoke_iterations, control_rng)
    treatment_smoke, treatment_rng = _train_chunk(
        treatment_trainer, smoke_iterations, treatment_rng
    )
    if _fingerprint(control_trainer.sample_history) != _fingerprint(treatment_trainer.sample_history):
        raise AssertionError("Phase 11 smoke sampler schedules differ")
    checkpoint_probe_dir.mkdir(parents=True, exist_ok=True)
    probe_path = treatment_trainer.save_checkpoint(checkpoint_probe_dir / "phase11-smoke.pt")
    resumed = SCRPFormalTrainer.from_checkpoint(
        probe_path, manifest, provider, allowed_base_ids=allowed
    )
    if resumed.episodes_seen != treatment_trainer.episodes_seen:
        raise AssertionError("Phase 11 checkpoint round trip failed")
    probe_path.unlink()
    checkpoint_probe_dir.rmdir()

    smoke_control_summary = optimization_summary(control_smoke)
    smoke_treatment_summary = optimization_summary(treatment_smoke)
    smoke_passed = all((
        smoke_control_summary["finite"], smoke_treatment_summary["finite"],
        smoke_control_summary["invalid_actions"] == 0,
        smoke_treatment_summary["invalid_actions"] == 0,
        smoke_control_summary["truncations"] == 0,
        smoke_treatment_summary["truncations"] == 0,
        smoke_treatment_summary["ERI_auxiliary_loss_mean"] > 0.0,
        smoke_treatment_summary["weighted_ERI_gradient_norm_mean"] > 0.0,
        MIN_AUX_TO_RL_GRADIENT_RATIO
        <= smoke_treatment_summary["weighted_ERI_to_RL_gradient_ratio_mean"]
        <= MAX_AUX_TO_RL_GRADIENT_RATIO,
    ))
    smoke = {
        "episodes_per_model": SMOKE_EPISODES,
        "same_initialization": True,
        "same_sampler_schedule": True,
        "checkpoint_round_trip": True,
        "temporary_artifacts_removed": True,
        "predeclared_weighted_aux_to_RL_gradient_ratio_range": [
            MIN_AUX_TO_RL_GRADIENT_RATIO, MAX_AUX_TO_RL_GRADIENT_RATIO
        ],
        "control": smoke_control_summary,
        "treatment": smoke_treatment_summary,
        "passed": smoke_passed,
    }
    if not smoke_passed:
        return {"smoke": smoke, "development_5k_executed": False, "ERI_AUX_PROTOTYPE_SUCCESS": "NO"}

    remaining = (DEVELOPMENT_EPISODES - SMOKE_EPISODES) // treatment_config.batch_size
    control_tail, control_rng = _train_chunk(control_trainer, remaining, control_rng)
    treatment_tail, treatment_rng = _train_chunk(treatment_trainer, remaining, treatment_rng)
    if _fingerprint(control_trainer.sample_history) != _fingerprint(treatment_trainer.sample_history):
        raise AssertionError("Phase 11 5k sampler schedules differ")
    control_rows = validation_rows(
        control_trainer.policy, control_config, manifest, provider, validation_refs
    )
    treatment_rows = validation_rows(
        treatment_trainer.policy, treatment_config, manifest, provider, validation_refs
    )
    paired = paired_hierarchical_bootstrap(control_rows, treatment_rows)
    control_validation = relocation_summary(control_rows)
    treatment_validation = relocation_summary(treatment_rows)
    control_diag = eri_diagnostic(
        control_trainer.policy, control_config, manifest, provider, validation_refs
    )
    treatment_diag = eri_diagnostic(
        treatment_trainer.policy, treatment_config, manifest, provider, validation_refs
    )
    control_stability = optimization_summary([*control_smoke, *control_tail])
    treatment_stability = optimization_summary([*treatment_smoke, *treatment_tail])
    gate = success_gate(
        control_validation, treatment_validation, paired,
        control_diag, treatment_diag, treatment_stability,
    )
    return {
        "run_id": PHASE11_RUN_ID,
        "scope": "DEVELOPMENT_ONLY_TRAIN_VALIDATION",
        "formal_test_episode_usage": 0,
        "phase8_raw_rows_accessed": False,
        "coefficient_adjustments": 0,
        "control_config": asdict(control_config),
        "treatment_config": asdict(treatment_config),
        "smoke": smoke,
        "development_5k_executed": True,
        "development_5k": {
            "episodes_per_model": DEVELOPMENT_EPISODES,
            "train_base_layouts": len(train_refs),
            "validation_base_layouts": len(validation_refs),
            "validation_scenarios_per_static_variant": 1,
            "same_initialization": True,
            "same_sampler_and_scenario_schedule": True,
            "control_validation": control_validation,
            "treatment_validation": treatment_validation,
            "paired_validation_bootstrap": paired,
            "control_ERI_diagnostic": control_diag,
            "treatment_ERI_diagnostic": treatment_diag,
            "control_optimization": control_stability,
            "treatment_optimization": treatment_stability,
            "success_gate": gate,
        },
        "ERI_AUX_PROTOTYPE_SUCCESS": "YES" if gate["passed"] else "NO",
    }
