"""Train/validation-only Phase 10 order-aware architecture prototype."""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict
from typing import Mapping, Sequence

import numpy as np
import torch

from experiments.baselines import ERIBaseline
from experiments.posttest_analysis import (
    _policy_outputs,
    _representation_probe,
    _training_sample,
    fixed_development_refs,
    parse_parameter_group,
)
from experiments.protocol import BaseInstanceRef, ScenarioSeedSchedule, SplitManifest
from hier_pg.network import O2_ORDER_XATTN_V1, O2_SHARED_ENCODER_V1
from scrp.environment import SCRPEnvironment
from scrp.formal_training import (
    FormalTrainingConfig,
    SCRPFormalTrainer,
    make_node_padding_mask,
    make_scrp_policy,
    run_formal_episode,
)
from scrp.models import Container, SCRPConfig, SCRPInstance
from scrp.rl_adapter import SCRPRLAdapter
from scrp.scenario import Scenario


PHASE10_RUN_ID = "phase10-order-aware-prototype-seed20260816-v1"
SMOKE_EPISODES = 1_000
DEVELOPMENT_EPISODES = 5_000
PERMUTATION_ACTION_RATE_MIN = 0.01
ORDER_ABLATION_ACTION_RATE_MIN = 0.02
SENSITIVITY_MULTIPLIER_MIN = 2.0
STRICTLY_WORSE_ERI_RATE_TOLERANCE = 0.01
VALIDATION_RELOCATION_TOLERANCE = 0.25


def parameter_count(policy) -> int:
    return sum(parameter.numel() for parameter in policy.parameters())


def architecture_record(control, treatment) -> dict[str, object]:
    control_count = parameter_count(control)
    treatment_count = parameter_count(treatment)
    return {
        "control_version": control.scrp_architecture_version,
        "treatment_version": treatment.scrp_architecture_version,
        "control_flow": (
            "stack/order/context nodes -> shared Transformer encoder -> "
            "global-only LOW query -> pointer over first S stack nodes"
        ),
        "treatment_flow": (
            "stack+context encoder; order encoder; explicit masked cross-attention "
            "from S stack queries to Mmax revealed-order keys/values; unchanged LOW "
            "pointer over first S stack nodes"
        ),
        "control_parameters": control_count,
        "treatment_parameters": treatment_count,
        "parameter_increase": treatment_count - control_count,
        "parameter_increase_percent": 100.0 * (treatment_count - control_count) / control_count,
        "embed_dim": 32,
        "encoder_layers": 1,
        "attention_heads": 4,
        "ffn_dim": 64,
        "Mmax": 6,
        "explicit_rank_feature_retained": True,
        "raw_container_id_used": False,
    }


class _FixedScenarioSampler:
    def __init__(self, current_order, future_order):
        self.current_order = tuple(current_order)
        self.future_order = tuple(future_order)

    def sample(self, instance, root_seed):
        return Scenario(
            root_seed=root_seed,
            order_seeds={1: root_seed + 1, 2: root_seed + 2},
            hidden_orders={1: self.current_order, 2: self.future_order},
            scenario_id=f"phase10-{self.current_order}-{self.future_order}",
        )


def _probe_instance() -> SCRPInstance:
    return SCRPInstance(
        "phase10-order-probe",
        4,
        4,
        tuple(
            [Container(index, 1) for index in (1, 2, 3)]
            + [Container(index, 2) for index in (4, 5, 6)]
        ),
        ((1, 4), (2, 5), (3, 6), ()),
        (1, 2),
    )


def _probe_observation(current_order, future_order):
    instance = _probe_instance()
    config = SCRPConfig(4, 4)
    core = SCRPEnvironment(
        config, instance, _FixedScenarioSampler(current_order, future_order)
    )
    adapter = SCRPRLAdapter(core, observation_version="O2", o2_mmax=6)
    observation, info = adapter.reset(seed=20260816)
    return instance, observation, np.asarray(info["action_mask"], dtype=bool)


def controlled_untrained_probe(policy) -> dict[str, object]:
    """Prove expressivity, hidden-order invariance, and padding safety."""

    instance, original, legal = _probe_observation((1, 2, 3), (4, 5, 6))
    _, hidden_changed, hidden_legal = _probe_observation((1, 2, 3), (6, 5, 4))
    _, revealed_changed, revealed_legal = _probe_observation((1, 3, 2), (4, 5, 6))
    if not np.array_equal(legal, hidden_legal) or not np.array_equal(legal, revealed_legal):
        raise AssertionError("controlled probe legal sets differ")
    original_t = torch.tensor(original).unsqueeze(0)
    hidden_t = torch.tensor(hidden_changed).unsqueeze(0)
    revealed_t = torch.tensor(revealed_changed).unsqueeze(0)
    mask_original = make_node_padding_mask(original_t, "O2", instance.num_stacks)
    mask_hidden = make_node_padding_mask(hidden_t, "O2", instance.num_stacks)
    mask_revealed = make_node_padding_mask(revealed_t, "O2", instance.num_stacks)
    p_original, e_original = _policy_outputs(
        policy, original, legal, instance.num_stacks, mask_original
    )
    p_hidden, e_hidden = _policy_outputs(
        policy, hidden_changed, legal, instance.num_stacks, mask_hidden
    )
    p_revealed, e_revealed = _policy_outputs(
        policy, revealed_changed, legal, instance.num_stacks, mask_revealed
    )
    nodes = original.reshape(-1, 12).copy()
    padding = mask_original[0].cpu().numpy()
    nodes[padding, 1:11] = 0.731
    p_padding, _ = _policy_outputs(
        policy, nodes.reshape(-1), legal, instance.num_stacks, mask_original
    )
    revealed_logit_effect = float(np.max(np.abs(p_original - p_revealed)))
    revealed_embedding_effect = float(
        np.linalg.norm(e_original - e_revealed) / math.sqrt(e_original.size)
    )
    hidden_logit_effect = float(np.max(np.abs(p_original - p_hidden)))
    hidden_embedding_effect = float(
        np.linalg.norm(e_original - e_hidden) / math.sqrt(e_original.size)
    )
    padding_effect = float(np.max(np.abs(p_original - p_padding)))
    return {
        "revealed_order_logit_probability_max_abs": revealed_logit_effect,
        "revealed_order_stack_embedding_RMS": revealed_embedding_effect,
        "hidden_future_order_logit_probability_max_abs": hidden_logit_effect,
        "hidden_future_order_stack_embedding_RMS": hidden_embedding_effect,
        "padding_perturbation_max_abs": padding_effect,
        "revealed_order_nonzero": revealed_logit_effect > 1e-8,
        "hidden_future_order_invariant": hidden_logit_effect == 0.0,
        "padding_invariant": padding_effect == 0.0,
        "passed": bool(
            revealed_logit_effect > 1e-8
            and hidden_logit_effect == 0.0
            and padding_effect == 0.0
        ),
    }


def _policy(
    config: FormalTrainingConfig,
    architecture_version: str,
):
    torch.manual_seed(config.seed)
    return make_scrp_policy(
        "O2", 5, 3,
        Mmax=6,
        embed_dim=config.embed_dim,
        num_encoder_layers=config.num_encoder_layers,
        num_heads=config.num_heads,
        ffn_dim=config.ffn_dim,
        clip_constant=config.clip_constant,
        architecture_version=architecture_version,
        device="cpu",
    )


def _fingerprint(samples) -> list[tuple[str, str, int]]:
    return [
        (sample.base_instance_id, sample.variant, sample.scenario_seed)
        for sample in samples
    ]


def _stable_metrics(metrics) -> dict[str, object]:
    fields = ("loss", "policy_loss", "entropy", "grad_norm", "mean_advantage")
    finite = all(
        math.isfinite(float(getattr(metric, field)))
        for metric in metrics
        for field in fields
    )
    return {
        "iterations": len(metrics),
        "finite": finite,
        "invalid_actions": sum(metric.invalid_actions for metric in metrics),
        "truncations": sum(metric.truncations for metric in metrics),
        "scenario_mismatches": sum(metric.scenario_mismatches for metric in metrics),
        "entropy_mean": float(np.mean([metric.entropy for metric in metrics])),
        "entropy_min": min(metric.entropy for metric in metrics),
        "entropy_max": max(metric.entropy for metric in metrics),
        "grad_norm_before_clip_mean": float(np.mean([
            metric.grad_norm for metric in metrics
        ])),
        "grad_norm_before_clip_max": max(metric.grad_norm for metric in metrics),
        "mean_policy_relocations": float(np.mean([
            metric.mean_policy_relocations for metric in metrics
        ])),
        "baseline_refreshes_at_end": metrics[-1].baseline_updates,
        "passed": bool(
            finite
            and not any(metric.invalid_actions for metric in metrics)
            and not any(metric.truncations for metric in metrics)
            and not any(metric.scenario_mismatches for metric in metrics)
        ),
    }


def _train_chunk(trainer, iterations: int, rng_state):
    torch.set_rng_state(rng_state.clone())
    metrics = trainer.train_iterations(iterations)
    return metrics, torch.get_rng_state().clone()


def _validation_relocations(
    policy,
    config: FormalTrainingConfig,
    manifest: SplitManifest,
    provider,
    refs: Sequence[BaseInstanceRef],
):
    schedule = ScenarioSeedSchedule(manifest)
    rows = []
    was_training = policy.training
    policy.eval()
    try:
        for ref in refs:
            group = parse_parameter_group(ref.parameter_group)
            for dataset in ("DS1", "DS2"):
                sample = _training_sample(
                    ref, dataset,
                    schedule.seed_for("validation", ref.base_instance_id, 0), 0,
                )
                trajectory = run_formal_episode(
                    provider(sample), sample, policy, config, greedy=True, device="cpu"
                )
                if not trajectory.terminated or trajectory.truncated or trajectory.invalid_actions:
                    raise RuntimeError("Phase 10 validation rollout failed")
                rows.append({
                    "base_instance_id": ref.base_instance_id,
                    "dataset": dataset,
                    **group,
                    "relocations": trajectory.relocations,
                })
    finally:
        policy.train(was_training)
    return rows


def _relocation_summary(rows) -> dict[str, object]:
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


def _eri_and_sensitivity(
    policy,
    config: FormalTrainingConfig,
    manifest: SplitManifest,
    provider,
    refs: Sequence[BaseInstanceRef],
) -> dict[str, object]:
    schedule = ScenarioSeedSchedule(manifest)
    eri = ERIBaseline()
    decisions = 0
    exact = score_equal = score_worse = 0
    score_gaps = []
    probes = []
    for ref in refs:
        for dataset in ("DS1", "DS2"):
            sample = _training_sample(
                ref, dataset,
                schedule.seed_for("validation", ref.base_instance_id, 0), 0,
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
            adapter = SCRPRLAdapter(core, observation_version="O2", o2_mmax=6)
            observation, info = adapter.reset(seed=sample.scenario_seed)
            eri.reset(config.seed)
            while not info["terminated"]:
                legal_mask = np.asarray(info["action_mask"], dtype=bool)
                legal = tuple(np.flatnonzero(legal_mask).tolist())
                node_mask = make_node_padding_mask(
                    torch.tensor(observation).unsqueeze(0), "O2", instance.num_stacks
                )
                probabilities, _ = _policy_outputs(
                    policy, observation, legal_mask, instance.num_stacks, node_mask
                )
                rl_action = int(np.argmax(probabilities))
                eri_action = eri.select_destination(instance, core.state, legal)
                state = core.state
                location = state.locations[state.current_target_id]
                blocker_id = state.stacks[location.stack_id].top_id
                gap = float(
                    eri.destination_score(instance, state, blocker_id, rl_action)
                    - eri.destination_score(instance, state, blocker_id, eri_action)
                )
                decisions += 1
                exact += int(rl_action == eri_action)
                score_equal += int(abs(gap) < 1e-12)
                score_worse += int(gap > 1e-12)
                score_gaps.append(gap)
                probe = _representation_probe(
                    policy, observation, legal_mask, instance.num_stacks, node_mask
                )
                if "permutation_tv" in probe:
                    probes.append(probe)
                observation, _, _, _, info = adapter.step(eri_action)
    return {
        "episodes": len(refs) * 2,
        "public_decision_states": decisions,
        "exact_action_agreement_rate": exact / decisions,
        "ERI_score_equivalent_action_rate": score_equal / decisions,
        "strictly_worse_ERI_score_action_rate": score_worse / decisions,
        "mean_ERI_score_gap": float(np.mean(score_gaps)),
        "eligible_sensitivity_states": len(probes),
        "permutation_TV_mean": float(np.mean([
            probe["permutation_tv"] for probe in probes
        ])),
        "permutation_action_change_rate": float(np.mean([
            probe["permutation_action_changed"] for probe in probes
        ])),
        "order_ablation_TV_mean": float(np.mean([
            probe["order_ablation_tv"] for probe in probes
        ])),
        "order_ablation_action_change_rate": float(np.mean([
            probe["order_ablation_action_changed"] for probe in probes
        ])),
        "max_padding_perturbation_effect": max(
            probe["padding_perturbation_max_abs"] for probe in probes
        ),
    }


def _paired_bootstrap(deltas, *, seed=20260816, repetitions=10_000):
    values = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(repetitions, len(values)), replace=True).mean(axis=1)
    return {
        "label": "DEVELOPMENT_ONLY_EXPLORATORY",
        "paired_mean_treatment_minus_control": float(values.mean()),
        "repetitions": repetitions,
        "seed": seed,
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
    }


def success_gate(control, treatment, control_validation, treatment_validation):
    permutation_control = control["permutation_action_change_rate"]
    permutation_treatment = treatment["permutation_action_change_rate"]
    ablation_control = control["order_ablation_action_change_rate"]
    ablation_treatment = treatment["order_ablation_action_change_rate"]
    checks = {
        "permutation_absolute": permutation_treatment >= PERMUTATION_ACTION_RATE_MIN,
        "permutation_relative": permutation_treatment >= (
            SENSITIVITY_MULTIPLIER_MIN * max(permutation_control, 1e-12)
        ),
        "ablation_absolute": ablation_treatment >= ORDER_ABLATION_ACTION_RATE_MIN,
        "ablation_relative": ablation_treatment >= (
            SENSITIVITY_MULTIPLIER_MIN * max(ablation_control, 1e-12)
        ),
        "ERI_error_not_increased": treatment["strictly_worse_ERI_score_action_rate"] <= (
            control["strictly_worse_ERI_score_action_rate"]
            + STRICTLY_WORSE_ERI_RATE_TOLERANCE
        ),
        "validation_not_worse": treatment_validation["mean_relocations"] <= (
            control_validation["mean_relocations"] + VALIDATION_RELOCATION_TOLERANCE
        ),
        "padding_invariant": treatment["max_padding_perturbation_effect"] == 0.0,
    }
    return {
        "frozen_thresholds": {
            "permutation_action_rate_min": PERMUTATION_ACTION_RATE_MIN,
            "order_ablation_action_rate_min": ORDER_ABLATION_ACTION_RATE_MIN,
            "relative_sensitivity_multiplier_min": SENSITIVITY_MULTIPLIER_MIN,
            "strictly_worse_ERI_rate_tolerance": STRICTLY_WORSE_ERI_RATE_TOLERANCE,
            "validation_relocation_tolerance": VALIDATION_RELOCATION_TOLERANCE,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def run_development_comparison(
    manifest: SplitManifest,
    provider,
    config: FormalTrainingConfig,
) -> dict[str, object]:
    train_refs = fixed_development_refs(manifest, "train")
    validation_refs = fixed_development_refs(manifest, "validation")
    allowed = tuple(ref.base_instance_id for ref in train_refs)
    control_policy = _policy(config, O2_SHARED_ENCODER_V1)
    treatment_policy = _policy(config, O2_ORDER_XATTN_V1)
    architecture = architecture_record(control_policy, treatment_policy)
    untrained_probe = controlled_untrained_probe(treatment_policy)
    if not untrained_probe["passed"]:
        raise RuntimeError("untrained order-aware sensitivity gate failed")

    control_trainer = SCRPFormalTrainer(
        config, manifest, provider, allowed_base_ids=allowed, policy=control_policy
    )
    treatment_trainer = SCRPFormalTrainer(
        config, manifest, provider, allowed_base_ids=allowed, policy=treatment_policy
    )
    common_rng = torch.Generator().manual_seed(config.seed + 10).get_state()
    control_rng = common_rng.clone()
    treatment_rng = common_rng.clone()

    smoke_iterations = SMOKE_EPISODES // config.batch_size
    control_smoke_metrics, control_rng = _train_chunk(
        control_trainer, smoke_iterations, control_rng
    )
    treatment_smoke_metrics, treatment_rng = _train_chunk(
        treatment_trainer, smoke_iterations, treatment_rng
    )
    if _fingerprint(control_trainer.sample_history) != _fingerprint(
        treatment_trainer.sample_history
    ):
        raise AssertionError("control/treatment sampler schedules differ at 1k")
    smoke_control_rows = _validation_relocations(
        control_trainer.policy, config, manifest, provider, validation_refs
    )
    smoke_treatment_rows = _validation_relocations(
        treatment_trainer.policy, config, manifest, provider, validation_refs
    )
    smoke = {
        "episodes_per_model": SMOKE_EPISODES,
        "same_sampler_schedule": True,
        "control_training": _stable_metrics(control_smoke_metrics),
        "treatment_training": _stable_metrics(treatment_smoke_metrics),
        "control_validation": _relocation_summary(smoke_control_rows),
        "treatment_validation": _relocation_summary(smoke_treatment_rows),
    }
    smoke["passed"] = bool(
        smoke["control_training"]["passed"]
        and smoke["treatment_training"]["passed"]
    )
    if not smoke["passed"]:
        return {
            "architecture": architecture,
            "untrained_controlled_probe": untrained_probe,
            "smoke_1k": smoke,
            "development_5k_executed": False,
            "ORDER_AWARE_PROTOTYPE_SUCCESS": "NO",
        }

    remaining_iterations = (DEVELOPMENT_EPISODES - SMOKE_EPISODES) // config.batch_size
    control_tail, control_rng = _train_chunk(
        control_trainer, remaining_iterations, control_rng
    )
    treatment_tail, treatment_rng = _train_chunk(
        treatment_trainer, remaining_iterations, treatment_rng
    )
    if _fingerprint(control_trainer.sample_history) != _fingerprint(
        treatment_trainer.sample_history
    ):
        raise AssertionError("control/treatment sampler schedules differ at 5k")

    control_rows = _validation_relocations(
        control_trainer.policy, config, manifest, provider, validation_refs
    )
    treatment_rows = _validation_relocations(
        treatment_trainer.policy, config, manifest, provider, validation_refs
    )
    control_validation = _relocation_summary(control_rows)
    treatment_validation = _relocation_summary(treatment_rows)
    paired = [
        treatment["relocations"] - control["relocations"]
        for control, treatment in zip(control_rows, treatment_rows)
    ]
    control_diagnostic = _eri_and_sensitivity(
        control_trainer.policy, config, manifest, provider, validation_refs
    )
    treatment_diagnostic = _eri_and_sensitivity(
        treatment_trainer.policy, config, manifest, provider, validation_refs
    )
    gate = success_gate(
        control_diagnostic, treatment_diagnostic,
        control_validation, treatment_validation,
    )
    return {
        "architecture": architecture,
        "untrained_controlled_probe": untrained_probe,
        "smoke_1k": smoke,
        "development_5k_executed": True,
        "development_5k": {
            "episodes_per_model": DEVELOPMENT_EPISODES,
            "same_sampler_schedule": True,
            "train_base_layouts": len(train_refs),
            "validation_base_layouts": len(validation_refs),
            "validation_scenarios_per_static_variant": 1,
            "control_training": _stable_metrics(
                [*control_smoke_metrics, *control_tail]
            ),
            "treatment_training": _stable_metrics(
                [*treatment_smoke_metrics, *treatment_tail]
            ),
            "control_FGB_refresh_history": [
                asdict(row) for row in control_trainer.baseline_refresh_history
            ],
            "treatment_FGB_refresh_history": [
                asdict(row) for row in treatment_trainer.baseline_refresh_history
            ],
            "control_validation": control_validation,
            "treatment_validation": treatment_validation,
            "paired_validation_bootstrap": _paired_bootstrap(paired),
            "control_action_and_order_diagnostic": control_diagnostic,
            "treatment_action_and_order_diagnostic": treatment_diagnostic,
            "success_gate": gate,
        },
        "ORDER_AWARE_PROTOTYPE_SUCCESS": "YES" if gate["passed"] else "NO",
    }
