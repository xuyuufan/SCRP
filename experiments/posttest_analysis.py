"""Development-only Phase 9 diagnostics using train and validation splits."""

from __future__ import annotations

import math
import re
import statistics
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from experiments.baselines import ERIBaseline
from experiments.protocol import BaseInstanceRef, ScenarioSeedSchedule, SplitManifest
from scrp.environment import SCRPEnvironment
from scrp.formal_training import (
    FormalTrainingConfig,
    SCRPFormalTrainer,
    TrainingSample,
    frozen_greedy_advantages,
    make_node_padding_mask,
    make_scrp_policy,
    run_formal_episode,
)
from scrp.models import SCRPConfig, SCRPInstance
from scrp.rl_adapter import SCRPRLAdapter


PHASE9_RUN_ID = "phase9-development-diagnostics-seed20260816-v1"
DEVELOPMENT_SPLITS = ("train", "validation")
_GROUP_PATTERN = re.compile(
    r"^S(?P<S>\d+)_T(?P<T>\d+)_mu(?P<fill>0\.50|0\.67)$"
)


def parse_parameter_group(group: str) -> dict[str, int | float]:
    match = _GROUP_PATTERN.fullmatch(group)
    if not match:
        raise ValueError(f"invalid parameter group {group!r}")
    return {
        "S": int(match.group("S")),
        "T": int(match.group("T")),
        "fill": float(match.group("fill")),
    }


def fixed_development_refs(
    manifest: SplitManifest,
    split: str,
    *,
    per_group: int = 1,
) -> tuple[BaseInstanceRef, ...]:
    """Select a deterministic small subset and reject formal-test access."""

    if split not in DEVELOPMENT_SPLITS:
        raise ValueError("Phase 9 diagnostics permit only train/validation splits")
    if per_group <= 0:
        raise ValueError("per_group must be positive")
    selected = []
    for group in sorted(manifest.groups):
        refs = sorted(
            manifest.groups[group][split], key=lambda ref: ref.base_instance_id
        )
        if len(refs) < per_group:
            raise ValueError(f"{group}/{split} has fewer than {per_group} refs")
        selected.extend(refs[:per_group])
    return tuple(selected)


def _mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _sample_variance(values: Sequence[float]) -> float:
    return float(statistics.variance(values)) if len(values) > 1 else 0.0


def audit_training_history(
    training: Mapping[str, object],
    validation: Mapping[str, object],
    completion: Mapping[str, object],
) -> dict[str, object]:
    """Summarize the frozen Phase 7B train/validation artifacts only."""

    if any(
        int(record.get("formal_test_episode_usage", 0)) != 0
        for record in (training, validation, completion)
    ):
        raise ValueError("training audit refuses artifacts with formal-test usage")
    windows = list(training["windows"])
    history = list(validation["history"])
    expected_episodes = list(range(2_500, 25_001, 2_500))
    episodes = [int(row["training_episode"]) for row in history]
    if episodes != expected_episodes:
        raise ValueError("validation checkpoint schedule drift")
    scores = [float(row["selection_score"]) for row in history]
    best_index = int(np.argmin(scores))
    best_episode = episodes[best_index]
    if best_episode != int(completion["best_validation_checkpoint"]["checkpoint_episode"]):
        raise ValueError("validation history disagrees with selected checkpoint")

    episode_per_iteration = int(windows[-1]["episode"]) / int(windows[-1]["iteration"])
    refreshes = [dict(row) for row in completion["baseline_refresh_history"]]
    refresh_episodes = [
        int(round(int(row["iteration"]) * episode_per_iteration))
        for row in refreshes
    ]
    refresh_gaps = [
        current - previous
        for previous, current in zip([0, *refresh_episodes[:-1]], refresh_episodes)
    ]

    periods = []
    previous_refresh_total = 0
    for end in expected_episodes:
        period_windows = [
            row for row in windows if end - 2_500 < int(row["episode"]) <= end
        ]
        refresh_total = max(
            (int(row["baseline_updates"]) for row in period_windows), default=0
        )
        periods.append({
            "end_episode": end,
            "validation_score": scores[expected_episodes.index(end)],
            "DS1_validation_mean": float(
                history[expected_episodes.index(end)]["DS1"]
                ["equal_instance_distribution"]["mean"]
            ),
            "DS2_validation_mean": float(
                history[expected_episodes.index(end)]["DS2"]
                ["equal_instance_distribution"]["mean"]
            ),
            "training_policy_relocations": _mean([
                float(row["mean_policy_relocations"]) for row in period_windows
            ]),
            "training_baseline_relocations": _mean([
                float(row["mean_baseline_relocations"]) for row in period_windows
            ]),
            "mean_advantage": _mean([
                float(row["mean_advantage"]) for row in period_windows
            ]),
            "entropy": _mean([float(row["entropy"]) for row in period_windows]),
            "grad_norm_before_clip": _mean([
                float(row["grad_norm"]) for row in period_windows
            ]),
            "baseline_refreshes": refresh_total - previous_refresh_total,
        })
        previous_refresh_total = refresh_total

    entropies = [float(row["entropy"]) for row in windows]
    gradients = [float(row["grad_norm"]) for row in windows]
    advantages = [float(row["mean_advantage"]) for row in windows]
    buckets = {
        str(key): int(value)
        for key, value in completion["S_bucket_episode_counts"].items()
    }
    bucket_values = list(buckets.values())
    variants = {
        str(key): int(value)
        for key, value in completion["variant_episode_counts"].items()
    }
    post_best = [
        {"episode": episode, "score": score, "degradation_from_best": score - scores[best_index]}
        for episode, score in zip(episodes[best_index + 1 :], scores[best_index + 1 :])
    ]
    first_entropy = _mean(entropies[:25])
    last_entropy = _mean(entropies[-25:])
    return {
        "source": "committed_phase7b_compact_train_validation_artifacts",
        "formal_test_used": False,
        "validation_checkpoints": [
            {
                "episode": episode,
                "DS1_mean": float(row["DS1"]["equal_instance_distribution"]["mean"]),
                "DS2_mean": float(row["DS2"]["equal_instance_distribution"]["mean"]),
                "selection_score": float(row["selection_score"]),
                "baseline_state_version": int(row["baseline_state_version"]),
            }
            for episode, row in zip(episodes, history)
        ],
        "best_episode": best_episode,
        "best_score": scores[best_index],
        "post_best_degradation": post_best,
        "periods": periods,
        "baseline_refresh": {
            "count": len(refreshes),
            "episodes": refresh_episodes,
            "before_or_at_8160": sum(episode <= 8_160 for episode in refresh_episodes),
            "between_8161_and_14123": sum(
                8_160 < episode < 14_124 for episode in refresh_episodes
            ),
            "after_or_at_17968": sum(episode >= 17_968 for episode in refresh_episodes),
            "median_gap_episodes": float(statistics.median(refresh_gaps)),
            "max_gap_episodes": max(refresh_gaps),
            "paired_sample_sizes": sorted({int(row["sample_size"]) for row in refreshes}),
            "assessment": (
                "n=4 checkpoint-local tests create discrete, noisy refresh evidence; "
                "refreshes cluster early, disappear for a long middle interval, then cluster late"
            ),
        },
        "optimization": {
            "entropy_mean": _mean(entropies),
            "entropy_min": min(entropies),
            "entropy_max": max(entropies),
            "entropy_windows_below_0_5": sum(value < 0.5 for value in entropies),
            "entropy_first_2500_mean": first_entropy,
            "entropy_last_2500_mean": last_entropy,
            "entropy_collapse_detected": bool(
                last_entropy < 0.5 or last_entropy < 0.5 * first_entropy
            ),
            "grad_norm_mean_before_clip": _mean(gradients),
            "grad_norm_max_before_clip": max(gradients),
            "grad_windows_above_clip_0_5": sum(value > 0.5 for value in gradients),
            "grad_windows_above_5": sum(value > 5.0 for value in gradients),
            "configured_clip": 0.5,
            "mean_advantage_range": [min(advantages), max(advantages)],
            "advantage_variance_available_in_compact_trace": False,
        },
        "sampling_balance": {
            "S_bucket_counts": buckets,
            "S_bucket_max_min_ratio": max(bucket_values) / min(bucket_values),
            "S_bucket_coefficient_of_variation": float(
                np.std(bucket_values) / np.mean(bucket_values)
            ),
            "variant_counts": variants,
            "variant_absolute_imbalance": abs(variants["DS1"] - variants["DS2"]),
        },
        "interpretation": (
            "15k is the minimum of the frozen validation trajectory. Later checkpoints "
            "degrade at 17.5k/20k and only partially recover, while noisy baseline refresh "
            "timing and frequently clipped gradients provide plausible optimization causes."
        ),
    }


def _training_sample(
    ref: BaseInstanceRef,
    variant: str,
    seed: int,
    visit_index: int,
) -> TrainingSample:
    group = parse_parameter_group(ref.parameter_group)
    return TrainingSample(
        base_instance_id=ref.base_instance_id,
        instance_id=ref.ds1_instance_id if variant == "DS1" else ref.ds2_instance_id,
        variant=variant,
        scenario_seed=seed,
        visit_index=visit_index,
        num_stacks=int(group["S"]),
    )


def _load_policy(
    state_dict: Mapping[str, torch.Tensor],
    config: FormalTrainingConfig,
    *,
    observation_version: str | None = None,
):
    version = observation_version or config.observation_version
    policy = make_scrp_policy(
        version, 5, 3,
        Mmax=config.Mmax or 6,
        embed_dim=config.embed_dim,
        num_encoder_layers=config.num_encoder_layers,
        num_heads=config.num_heads,
        ffn_dim=config.ffn_dim,
        clip_constant=config.clip_constant,
        device="cpu",
    )
    policy.load_state_dict(state_dict, strict=True)
    policy.eval()
    return policy


def _policy_outputs(policy, observation, legal, num_stacks, node_mask=None):
    obs_t = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
    forbidden = torch.tensor(~np.asarray(legal, dtype=bool)).unsqueeze(0)
    with torch.no_grad():
        encoded = policy.encode(obs_t, node_mask)
        query = policy.low_decoder._build_query(encoded, node_mask)
        context = policy.low_decoder.cross_attn(
            query, encoded, encoded, node_mask
        )
        context = policy.low_decoder.norm(query + context)
        log_probs = policy.low_decoder._pointer_scores(
            context, encoded, forbidden
        )
    return log_probs.exp()[0].cpu().numpy(), encoded[0, :num_stacks].cpu().numpy()


def _controlled_future_permutation(observation, num_stacks, node_mask):
    nodes = np.asarray(observation, dtype=np.float32).reshape(-1, 12).copy()
    mask = node_mask[0].cpu().numpy()
    real_order = [
        index for index in range(num_stacks, num_stacks + 6) if not mask[index]
    ]
    future = real_order[1:]
    if len(future) < 2:
        return None
    # Preserve the current target and rank slots while reversing which future
    # public location occupies each revealed-order rank.
    payload = nodes[future, 2:10].copy()[::-1]
    nodes[future, 2:10] = payload
    return nodes.reshape(-1)


def _representation_probe(policy, observation, legal, num_stacks, node_mask):
    original_probs, original_embeddings = _policy_outputs(
        policy, observation, legal, num_stacks, node_mask
    )
    result = {}
    permuted = _controlled_future_permutation(
        observation, num_stacks, node_mask
    )
    if permuted is not None:
        permuted_probs, permuted_embeddings = _policy_outputs(
            policy, permuted, legal, num_stacks, node_mask
        )
        result["permutation_tv"] = float(
            0.5 * np.abs(original_probs - permuted_probs).sum()
        )
        result["permutation_action_changed"] = bool(
            np.argmax(original_probs) != np.argmax(permuted_probs)
        )
        result["permutation_stack_embedding_l2"] = float(
            np.linalg.norm(original_embeddings - permuted_embeddings)
            / math.sqrt(original_embeddings.size)
        )

    ablated_mask = node_mask.clone()
    ablated_mask[:, num_stacks : num_stacks + 6] = True
    ablated_probs, _ = _policy_outputs(
        policy, observation, legal, num_stacks, ablated_mask
    )
    result["order_ablation_tv"] = float(
        0.5 * np.abs(original_probs - ablated_probs).sum()
    )
    result["order_ablation_action_changed"] = bool(
        np.argmax(original_probs) != np.argmax(ablated_probs)
    )

    nodes = np.asarray(observation, dtype=np.float32).reshape(-1, 12).copy()
    padding = node_mask[0].cpu().numpy()
    nodes[padding, 1:11] = 0.731
    padded_probs, _ = _policy_outputs(
        policy, nodes.reshape(-1), legal, num_stacks, node_mask
    )
    result["padding_perturbation_max_abs"] = float(
        np.max(np.abs(original_probs - padded_probs))
    )
    return result


def _aggregate_records(
    records: Sequence[Mapping[str, object]], keys: Sequence[str]
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        grouped[tuple(record[key] for key in keys)].append(record)
    result = []
    for values, rows in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        result.append({
            **dict(zip(keys, values)),
            "states": len(rows),
            "agreement_rate": _mean([float(row["agreement"]) for row in rows]),
            "mean_ERI_score_gap_of_RL_action": _mean([
                float(row["eri_score_gap"]) for row in rows
            ]),
            "ERI_score_equivalent_rate": _mean([
                float(abs(float(row["eri_score_gap"])) < 1e-12) for row in rows
            ]),
            "positive_ERI_score_gap_rate": _mean([
                float(float(row["eri_score_gap"]) > 1e-12) for row in rows
            ]),
        })
    return result


def run_eri_imitation_and_representation_diagnostic(
    manifest: SplitManifest,
    provider,
    checkpoint: Mapping[str, object],
    config: FormalTrainingConfig,
    refs_by_split: Mapping[str, Sequence[BaseInstanceRef]],
) -> dict[str, object]:
    policy = _load_policy(checkpoint["model_state_dict"], config)
    schedule = ScenarioSeedSchedule(manifest)
    eri = ERIBaseline()
    records = []
    probes = []
    episode_count = 0
    for split, refs in refs_by_split.items():
        if split not in DEVELOPMENT_SPLITS:
            raise ValueError("imitation diagnostic refuses non-development split")
        for ref in refs:
            group = parse_parameter_group(ref.parameter_group)
            for variant in ("DS1", "DS2"):
                sample = _training_sample(
                    ref, variant,
                    schedule.seed_for(split, ref.base_instance_id, 0), 0,
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
                while not info.get("terminated", False):
                    legal_mask = np.asarray(info["action_mask"], dtype=bool)
                    legal = tuple(np.flatnonzero(legal_mask).tolist())
                    node_mask = make_node_padding_mask(
                        torch.tensor(observation).unsqueeze(0), "O2",
                        instance.num_stacks, Mmax=6,
                    )
                    probs, _ = _policy_outputs(
                        policy, observation, legal_mask, instance.num_stacks, node_mask
                    )
                    rl_action = int(np.argmax(probs))
                    eri_action = eri.select_destination(instance, core.state, legal)
                    state = core.state
                    target_location = state.locations[state.current_target_id]
                    target_stack = state.stacks[target_location.stack_id]
                    blocker_id = target_stack.top_id
                    current_batch = instance.batch_order[state.current_batch_index]
                    records.append({
                        "split": split,
                        "dataset": variant,
                        **group,
                        "target_tier": target_location.tier,
                        "blockers_above": target_stack.height - target_location.tier - 1,
                        "current_batch_size": instance.batch_sizes[current_batch],
                        "agreement": int(rl_action == eri_action),
                        "eri_score_gap": float(
                            eri.destination_score(instance, state, blocker_id, rl_action)
                            - eri.destination_score(instance, state, blocker_id, eri_action)
                        ),
                    })
                    probe = _representation_probe(
                        policy, observation, legal_mask, instance.num_stacks, node_mask
                    )
                    if "permutation_tv" in probe:
                        probes.append(probe)
                    observation, _, _, _, info = adapter.step(eri_action)
                episode_count += 1

    grouped = {
        "split_dataset": _aggregate_records(records, ("split", "dataset")),
        "S": _aggregate_records(records, ("S",)),
        "T": _aggregate_records(records, ("T",)),
        "fill": _aggregate_records(records, ("fill",)),
        "dataset_batch_size": _aggregate_records(
            records, ("dataset", "current_batch_size")
        ),
        "target_tier_blockers": _aggregate_records(
            records, ("target_tier", "blockers_above")
        ),
    }
    candidate_groups = [
        row for name in ("S", "T", "fill", "dataset_batch_size", "target_tier_blockers")
        for row in grouped[name] if int(row["states"]) >= 20
    ]
    worst = sorted(
        candidate_groups,
        key=lambda row: (float(row["agreement_rate"]), -int(row["states"])),
    )[:10]
    return {
        "splits": list(refs_by_split),
        "episodes": episode_count,
        "public_decision_states": len(records),
        "overall_action_agreement": _mean([
            float(row["agreement"]) for row in records
        ]),
        "overall_mean_ERI_score_gap_of_RL_action": _mean([
            float(row["eri_score_gap"]) for row in records
        ]),
        "grouped": grouped,
        "lowest_agreement_state_groups_min_20_states": worst,
        "representation_usage": {
            "eligible_future_permutation_states": len(probes),
            "mean_controlled_permutation_TV": _mean([
                float(row["permutation_tv"]) for row in probes
            ]),
            "controlled_permutation_action_change_rate": _mean([
                float(row["permutation_action_changed"]) for row in probes
            ]),
            "mean_stack_embedding_change_L2": _mean([
                float(row["permutation_stack_embedding_l2"]) for row in probes
            ]),
            "mean_order_ablation_TV": _mean([
                float(row["order_ablation_tv"]) for row in probes
            ]),
            "order_ablation_action_change_rate": _mean([
                float(row["order_ablation_action_changed"]) for row in probes
            ]),
            "max_padding_perturbation_effect": max(
                (float(row["padding_perturbation_max_abs"]) for row in probes),
                default=0.0,
            ),
            "probe_interpretation": (
                "controlled future-order permutation and order-node ablation measure "
                "whether revealed-order nodes affect logits/actions; padding perturbation "
                "must remain numerically inert under the mask"
            ),
        },
        "trajectory_policy": "ERI-guided public states; RL action is compared but not executed",
        "raw_state_records_retained": False,
    }


def run_checkpoint_baseline_diagnostic(
    manifest: SplitManifest,
    provider,
    checkpoint: Mapping[str, object],
    config: FormalTrainingConfig,
    validation_refs: Sequence[BaseInstanceRef],
) -> dict[str, object]:
    policy = _load_policy(checkpoint["model_state_dict"], config)
    baseline = _load_policy(checkpoint["baseline_state"], config)
    schedule = ScenarioSeedSchedule(manifest)
    episode_deltas = []
    decision_advantages = []
    group_rows = []
    for ref in validation_refs:
        group = parse_parameter_group(ref.parameter_group)
        for variant in ("DS1", "DS2"):
            sample = _training_sample(
                ref, variant,
                schedule.seed_for("validation", ref.base_instance_id, 0), 0,
            )
            instance = provider(sample)
            policy_run = run_formal_episode(
                instance, sample, policy, config, greedy=True, device="cpu"
            )
            baseline_run = run_formal_episode(
                instance, sample, baseline, config, greedy=True, device="cpu"
            )
            if policy_run.scenario_id != baseline_run.scenario_id:
                raise AssertionError("checkpoint policy/baseline CRN mismatch")
            delta = policy_run.relocations - baseline_run.relocations
            episode_deltas.append(float(delta))
            decision_advantages.extend(frozen_greedy_advantages(
                policy_run.rewards, baseline_run.episode_return, config.gamma
            ))
            group_rows.append({
                "dataset": variant,
                **group,
                "delta": float(delta),
            })
    def grouped_delta(key):
        values = defaultdict(list)
        for row in group_rows:
            values[row[key]].append(float(row["delta"]))
        return [
            {
                key: value,
                "episodes": len(rows),
                "mean_policy_minus_baseline_relocations": _mean(rows),
                "policy_not_worse_rate": _mean([float(row <= 0) for row in rows]),
            }
            for value, rows in sorted(values.items(), key=lambda item: str(item[0]))
        ]
    return {
        "split": "validation",
        "episodes": len(episode_deltas),
        "policy_minus_frozen_baseline_relocations_mean": _mean(episode_deltas),
        "policy_minus_frozen_baseline_relocations_variance": _sample_variance(
            episode_deltas
        ),
        "decision_advantage_count": len(decision_advantages),
        "decision_advantage_mean": _mean(decision_advantages),
        "decision_advantage_variance": _sample_variance(decision_advantages),
        "by_dataset": grouped_delta("dataset"),
        "by_S": grouped_delta("S"),
        "by_T": grouped_delta("T"),
        "by_fill": grouped_delta("fill"),
    }


def _evaluate_policy_on_refs(
    policy,
    config: FormalTrainingConfig,
    manifest: SplitManifest,
    provider,
    refs: Sequence[BaseInstanceRef],
) -> dict[str, object]:
    schedule = ScenarioSeedSchedule(manifest)
    rows = []
    policy.eval()
    for ref in refs:
        group = parse_parameter_group(ref.parameter_group)
        for variant in ("DS1", "DS2"):
            sample = _training_sample(
                ref, variant,
                schedule.seed_for("validation", ref.base_instance_id, 0), 0,
            )
            run = run_formal_episode(
                provider(sample), sample, policy, config, greedy=True, device="cpu"
            )
            if not run.terminated or run.truncated or run.invalid_actions:
                raise RuntimeError("development validation rollout failed")
            rows.append({"dataset": variant, **group, "relocations": run.relocations})
    return {
        "episodes": len(rows),
        "mean_relocations": _mean([float(row["relocations"]) for row in rows]),
        "DS1_mean": _mean([
            float(row["relocations"]) for row in rows if row["dataset"] == "DS1"
        ]),
        "DS2_mean": _mean([
            float(row["relocations"]) for row in rows if row["dataset"] == "DS2"
        ]),
        "by_S": [
            {
                "S": S,
                "episodes": len(subset),
                "mean_relocations": _mean([float(row["relocations"]) for row in subset]),
            }
            for S in sorted({int(row["S"]) for row in rows})
            for subset in [[row for row in rows if int(row["S"]) == S]]
        ],
    }


def run_o1_o2_development_ablation(
    manifest: SplitManifest,
    provider,
    base_config: FormalTrainingConfig,
    train_refs: Sequence[BaseInstanceRef],
    validation_refs: Sequence[BaseInstanceRef],
    *,
    training_episodes: int = 400,
) -> dict[str, object]:
    if training_episodes <= 0 or training_episodes % base_config.batch_size:
        raise ValueError("training_episodes must be a positive multiple of batch size")
    allowed = tuple(ref.base_instance_id for ref in train_refs)
    results = {}
    sample_schedule = None
    for version in ("O1", "O2"):
        config = replace(
            base_config,
            observation_version=version,
            Mmax=None if version == "O1" else 6,
        )
        trainer = SCRPFormalTrainer(
            config, manifest, provider, allowed_base_ids=allowed
        )
        metrics = trainer.train_iterations(training_episodes // config.batch_size)
        schedule_fingerprint = [
            [sample.base_instance_id, sample.variant, sample.scenario_seed]
            for sample in trainer.sample_history
        ]
        if sample_schedule is None:
            sample_schedule = schedule_fingerprint
        elif sample_schedule != schedule_fingerprint:
            raise AssertionError("O1/O2 ablation training schedules differ")
        validation_result = _evaluate_policy_on_refs(
            trainer.policy, config, manifest, provider, validation_refs
        )
        results[version] = {
            "training_episodes": trainer.episodes_seen,
            "training_iterations": trainer.iteration,
            "baseline_refreshes": trainer.baseline_updates,
            "last_100_episode_proxy": {
                "policy_relocations": _mean([
                    metric.mean_policy_relocations for metric in metrics[-25:]
                ]),
                "entropy": _mean([metric.entropy for metric in metrics[-25:]]),
                "grad_norm_before_clip": _mean([
                    metric.grad_norm for metric in metrics[-25:]
                ]),
            },
            "validation": validation_result,
        }
    o1 = float(results["O1"]["validation"]["mean_relocations"])
    o2 = float(results["O2"]["validation"]["mean_relocations"])
    return {
        "status": "development_only_small_budget_ablation",
        "seed": base_config.seed,
        "same_training_schedule": True,
        "train_base_layouts": len(train_refs),
        "validation_base_layouts": len(validation_refs),
        "training_episodes_per_observation": training_episodes,
        "validation_scenarios_per_static_variant": 1,
        "formal_test_used": False,
        "model_artifacts_saved": False,
        "O1": results["O1"],
        "O2": results["O2"],
        "O2_minus_O1_validation_relocations": o2 - o1,
        "O2_representation_outperforms_O1_at_this_budget": o2 < o1,
        "specific_revealed_order_causality_established": False,
        "interpretation_scope": (
            "answers whether the complete O2 representation improves over O1 under "
            "this fixed small development budget. Because the adapters differ beyond "
            "only the order nodes, this does not by itself establish that the model "
            "uses the specific revealed-order sequence."
        ),
    }
