"""Execute the one-shot Phase 8 paired formal test: frozen O2 RL versus ERI."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from experiments.baselines import ERIBaseline, run_baseline_episode
from experiments.formal_run import CANDIDATE_CONFIG_PATH, committed_file_sha256
from experiments.formal_test import (
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_SEED,
    ERI_ALGORITHM,
    FORMAL_TEST_RUN_ID,
    RL_ALGORITHM,
    aggregate_per_instance,
    assert_compact_artifact_schema,
    atomic_write_json,
    build_formal_test_coordinates,
    create_formal_test_identity,
    dataset_summary,
    file_sha256,
    hierarchical_paired_bootstrap,
    pair_and_validate_primary_results,
    parameter_group_summary,
    robustness_statistics,
    verify_file_sha256,
)
from experiments.protocol import ScenarioResult, load_protocol_config, load_split_manifest
from scrp.datasets import merge_adjacent_batches, parse_ku_crptw
from scrp.environment import SCRPEnvironment
from scrp.formal_training import (
    TrainingSample,
    load_formal_training_config,
    make_scrp_policy,
    run_formal_episode,
)
from scrp.models import SCRPConfig


def _load_static_instances(source_root: Path, coordinates):
    paths = {path.stem: path for path in source_root.rglob("*.txt")}
    if len(paths) != 1_440:
        raise RuntimeError(f"expected 1,440 source instances, observed {len(paths)}")
    cache = {}
    for coordinate in coordinates:
        key = (coordinate.dataset, coordinate.base_instance_id)
        if key in cache:
            continue
        try:
            ds1 = parse_ku_crptw(paths[coordinate.base_instance_id])
        except KeyError as error:
            raise RuntimeError(
                f"missing Ku source instance {coordinate.base_instance_id}"
            ) from error
        instance = ds1 if coordinate.dataset == "DS1" else merge_adjacent_batches(ds1)
        if instance.instance_id != coordinate.instance_id:
            raise RuntimeError("materialized static instance ID mismatch")
        cache[key] = instance
    if len(cache) != 480:
        raise RuntimeError(f"expected 480 static variants, observed {len(cache)}")
    return cache


def _checkpoint_integrity(checkpoint, identity, training_config) -> dict[str, object]:
    expected = {
        "episodes_seen": 15_000,
        "root_seed": identity.root_seed,
        "observation_version": "O2",
        "feature_dim": 12,
        "Mmax": 6,
        "dataset_version": identity.dataset_version,
        "split_manifest_version": identity.split_manifest_version,
        "training_protocol_version": training_config.training_protocol_version,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise RuntimeError(
                f"checkpoint metadata mismatch for {key}: "
                f"expected {value!r}, observed {checkpoint.get(key)!r}"
            )
    if list(checkpoint.get("S_bucket_metadata", ())) != [5, 6, 7, 8, 9, 10]:
        raise RuntimeError("checkpoint S-bucket support mismatch")
    if checkpoint.get("config_snapshot") != training_config.__dict__:
        raise RuntimeError("checkpoint config snapshot mismatch")
    return {
        **expected,
        "S_bucket_support": [5, 6, 7, 8, 9, 10],
        "policy_architecture": {
            "embed_dim": training_config.embed_dim,
            "num_encoder_layers": training_config.num_encoder_layers,
            "num_heads": training_config.num_heads,
            "ffn_dim": training_config.ffn_dim,
            "clip_constant": training_config.clip_constant,
            "feature_dim": training_config.feature_dim,
        },
        "candidate_count": "dynamic_equal_to_S",
        "passed": True,
    }


class _FrozenPolicyPool:
    def __init__(self, state_dict, config):
        self.state_dict = state_dict
        self.config = config
        self.policies = {}

    def policy_for(self, num_stacks: int, max_tiers: int):
        key = (num_stacks, max_tiers)
        if key not in self.policies:
            policy = make_scrp_policy(
                "O2", num_stacks, max_tiers, Mmax=6,
                embed_dim=self.config.embed_dim,
                num_encoder_layers=self.config.num_encoder_layers,
                num_heads=self.config.num_heads,
                ffn_dim=self.config.ffn_dim,
                clip_constant=self.config.clip_constant,
                device="cpu",
            )
            policy.load_state_dict(self.state_dict, strict=True)
            policy.eval()
            if policy.scrp_candidate_count != num_stacks:
                raise RuntimeError("policy candidate count does not equal S")
            self.policies[key] = policy
        return self.policies[key]

    def run(self, instance, coordinate):
        sample = TrainingSample(
            base_instance_id=coordinate.base_instance_id,
            instance_id=coordinate.instance_id,
            variant=coordinate.dataset,
            scenario_seed=coordinate.scenario_seed,
            visit_index=0,
            num_stacks=instance.num_stacks,
        )
        return run_formal_episode(
            instance,
            sample,
            self.policy_for(instance.num_stacks, instance.max_tiers),
            self.config,
            greedy=True,
            device="cpu",
        )


def _scenario_result(coordinate, algorithm, relocations, scenario_id):
    return ScenarioResult(
        dataset=coordinate.dataset,
        split="test",
        instance_id=coordinate.instance_id,
        base_instance_id=coordinate.base_instance_id,
        parameter_group=coordinate.parameter_group,
        scenario_seed=coordinate.scenario_seed,
        scenario_id=scenario_id,
        algorithm=algorithm,
        relocations=relocations,
        terminated=True,
        truncated=False,
    )


def _write_paper_tables(path: Path, ds1, ds2, statistics):
    lines = [
        "# Phase 8 Formal Test Tables",
        "",
        "Delta is RL minus ERI; negative values favor RL.",
        "",
        "| Dataset | Algorithm | Mean relocations | Difference vs ERI | 95% CI |",
        "|---|---|---:|---:|---:|",
    ]
    for summary in (ds1, ds2):
        dataset = summary["dataset"]
        rl_mean = summary["algorithms"][RL_ALGORITHM]["scenario_distribution"]["mean"]
        eri_mean = summary["algorithms"][ERI_ALGORITHM]["scenario_distribution"]["mean"]
        boot = summary["paired_delta"]["hierarchical_bootstrap"]
        lines.extend([
            f"| {dataset} | ERI | {eri_mean:.6f} | 0 | [0, 0] |",
            (
                f"| {dataset} | RL O2 | {rl_mean:.6f} | "
                f"{boot['mean_delta_RL_minus_ERI']:.6f} | "
                f"[{boot['ci95_low']:.6f}, {boot['ci95_high']:.6f}] |"
            ),
        ])
    lines.extend([
        "",
        "| Dataset | Mean delta (RL-ERI) | 95% CI | Wilcoxon p | Paired t p |",
        "|---|---:|---:|---:|---:|",
    ])
    for dataset in ("DS1", "DS2"):
        item = statistics[dataset]
        boot = item["hierarchical_bootstrap"]
        robust = item["robustness"]
        lines.append(
            f"| {dataset} | {boot['mean_delta_RL_minus_ERI']:.6f} | "
            f"[{boot['ci95_low']:.6f}, {boot['ci95_high']:.6f}] | "
            f"{robust['wilcoxon_signed_rank']['p_value_two_sided']:.6g} | "
            f"{robust['paired_t_test']['p_value_two_sided']:.6g} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path(
            "checkpoints/formal-o2-mixed-seed20260816-run1/best-validation.pt"
        ),
    )
    parser.add_argument(
        "--identity-output", type=Path,
        default=Path(
            "experiments/formal_test_runs/"
            f"{FORMAL_TEST_RUN_ID}/run_identity.json"
        ),
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("experiments/splits/scrp_split_v1.json"),
    )
    parser.add_argument(
        "--protocol", type=Path,
        default=Path("experiments/configs/formal_protocol_v1.json"),
    )
    parser.add_argument(
        "--summary-root", type=Path, default=Path("experiments/summaries"),
    )
    parser.add_argument(
        "--raw-root", type=Path, default=Path("experiments/raw_results"),
    )
    args = parser.parse_args()

    run_id = FORMAL_TEST_RUN_ID
    summary_paths = {
        "DS1": args.summary_root / f"{run_id}-ds1.json",
        "DS2": args.summary_root / f"{run_id}-ds2.json",
        "groups": args.summary_root / f"{run_id}-parameter-groups.json",
        "statistics": args.summary_root / f"{run_id}-statistics.json",
        "completion": args.summary_root / f"{run_id}-completion.json",
        "tables": args.summary_root / f"{run_id}-paper-tables.md",
    }
    raw_dir = args.raw_root / run_id
    raw_path = raw_dir / "primary-results.jsonl"
    partial_path = raw_dir / "primary-results.jsonl.partial"
    failure_path = raw_dir / "failure.json"
    forbidden_existing = [
        args.identity_output, raw_path, partial_path, failure_path,
        *summary_paths.values(),
    ]
    existing = [str(path) for path in forbidden_existing if path.exists()]
    if existing:
        raise RuntimeError(
            "formal test is one-shot and refuses existing outputs: " + ", ".join(existing)
        )

    # This is written before loading the split manifest or producing any result.
    identity = create_formal_test_identity(
        args.identity_output, checkpoint_path=args.checkpoint,
    )
    results = []
    started = time.perf_counter()
    try:
        checkpoint_path = Path(identity.checkpoint_path)
        checkpoint_hash = verify_file_sha256(
            checkpoint_path, identity.checkpoint_sha256
        )
        if committed_file_sha256(identity.code_sha, CANDIDATE_CONFIG_PATH) != identity.config_sha256:
            raise RuntimeError("committed candidate config hash mismatch")
        training_config = load_formal_training_config(CANDIDATE_CONFIG_PATH)
        protocol = load_protocol_config(args.protocol)
        manifest = load_split_manifest(args.manifest)
        if protocol.protocol_version != identity.formal_test_protocol_version:
            raise RuntimeError("formal-test protocol version mismatch")
        if protocol.formal_test_scenarios_per_instance != 50:
            raise RuntimeError("formal-test protocol K drift")
        if manifest.protocol_version != identity.split_manifest_version:
            raise RuntimeError("split manifest version mismatch")
        if manifest.dataset_version != identity.dataset_version:
            raise RuntimeError("dataset version mismatch")

        # SHA is verified before torch is allowed to deserialize the checkpoint.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_audit = _checkpoint_integrity(checkpoint, identity, training_config)
        coordinates = build_formal_test_coordinates(manifest)
        instances = _load_static_instances(args.source_root, coordinates)
        policy_pool = _FrozenPolicyPool(checkpoint["model_state_dict"], training_config)
        raw_dir.mkdir(parents=True, exist_ok=True)

        with partial_path.open("x", encoding="utf-8", newline="\n") as raw_stream:
            for index, coordinate in enumerate(coordinates, start=1):
                instance = instances[(coordinate.dataset, coordinate.base_instance_id)]
                rl = policy_pool.run(instance, coordinate)
                if not rl.terminated or rl.truncated or rl.invalid_actions:
                    raise RuntimeError("RL formal-test rollout failed integrity checks")
                eri_env = SCRPEnvironment(
                    SCRPConfig(
                        instance.num_stacks,
                        instance.max_tiers,
                        root_seed=identity.root_seed,
                        max_steps=training_config.max_steps,
                        validate_after_transition=True,
                    ),
                    instance,
                )
                eri = run_baseline_episode(
                    eri_env,
                    ERIBaseline(),
                    coordinate.scenario_seed,
                    action_seed=identity.root_seed,
                )
                if (
                    not eri.terminated or eri.truncated or eri.invalid_action_count
                    or rl.scenario_id != eri.scenario_id
                ):
                    raise RuntimeError("formal-test CRN/integrity failure")
                rows = (
                    _scenario_result(
                        coordinate, RL_ALGORITHM, rl.relocations, rl.scenario_id
                    ),
                    _scenario_result(
                        coordinate, ERI_ALGORITHM, eri.relocations, eri.scenario_id
                    ),
                )
                for row in rows:
                    results.append(row)
                    raw_stream.write(json.dumps(row.to_record(), sort_keys=True) + "\n")
                if index % 500 == 0:
                    raw_stream.flush()
                    print(json.dumps({
                        "completed_coordinates": index,
                        "planned_coordinates": len(coordinates),
                        "raw_rows": len(results),
                        "wall_seconds": time.perf_counter() - started,
                    }), flush=True)

        pairs = pair_and_validate_primary_results(results, coordinates)
        os.replace(partial_path, raw_path)
        raw_hash = file_sha256(raw_path)
        per_instance = aggregate_per_instance(pairs)
        dataset_summaries = {}
        statistics = {}
        for dataset in ("DS1", "DS2"):
            dataset_pairs = tuple(pair for pair in pairs if pair.dataset == dataset)
            dataset_instances = tuple(
                row for row in per_instance if row["dataset"] == dataset
            )
            bootstrap = hierarchical_paired_bootstrap(
                dataset_pairs,
                repetitions=BOOTSTRAP_REPETITIONS,
                seed=BOOTSTRAP_SEED,
            )
            robustness = robustness_statistics(dataset_instances)
            summary = dataset_summary(
                dataset, dataset_pairs, dataset_instances, bootstrap, robustness
            )
            assert_compact_artifact_schema(summary)
            dataset_summaries[dataset] = summary
            statistics[dataset] = {
                "hierarchical_bootstrap": bootstrap,
                "robustness": robustness,
                "instance_win_tie_loss": summary["instance_win_tie_loss"],
            }
        groups = parameter_group_summary(per_instance)
        strongest_weakest = {}
        for dataset in ("DS1", "DS2"):
            subset = [row for row in groups if row["dataset"] == dataset]
            strongest_weakest[dataset] = {
                "strongest_RL_group": min(
                    subset, key=lambda row: row["mean_delta_RL_minus_ERI"]
                ),
                "weakest_RL_group": max(
                    subset, key=lambda row: row["mean_delta_RL_minus_ERI"]
                ),
            }

        wall_seconds = time.perf_counter() - started
        integrity = {
            "invalid_actions": 0,
            "truncated": 0,
            "scenario_mismatches": 0,
            "scheduled_coordinates": 24_000,
            "rows_per_primary_algorithm": {
                RL_ALGORITHM: 24_000,
                ERI_ALGORITHM: 24_000,
            },
            "total_rows": 48_000,
            "duplicates": 0,
            "missing_coordinates": 0,
            "missing_algorithm_pairs": 0,
            "passed": True,
        }
        raw_artifact = {
            "gitignored_path": raw_path.as_posix(),
            "sha256": raw_hash,
            "bytes": raw_path.stat().st_size,
            "rows": 48_000,
        }
        for dataset in ("DS1", "DS2"):
            atomic_write_json(summary_paths[dataset], {
                "run_id": run_id,
                "raw_artifact": raw_artifact,
                **dataset_summaries[dataset],
            })
        atomic_write_json(summary_paths["groups"], {
            "run_id": run_id,
            "interpretation": "descriptive heterogeneity; n=5 test bases per group",
            "groups": list(groups),
            "strongest_weakest": strongest_weakest,
            "raw_artifact": raw_artifact,
        })
        statistics_payload = {
            "run_id": run_id,
            "paired_delta_sign_convention": "RL_minus_ERI; negative_favors_RL",
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "inference_unit": "static_instance_with_nested_paired_scenarios",
            "naive_scenario_independent_t_test_performed": False,
            "DS1": statistics["DS1"],
            "DS2": statistics["DS2"],
            "raw_artifact": raw_artifact,
        }
        atomic_write_json(summary_paths["statistics"], statistics_payload)
        _write_paper_tables(
            summary_paths["tables"],
            dataset_summaries["DS1"], dataset_summaries["DS2"], statistics,
        )
        completion = {
            "run_id": run_id,
            "code_sha": identity.code_sha,
            "training_run_id": identity.training_run_id,
            "checkpoint_episode": identity.checkpoint_episode,
            "checkpoint_sha256": checkpoint_hash,
            "config_sha256": identity.config_sha256,
            "formal_test_protocol_version": identity.formal_test_protocol_version,
            "checkpoint_integrity": checkpoint_audit,
            "integrity": integrity,
            "raw_artifact": raw_artifact,
            "wall_seconds": wall_seconds,
            "episodes_per_second": 48_000 / wall_seconds,
            "primary_results": {
                dataset: {
                    "RL_mean": dataset_summaries[dataset]["algorithms"]
                        [RL_ALGORITHM]["scenario_distribution"]["mean"],
                    "ERI_mean": dataset_summaries[dataset]["algorithms"]
                        [ERI_ALGORITHM]["scenario_distribution"]["mean"],
                    "paired_delta": statistics[dataset]["hierarchical_bootstrap"],
                    "robustness": statistics[dataset]["robustness"],
                    "instance_win_tie_loss": statistics[dataset]
                        ["instance_win_tie_loss"],
                }
                for dataset in ("DS1", "DS2")
            },
            "strongest_weakest_parameter_groups": strongest_weakest,
            "test_driven_adjustment_performed": False,
            "training_performed": False,
            "FORMAL_TEST_COMPLETE": "YES",
        }
        atomic_write_json(summary_paths["completion"], completion)
        print(json.dumps(completion, indent=2), flush=True)
    except BaseException as error:
        atomic_write_json(failure_path, {
            "run_id": run_id,
            "error_type": type(error).__name__,
            "error": str(error),
            "completed_rows": len(results),
            "inference_performed": False,
            "FORMAL_TEST_COMPLETE": "NO",
        })
        raise


if __name__ == "__main__":
    main()
