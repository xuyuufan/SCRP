"""Run the non-formal Phase 4 baseline integration subset.

This intentionally evaluates only 12 base layouts, both DS1/DS2 variants and
three scenarios. It validates the harness; it is not a formal performance run.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from experiments.baselines import (
    ERIBaseline,
    MinBlockingGreedyBaseline,
    RandomLegalBaseline,
)
from experiments.evaluation import (
    BaselineAlgorithm,
    EvaluationCase,
    LowPolicyAlgorithm,
    aggregate_relocations,
    assert_paired_scenarios,
    evaluate_algorithm_on_schedule,
    save_raw_results,
)
from experiments.protocol import ScenarioSeedSchedule, load_split_manifest
from scrp.datasets import merge_adjacent_batches, parse_ku_crptw
from scrp.training import make_scrp_o1_policy


def _load_policy(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["training_config"]
    policy = make_scrp_o1_policy(
        embed_dim=config["embed_dim"],
        num_encoder_layers=config["num_encoder_layers"],
        num_heads=config["num_heads"],
        ffn_dim=config["ffn_dim"],
        clip_constant=config["clip_constant"],
    )
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()
    return policy


def _build_cases(source_root: Path, manifest_path: Path) -> tuple[EvaluationCase, ...]:
    manifest = load_split_manifest(manifest_path)
    schedule = ScenarioSeedSchedule(manifest)
    source_by_name = {path.name: path for path in source_root.rglob("*.txt")}
    selected = [
        ref
        for group in sorted(manifest.groups)[:4]
        for ref in manifest.groups[group]["train"][:3]
    ]
    cases = []
    for ref in selected:
        source = source_by_name[f"{ref.original_instance_id}.txt"]
        ds1 = parse_ku_crptw(source)
        ds2 = merge_adjacent_batches(ds1)
        seeds = schedule.seeds("train", ref.base_instance_id, 3)
        for dataset, instance in (("DS1", ds1), ("DS2", ds2)):
            cases.append(
                EvaluationCase(
                    instance=instance,
                    dataset=dataset,
                    split="train",
                    base_instance_id=ref.base_instance_id,
                    parameter_group=ref.parameter_group,
                    scenario_seeds=seeds,
                )
            )
    return tuple(cases)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=Path("experiments/splits/scrp_split_v1.json")
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("scrp/trained_models/scrp_phase_2_5_sanity.pt"),
    )
    parser.add_argument(
        "--raw-output", type=Path,
        default=Path("experiments/raw_results/phase4_development.jsonl"),
    )
    parser.add_argument(
        "--summary-output", type=Path,
        default=Path("experiments/summaries/phase4_development_summary.json"),
    )
    args = parser.parse_args()

    cases = _build_cases(args.source_root, args.manifest)
    algorithms = (
        BaselineAlgorithm(RandomLegalBaseline, action_seed_root=40),
        BaselineAlgorithm(MinBlockingGreedyBaseline),
        BaselineAlgorithm(ERIBaseline),
        LowPolicyAlgorithm(_load_policy(args.checkpoint)),
    )
    result_sets = tuple(evaluate_algorithm_on_schedule(a, cases) for a in algorithms)
    assert_paired_scenarios(*result_sets)
    all_results = tuple(result for results in result_sets for result in results)
    save_raw_results(all_results, args.raw_output)

    payload = {
        "status": "development_integration_only",
        "performance_claim": False,
        "base_layout_count": len({case.base_instance_id for case in cases}),
        "static_artifact_count": len(cases),
        "scenarios_per_artifact": 3,
        "algorithms": [algorithm.name for algorithm in algorithms],
        "scenario_results_per_algorithm": len(result_sets[0]),
        "paired_scenario_ids": True,
        "all_terminated": all(result.terminated for result in all_results),
        "invalid_action_count": 0,
        "truncated_count": sum(result.truncated for result in all_results),
        "summaries": [asdict(item) for item in aggregate_relocations(all_results)],
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
