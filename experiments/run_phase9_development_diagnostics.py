"""Run Phase 9 post-test diagnostics without accessing formal-test raw rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.formal_run import CachedKuProvider, atomic_write_json, file_sha256
from experiments.posttest_analysis import (
    PHASE9_RUN_ID,
    audit_training_history,
    fixed_development_refs,
    run_checkpoint_baseline_diagnostic,
    run_eri_imitation_and_representation_diagnostic,
    run_o1_o2_development_ablation,
)
from experiments.protocol import load_split_manifest
from scrp.formal_training import load_formal_training_config


EXPECTED_CHECKPOINT_SHA256 = (
    "1dbcb20686840df3d392a89a66cb28b79a3a7531d4669300bf09818a714ed255"
)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root", type=Path,
        default=Path("tmp/StochasticCRP/crptw_instance"),
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path(
            "checkpoints/formal-o2-mixed-seed20260816-run1/best-validation.pt"
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("experiments/summaries/phase9_posttest_analysis.json"),
    )
    parser.add_argument("--ablation-training-episodes", type=int, default=400)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite Phase 9 output {args.output}")
    checkpoint_hash = file_sha256(args.checkpoint)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("frozen best-validation checkpoint hash mismatch")

    manifest = load_split_manifest("experiments/splits/scrp_split_v1.json")
    config = load_formal_training_config(
        "experiments/configs/training_protocol_v1_candidate.json"
    )
    provider = CachedKuProvider(args.source_root)
    train_refs = fixed_development_refs(manifest, "train")
    validation_refs = fixed_development_refs(manifest, "validation")

    summary_root = Path("experiments/summaries")
    training_audit = audit_training_history(
        _load_json(summary_root / "formal-o2-mixed-seed20260816-run1-training.json"),
        _load_json(summary_root / "formal-o2-mixed-seed20260816-run1-validation.json"),
        _load_json(summary_root / "formal-o2-mixed-seed20260816-run1-completion.json"),
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    imitation = run_eri_imitation_and_representation_diagnostic(
        manifest, provider, checkpoint, config,
        {"train": train_refs, "validation": validation_refs},
    )
    baseline = run_checkpoint_baseline_diagnostic(
        manifest, provider, checkpoint, config, validation_refs
    )
    ablation = run_o1_o2_development_ablation(
        manifest, provider, config, train_refs, validation_refs,
        training_episodes=args.ablation_training_episodes,
    )
    payload = {
        "run_id": PHASE9_RUN_ID,
        "status": "development_only_posttest_diagnostics",
        "splits_used": ["train", "validation"],
        "formal_test_raw_rows_accessed": False,
        "formal_test_evaluation_performed": False,
        "formal_model_training_performed": False,
        "development_ablation_models_saved": False,
        "checkpoint_episode": 15_000,
        "checkpoint_sha256": checkpoint_hash,
        "training_curve_audit": training_audit,
        "checkpoint_vs_frozen_baseline": baseline,
        "O1_vs_O2_development_ablation": ablation,
        "ERI_action_imitation_and_error_states": imitation,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
