"""Run the Phase 10 train/validation-only order-aware development prototype."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.formal_run import CachedKuProvider, atomic_write_json
from experiments.order_aware_prototype import PHASE10_RUN_ID, run_development_comparison
from experiments.protocol import load_split_manifest
from scrp.formal_training import load_formal_training_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root", type=Path,
        default=Path("tmp/StochasticCRP/crptw_instance"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("experiments/summaries/phase10_order_aware_prototype.json"),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite Phase 10 output {args.output}")

    manifest = load_split_manifest("experiments/splits/scrp_split_v1.json")
    config = load_formal_training_config(
        "experiments/configs/training_protocol_v1_candidate.json"
    )
    provider = CachedKuProvider(args.source_root)
    comparison = run_development_comparison(manifest, provider, config)
    payload = {
        "run_id": PHASE10_RUN_ID,
        "status": "DEVELOPMENT_ONLY",
        "base_protocol": "Phase 7B optimizer/hyperparameters/FGB unchanged",
        "primary_change": "order-aware cross-attention only",
        "ERI_auxiliary_objective_used": False,
        "FGB_semantics_changed": False,
        "splits_used": ["train", "validation"],
        "formal_test_raw_rows_accessed": False,
        "formal_test_evaluation_performed": False,
        "formal_test_split_used": False,
        "development_checkpoints_saved": False,
        "cleanup": {"temporary_checkpoints": [], "models_retained": 0},
        **comparison,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
