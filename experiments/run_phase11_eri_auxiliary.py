"""Run the fixed train/validation-only Phase 11 comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.formal_run import CachedKuProvider, atomic_write_json
from experiments.phase11_eri_auxiliary import run_comparison
from experiments.protocol import load_split_manifest
from scrp.formal_training import load_formal_training_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("tmp/StochasticCRP/crptw_instance"))
    parser.add_argument("--config", type=Path, default=Path("experiments/configs/phase11_eri_aux_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("experiments/summaries/phase11_eri_auxiliary.json"))
    parser.add_argument("--checkpoint-probe-dir", type=Path, default=Path("tmp/phase11-smoke-checkpoint"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite Phase 11 result {args.output}")
    manifest = load_split_manifest("experiments/splits/scrp_split_v1.json")
    config = load_formal_training_config(args.config)
    result = run_comparison(
        manifest, CachedKuProvider(args.source_root), config, args.checkpoint_probe_dir
    )
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
