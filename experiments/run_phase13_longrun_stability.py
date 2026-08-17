"""Run the Phase 13 CUDA smoke gate or frozen long-run experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from experiments.formal_run import CachedKuProvider, atomic_write_json
from experiments.phase13_longrun_stability import (
    load_phase13_protocol,
    run_longrun_stability,
    run_phase13_smoke,
)
from experiments.protocol import load_split_manifest


def _code_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "run"), required=True)
    parser.add_argument("--source-root", type=Path, default=Path("tmp/StochasticCRP/crptw_instance"))
    parser.add_argument("--protocol", type=Path, default=Path("experiments/configs/phase13_longrun_stability_v1.json"))
    parser.add_argument("--smoke-output", type=Path, default=Path("tmp/phase13_cuda_smoke.json"))
    parser.add_argument("--output", type=Path, default=Path("experiments/summaries/phase13_longrun_stability.json"))
    parser.add_argument("--checkpoint-probe-dir", type=Path, default=Path("tmp/phase13-smoke-checkpoint"))
    args = parser.parse_args()
    _, _, config = load_phase13_protocol(args.protocol)
    manifest = load_split_manifest("experiments/splits/scrp_split_v1.json")
    provider = CachedKuProvider(args.source_root)
    if args.mode == "smoke":
        result = run_phase13_smoke(manifest, provider, config, args.checkpoint_probe_dir)
        destination = args.smoke_output
    else:
        if not args.smoke_output.exists():
            raise FileNotFoundError("Phase 13 smoke record is required before the formal run")
        result = run_longrun_stability(
            manifest, provider, config, code_sha=_code_sha(),
            smoke_record=json.loads(args.smoke_output.read_text(encoding="utf-8")),
        )
        destination = args.output
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite Phase 13 artifact {destination}")
    atomic_write_json(destination, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
