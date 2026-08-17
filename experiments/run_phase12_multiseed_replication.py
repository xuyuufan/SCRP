"""Run Phase 12 CUDA gates or the frozen five-seed replication."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from experiments.formal_run import CachedKuProvider, atomic_write_json
from experiments.phase12_multiseed_replication import (
    load_phase12_protocol,
    run_cuda_smoke,
    run_multiseed_replication,
    run_timing_probe,
)
from experiments.protocol import load_split_manifest


def _code_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "timing", "run"), required=True)
    parser.add_argument("--source-root", type=Path, default=Path("tmp/StochasticCRP/crptw_instance"))
    parser.add_argument("--protocol", type=Path, default=Path("experiments/configs/phase12_multiseed_v1.json"))
    parser.add_argument("--smoke-output", type=Path, default=Path("tmp/phase12_cuda_smoke.json"))
    parser.add_argument("--timing-output", type=Path, default=Path("tmp/phase12_cuda_timing.json"))
    parser.add_argument("--output", type=Path, default=Path("experiments/summaries/phase12_multiseed_replication.json"))
    parser.add_argument("--checkpoint-probe-dir", type=Path, default=Path("tmp/phase12-smoke-checkpoint"))
    args = parser.parse_args()
    _, config = load_phase12_protocol(args.protocol)
    manifest = load_split_manifest("experiments/splits/scrp_split_v1.json")
    provider = CachedKuProvider(args.source_root)
    if args.mode == "smoke":
        result = run_cuda_smoke(manifest, provider, config, args.checkpoint_probe_dir)
        destination = args.smoke_output
    elif args.mode == "timing":
        result = run_timing_probe(manifest, provider, config)
        destination = args.timing_output
    else:
        if not args.smoke_output.exists() or not args.timing_output.exists():
            raise FileNotFoundError("smoke and timing records are required before the main run")
        result = run_multiseed_replication(
            manifest, provider, config, code_sha=_code_sha(),
            smoke_record=_load_json(args.smoke_output),
            timing_record=_load_json(args.timing_output),
        )
        destination = args.output
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite Phase 12 artifact {destination}")
    atomic_write_json(destination, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
