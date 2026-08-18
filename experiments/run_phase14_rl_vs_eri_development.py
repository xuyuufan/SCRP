"""Run the frozen Phase 14 preflight or authorized development evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.formal_run import CachedKuProvider, atomic_write_json
from experiments.phase14_rl_vs_eri_development import (
    Phase14PreflightError,
    blocked_summary,
    checkpoint_preflight,
    load_phase14_protocol,
    run_development_evaluation,
)
from experiments.protocol import load_split_manifest


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "run"), required=True)
    parser.add_argument(
        "--protocol", type=Path,
        default=Path("experiments/configs/phase14_rl_vs_eri_development_v1.json"),
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("experiments/splits/scrp_split_v1.json")
    )
    parser.add_argument(
        "--source-root", type=Path, default=Path("tmp/StochasticCRP/crptw_instance")
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("experiments/summaries/phase14_rl_vs_eri_development.json"),
    )
    parser.add_argument(
        "--raw-output", type=Path,
        default=Path("experiments/raw_results/phase14_rl_vs_eri_development.jsonl"),
    )
    parser.add_argument(
        "--action-output", type=Path,
        default=Path("experiments/raw_results/phase14_rl_vs_eri_actions.jsonl"),
    )
    args = parser.parse_args()

    protocol = load_phase14_protocol(args.protocol)
    manifest = load_split_manifest(args.manifest)
    repository_root = Path.cwd()
    preflight = checkpoint_preflight(protocol, repository_root)
    if args.mode == "preflight" or not preflight["ready"]:
        summary = blocked_summary(protocol, manifest, preflight)
        atomic_write_json(args.output, summary)
        print(json.dumps(summary, indent=2))
        if args.mode == "run":
            raise Phase14PreflightError(summary["blocker"])
        return

    provider = CachedKuProvider(args.source_root)
    summary, rows, action_rows = run_development_evaluation(
        protocol, manifest, provider, repository_root
    )
    atomic_write_json(args.output, summary)
    _write_jsonl(args.raw_output, rows)
    _write_jsonl(args.action_output, action_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
