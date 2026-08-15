"""CLI for generating the frozen Phase 3.5 split manifest."""

from __future__ import annotations

import argparse

from .protocol import build_split_manifest, discover_ku_base_instances, save_split_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", help="path to the public crptw_instance directory")
    parser.add_argument("output", help="destination JSON manifest")
    parser.add_argument("--split-seed", type=int, default=35_2026)
    args = parser.parse_args()
    refs = discover_ku_base_instances(args.source_root)
    manifest = build_split_manifest(refs, split_seed=args.split_seed)
    save_split_manifest(manifest, args.output)
    print(
        f"saved {manifest.num_base_instances} base instances across "
        f"{manifest.num_groups} groups to {args.output}"
    )


if __name__ == "__main__":
    main()
