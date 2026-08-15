"""Non-training diagnostic for O1 information loss and O2 coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scrp import (
    Container,
    O1ObservationAdapter,
    O2ObservationAdapter,
    SCRPConfig,
    SCRPEnvironment,
    SCRPInstance,
    SCRP_O2_MMAX,
    Scenario,
    load_instance_json,
    merge_adjacent_batches,
    parse_ku_crptw,
)


class _FixedSampler:
    def __init__(self, order):
        self.order = order

    def sample(self, instance, root_seed):
        return Scenario(
            root_seed,
            {1: root_seed + 1, 2: root_seed + 2},
            {1: self.order, 2: (5,)},
            f"phase5-diagnostic-{root_seed}",
        )


def _collision_observations(order):
    instance = SCRPInstance(
        "phase5-o1-collision",
        3,
        4,
        tuple(
            Container(container_id, 1 if container_id <= 4 else 2)
            for container_id in range(1, 6)
        ),
        ((1, 5), (2, 3, 4), ()),
        (1, 2),
    )
    config = SCRPConfig(3, 4)
    env = SCRPEnvironment(config, instance, _FixedSampler(order))
    state = env.reset(seed=20260816)
    o1 = O1ObservationAdapter(instance, config).build(state)
    o2_adapter = O2ObservationAdapter(instance, config)
    o2 = o2_adapter.build(state)
    return o1, o2, o2_adapter


def _audit_mmax(source_root: Path):
    files = sorted(source_root.rglob("*.txt"))
    ds1_max = 0
    ds2_max = 0
    for path in files:
        ds1 = parse_ku_crptw(path)
        ds2 = merge_adjacent_batches(ds1)
        ds1_max = max(ds1_max, max(ds1.batch_sizes.values()))
        ds2_max = max(ds2_max, max(ds2.batch_sizes.values()))
    return len(files), ds1_max, ds2_max


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/summaries/phase5_o2_diagnostic.json"),
    )
    args = parser.parse_args()

    file_count, ds1_max, ds2_max = _audit_mmax(args.source_root)
    o1_a, o2_a, collision_adapter = _collision_observations((1, 2, 3, 4))
    o1_b, o2_b, _ = _collision_observations((1, 2, 4, 3))

    artifacts = []
    for dataset, path in (
        ("DS1", Path("data/phase3_sanity/S05_T03_mu050/ds1_001.json")),
        ("DS2", Path("data/phase3_sanity/S07_T04_mu067/ds2_001.json")),
    ):
        instance = load_instance_json(path)
        config = SCRPConfig(instance.num_stacks, instance.max_tiers)
        state = SCRPEnvironment(config, instance).reset(seed=20260816)
        o1 = O1ObservationAdapter(instance, config).build(state)
        o2_adapter = O2ObservationAdapter(instance, config)
        o2 = o2_adapter.build(state)
        nodes = o2.reshape(o2_adapter.node_shape)
        order_nodes = nodes[
            instance.num_stacks:instance.num_stacks + SCRP_O2_MMAX
        ]
        artifacts.append(
            {
                "dataset": dataset,
                "instance_id": instance.instance_id,
                "max_batch_size": max(instance.batch_sizes.values()),
                "O1_shape": list(o1.shape),
                "O2_shape": list(o2.shape),
                "real_revealed_nodes": int(np.sum(order_nodes[:, 11] == 0.0)),
                "padding_nodes": int(np.sum(order_nodes[:, 11] == 1.0)),
            }
        )

    payload = {
        "status": "diagnostic_only",
        "training_performed": False,
        "performance_claim": False,
        "source_instance_count": file_count,
        "Mmax_DS1": ds1_max,
        "Mmax_DS2": ds2_max,
        "Mmax_combined": max(ds1_max, ds2_max),
        "configured_Mmax": SCRP_O2_MMAX,
        "collision": {
            "physical_bay_identical": True,
            "current_target_identical": True,
            "revealed_orders_different": True,
            "O1_shape": list(o1_a.shape),
            "O2_shape": list(o2_a.shape),
            "O1_identical": bool(np.array_equal(o1_a, o1_b)),
            "O2_identical": bool(np.array_equal(o2_a, o2_b)),
            "O2_node_shape": list(collision_adapter.node_shape),
        },
        "artifacts": artifacts,
    }
    if file_count != 1_440 or (ds1_max, ds2_max) != (4, 6):
        raise AssertionError("dataset audit no longer matches frozen O2 bounds")
    if not payload["collision"]["O1_identical"] or payload["collision"]["O2_identical"]:
        raise AssertionError("O1/O2 collision diagnostic failed")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
