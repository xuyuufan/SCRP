"""Deterministic, action-path-independent hidden-order sampling."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, Mapping, Tuple

from .models import SCRPInstance


@dataclass(frozen=True)
class Scenario:
    root_seed: int
    order_seeds: Mapping[int, int]
    hidden_orders: Mapping[int, Tuple[int, ...]]
    scenario_id: str

    def __post_init__(self) -> None:
        # Copy first so callers retaining the input dictionaries cannot mutate
        # the scenario through an alias after construction.
        immutable_seeds = MappingProxyType(dict(self.order_seeds))
        immutable_orders = MappingProxyType({
            batch_id: tuple(order)
            for batch_id, order in self.hidden_orders.items()
        })
        object.__setattr__(self, "order_seeds", immutable_seeds)
        object.__setattr__(self, "hidden_orders", immutable_orders)


class ScenarioSampler:
    """Pre-sample one independent uniform permutation per ordered batch.

    ``random.Random(seed).shuffle`` is reproducible for the same Python
    implementation/version used by an experiment. If future benchmarks need
    bit-for-bit reproduction across Python versions, the permutation algorithm
    and RNG implementation must be fixed explicitly instead of relying on the
    standard-library implementation.
    """

    @staticmethod
    def derive_order_seed(root_seed: int, batch_id: int) -> int:
        payload = f"scrp-phase1|root={root_seed}|order|batch={batch_id}".encode("utf-8")
        return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")

    def sample(self, instance: SCRPInstance, root_seed: int) -> Scenario:
        order_seeds: Dict[int, int] = {}
        hidden_orders: Dict[int, Tuple[int, ...]] = {}

        for batch_id in instance.batch_order:
            order_seed = self.derive_order_seed(root_seed, batch_id)
            members = sorted(instance.containers_by_batch[batch_id])
            rng = random.Random(order_seed)
            rng.shuffle(members)
            order_seeds[batch_id] = order_seed
            hidden_orders[batch_id] = tuple(members)

        identity = {
            "instance_id": instance.instance_id,
            "root_seed": root_seed,
            "hidden_orders": [
                [batch_id, list(hidden_orders[batch_id])]
                for batch_id in instance.batch_order
            ],
        }
        encoded = json.dumps(identity, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        scenario_id = hashlib.sha256(encoded).hexdigest()
        return Scenario(
            root_seed=root_seed,
            order_seeds=order_seeds,
            hidden_orders=hidden_orders,
            scenario_id=scenario_id,
        )
