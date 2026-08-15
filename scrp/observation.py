"""Phase 2 O1 observation adapter for SCRP integration smoke tests.

O1 is deliberately an approximation: it summarizes the currently revealed
order but does not encode the full revealed permutation losslessly. It is not
a final full-information SCRP observation; an O2/sequence representation
remains future work.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .models import SCRPConfig, SCRPInstance, SCRPState


SCRP_O2_MMAX = 6
SCRP_O2_FEATURE_NAMES = (
    "node_type",
    "node_index_or_revealed_rank",
    "stack_index_or_height",
    "free_space_or_tier",
    "top_batch_or_blockers_above",
    "contains_target_or_current_batch_rank",
    "top_is_target_or_is_current_target",
    "target_tier_or_is_top",
    "target_blockers_or_stack_height",
    "current_batch_fraction_or_free_space",
    "earliest_rank_or_remaining_rank",
    "padding",
)
# Every O2 value is deliberately normalized to [0, 1]. Keeping this explicit
# prevents future features from silently inheriting the old O1 all-ones scale.
SCRP_O2_FEATURE_SCALE = (1.0,) * len(SCRP_O2_FEATURE_NAMES)
# Public Phase 5 name used by experiment code and protocol documentation.
O2_FEATURE_SCALE = SCRP_O2_FEATURE_SCALE


@dataclass(frozen=True)
class SCRPObservationConfig:
    """Serializable observation-shape contract, independent of checkpoints."""

    observation_version: str = "O1"
    feature_dim: int = 12
    mmax: int | None = None
    dataset_version: str = "unspecified"

    def __post_init__(self) -> None:
        if self.observation_version not in {"O1", "O2"}:
            raise ValueError("observation_version must be O1 or O2")
        if self.feature_dim != 12:
            raise ValueError("current SCRP observation feature_dim must be 12")
        if self.observation_version == "O1" and self.mmax is not None:
            raise ValueError("O1 does not define Mmax")
        if self.observation_version == "O2" and (
            isinstance(self.mmax, bool) or not isinstance(self.mmax, int) or self.mmax <= 0
        ):
            raise ValueError("O2 requires a positive integer Mmax")
        if not self.dataset_version:
            raise ValueError("dataset_version must be non-empty")

    def to_record(self) -> dict[str, Any]:
        return {
            "observation_version": self.observation_version,
            "feature_dim": self.feature_dim,
            "Mmax": self.mmax,
            "dataset_version": self.dataset_version,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "SCRPObservationConfig":
        expected = {"observation_version", "feature_dim", "Mmax", "dataset_version"}
        if set(record) != expected:
            raise ValueError("observation config keys mismatch")
        return cls(
            observation_version=str(record["observation_version"]),
            feature_dim=int(record["feature_dim"]),
            mmax=None if record["Mmax"] is None else int(record["Mmax"]),
            dataset_version=str(record["dataset_version"]),
        )


def save_observation_config(
    config: SCRPObservationConfig, path: str | Path
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(config.to_record(), indent=2) + "\n", encoding="utf-8"
    )
    return destination


def load_observation_config(path: str | Path) -> SCRPObservationConfig:
    source = Path(path)
    try:
        record = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read observation config {source}: {error}") from error
    return SCRPObservationConfig.from_record(record)


class O1ObservationAdapter:
    """Encode S stack nodes plus one context node, with 12 features each."""

    FEATURES_PER_NODE = 12
    CONTEXT_MARKER = 2.0
    VERSION_MARKER = 1.0
    VERSION = "O1"

    def __init__(self, instance: SCRPInstance, config: SCRPConfig) -> None:
        self.instance = instance
        self.config = config
        self._batch_rank = {
            batch_id: rank for rank, batch_id in enumerate(instance.batch_order)
        }

    @property
    def shape(self) -> tuple[int]:
        return ((self.instance.num_stacks + 1) * self.FEATURES_PER_NODE,)

    def build(self, state: SCRPState) -> np.ndarray:
        nodes = [self._stack_node(state, stack_id) for stack_id in range(self.instance.num_stacks)]
        nodes.append(self._context_node(state))
        observation = np.asarray(nodes, dtype=np.float32).reshape(-1)
        if observation.shape != self.shape:
            raise RuntimeError(
                f"O1 observation shape mismatch: expected {self.shape}, got {observation.shape}"
            )
        if not np.isfinite(observation).all():
            raise RuntimeError("O1 observation contains NaN or infinity")
        return observation

    def _current_batch(self, state: SCRPState):
        if state.terminated or state.current_batch_index >= self.instance.num_batches:
            return None, (), 0
        batch_id = self.instance.batch_order[state.current_batch_index]
        revealed_order = state.revealed_orders[batch_id]
        return batch_id, revealed_order, len(revealed_order)

    def _stack_node(self, state: SCRPState, stack_id: int) -> list[float]:
        stack = state.stacks[stack_id]
        S = self.instance.num_stacks
        T = self.instance.max_tiers
        K = self.instance.num_batches
        batch_id, revealed_order, batch_size = self._current_batch(state)
        target_id = state.current_target_id
        target_location = state.locations[target_id] if target_id is not None else None

        stack_id_norm = stack_id / max(S - 1, 1)
        height_norm = stack.height / T
        free_space_norm = (T - stack.height) / T

        if stack.is_empty:
            top_batch_norm = 0.0
        else:
            top_batch_id = self.instance.container_by_id[stack.top_id].batch_id
            top_batch_norm = (self._batch_rank[top_batch_id] + 1) / K

        contains_target = target_id is not None and target_id in stack.containers
        top_is_target = contains_target and stack.top_id == target_id
        if contains_target:
            target_tier = stack.containers.index(target_id)
            blockers_above = stack.height - target_tier - 1
            target_tier_norm = target_tier / max(T - 1, 1)
            blockers_norm = blockers_above / T
        else:
            target_tier_norm = 0.0
            blockers_norm = 0.0

        if batch_id is None:
            current_batch_count_norm = 0.0
            earliest_revealed_rank_norm = 1.0
        else:
            current_members = set(self.instance.containers_by_batch[batch_id])
            current_batch_count = sum(cid in current_members for cid in stack.containers)
            current_batch_count_norm = current_batch_count / batch_size
            remaining_rank = {
                container_id: rank
                for rank, container_id in enumerate(revealed_order)
                if rank >= state.order_position
            }
            ranks_in_stack = [
                remaining_rank[cid] for cid in stack.containers if cid in remaining_rank
            ]
            earliest_revealed_rank_norm = (
                min(ranks_in_stack) / max(batch_size - 1, 1)
                if ranks_in_stack else 1.0
            )

        disorder_pairs = 0
        for lower_index, lower_id in enumerate(stack.containers):
            lower_batch = self.instance.container_by_id[lower_id].batch_id
            for upper_id in stack.containers[lower_index + 1:]:
                upper_batch = self.instance.container_by_id[upper_id].batch_id
                if self._batch_rank[lower_batch] < self._batch_rank[upper_batch]:
                    disorder_pairs += 1
        max_pairs = max(T * (T - 1) / 2, 1)
        disorder_norm = disorder_pairs / max_pairs

        is_source = target_location is not None and target_location.stack_id == stack_id
        return [
            stack_id_norm,
            height_norm,
            free_space_norm,
            top_batch_norm,
            float(top_is_target),
            float(contains_target),
            target_tier_norm,
            blockers_norm,
            current_batch_count_norm,
            earliest_revealed_rank_norm,
            disorder_norm,
            float(is_source),
        ]

    def _context_node(self, state: SCRPState) -> list[float]:
        S = self.instance.num_stacks
        T = self.instance.max_tiers
        N = self.instance.num_containers
        K = self.instance.num_batches
        batch_id, revealed_order, batch_size = self._current_batch(state)
        target_id = state.current_target_id
        target_location = state.locations[target_id] if target_id is not None else None

        if batch_id is None:
            current_batch_norm = 1.0 if state.terminated else 0.0
            progress_norm = 1.0 if state.terminated else 0.0
            batch_remaining_norm = 0.0
            target_stack_norm = 0.0
            target_depth_norm = 0.0
            current_batch_size_norm = 0.0
            current_target_rank_norm = 0.0
            revealed_flag = 0.0
        else:
            current_batch_norm = state.current_batch_index / max(K - 1, 1)
            progress_norm = state.order_position / batch_size
            batch_remaining_norm = (batch_size - state.order_position) / batch_size
            target_stack_norm = target_location.stack_id / max(S - 1, 1)
            target_stack = state.stacks[target_location.stack_id]
            blockers = target_stack.height - target_location.tier - 1
            target_depth_norm = blockers / T
            current_batch_size_norm = batch_size / N
            current_target_rank_norm = state.order_position / max(batch_size - 1, 1)
            revealed_flag = float(batch_id in state.revealed_orders)

        relocation_count_norm = min(state.relocation_count / self.config.max_steps, 1.0)
        total_remaining_norm = (N - state.retrieval_count) / N
        return [
            self.CONTEXT_MARKER,
            current_batch_norm,
            progress_norm,
            batch_remaining_norm,
            target_stack_norm,
            target_depth_norm,
            relocation_count_norm,
            total_remaining_norm,
            current_batch_size_norm,
            current_target_rank_norm,
            revealed_flag,
            self.VERSION_MARKER,
        ]


class O2ObservationAdapter:
    """Encode stack actions, full remaining revealed order, and context.

    Node layout is ``[S stack nodes] + [Mmax order nodes] + [1 context]``.
    Container IDs never appear as numerical features. An order container is
    identified losslessly within the public state by its unique stack/tier
    location together with its explicit revealed rank.
    """

    FEATURES_PER_NODE = len(SCRP_O2_FEATURE_NAMES)
    VERSION = "O2"
    STACK_NODE_TYPE = 0.0
    ORDER_NODE_TYPE = 0.5
    CONTEXT_NODE_TYPE = 1.0

    def __init__(
        self,
        instance: SCRPInstance,
        config: SCRPConfig,
        *,
        mmax: int = SCRP_O2_MMAX,
    ) -> None:
        if isinstance(mmax, bool) or not isinstance(mmax, int) or mmax <= 0:
            raise ValueError("mmax must be a positive integer")
        observed_max = max(instance.batch_sizes.values())
        if observed_max > mmax:
            raise ValueError(
                f"instance batch size {observed_max} exceeds O2 Mmax={mmax}"
            )
        self.instance = instance
        self.config = config
        self.mmax = mmax
        self._batch_rank = {
            batch_id: rank for rank, batch_id in enumerate(instance.batch_order)
        }

    @property
    def shape(self) -> tuple[int]:
        return (
            (self.instance.num_stacks + self.mmax + 1) * self.FEATURES_PER_NODE,
        )

    @property
    def node_shape(self) -> tuple[int, int]:
        return (
            self.instance.num_stacks + self.mmax + 1,
            self.FEATURES_PER_NODE,
        )

    def build(self, state: SCRPState) -> np.ndarray:
        nodes = [
            self._stack_node(state, stack_id)
            for stack_id in range(self.instance.num_stacks)
        ]
        remaining = self._remaining_revealed_order(state)
        nodes.extend(
            self._order_node(state, container_id, remaining_index, len(remaining))
            for remaining_index, container_id in enumerate(remaining)
        )
        nodes.extend(self._padding_node() for _ in range(self.mmax - len(remaining)))
        nodes.append(self._context_node(state, len(remaining)))
        observation = np.asarray(nodes, dtype=np.float32).reshape(-1)
        if observation.shape != self.shape:
            raise RuntimeError(
                f"O2 observation shape mismatch: expected {self.shape}, got {observation.shape}"
            )
        if not np.isfinite(observation).all():
            raise RuntimeError("O2 observation contains NaN or infinity")
        if np.any(observation < 0.0) or np.any(observation > 1.0):
            raise RuntimeError("O2 observation features must remain in [0, 1]")
        return observation

    def _current_batch(self, state: SCRPState):
        if state.terminated or state.current_batch_index >= self.instance.num_batches:
            return None, (), 0
        batch_id = self.instance.batch_order[state.current_batch_index]
        revealed_order = state.revealed_orders[batch_id]
        return batch_id, revealed_order, len(revealed_order)

    def _remaining_revealed_order(self, state: SCRPState) -> tuple[int, ...]:
        _, revealed_order, _ = self._current_batch(state)
        if not revealed_order:
            return ()
        return tuple(revealed_order[state.order_position:])

    def _stack_node(self, state: SCRPState, stack_id: int) -> list[float]:
        stack = state.stacks[stack_id]
        S = self.instance.num_stacks
        T = self.instance.max_tiers
        K = self.instance.num_batches
        batch_id, revealed_order, batch_size = self._current_batch(state)
        target_id = state.current_target_id
        target_location = state.locations[target_id] if target_id is not None else None

        if stack.is_empty:
            top_batch_norm = 0.0
        else:
            top_batch = self.instance.container_by_id[stack.top_id].batch_id
            top_batch_norm = (self._batch_rank[top_batch] + 1) / K

        contains_target = target_id is not None and target_id in stack.containers
        top_is_target = contains_target and stack.top_id == target_id
        if contains_target:
            target_tier = stack.containers.index(target_id)
            target_tier_norm = target_tier / max(T - 1, 1)
            target_blockers_norm = (stack.height - target_tier - 1) / T
        else:
            target_tier_norm = 0.0
            target_blockers_norm = 0.0

        if batch_id is None:
            current_fraction = 0.0
            earliest_rank_norm = 1.0
        else:
            current_members = set(self.instance.containers_by_batch[batch_id])
            current_fraction = sum(
                container_id in current_members for container_id in stack.containers
            ) / batch_size
            remaining_rank = {
                container_id: rank
                for rank, container_id in enumerate(revealed_order)
                if rank >= state.order_position
            }
            ranks = [
                remaining_rank[container_id]
                for container_id in stack.containers
                if container_id in remaining_rank
            ]
            earliest_rank_norm = (
                min(ranks) / max(batch_size - 1, 1) if ranks else 1.0
            )

        return [
            self.STACK_NODE_TYPE,
            stack_id / max(S - 1, 1),
            stack.height / T,
            (T - stack.height) / T,
            top_batch_norm,
            float(contains_target),
            float(top_is_target),
            target_tier_norm,
            target_blockers_norm,
            current_fraction,
            earliest_rank_norm,
            0.0,
        ]

    def _order_node(
        self,
        state: SCRPState,
        container_id: int,
        remaining_index: int,
        remaining_count: int,
    ) -> list[float]:
        location = state.locations[container_id]
        if location is None:
            raise RuntimeError("remaining revealed container has no live location")
        stack = state.stacks[location.stack_id]
        batch_id, revealed_order, batch_size = self._current_batch(state)
        if batch_id is None:
            raise RuntimeError("order node requested without a current batch")
        full_rank = revealed_order.index(container_id)
        return [
            self.ORDER_NODE_TYPE,
            full_rank / max(batch_size - 1, 1),
            location.stack_id / max(self.instance.num_stacks - 1, 1),
            location.tier / max(self.instance.max_tiers - 1, 1),
            (stack.height - location.tier - 1) / self.instance.max_tiers,
            self._batch_rank[batch_id] / max(self.instance.num_batches - 1, 1),
            float(container_id == state.current_target_id),
            float(location.tier == stack.height - 1),
            stack.height / self.instance.max_tiers,
            (self.instance.max_tiers - stack.height) / self.instance.max_tiers,
            remaining_index / max(remaining_count - 1, 1),
            0.0,
        ]

    def _padding_node(self) -> list[float]:
        return [self.ORDER_NODE_TYPE] + [0.0] * 10 + [1.0]

    def _context_node(self, state: SCRPState, remaining_count: int) -> list[float]:
        S = self.instance.num_stacks
        T = self.instance.max_tiers
        N = self.instance.num_containers
        K = self.instance.num_batches
        batch_id, revealed_order, batch_size = self._current_batch(state)
        target_id = state.current_target_id
        target_location = state.locations[target_id] if target_id is not None else None
        if batch_id is None:
            current_batch_norm = 1.0 if state.terminated else 0.0
            progress_norm = 1.0 if state.terminated else 0.0
            remaining_fraction = 0.0
            target_stack_norm = 0.0
            target_depth_norm = 0.0
            current_batch_size_norm = 0.0
            remaining_count_norm = 0.0
            revealed_flag = 0.0
        else:
            current_batch_norm = state.current_batch_index / max(K - 1, 1)
            progress_norm = state.order_position / batch_size
            remaining_fraction = remaining_count / batch_size
            target_stack_norm = target_location.stack_id / max(S - 1, 1)
            target_stack = state.stacks[target_location.stack_id]
            target_depth_norm = (
                target_stack.height - target_location.tier - 1
            ) / T
            current_batch_size_norm = batch_size / self.mmax
            remaining_count_norm = remaining_count / self.mmax
            revealed_flag = float(batch_id in state.revealed_orders)

        return [
            self.CONTEXT_NODE_TYPE,
            current_batch_norm,
            progress_norm,
            remaining_fraction,
            target_stack_norm,
            target_depth_norm,
            min(state.relocation_count / self.config.max_steps, 1.0),
            (N - state.retrieval_count) / N,
            current_batch_size_norm,
            remaining_count_norm,
            revealed_flag,
            0.0,
        ]
