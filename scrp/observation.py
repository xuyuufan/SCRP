"""Phase 2 O1 observation adapter for SCRP integration smoke tests.

O1 is deliberately an approximation: it summarizes the currently revealed
order but does not encode the full revealed permutation losslessly. It is not
a final full-information SCRP observation; an O2/sequence representation
remains future work.
"""

from __future__ import annotations

import numpy as np

from .models import SCRPConfig, SCRPInstance, SCRPState


class O1ObservationAdapter:
    """Encode S stack nodes plus one context node, with 12 features each."""

    FEATURES_PER_NODE = 12
    CONTEXT_MARKER = 2.0
    VERSION_MARKER = 1.0

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
