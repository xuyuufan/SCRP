"""Transparent public-information greedy baseline for SCRP."""

from __future__ import annotations

from fractions import Fraction
from typing import Dict, Tuple

from scrp.models import SCRPInstance, SCRPState

from .base import BaselineActionError


class MinBlockingGreedyBaseline:
    """Minimize newly introduced public-information blocking risk.

    For relocated blocker ``c`` and each container ``x`` already in destination
    stack ``s``, ``p(x < c)`` is 1 when public precedence proves that ``x`` is
    retrieved first, 0 when it proves the opposite, and 1/2 when both are in the
    same unrevealed batch. The primary score is ``sum_x p(x < c)``.

    Ties minimize free slots after the move (pack an already used stack and
    preserve flexible capacity), then minimize the stable stack ID. This is a
    project baseline, not a claimed reproduction of ERI, EM, EG, or RIRH.
    """

    name = "min_blocking_greedy_v1"

    def reset(self, action_seed: int) -> None:
        if isinstance(action_seed, bool) or not isinstance(action_seed, int):
            raise ValueError("action_seed must be an integer")

    def select_destination(
        self,
        instance: SCRPInstance,
        state: SCRPState,
        legal_destinations: Tuple[int, ...],
    ) -> int:
        if not legal_destinations:
            raise BaselineActionError("non-terminal decision has no legal destination")
        if state.current_target_id is None or state.terminated:
            raise BaselineActionError("baseline selection requires a live decision state")

        target_location = state.locations[state.current_target_id]
        if target_location is None:
            raise BaselineActionError("current target has no live location")
        blocker_id = state.stacks[target_location.stack_id].top_id
        batch_rank = {batch_id: rank for rank, batch_id in enumerate(instance.batch_order)}
        revealed_rank: Dict[int, Dict[int, int]] = {
            batch_id: {container_id: rank for rank, container_id in enumerate(order)}
            for batch_id, order in state.revealed_orders.items()
        }

        def precedes_probability(lower_id: int) -> Fraction:
            lower_batch = instance.container_by_id[lower_id].batch_id
            blocker_batch = instance.container_by_id[blocker_id].batch_id
            if batch_rank[lower_batch] < batch_rank[blocker_batch]:
                return Fraction(1)
            if batch_rank[lower_batch] > batch_rank[blocker_batch]:
                return Fraction(0)
            order = revealed_rank.get(lower_batch)
            if order is None:
                return Fraction(1, 2)
            return Fraction(int(order[lower_id] < order[blocker_id]))

        legal = tuple(legal_destinations)
        if len(set(legal)) != len(legal):
            raise BaselineActionError("legal destinations must be unique")
        try:
            return min(
                legal,
                key=lambda destination: (
                    sum(
                        (precedes_probability(container_id)
                         for container_id in state.stacks[destination].containers),
                        Fraction(0),
                    ),
                    instance.max_tiers - (state.stacks[destination].height + 1),
                    destination,
                ),
            )
        except (IndexError, KeyError) as error:
            raise BaselineActionError("public state/instance mismatch") from error
