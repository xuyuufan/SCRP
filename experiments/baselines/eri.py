"""Clean-room reproduction of the published Expected Reshuffling Index."""

from __future__ import annotations

from fractions import Fraction
from typing import Dict, Tuple

from scrp.models import SCRPInstance, SCRPState

from .base import BaselineActionError


class ERIBaseline:
    """Select the legal destination with minimum published ERI score.

    The implementation follows the per-container equation stated by Ku and
    Arthanari and Galle et al. Publicly known earlier precedence contributes
    one, later precedence contributes zero, and an unresolved order within the
    same future batch contributes one-half. Ties prefer the tallest stack and
    then the smallest (leftmost) stack ID, as specified by Galle et al.
    """

    name = "eri_reproduction_v1"

    def reset(self, action_seed: int) -> None:
        """Validate the common API seed; ERI itself is deterministic."""

        if isinstance(action_seed, bool) or not isinstance(action_seed, int):
            raise ValueError("action_seed must be an integer")

    @staticmethod
    def _revealed_positions(state: SCRPState) -> Dict[int, Dict[int, int]]:
        return {
            batch_id: {
                container_id: position
                for position, container_id in enumerate(order)
            }
            for batch_id, order in state.revealed_orders.items()
        }

    @classmethod
    def destination_score(
        cls,
        instance: SCRPInstance,
        state: SCRPState,
        blocker_id: int,
        destination_stack_id: int,
    ) -> Fraction:
        """Return the exact public-information ERI score for one destination."""

        batch_position = {
            batch_id: position
            for position, batch_id in enumerate(instance.batch_order)
        }
        revealed_positions = cls._revealed_positions(state)
        try:
            blocker_batch = instance.container_by_id[blocker_id].batch_id
            blocker_batch_position = batch_position[blocker_batch]
            destination = state.stacks[destination_stack_id]
        except (IndexError, KeyError) as error:
            raise BaselineActionError("public state/instance mismatch") from error

        score = Fraction(0)
        for lower_id in destination.containers:
            try:
                lower_batch = instance.container_by_id[lower_id].batch_id
                lower_batch_position = batch_position[lower_batch]
            except KeyError as error:
                raise BaselineActionError("public state contains an unknown container") from error

            if lower_batch_position < blocker_batch_position:
                score += 1
            elif lower_batch_position == blocker_batch_position:
                revealed = revealed_positions.get(lower_batch)
                if revealed is None:
                    score += Fraction(1, 2)
                else:
                    try:
                        score += int(revealed[lower_id] < revealed[blocker_id])
                    except KeyError as error:
                        raise BaselineActionError(
                            "revealed order is incomplete for a compared batch"
                        ) from error
            # A later batch contributes zero.
        return score

    def select_destination(
        self,
        instance: SCRPInstance,
        state: SCRPState,
        legal_destinations: Tuple[int, ...],
    ) -> int:
        if not legal_destinations:
            raise BaselineActionError("non-terminal decision has no legal destination")
        if state.terminated or state.current_target_id is None:
            raise BaselineActionError("ERI selection requires a live decision state")
        if len(set(legal_destinations)) != len(legal_destinations):
            raise BaselineActionError("legal destinations must be unique")

        try:
            target_location = state.locations[state.current_target_id]
            if target_location is None:
                raise BaselineActionError("current target has no live location")
            blocker_id = state.stacks[target_location.stack_id].top_id
            return min(
                legal_destinations,
                key=lambda destination: (
                    self.destination_score(
                        instance, state, blocker_id, destination
                    ),
                    -state.stacks[destination].height,
                    destination,
                ),
            )
        except (IndexError, KeyError) as error:
            raise BaselineActionError("public state/instance mismatch") from error
