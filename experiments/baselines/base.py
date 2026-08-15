"""Public-state-only interface shared by all SCRP baselines."""

from __future__ import annotations

from typing import Protocol, Tuple, runtime_checkable

from scrp.models import SCRPInstance, SCRPState


class BaselineActionError(RuntimeError):
    """Raised when a baseline violates the legal-destination contract."""


@runtime_checkable
class SCRPBaseline(Protocol):
    """Thin baseline contract with no access to Scenario or environment internals.

    ``instance`` is immutable public static data. ``state`` is the detached public
    state returned by :class:`SCRPEnvironment`, and ``legal_destinations`` is the
    environment-computed legal action set. The action RNG is reset independently
    of the scenario RNG at the start of every episode.
    """

    name: str

    def reset(self, action_seed: int) -> None:
        """Reset algorithm-side randomness for one episode."""

    def select_destination(
        self,
        instance: SCRPInstance,
        state: SCRPState,
        legal_destinations: Tuple[int, ...],
    ) -> int:
        """Select exactly one member of ``legal_destinations``."""
