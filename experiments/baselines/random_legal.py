"""Uniform random selection over the current legal destinations."""

from __future__ import annotations

import random
from typing import Tuple

from scrp.models import SCRPInstance, SCRPState

from .base import BaselineActionError


class RandomLegalBaseline:
    """Choose uniformly from legal destinations using an action-only RNG."""

    name = "random_legal_v1"

    def __init__(self) -> None:
        self._rng: random.Random | None = None

    def reset(self, action_seed: int) -> None:
        if isinstance(action_seed, bool) or not isinstance(action_seed, int):
            raise ValueError("action_seed must be an integer")
        self._rng = random.Random(action_seed)

    def select_destination(
        self,
        instance: SCRPInstance,
        state: SCRPState,
        legal_destinations: Tuple[int, ...],
    ) -> int:
        del instance, state
        if self._rng is None:
            raise RuntimeError("reset(action_seed) must be called before selection")
        if not legal_destinations:
            raise BaselineActionError("non-terminal decision has no legal destination")
        return self._rng.choice(legal_destinations)
