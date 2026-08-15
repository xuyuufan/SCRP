"""Thin Phase 2 RL compatibility adapter around the exact SCRP core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .environment import SCRPEnvironment
from .models import SCRPState
from .observation import O1ObservationAdapter


@dataclass(frozen=True)
class _DiscreteActionSpace:
    """Minimal subset of Gymnasium Discrete used by the current runner."""

    n: int


class SCRPRLAdapter:
    """Expose the Phase 1 core through the runner's reset/step protocol.

    Every SCRP decision is a LOW-style destination decision. The adapter does
    not add ``_mode`` or recreate HIGH/source selection. Current Phase 2 smoke
    integration calls HierPolicyNetwork with ``mode="low"`` explicitly.

    O1 is a lossy revealed-order summary intended only for integration tests;
    it is not the final full-information SCRP observation.
    """

    def __init__(
        self,
        core_env: SCRPEnvironment,
        observation_adapter: Optional[O1ObservationAdapter] = None,
    ) -> None:
        self.core_env = core_env
        self.observation_adapter = observation_adapter or O1ObservationAdapter(
            core_env.instance, core_env.config
        )
        self.action_space = _DiscreteActionSpace(core_env.instance.num_stacks)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        state = self.core_env.reset(seed=seed)
        return self.observation_adapter.build(state), self._build_info(state)

    def step(self, action: int):
        # StepLimitError intentionally propagates: Phase 2 smoke integration
        # must not confuse a safety-limit failure with normal truncation.
        result = self.core_env.step(action)
        observation = self.observation_adapter.build(result.state)
        info = self._build_info(result.state)
        return observation, result.reward, result.terminated, False, info

    def get_action_mask(self) -> np.ndarray:
        state = self.core_env.state
        mask = np.zeros(self.action_space.n, dtype=np.bool_)
        if not state.terminated:
            legal = self.core_env.legal_destinations()
            mask[list(legal)] = True
            if not mask.any():
                raise RuntimeError("non-terminal SCRP state has no legal action")
        return mask

    def get_metrics(self) -> Dict[str, Any]:
        state = self.core_env.state
        return {
            "shifters": state.relocation_count,
            "relocation_count": state.relocation_count,
            "retrieval_count": state.retrieval_count,
            "total_reward": state.total_reward,
            "terminated": state.terminated,
            "scenario_id": self.core_env.scenario_id,
        }

    def get_state_snapshot(self) -> Dict[str, Any]:
        state = self.core_env.state
        current_batch = (
            None
            if state.terminated
            else self.core_env.instance.batch_order[state.current_batch_index]
        )
        return {
            "stacks": [list(stack.containers) for stack in state.stacks],
            "locations": {
                str(container_id): (
                    None
                    if location is None
                    else {"stack_id": location.stack_id, "tier": location.tier}
                )
                for container_id, location in state.locations.items()
            },
            "current_batch": current_batch,
            "current_target_id": state.current_target_id,
            "revealed_orders": {
                str(batch_id): list(order)
                for batch_id, order in state.revealed_orders.items()
            },
            "relocation_count": state.relocation_count,
            "retrieval_count": state.retrieval_count,
            "total_reward": state.total_reward,
            "terminated": state.terminated,
            "scenario_id": self.core_env.scenario_id,
        }

    def _build_info(self, state: SCRPState) -> Dict[str, Any]:
        return {
            "action_mask": self.get_action_mask(),
            "terminated": state.terminated,
            "current_target_id": state.current_target_id,
        }
