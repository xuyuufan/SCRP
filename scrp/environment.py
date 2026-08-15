"""Exact Phase 1 SCRP transition core, without Gymnasium or RL dependencies."""

from __future__ import annotations

import copy
from typing import List, Optional, Protocol, Tuple

from .models import (
    EpisodeTerminatedError,
    EventKind,
    InvalidActionError,
    Location,
    NoLegalRelocationError,
    Phase,
    SCRPConfig,
    SCRPInstance,
    SCRPState,
    Stack,
    StateInvariantError,
    StepLimitError,
    TransitionEvent,
    TransitionResult,
)
from .scenario import Scenario, ScenarioSampler


class _ScenarioSamplerLike(Protocol):
    def sample(self, instance: SCRPInstance, root_seed: int) -> Scenario:
        ...


class SCRPEnvironment:
    """Restricted SCRP environment exposing only relocation destinations."""

    def __init__(
        self,
        config: SCRPConfig,
        instance: SCRPInstance,
        scenario_sampler: Optional[_ScenarioSamplerLike] = None,
    ) -> None:
        if config.num_stacks != instance.num_stacks:
            raise ValueError("config.num_stacks does not match instance")
        if config.max_tiers != instance.max_tiers:
            raise ValueError("config.max_tiers does not match instance")
        self.config = config
        self.instance = instance
        self._scenario_sampler = scenario_sampler or ScenarioSampler()
        self._scenario: Optional[Scenario] = None
        self._state: Optional[SCRPState] = None

    @property
    def state(self) -> SCRPState:
        """Return a detached visible-state copy; hidden future orders are absent."""
        if self._state is None:
            raise RuntimeError("reset must be called before reading state")
        return copy.deepcopy(self._state)

    @property
    def scenario_id(self) -> str:
        if self._scenario is None:
            raise RuntimeError("reset must be called before reading scenario_id")
        return self._scenario.scenario_id

    def reset(self, seed: Optional[int] = None) -> SCRPState:
        root_seed = self.config.root_seed if seed is None else seed
        previous_state = self._state
        previous_scenario = self._scenario
        try:
            scenario = self._scenario_sampler.sample(self.instance, root_seed)
            stacks = [
                Stack(stack_id, self.instance.max_tiers, list(container_ids))
                for stack_id, container_ids in enumerate(self.instance.initial_stacks)
            ]
            locations = {
                container.container_id: None for container in self.instance.containers
            }
            for stack in stacks:
                for tier, container_id in enumerate(stack.containers):
                    locations[container_id] = Location(stack.stack_id, tier)

            self._scenario = scenario
            self._state = SCRPState(
                stacks=stacks,
                locations=locations,
                current_batch_index=0,
                revealed_orders={},
                order_position=0,
                current_target_id=None,
                phase=Phase.NEEDS_RELOCATION,
                relocation_count=0,
                retrieval_count=0,
                step_count=0,
                total_reward=0,
                retrieved_order=[],
                terminated=False,
            )
            events: List[TransitionEvent] = []
            self._auto_advance(events)
            if self.config.validate_after_transition:
                self._validate_state()
            return self.state
        except Exception:
            self._state = previous_state
            self._scenario = previous_scenario
            raise

    def step(self, destination_stack_id: int) -> TransitionResult:
        state = self._require_state()
        if state.terminated:
            raise EpisodeTerminatedError("cannot step a terminated episode")
        if state.step_count >= self.config.max_steps:
            raise StepLimitError(f"max_steps={self.config.max_steps} reached")

        source_stack_id, _ = self._current_source_and_blocker()
        self._validate_destination(destination_stack_id, source_stack_id)

        rollback_state = copy.deepcopy(state)
        events: List[TransitionEvent] = []
        try:
            source = state.stacks[source_stack_id]
            destination = state.stacks[destination_stack_id]
            blocker_id = source.pop()
            destination.push(blocker_id)
            state.locations[blocker_id] = Location(
                stack_id=destination_stack_id,
                tier=destination.height - 1,
            )
            state.relocation_count += 1
            state.step_count += 1
            state.total_reward -= 1
            events.append(
                TransitionEvent(
                    kind=EventKind.RELOCATE,
                    container_id=blocker_id,
                    source_stack_id=source_stack_id,
                    destination_stack_id=destination_stack_id,
                )
            )

            self._auto_advance(events)
            if self.config.validate_after_transition:
                self._validate_state()
        except Exception:
            self._state = rollback_state
            raise

        return TransitionResult(
            reward=-1,
            terminated=state.terminated,
            events=tuple(events),
            state=self.state,
        )

    def legal_destinations(self) -> Tuple[int, ...]:
        state = self._require_state()
        if state.terminated:
            return ()
        source_stack_id, _ = self._current_source_and_blocker()
        return tuple(
            stack.stack_id
            for stack in state.stacks
            if stack.stack_id != source_stack_id and not stack.is_full
        )

    def _require_state(self) -> SCRPState:
        if self._state is None:
            raise RuntimeError("reset must be called before step")
        return self._state

    def _validate_destination(self, destination_stack_id: int, source_stack_id: int) -> None:
        state = self._require_state()
        if isinstance(destination_stack_id, bool) or not isinstance(destination_stack_id, int):
            raise InvalidActionError("destination_stack_id must be an integer")
        if not 0 <= destination_stack_id < self.instance.num_stacks:
            raise InvalidActionError(
                f"destination stack {destination_stack_id} is outside "
                f"0..{self.instance.num_stacks - 1}"
            )
        if destination_stack_id == source_stack_id:
            raise InvalidActionError("destination stack cannot equal source stack")
        if state.stacks[destination_stack_id].is_full:
            raise InvalidActionError(
                f"destination stack {destination_stack_id} is full"
            )

    def _current_source_and_blocker(self) -> Tuple[int, int]:
        state = self._require_state()
        target_id = state.current_target_id
        if target_id is None:
            raise StateInvariantError("non-terminal state has no current target")
        location = state.locations[target_id]
        if location is None:
            raise StateInvariantError("current target has already been retrieved")
        source = state.stacks[location.stack_id]
        blocker_id = source.top_id
        if blocker_id == target_id:
            raise StateInvariantError(
                "accessible target escaped automatic retrieval and reached decision state"
            )
        return source.stack_id, blocker_id

    def _auto_advance(self, events: List[TransitionEvent]) -> None:
        state = self._require_state()
        scenario = self._scenario
        if scenario is None:
            raise StateInvariantError("scenario is missing")

        while True:
            if state.retrieval_count == self.instance.num_containers:
                state.current_target_id = None
                state.phase = Phase.TERMINATED
                state.terminated = True
                events.append(TransitionEvent(kind=EventKind.TERMINATE))
                return

            if state.current_batch_index >= self.instance.num_batches:
                raise StateInvariantError(
                    "batch sequence exhausted before all containers were retrieved"
                )

            batch_id = self.instance.batch_order[state.current_batch_index]
            if batch_id not in state.revealed_orders:
                state.revealed_orders[batch_id] = tuple(scenario.hidden_orders[batch_id])
                state.order_position = 0
                events.append(
                    TransitionEvent(kind=EventKind.REVEAL_BATCH, batch_id=batch_id)
                )

            revealed_order = state.revealed_orders[batch_id]
            if not 0 <= state.order_position < len(revealed_order):
                raise StateInvariantError("order_position is outside current revealed order")
            target_id = revealed_order[state.order_position]
            state.current_target_id = target_id
            location = state.locations[target_id]
            if location is None:
                raise StateInvariantError("next target is already marked retrieved")
            source = state.stacks[location.stack_id]

            if source.top_id != target_id:
                state.phase = Phase.NEEDS_RELOCATION
                state.terminated = False
                if not any(
                    stack.stack_id != source.stack_id and not stack.is_full
                    for stack in state.stacks
                ):
                    raise NoLegalRelocationError(
                        f"blocked target {target_id} in stack {source.stack_id} "
                        "has no legal destination"
                    )
                return

            retrieved_id = source.pop()
            state.locations[retrieved_id] = None
            state.retrieval_count += 1
            state.retrieved_order.append(retrieved_id)
            events.append(
                TransitionEvent(
                    kind=EventKind.RETRIEVE,
                    container_id=retrieved_id,
                    batch_id=batch_id,
                    source_stack_id=source.stack_id,
                )
            )
            state.order_position += 1

            if state.order_position == len(revealed_order):
                events.append(
                    TransitionEvent(kind=EventKind.FINISH_BATCH, batch_id=batch_id)
                )
                state.current_batch_index += 1
                state.order_position = 0
                state.current_target_id = None

    def _validate_state(self) -> None:
        state = self._require_state()
        scenario = self._scenario
        if scenario is None:
            raise StateInvariantError("scenario is missing")

        if len(state.stacks) != self.instance.num_stacks:
            raise StateInvariantError("state has incorrect stack count")

        expected_locations = {}
        seen = set()
        for expected_stack_id, stack in enumerate(state.stacks):
            if stack.stack_id != expected_stack_id:
                raise StateInvariantError("stack IDs are not aligned with list indexes")
            if stack.height > stack.capacity or stack.capacity != self.instance.max_tiers:
                raise StateInvariantError("stack capacity invariant broken")
            for tier, container_id in enumerate(stack.containers):
                if container_id not in self.instance.container_by_id:
                    raise StateInvariantError(f"unknown container {container_id} in state")
                if container_id in seen:
                    raise StateInvariantError(f"container {container_id} appears twice")
                seen.add(container_id)
                expected_locations[container_id] = Location(stack.stack_id, tier)

        all_ids = set(self.instance.container_by_id)
        retrieved_ids = set(state.retrieved_order)
        if len(retrieved_ids) != len(state.retrieved_order):
            raise StateInvariantError("retrieved_order contains duplicates")
        if seen & retrieved_ids or seen | retrieved_ids != all_ids:
            raise StateInvariantError("remaining/retrieved container conservation broken")
        if state.retrieval_count != len(state.retrieved_order):
            raise StateInvariantError("retrieval_count does not match retrieved_order")
        if state.locations.keys() != self.instance.container_by_id.keys():
            raise StateInvariantError("location mapping has incorrect container IDs")
        for container_id in all_ids:
            expected = expected_locations.get(container_id)
            if state.locations[container_id] != expected:
                raise StateInvariantError(
                    f"location mismatch for container {container_id}: "
                    f"expected {expected}, got {state.locations[container_id]}"
                )

        revealed_batch_ids = tuple(state.revealed_orders)
        expected_prefix = self.instance.batch_order[: len(revealed_batch_ids)]
        if revealed_batch_ids != expected_prefix:
            raise StateInvariantError("revealed batches are not an ordered prefix")
        for batch_id, order in state.revealed_orders.items():
            if tuple(order) != tuple(scenario.hidden_orders[batch_id]):
                raise StateInvariantError("revealed order differs from sampled hidden order")
            if set(order) != set(self.instance.containers_by_batch[batch_id]):
                raise StateInvariantError("revealed order is not a batch permutation")

        expected_retrieved_prefix = []
        for batch_id in self.instance.batch_order:
            if batch_id not in state.revealed_orders:
                break
            order = state.revealed_orders[batch_id]
            if self.instance.batch_order.index(batch_id) < state.current_batch_index:
                expected_retrieved_prefix.extend(order)
            elif self.instance.batch_order.index(batch_id) == state.current_batch_index:
                expected_retrieved_prefix.extend(order[: state.order_position])
        if state.retrieved_order != expected_retrieved_prefix:
            raise StateInvariantError("retrieved_order is not the revealed-order prefix")

        if state.relocation_count != state.step_count:
            raise StateInvariantError("one relocation per step invariant broken")
        if state.total_reward != -state.relocation_count:
            raise StateInvariantError("reward/relocation invariant broken")

        if state.terminated:
            if state.phase is not Phase.TERMINATED:
                raise StateInvariantError("terminated state has wrong phase")
            if state.retrieval_count != self.instance.num_containers:
                raise StateInvariantError("episode terminated before all retrievals")
            if state.current_target_id is not None or any(not stack.is_empty for stack in state.stacks):
                raise StateInvariantError("terminated state still has a target or containers")
            return

        if state.phase is not Phase.NEEDS_RELOCATION:
            raise StateInvariantError("non-terminal state has wrong phase")
        if not 0 <= state.current_batch_index < self.instance.num_batches:
            raise StateInvariantError("current_batch_index outside batch_order")
        batch_id = self.instance.batch_order[state.current_batch_index]
        if batch_id not in state.revealed_orders:
            raise StateInvariantError("current batch is not revealed")
        order = state.revealed_orders[batch_id]
        if not 0 <= state.order_position < len(order):
            raise StateInvariantError("current order_position is invalid")
        if state.current_target_id != order[state.order_position]:
            raise StateInvariantError("current target does not match revealed order")
        self._current_source_and_blocker()
        if not self.legal_destinations():
            raise NoLegalRelocationError("non-terminal state has no legal destination")
