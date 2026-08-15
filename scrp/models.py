"""Data models and validation for the minimal SCRP Phase 1 core."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Tuple


class SCRPError(Exception):
    """Base exception for the Phase 1 SCRP core."""


class InstanceValidationError(SCRPError, ValueError):
    """Raised when a static SCRP instance is inconsistent."""


class InvalidActionError(SCRPError, ValueError):
    """Raised when a destination stack is not a legal relocation action."""


class EpisodeTerminatedError(SCRPError, RuntimeError):
    """Raised when step is called after normal termination."""


class NoLegalRelocationError(SCRPError, RuntimeError):
    """Raised when a blocked target has no legal destination stack."""


class StateInvariantError(SCRPError, RuntimeError):
    """Raised when dynamic state violates an SCRP invariant."""


class StackError(SCRPError, RuntimeError):
    """Raised for stack underflow or overflow."""


class StepLimitError(SCRPError, RuntimeError):
    """Raised when the Phase 1 safety step limit is reached."""


class Phase(str, Enum):
    NEEDS_RELOCATION = "NEEDS_RELOCATION"
    TERMINATED = "TERMINATED"


class EventKind(str, Enum):
    RELOCATE = "RELOCATE"
    RETRIEVE = "RETRIEVE"
    FINISH_BATCH = "FINISH_BATCH"
    REVEAL_BATCH = "REVEAL_BATCH"
    TERMINATE = "TERMINATE"


@dataclass(frozen=True)
class Container:
    container_id: int
    batch_id: int


@dataclass(frozen=True)
class Location:
    stack_id: int
    tier: int


@dataclass
class Stack:
    """A stack whose container IDs are stored from bottom to top."""

    stack_id: int
    capacity: int
    containers: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.containers = list(self.containers)
        if self.capacity <= 0:
            raise StackError("stack capacity must be positive")
        if len(self.containers) > self.capacity:
            raise StackError(
                f"stack {self.stack_id} height {len(self.containers)} exceeds "
                f"capacity {self.capacity}"
            )

    @property
    def height(self) -> int:
        return len(self.containers)

    @property
    def is_empty(self) -> bool:
        return not self.containers

    @property
    def is_full(self) -> bool:
        return self.height == self.capacity

    @property
    def top_id(self) -> int:
        if self.is_empty:
            raise StackError(f"cannot read top of empty stack {self.stack_id}")
        return self.containers[-1]

    def push(self, container_id: int) -> None:
        if self.is_full:
            raise StackError(f"cannot push to full stack {self.stack_id}")
        self.containers.append(container_id)

    def pop(self) -> int:
        if self.is_empty:
            raise StackError(f"cannot pop empty stack {self.stack_id}")
        return self.containers.pop()


@dataclass(frozen=True)
class SCRPInstance:
    instance_id: str
    num_stacks: int
    max_tiers: int
    containers: Tuple[Container, ...]
    initial_stacks: Tuple[Tuple[int, ...], ...]
    batch_order: Tuple[int, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    container_by_id: Dict[int, Container] = field(init=False, repr=False)
    containers_by_batch: Dict[int, Tuple[int, ...]] = field(init=False, repr=False)
    batch_sizes: Dict[int, int] = field(init=False)

    def __post_init__(self) -> None:
        containers = tuple(self.containers)
        initial_stacks = tuple(tuple(stack) for stack in self.initial_stacks)
        batch_order = tuple(self.batch_order)
        object.__setattr__(self, "containers", containers)
        object.__setattr__(self, "initial_stacks", initial_stacks)
        object.__setattr__(self, "batch_order", batch_order)

        if not self.instance_id:
            raise InstanceValidationError("instance_id must not be empty")
        if self.num_stacks <= 0 or self.max_tiers <= 0:
            raise InstanceValidationError("num_stacks and max_tiers must be positive")
        if len(initial_stacks) != self.num_stacks:
            raise InstanceValidationError(
                f"expected {self.num_stacks} stacks, got {len(initial_stacks)}"
            )
        if not containers:
            raise InstanceValidationError("instance must contain at least one container")
        if not batch_order or len(set(batch_order)) != len(batch_order):
            raise InstanceValidationError("batch_order must contain unique batch IDs")

        ids = [container.container_id for container in containers]
        if len(ids) != len(set(ids)):
            raise InstanceValidationError("container IDs must be unique")
        container_by_id = {container.container_id: container for container in containers}

        legal_batches = set(batch_order)
        unknown_batches = {
            container.batch_id for container in containers
            if container.batch_id not in legal_batches
        }
        if unknown_batches:
            raise InstanceValidationError(
                f"containers reference batch IDs outside batch_order: {sorted(unknown_batches)}"
            )

        containers_by_batch: Dict[int, List[int]] = {batch_id: [] for batch_id in batch_order}
        for container in containers:
            containers_by_batch[container.batch_id].append(container.container_id)
        empty_batches = [batch_id for batch_id, members in containers_by_batch.items() if not members]
        if empty_batches:
            raise InstanceValidationError(f"batches must be non-empty: {empty_batches}")

        layout_ids: List[int] = []
        for stack_id, stack in enumerate(initial_stacks):
            if len(stack) > self.max_tiers:
                raise InstanceValidationError(
                    f"stack {stack_id} exceeds max_tiers {self.max_tiers}"
                )
            layout_ids.extend(stack)
        unknown_ids = set(layout_ids) - set(ids)
        if unknown_ids:
            raise InstanceValidationError(
                f"layout contains unknown container IDs: {sorted(unknown_ids)}"
            )
        if len(layout_ids) != len(set(layout_ids)):
            raise InstanceValidationError("a container appears more than once in initial_stacks")
        missing_ids = set(ids) - set(layout_ids)
        if missing_ids:
            raise InstanceValidationError(
                f"containers missing from initial_stacks: {sorted(missing_ids)}"
            )
        if len(ids) > self.num_stacks * self.max_tiers:
            raise InstanceValidationError("container count exceeds total bay capacity")

        frozen_by_batch = {
            batch_id: tuple(members) for batch_id, members in containers_by_batch.items()
        }
        batch_sizes = {
            batch_id: len(members) for batch_id, members in frozen_by_batch.items()
        }
        if sum(batch_sizes.values()) != len(containers):
            raise InstanceValidationError("batch sizes do not sum to container count")

        object.__setattr__(self, "container_by_id", container_by_id)
        object.__setattr__(self, "containers_by_batch", frozen_by_batch)
        object.__setattr__(self, "batch_sizes", batch_sizes)

    @property
    def num_containers(self) -> int:
        return len(self.containers)

    @property
    def num_batches(self) -> int:
        return len(self.batch_order)


def is_guaranteed_restricted_feasible(instance: SCRPInstance) -> bool:
    """Return whether the instance satisfies the paper benchmark guarantee.

    This is intentionally an opt-in checker rather than a general instance
    invariant: stress and invalid-state tests may use any layout satisfying
    the physical capacity constraint C <= S*T.
    """

    guaranteed_capacity = (
        instance.num_stacks * instance.max_tiers - (instance.max_tiers - 1)
    )
    return instance.num_containers <= guaranteed_capacity


@dataclass(frozen=True)
class SCRPConfig:
    num_stacks: int
    max_tiers: int
    root_seed: int = 0
    max_steps: int = 10_000
    validate_after_transition: bool = True

    def __post_init__(self) -> None:
        if self.num_stacks <= 0 or self.max_tiers <= 0:
            raise ValueError("num_stacks and max_tiers must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")


@dataclass
class SCRPState:
    stacks: List[Stack]
    locations: Dict[int, Optional[Location]]
    current_batch_index: int
    revealed_orders: Dict[int, Tuple[int, ...]]
    order_position: int
    current_target_id: Optional[int]
    phase: Phase
    relocation_count: int
    retrieval_count: int
    step_count: int
    total_reward: int
    retrieved_order: List[int]
    terminated: bool

    def location_of(self, container_id: int) -> Optional[Location]:
        if container_id not in self.locations:
            raise KeyError(f"unknown container ID {container_id}")
        return self.locations[container_id]


@dataclass(frozen=True)
class TransitionEvent:
    kind: EventKind
    container_id: Optional[int] = None
    batch_id: Optional[int] = None
    source_stack_id: Optional[int] = None
    destination_stack_id: Optional[int] = None


@dataclass(frozen=True)
class TransitionResult:
    reward: int
    terminated: bool
    events: Tuple[TransitionEvent, ...]
    state: SCRPState
