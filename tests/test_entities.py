from dataclasses import FrozenInstanceError

import pytest

from scrp import (
    Container,
    InstanceValidationError,
    Location,
    SCRPConfig,
    SCRPEnvironment,
    SCRPInstance,
    Stack,
    is_guaranteed_restricted_feasible,
)
from scrp.models import StackError


def make_instance(**overrides):
    values = {
        "instance_id": "entities",
        "num_stacks": 3,
        "max_tiers": 3,
        "containers": (
            Container(1, 1),
            Container(2, 1),
            Container(3, 2),
        ),
        "initial_stacks": ((1,), (2,), (3,)),
        "batch_order": (1, 2),
        "metadata": {"kind": "test"},
    }
    values.update(overrides)
    return SCRPInstance(**values)


def test_container_is_immutable_and_has_no_dynamic_location():
    container = Container(1, 2)
    assert container.container_id == 1
    assert container.batch_id == 2
    assert not hasattr(container, "stack_id")
    with pytest.raises(FrozenInstanceError):
        container.batch_id = 3


def test_container_ids_must_be_unique():
    with pytest.raises(InstanceValidationError, match="unique"):
        make_instance(
            containers=(Container(1, 1), Container(1, 1), Container(3, 2)),
            initial_stacks=((1,), (), (3,)),
        )


def test_batch_membership_and_derived_fields_are_correct():
    instance = make_instance()
    assert instance.container_by_id[3].batch_id == 2
    assert instance.containers_by_batch == {1: (1, 2), 2: (3,)}
    assert instance.batch_sizes == {1: 2, 2: 1}
    assert instance.num_containers == 3
    assert instance.num_batches == 2


def test_unknown_batch_and_empty_batch_are_rejected():
    with pytest.raises(InstanceValidationError, match="outside batch_order"):
        make_instance(containers=(Container(1, 1), Container(2, 3), Container(3, 2)))
    with pytest.raises(InstanceValidationError, match="non-empty"):
        make_instance(batch_order=(1, 2, 3))


@pytest.mark.parametrize(
    "initial_stacks, message",
    [
        (((1,), (2,)), "expected 3 stacks"),
        (((1,), (2,), (3, 99)), "unknown container"),
        (((1,), (2,), ()), "missing from initial_stacks"),
        (((1,), (2,), (3, 1)), "more than once"),
        (((1, 2, 3), (), ()), "exceeds max_tiers"),
    ],
)
def test_invalid_layouts_are_rejected(initial_stacks, message):
    max_tiers = 2 if message == "exceeds max_tiers" else 3
    with pytest.raises(InstanceValidationError, match=message):
        make_instance(initial_stacks=initial_stacks, max_tiers=max_tiers)


def test_stack_push_pop_and_bottom_to_top_semantics():
    stack = Stack(stack_id=0, capacity=3, containers=[10, 11])
    assert stack.height == 2
    assert stack.top_id == 11
    stack.push(12)
    assert stack.is_full
    assert stack.containers == [10, 11, 12]
    assert stack.pop() == 12
    assert stack.containers == [10, 11]


def test_stack_rejects_overflow_and_underflow():
    full = Stack(0, 1, [1])
    with pytest.raises(StackError, match="full"):
        full.push(2)
    empty = Stack(1, 1)
    with pytest.raises(StackError, match="empty"):
        empty.pop()
    with pytest.raises(StackError, match="empty"):
        _ = empty.top_id


def test_reset_builds_consistent_zero_based_locations():
    instance = make_instance(initial_stacks=((1, 2), (), (3,)))
    config = SCRPConfig(3, 3, root_seed=1)
    env = SCRPEnvironment(config, instance)
    state = env.reset()
    # The sampled order may auto-retrieve, so verify every remaining location.
    for stack in state.stacks:
        for tier, container_id in enumerate(stack.containers):
            assert state.location_of(container_id) == Location(stack.stack_id, tier)
    for container_id in state.retrieved_order:
        assert state.location_of(container_id) is None


def test_environment_rejects_config_instance_shape_mismatch():
    instance = make_instance()
    with pytest.raises(ValueError, match="num_stacks"):
        SCRPEnvironment(SCRPConfig(2, 3), instance)
    with pytest.raises(ValueError, match="max_tiers"):
        SCRPEnvironment(SCRPConfig(3, 4), instance)


def _capacity_instance(container_count):
    ids = list(range(1, container_count + 1))
    return SCRPInstance(
        instance_id=f"capacity-{container_count}",
        num_stacks=3,
        max_tiers=3,
        containers=tuple(Container(container_id, 1) for container_id in ids),
        initial_stacks=tuple(
            tuple(ids[start:start + 3]) for start in range(0, 9, 3)
        ),
        batch_order=(1,),
    )


def test_paper_guaranteed_feasibility_boundary_and_general_instance_scope():
    boundary = _capacity_instance(7)  # 3*3 - (3-1)
    boundary_plus_one = _capacity_instance(8)
    assert is_guaranteed_restricted_feasible(boundary)
    assert not is_guaranteed_restricted_feasible(boundary_plus_one)
    # The checker is opt-in: a physical-capacity-valid stress instance remains legal.
    assert boundary_plus_one.num_containers == 8
