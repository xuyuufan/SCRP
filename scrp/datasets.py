"""Auditable loaders and serialization for published SCRP instances.

The static instance is deliberately separate from stochastic scenarios.  This
module never serializes sampled within-batch permutations, order seeds, or
scenario identifiers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

from .models import (
    Container,
    InstanceValidationError,
    SCRPInstance,
    is_guaranteed_restricted_feasible,
)


STATIC_INSTANCE_SCHEMA_VERSION = "scrp-static-instance-v1"

_STATIC_KEYS = {
    "schema_version",
    "instance_id",
    "source_dataset",
    "num_stacks",
    "max_tiers",
    "num_containers",
    "batch_order",
    "stacks",
    "container_batch",
    "metadata",
}

_FUTURE_ORDER_KEYS = {
    "exact_retrieval_order",
    "hidden_order",
    "hidden_orders",
    "hidden_permutation",
    "order_seed",
    "order_seeds",
    "retrieval_permutation",
    "revealed_order",
    "scenario_id",
    "scenario_seed",
}


def _reject_future_order_data(value: Any, path: str = "record") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if normalized in _FUTURE_ORDER_KEYS:
                raise InstanceValidationError(
                    f"static instance must not contain future-order field {path}.{key}"
                )
            _reject_future_order_data(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_future_order_data(child, f"{path}[{index}]")


def validate_paper_instance(instance: SCRPInstance) -> None:
    """Apply benchmark-only checks in addition to ``SCRPInstance`` checks."""

    if not is_guaranteed_restricted_feasible(instance):
        limit = instance.num_stacks * instance.max_tiers - (instance.max_tiers - 1)
        raise InstanceValidationError(
            f"paper benchmark requires N <= S*T-(T-1): "
            f"{instance.num_containers} > {limit}"
        )
    _reject_future_order_data(instance.metadata, "metadata")


def instance_to_record(instance: SCRPInstance) -> Dict[str, Any]:
    """Convert an instance to the stable, scenario-free JSON record."""

    validate_paper_instance(instance)
    metadata = dict(instance.metadata)
    source_dataset = metadata.pop("source_dataset", None)
    if not isinstance(source_dataset, str) or not source_dataset:
        raise InstanceValidationError("metadata.source_dataset must be a non-empty string")

    record: Dict[str, Any] = {
        "schema_version": STATIC_INSTANCE_SCHEMA_VERSION,
        "instance_id": instance.instance_id,
        "source_dataset": source_dataset,
        "num_stacks": instance.num_stacks,
        "max_tiers": instance.max_tiers,
        "num_containers": instance.num_containers,
        "batch_order": list(instance.batch_order),
        "stacks": [list(stack) for stack in instance.initial_stacks],
        "container_batch": {
            str(container.container_id): container.batch_id
            for container in sorted(instance.containers, key=lambda item: item.container_id)
        },
        "metadata": metadata,
    }
    _reject_future_order_data(record)
    try:
        # Also rejects values that cannot be persisted faithfully as JSON.
        return json.loads(json.dumps(record, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise InstanceValidationError(f"instance metadata is not JSON-compatible: {error}") from error


def instance_from_record(record: Mapping[str, Any]) -> SCRPInstance:
    """Load and strictly validate one static JSON record."""

    if not isinstance(record, Mapping):
        raise InstanceValidationError("static instance record must be a JSON object")
    unknown = set(record) - _STATIC_KEYS
    missing = _STATIC_KEYS - set(record)
    if unknown or missing:
        raise InstanceValidationError(
            f"static schema keys mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    if record["schema_version"] != STATIC_INSTANCE_SCHEMA_VERSION:
        raise InstanceValidationError(
            f"unsupported schema_version {record['schema_version']!r}"
        )
    _reject_future_order_data(record)

    source_dataset = record["source_dataset"]
    if not isinstance(source_dataset, str) or not source_dataset:
        raise InstanceValidationError("source_dataset must be a non-empty string")
    metadata = record["metadata"]
    if not isinstance(metadata, Mapping):
        raise InstanceValidationError("metadata must be a JSON object")
    if "source_dataset" in metadata:
        raise InstanceValidationError("source_dataset must only appear at the record top level")

    try:
        container_batch = {
            int(container_id): int(batch_id)
            for container_id, batch_id in record["container_batch"].items()
        }
        containers = tuple(
            Container(container_id, batch_id)
            for container_id, batch_id in sorted(container_batch.items())
        )
        instance = SCRPInstance(
            instance_id=str(record["instance_id"]),
            num_stacks=int(record["num_stacks"]),
            max_tiers=int(record["max_tiers"]),
            containers=containers,
            initial_stacks=tuple(
                tuple(int(container_id) for container_id in stack)
                for stack in record["stacks"]
            ),
            batch_order=tuple(int(batch_id) for batch_id in record["batch_order"]),
            metadata={"source_dataset": source_dataset, **dict(metadata)},
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise InstanceValidationError(f"malformed static instance record: {error}") from error

    if instance.num_containers != record["num_containers"]:
        raise InstanceValidationError(
            f"num_containers={record['num_containers']} but parsed {instance.num_containers}"
        )
    validate_paper_instance(instance)
    return instance


def save_instance_json(instance: SCRPInstance, path: str | Path) -> Path:
    """Save one static instance. Parent creation is the only filesystem side effect."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(instance_to_record(instance), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination


def load_instance_json(path: str | Path) -> SCRPInstance:
    source = Path(path)
    try:
        record = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InstanceValidationError(f"cannot read static instance {source}: {error}") from error
    return instance_from_record(record)


def _fill_rate_from_original_id(original_id: str) -> float | None:
    # These prefixes are documented by the public StochasticCRP repository.
    if original_id.startswith("T271014_"):
        return 0.50
    if original_id.startswith("T281014_"):
        return 0.67
    return None


def parse_ku_crptw(path: str | Path) -> SCRPInstance:
    """Parse one exact Ku-Arthanari CRPTW benchmark text file.

    Source rows contain two identical values per occupied tier.  The public
    MATLAB reader uses the first value in every pair as the time-window label.
    A mismatch is rejected because it cannot be losslessly interpreted as the
    single batch membership required by ``SCRPInstance``.
    """

    source = Path(path)
    try:
        rows = [
            line.split()
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except OSError as error:
        raise InstanceValidationError(f"cannot read Ku CRPTW file {source}: {error}") from error
    if not rows or len(rows[0]) != 6:
        raise InstanceValidationError("Ku CRPTW header must contain exactly 6 fields")

    original_id = rows[0][0]
    try:
        bay_id, num_stacks, max_tiers, num_containers, num_windows = map(
            int, rows[0][1:]
        )
    except ValueError as error:
        raise InstanceValidationError("Ku CRPTW header contains a non-integer field") from error
    if len(rows) != num_stacks + 1:
        raise InstanceValidationError(
            f"header declares {num_stacks} stacks but file has {len(rows) - 1} stack rows"
        )

    label_stacks = []
    for expected_stack, row in enumerate(rows[1:], start=1):
        try:
            values = [int(value) for value in row]
        except ValueError as error:
            raise InstanceValidationError(
                f"stack row {expected_stack} contains a non-integer field"
            ) from error
        if len(values) < 3:
            raise InstanceValidationError(f"stack row {expected_stack} is too short")
        row_bay, stack_id, height = values[:3]
        tier_pairs = values[3:]
        if row_bay != bay_id or stack_id != expected_stack:
            raise InstanceValidationError(
                f"expected bay/stack {bay_id}/{expected_stack}, got {row_bay}/{stack_id}"
            )
        if not 0 <= height <= max_tiers or len(tier_pairs) != 2 * height:
            raise InstanceValidationError(
                f"stack {stack_id} height/payload does not match max_tiers={max_tiers}"
            )
        labels = []
        for tier in range(height):
            first, second = tier_pairs[2 * tier : 2 * tier + 2]
            if first != second:
                raise InstanceValidationError(
                    f"stack {stack_id}, tier {tier} has unequal source label pair "
                    f"({first}, {second})"
                )
            labels.append(first)
        label_stacks.append(tuple(labels))

    flat_labels = [label for stack in label_stacks for label in stack]
    if len(flat_labels) != num_containers:
        raise InstanceValidationError(
            f"header declares {num_containers} containers but layout has {len(flat_labels)}"
        )
    # A minority of the public files have a sixth header value that differs
    # from the number (and occasionally the maximum) of labels actually used.
    # Galle's reader ignores that header field.  The layout labels are the
    # authoritative data; sort them to preserve precedence and normalize them
    # to the non-empty batch IDs required by SCRPInstance.
    original_labels = sorted(set(flat_labels))
    if not original_labels or original_labels[0] <= 0:
        raise InstanceValidationError("time-window labels must be positive integers")
    label_to_batch = {
        original_label: batch_id
        for batch_id, original_label in enumerate(original_labels, start=1)
    }

    containers = []
    id_stacks = []
    next_id = 1
    for labels in label_stacks:
        stack_ids = []
        for original_label in labels:
            containers.append(Container(next_id, label_to_batch[original_label]))
            stack_ids.append(next_id)
            next_id += 1
        id_stacks.append(tuple(stack_ids))

    fill_rate = _fill_rate_from_original_id(original_id)
    converted_id = f"ku2016-{original_id}"
    instance = SCRPInstance(
        instance_id=converted_id,
        num_stacks=num_stacks,
        max_tiers=max_tiers,
        containers=tuple(containers),
        initial_stacks=tuple(id_stacks),
        batch_order=tuple(range(1, len(original_labels) + 1)),
        metadata={
            "source_dataset": "Ku2016_CRPTW_Galle2017_existing_Bacci2022_DS1",
            "paper": ["Ku & Arthanari (2016)", "Galle et al. (2017/2018)", "Bacci et al. (2022)"],
            "original_instance_id": original_id,
            "converted_instance_id": converted_id,
            "original_file": source.name,
            "parameter_group": f"S{num_stacks:02d}_T{max_tiers:02d}_mu{fill_rate}",
            "fill_rate": fill_rate,
            "original_bay_id": bay_id,
            "original_header_field_6": num_windows,
            "observed_num_time_windows": len(original_labels),
            "original_label_to_batch": {
                str(label): batch_id for label, batch_id in label_to_batch.items()
            },
            "id_assignment_rule": "stack-major, then bottom-to-top, consecutive integers from 1",
            "stack_orientation": "bottom-to-top",
            "source_row_rule": "first value of each validated-equal tier pair is the time-window label",
        },
    )
    validate_paper_instance(instance)
    return instance


def merge_adjacent_batches(
    instance: SCRPInstance,
    merge_factor: int = 2,
    *,
    source_dataset: str = "Galle2017_modified_Bacci2022_DS2",
) -> SCRPInstance:
    """Derive DS2 by merging adjacent DS1 batches (paper formula w'=ceil(w/gamma))."""

    if isinstance(merge_factor, bool) or not isinstance(merge_factor, int) or merge_factor <= 1:
        raise ValueError("merge_factor must be an integer greater than 1")
    batch_position = {
        batch_id: position for position, batch_id in enumerate(instance.batch_order, start=1)
    }
    mapped = {
        batch_id: (position - 1) // merge_factor + 1
        for batch_id, position in batch_position.items()
    }
    merged_order = tuple(range(1, max(mapped.values()) + 1))
    converted_id = f"{instance.instance_id}-merge{merge_factor}"
    metadata = dict(instance.metadata)
    metadata.update(
        {
            "source_dataset": source_dataset,
            "converted_instance_id": converted_id,
            "derived_from_instance_id": instance.instance_id,
            "batch_merge_factor": merge_factor,
            "batch_mapping_rule": "new batch position = ceil(original batch position / merge_factor)",
            "original_num_batches": instance.num_batches,
        }
    )
    merged = SCRPInstance(
        instance_id=converted_id,
        num_stacks=instance.num_stacks,
        max_tiers=instance.max_tiers,
        containers=tuple(
            Container(container.container_id, mapped[container.batch_id])
            for container in instance.containers
        ),
        initial_stacks=instance.initial_stacks,
        batch_order=merged_order,
        metadata=metadata,
    )
    validate_paper_instance(merged)
    return merged
