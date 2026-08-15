import copy
import json

import pytest

from scrp import (
    InstanceValidationError,
    ScenarioSampler,
    instance_from_record,
    instance_to_record,
    load_instance_json,
    merge_adjacent_batches,
    parse_ku_crptw,
    save_instance_json,
)


KU_SAMPLE = """T271014_0503_001 1 5 3 8 4
  1   1   1   3   3
  1   2   2   1   1   2   2
  1   3   1   1   1
  1   4   1   3   3
  1   5   3   4   4   4   4   1   1
"""


def make_ku_file(tmp_path):
    path = tmp_path / "T271014_0503_001.txt"
    path.write_text(KU_SAMPLE, encoding="utf-8")
    return path


def test_parse_exact_ku_format_count_capacity_ids_batches_and_orientation(tmp_path):
    instance = parse_ku_crptw(make_ku_file(tmp_path))
    assert (instance.num_stacks, instance.max_tiers, instance.num_containers) == (5, 3, 8)
    assert instance.initial_stacks == ((1,), (2, 3), (4,), (5,), (6, 7, 8))
    assert [instance.container_by_id[i].batch_id for i in range(1, 9)] == [3, 1, 2, 1, 3, 4, 4, 1]
    assert instance.batch_order == (1, 2, 3, 4)
    assert instance.batch_sizes == {1: 3, 2: 1, 3: 2, 4: 2}
    assert instance.metadata["fill_rate"] == 0.5
    assert instance.metadata["original_instance_id"] == "T271014_0503_001"
    assert len(set(instance.container_by_id)) == 8


def test_parser_rejects_nonidentical_source_label_pair(tmp_path):
    path = make_ku_file(tmp_path)
    path.write_text(KU_SAMPLE.replace("3   3", "3   4", 1), encoding="utf-8")
    with pytest.raises(InstanceValidationError, match="unequal source label pair"):
        parse_ku_crptw(path)


def test_parser_uses_observed_nonempty_batches_when_header_count_disagrees(tmp_path):
    path = make_ku_file(tmp_path)
    path.write_text(KU_SAMPLE.replace("8 4", "8 5", 1), encoding="utf-8")
    instance = parse_ku_crptw(path)
    assert instance.batch_order == (1, 2, 3, 4)
    assert instance.metadata["original_header_field_6"] == 5
    assert instance.metadata["observed_num_time_windows"] == 4


def test_parser_normalizes_gapped_labels_without_changing_precedence(tmp_path):
    path = make_ku_file(tmp_path)
    path.write_text(KU_SAMPLE.replace("4   4", "5   5"), encoding="utf-8")
    instance = parse_ku_crptw(path)
    assert instance.batch_order == (1, 2, 3, 4)
    assert instance.metadata["original_label_to_batch"] == {
        "1": 1, "2": 2, "3": 3, "5": 4
    }
    assert instance.container_by_id[6].batch_id == 4


def test_ds2_merge_preserves_ids_layout_and_container_count(tmp_path):
    ds1 = parse_ku_crptw(make_ku_file(tmp_path))
    ds2 = merge_adjacent_batches(ds1, 2)
    assert ds2.initial_stacks == ds1.initial_stacks
    assert [item.container_id for item in ds2.containers] == [
        item.container_id for item in ds1.containers
    ]
    assert ds2.num_containers == ds1.num_containers
    assert ds2.batch_order == (1, 2)
    assert [ds2.container_by_id[i].batch_id for i in range(1, 9)] == [2, 1, 1, 1, 2, 2, 2, 1]


def test_static_json_round_trip_is_identical(tmp_path):
    instance = parse_ku_crptw(make_ku_file(tmp_path))
    destination = save_instance_json(instance, tmp_path / "instance.json")
    assert load_instance_json(destination) == instance
    assert instance_from_record(instance_to_record(instance)) == instance


def test_static_record_contains_no_scenario_or_permutation(tmp_path):
    record = instance_to_record(parse_ku_crptw(make_ku_file(tmp_path)))
    encoded = json.dumps(record).lower()
    assert "scenario_seed" not in encoded
    assert "scenario_id" not in encoded
    assert "hidden_order" not in encoded
    assert "retrieval_permutation" not in encoded


@pytest.mark.parametrize("field", ["hidden_orders", "scenario_seed", "scenario_id", "order_seeds"])
def test_loader_rejects_future_order_leakage_in_metadata(tmp_path, field):
    record = instance_to_record(parse_ku_crptw(make_ku_file(tmp_path)))
    record["metadata"][field] = {"1": [1, 2]}
    with pytest.raises(InstanceValidationError, match="future-order field"):
        instance_from_record(record)


def test_loader_rejects_unknown_top_level_field(tmp_path):
    record = instance_to_record(parse_ku_crptw(make_ku_file(tmp_path)))
    record["permutation"] = [1, 2]
    with pytest.raises(InstanceValidationError, match="schema keys mismatch"):
        instance_from_record(record)


def test_loader_rejects_wrong_declared_container_count(tmp_path):
    record = instance_to_record(parse_ku_crptw(make_ku_file(tmp_path)))
    record["num_containers"] += 1
    with pytest.raises(InstanceValidationError, match="num_containers"):
        instance_from_record(record)


def test_same_static_instance_different_seed_does_not_mutate_static_data(tmp_path):
    instance = parse_ku_crptw(make_ku_file(tmp_path))
    before = instance_to_record(instance)
    sampler = ScenarioSampler()
    scenarios = [sampler.sample(instance, seed) for seed in range(8)]
    assert instance_to_record(instance) == before
    assert len({scenario.scenario_id for scenario in scenarios}) > 1
    assert len({tuple(scenario.hidden_orders.items()) for scenario in scenarios}) > 1


def test_same_static_instance_same_seed_has_identical_scenario(tmp_path):
    instance = parse_ku_crptw(make_ku_file(tmp_path))
    sampler = ScenarioSampler()
    assert sampler.sample(instance, 2026) == sampler.sample(instance, 2026)


def test_schema_rejects_layout_duplicate_and_missing_id(tmp_path):
    record = instance_to_record(parse_ku_crptw(make_ku_file(tmp_path)))
    broken = copy.deepcopy(record)
    broken["stacks"][0][0] = broken["stacks"][1][0]
    with pytest.raises(InstanceValidationError, match="more than once"):
        instance_from_record(broken)


def test_schema_rejects_stack_over_capacity(tmp_path):
    record = instance_to_record(parse_ku_crptw(make_ku_file(tmp_path)))
    record["max_tiers"] = 2
    with pytest.raises(InstanceValidationError, match="exceeds max_tiers"):
        instance_from_record(record)


def test_paper_feasibility_bound_is_enforced(tmp_path):
    record = instance_to_record(parse_ku_crptw(make_ku_file(tmp_path)))
    record["max_tiers"] = 1
    with pytest.raises(InstanceValidationError):
        instance_from_record(record)
