import json

from experiments import (
    ExperimentProtocolConfig,
    ScenarioResult,
    ScenarioSeedSchedule,
    build_split_manifest,
    load_protocol_config,
    load_split_manifest,
    save_protocol_config,
    save_split_manifest,
)
from scrp import SCRPConfig, SCRPEnvironment, load_instance_json


MANIFEST_PATH = "experiments/splits/scrp_split_v1.json"
CONFIG_PATH = "experiments/configs/formal_protocol_v1.json"


def test_all_1440_base_instances_are_assigned_exactly_once():
    manifest = load_split_manifest(MANIFEST_PATH)
    refs = manifest.refs()
    assert manifest.num_groups == 48
    assert manifest.num_base_instances == 1440
    assert len({ref.base_instance_id for ref in refs}) == 1440
    assert len({ref.ds1_instance_id for ref in refs}) == 1440
    assert len({ref.ds2_instance_id for ref in refs}) == 1440


def test_every_parameter_group_has_recommended_split_counts():
    manifest = load_split_manifest(MANIFEST_PATH)
    for assignments in manifest.groups.values():
        assert len(assignments["train"]) == 20
        assert len(assignments["validation"]) == 5
        assert len(assignments["test"]) == 5
    assert len(manifest.refs("train")) == 960
    assert len(manifest.refs("validation")) == 240
    assert len(manifest.refs("test")) == 240


def test_train_validation_test_have_no_base_or_derived_overlap():
    manifest = load_split_manifest(MANIFEST_PATH)
    for attribute in ("base_instance_id", "ds1_instance_id", "ds2_instance_id"):
        pools = [
            {getattr(ref, attribute) for ref in manifest.refs(split)}
            for split in ("train", "validation", "test")
        ]
        assert pools[0].isdisjoint(pools[1])
        assert pools[0].isdisjoint(pools[2])
        assert pools[1].isdisjoint(pools[2])


def test_ds1_ds2_pair_uses_one_base_split_assignment():
    manifest = load_split_manifest(MANIFEST_PATH)
    for ref in manifest.refs():
        split = manifest.split_for_base(ref.base_instance_id)
        assigned = manifest.groups[ref.parameter_group][split]
        matched = [item for item in assigned if item.base_instance_id == ref.base_instance_id]
        assert matched == [ref]


def test_same_split_seed_reconstructs_identical_manifest():
    manifest = load_split_manifest(MANIFEST_PATH)
    rebuilt = build_split_manifest(manifest.refs(), split_seed=manifest.split_seed)
    assert rebuilt == manifest


def test_different_split_seed_changes_assignment():
    manifest = load_split_manifest(MANIFEST_PATH)
    changed = build_split_manifest(manifest.refs(), split_seed=manifest.split_seed + 1)
    assert changed != manifest
    assert {
        ref.base_instance_id for ref in changed.refs("test")
    } != {
        ref.base_instance_id for ref in manifest.refs("test")
    }


def test_scenario_seed_streams_are_deterministic_and_disjoint():
    manifest = load_split_manifest(MANIFEST_PATH)
    first = ScenarioSeedSchedule(manifest)
    second = ScenarioSeedSchedule(manifest)
    pools = {}
    for split in ("train", "validation", "test"):
        ref = manifest.refs(split)[0]
        pools[split] = set(first.seeds(split, ref.base_instance_id, 100))
        assert first.seeds(split, ref.base_instance_id, 100) == second.seeds(
            split, ref.base_instance_id, 100
        )
    assert pools["train"].isdisjoint(pools["validation"])
    assert pools["train"].isdisjoint(pools["test"])
    assert pools["validation"].isdisjoint(pools["test"])


def test_test_seeds_are_fixed_and_ds1_ds2_share_base_schedule():
    manifest = load_split_manifest(MANIFEST_PATH)
    schedule = ScenarioSeedSchedule(manifest)
    ref = manifest.refs("test")[0]
    seeds = schedule.seeds("test", ref.base_instance_id, 50)
    assert seeds == schedule.seeds("test", ref.base_instance_id, 50)
    assert len(seeds) == len(set(seeds)) == 50


def test_crn_scenario_ids_pair_within_variant_not_across_ds1_ds2():
    ds1 = load_instance_json("data/phase3_sanity/S05_T03_mu050/ds1_001.json")
    ds2 = load_instance_json("data/phase3_sanity/S05_T03_mu050/ds2_001.json")
    scenario_seed = 3_000_000_000_000

    def paired_algorithm_ids(instance):
        config = SCRPConfig(instance.num_stacks, instance.max_tiers)
        first_algorithm_env = SCRPEnvironment(config, instance)
        second_algorithm_env = SCRPEnvironment(config, instance)
        first_algorithm_env.reset(seed=scenario_seed)
        second_algorithm_env.reset(seed=scenario_seed)
        return first_algorithm_env.scenario_id, second_algorithm_env.scenario_id

    ds1_first, ds1_second = paired_algorithm_ids(ds1)
    ds2_first, ds2_second = paired_algorithm_ids(ds2)

    assert ds1_first == ds1_second
    assert ds2_first == ds2_second
    assert ds1_first != ds2_first


def test_manifest_save_load_is_identical(tmp_path):
    manifest = load_split_manifest(MANIFEST_PATH)
    destination = save_split_manifest(manifest, tmp_path / "split.json")
    assert load_split_manifest(destination) == manifest


def test_static_split_manifest_contains_no_scenario_or_hidden_order():
    record = json.loads(open(MANIFEST_PATH, encoding="utf-8").read())
    encoded = json.dumps(record).lower()
    for forbidden in (
        "scenario_seed", "scenario_id", "hidden_order", "hidden_permutation", "order_seed"
    ):
        assert forbidden not in encoded


def test_scenario_result_schema_is_json_round_trip_safe():
    result = ScenarioResult(
        dataset="DS2",
        split="test",
        instance_id="ku2016-T271014_0503_001-merge2",
        base_instance_id="T271014_0503_001",
        parameter_group="S05_T03_mu0.50",
        scenario_seed=3_000_000_000_000,
        scenario_id="sha256-example",
        algorithm="rl_o1",
        relocations=7,
        terminated=True,
        truncated=False,
    )
    encoded = json.dumps(result.to_record())
    assert ScenarioResult.from_record(json.loads(encoded)) == result


def test_protocol_metadata_contains_observation_and_dataset_versions(tmp_path):
    config = load_protocol_config(CONFIG_PATH)
    assert config.observation_version == "O1"
    assert config.dataset_version
    assert config.split_manifest_version
    destination = save_protocol_config(config, tmp_path / "config.json")
    assert load_protocol_config(destination) == config
