from __future__ import annotations

import numpy as np
import pytest
import torch

from hier_pg.network import HierPolicyNetwork
from scrp import (
    Container,
    O1ObservationAdapter,
    O2ObservationAdapter,
    SCRPConfig,
    SCRPEnvironment,
    SCRPInstance,
    SCRPObservationConfig,
    SCRPRLAdapter,
    SCRP_O2_FEATURE_SCALE,
    SCRP_O2_MMAX,
    Scenario,
    load_instance_json,
    load_observation_config,
    save_observation_config,
)


class FixedSampler:
    def __init__(self, hidden_orders, scenario_id="o2-fixed"):
        self.hidden_orders = hidden_orders
        self.scenario_id = scenario_id

    def sample(self, instance, root_seed):
        return Scenario(
            root_seed,
            {batch: root_seed + batch for batch in instance.batch_order},
            self.hidden_orders,
            self.scenario_id,
        )


def collision_instance(id_offset=0):
    ids = [value + id_offset for value in range(1, 6)]
    return SCRPInstance(
        f"o1-collision-{id_offset}",
        3,
        4,
        tuple(
            Container(container_id, 1 if index < 4 else 2)
            for index, container_id in enumerate(ids)
        ),
        ((ids[0], ids[4]), (ids[1], ids[2], ids[3]), ()),
        (1, 2),
    )


def collision_env(order, *, id_offset=0):
    instance = collision_instance(id_offset)
    mapped_order = tuple(value + id_offset for value in order)
    return SCRPEnvironment(
        SCRPConfig(3, 4),
        instance,
        FixedSampler({1: mapped_order, 2: (5 + id_offset,)}),
    )


def reshape(adapter, observation):
    return observation.reshape(adapter.node_shape)


def test_o1_has_a_real_full_revealed_order_collision():
    # Containers 3 and 4 swap after the unchanged earliest member 2 in one
    # stack. O1 stores only that stack's earliest rank, so both tensors collide.
    env_a = collision_env((1, 2, 3, 4))
    env_b = collision_env((1, 2, 4, 3))
    state_a = env_a.reset(seed=1)
    state_b = env_b.reset(seed=1)
    o1_a = O1ObservationAdapter(env_a.instance, env_a.config).build(state_a)
    o1_b = O1ObservationAdapter(env_b.instance, env_b.config).build(state_b)
    assert state_a.current_target_id == state_b.current_target_id == 1
    assert [stack.containers for stack in state_a.stacks] == [
        stack.containers for stack in state_b.stacks
    ]
    assert state_a.revealed_orders != state_b.revealed_orders
    assert np.array_equal(o1_a, o1_b)


def test_o2_fixed_shape_and_float32_dtype():
    env = collision_env((1, 2, 3, 4))
    state = env.reset(seed=2)
    adapter = O2ObservationAdapter(env.instance, env.config)
    observation = adapter.build(state)
    assert observation.shape == ((3 + SCRP_O2_MMAX + 1) * 12,)
    assert observation.dtype == np.float32


def test_o2_node_layout_keeps_first_s_nodes_as_stack_actions():
    env = collision_env((1, 2, 3, 4))
    state = env.reset(seed=3)
    adapter = O2ObservationAdapter(env.instance, env.config)
    nodes = reshape(adapter, adapter.build(state))
    assert np.all(nodes[:3, 0] == adapter.STACK_NODE_TYPE)
    assert np.all(nodes[3:3 + SCRP_O2_MMAX, 0] == adapter.ORDER_NODE_TYPE)
    assert nodes[-1, 0] == adapter.CONTEXT_NODE_TYPE


def test_o2_encodes_full_revealed_order_by_unique_public_locations():
    env = collision_env((1, 2, 3, 4))
    state = env.reset(seed=4)
    adapter = O2ObservationAdapter(env.instance, env.config)
    order_nodes = reshape(adapter, adapter.build(state))[3:7]
    # Stack/tier pairs uniquely identify all four live containers without IDs.
    encoded_locations = [tuple(row[2:4]) for row in order_nodes]
    assert encoded_locations == [
        (0.0, 0.0),
        (0.5, 0.0),
        (0.5, 1 / 3),
        (0.5, 2 / 3),
    ]


def test_o2_explicit_rank_feature_distinguishes_colliding_orders():
    env_a = collision_env((1, 2, 3, 4))
    env_b = collision_env((1, 2, 4, 3))
    state_a = env_a.reset(seed=5)
    state_b = env_b.reset(seed=5)
    adapter_a = O2ObservationAdapter(env_a.instance, env_a.config)
    adapter_b = O2ObservationAdapter(env_b.instance, env_b.config)
    observation_a = adapter_a.build(state_a)
    observation_b = adapter_b.build(state_b)
    assert not np.array_equal(observation_a, observation_b)
    assert np.allclose(
        reshape(adapter_a, observation_a)[3:7, 1],
        [0.0, 1 / 3, 2 / 3, 1.0],
    )


def test_o2_does_not_use_raw_container_id_as_an_ordinal_feature():
    env_original = collision_env((1, 2, 3, 4), id_offset=0)
    env_relabelled = collision_env((1, 2, 3, 4), id_offset=100)
    original = O2ObservationAdapter(
        env_original.instance, env_original.config
    ).build(env_original.reset(seed=6))
    relabelled = O2ObservationAdapter(
        env_relabelled.instance, env_relabelled.config
    ).build(env_relabelled.reset(seed=6))
    assert np.array_equal(original, relabelled)


def test_o2_padding_is_explicit_and_cannot_be_a_real_order_node():
    env = collision_env((1, 2, 3, 4))
    state = env.reset(seed=7)
    adapter = O2ObservationAdapter(env.instance, env.config)
    order_nodes = reshape(adapter, adapter.build(state))[3:3 + SCRP_O2_MMAX]
    assert np.all(order_nodes[:4, 11] == 0.0)
    assert np.all(order_nodes[4:, 11] == 1.0)
    assert np.all(order_nodes[4:, 0] == adapter.ORDER_NODE_TYPE)
    assert np.all(order_nodes[4:, 1:11] == 0.0)


def test_o2_rank_semantics_do_not_depend_only_on_tensor_position():
    env = collision_env((1, 2, 3, 4))
    state = env.reset(seed=8)
    adapter = O2ObservationAdapter(env.instance, env.config)
    order_nodes = reshape(adapter, adapter.build(state))[3:7].copy()
    expected = order_nodes[np.argsort(order_nodes[:, 1])][:, 2:4]
    order_nodes[[1, 2]] = order_nodes[[2, 1]]
    reconstructed = order_nodes[np.argsort(order_nodes[:, 1])][:, 2:4]
    assert np.array_equal(reconstructed, expected)


def reveal_boundary_env(batch2_order, scenario_id):
    instance = SCRPInstance(
        "o2-reveal-boundary",
        4,
        3,
        tuple(
            Container(container_id, batch_id)
            for container_id, batch_id in (
                (1, 1), (2, 2), (3, 2), (4, 3), (5, 3), (6, 3)
            )
        ),
        ((1, 4), (2, 5), (3, 6), ()),
        (1, 2, 3),
    )
    return SCRPEnvironment(
        SCRPConfig(4, 3),
        instance,
        FixedSampler({1: (1,), 2: batch2_order, 3: (4, 5, 6)}, scenario_id),
    )


def test_o2_future_hidden_order_has_no_leakage_before_reveal():
    env_a = reveal_boundary_env((2, 3), "future-a")
    env_b = reveal_boundary_env((3, 2), "future-b")
    adapter_a = O2ObservationAdapter(env_a.instance, env_a.config)
    adapter_b = O2ObservationAdapter(env_b.instance, env_b.config)
    before_a = adapter_a.build(env_a.reset(seed=9))
    before_b = adapter_b.build(env_b.reset(seed=9))
    assert np.array_equal(before_a, before_b)


def test_o2_may_change_at_the_future_batch_reveal_boundary():
    env_a = reveal_boundary_env((2, 3), "future-a")
    env_b = reveal_boundary_env((3, 2), "future-b")
    adapter_a = O2ObservationAdapter(env_a.instance, env_a.config)
    adapter_b = O2ObservationAdapter(env_b.instance, env_b.config)
    env_a.reset(seed=10)
    env_b.reset(seed=10)
    after_a = adapter_a.build(env_a.step(3).state)
    after_b = adapter_b.build(env_b.step(3).state)
    assert not np.array_equal(after_a, after_b)
    assert env_a.state.revealed_orders[2] == (2, 3)
    assert env_b.state.revealed_orders[2] == (3, 2)


@pytest.mark.parametrize(
    "path,expected_max",
    [
        ("data/phase3_sanity/S05_T03_mu050/ds1_001.json", 3),
        ("data/phase3_sanity/S07_T04_mu067/ds2_001.json", 6),
    ],
)
def test_o2_supports_real_ds1_and_ds2_with_one_fixed_mmax(path, expected_max):
    instance = load_instance_json(path)
    assert max(instance.batch_sizes.values()) == expected_max
    core = SCRPEnvironment(
        SCRPConfig(instance.num_stacks, instance.max_tiers), instance
    )
    adapter = SCRPRLAdapter(core, observation_version="O2")
    observation, _ = adapter.reset(seed=11)
    assert observation.shape == ((instance.num_stacks + 6 + 1) * 12,)


def test_o2_rejects_an_instance_outside_the_audited_mmax_universe():
    instance = SCRPInstance(
        "batch-seven",
        3,
        3,
        tuple(Container(container_id, 1) for container_id in range(1, 8)),
        ((1, 2, 3), (4, 5), (6, 7)),
        (1,),
    )
    with pytest.raises(ValueError, match="exceeds O2 Mmax=6"):
        O2ObservationAdapter(instance, SCRPConfig(3, 3))


def make_o2_network():
    torch.manual_seed(12)
    return HierPolicyNetwork(
        embed_dim=32,
        num_enc_layers=1,
        num_heads=4,
        ffn_dim=64,
        feature_scale=torch.tensor(SCRP_O2_FEATURE_SCALE),
    ).eval()


def test_o2_network_low_forward_keeps_logits_and_greedy_action_in_stack_range():
    env = collision_env((1, 2, 3, 4))
    adapter = SCRPRLAdapter(env, observation_version="O2")
    observation, info = adapter.reset(seed=12)
    forbidden = torch.tensor(~info["action_mask"]).unsqueeze(0)
    network = make_o2_network()
    with torch.no_grad():
        encoded = network.encode(torch.tensor(observation).unsqueeze(0))
        action, log_probability = network.forward(
            torch.tensor(observation).unsqueeze(0),
            forbidden,
            greedy=True,
            mode="low",
            enc_out=encoded,
        )
        decoder = network.low_decoder
        query = decoder._build_query(encoded)
        context = decoder.norm(query + decoder.cross_attn(query, encoded, encoded))
        logits = decoder._pointer_scores(context, encoded, forbidden)
    assert encoded.shape[1] == env.instance.num_stacks + SCRP_O2_MMAX + 1
    assert logits.shape == (1, env.instance.num_stacks)
    assert 0 <= action.item() < env.instance.num_stacks
    assert info["action_mask"][action.item()]
    assert torch.isfinite(log_probability).all()


def test_o2_network_stochastic_sampling_can_only_return_legal_stack_actions():
    env = collision_env((1, 2, 3, 4))
    adapter = SCRPRLAdapter(env, observation_version="O2")
    observation, info = adapter.reset(seed=13)
    observation_tensor = torch.tensor(observation).unsqueeze(0)
    forbidden = torch.tensor(~info["action_mask"]).unsqueeze(0)
    network = make_o2_network()
    with torch.no_grad():
        actions = [
            int(network.forward(observation_tensor, forbidden, mode="low")[0].item())
            for _ in range(50)
        ]
    assert all(0 <= action < env.instance.num_stacks for action in actions)
    assert all(info["action_mask"][action] for action in actions)


def test_scrp_adapter_default_and_explicit_o1_are_backward_compatible():
    env_default = collision_env((1, 2, 3, 4))
    env_explicit = collision_env((1, 2, 3, 4))
    default_observation, default_info = SCRPRLAdapter(env_default).reset(seed=14)
    explicit_observation, explicit_info = SCRPRLAdapter(
        env_explicit, observation_version="O1"
    ).reset(seed=14)
    assert np.array_equal(default_observation, explicit_observation)
    assert np.array_equal(default_info["action_mask"], explicit_info["action_mask"])
    assert default_observation.shape == ((env_default.instance.num_stacks + 1) * 12,)


def test_o1_and_o2_metadata_are_versioned_without_breaking_old_fields():
    env_o1 = collision_env((1, 2, 3, 4))
    env_o2 = collision_env((1, 2, 3, 4))
    o1 = SCRPRLAdapter(env_o1).get_observation_metadata("dataset-v1")
    o2 = SCRPRLAdapter(
        env_o2, observation_version="O2"
    ).get_observation_metadata("dataset-v1")
    assert o1["observation_version"] == "O1" and "Mmax" not in o1
    assert o2["observation_version"] == "O2" and o2["Mmax"] == 6
    for key in (
        "problem_type", "num_stacks", "max_tiers", "feature_dim",
        "decision_mode", "dataset_version",
    ):
        assert o1[key] == o2[key]


def test_o2_observation_config_save_load_round_trip(tmp_path):
    config = SCRPObservationConfig(
        observation_version="O2",
        feature_dim=12,
        mmax=6,
        dataset_version="ku-galle-bacci-ds1-ds2-ec672df",
    )
    path = save_observation_config(config, tmp_path / "observation.json")
    assert load_observation_config(path) == config
    assert load_observation_config(path).to_record()["Mmax"] == 6


def test_checked_o2_observation_config_loads_with_audited_metadata():
    config = load_observation_config("experiments/configs/o2_observation_v1.json")
    assert config.observation_version == "O2"
    assert config.feature_dim == 12
    assert config.mmax == 6
    assert config.dataset_version == "ku-galle-bacci-ds1-ds2-ec672df"


def test_o2_feature_scale_is_explicit_and_all_features_are_normalized():
    env = collision_env((1, 2, 3, 4))
    adapter = O2ObservationAdapter(env.instance, env.config)
    observation = adapter.build(env.reset(seed=15))
    assert SCRP_O2_FEATURE_SCALE == (1.0,) * 12
    assert np.all((0.0 <= observation) & (observation <= 1.0))
