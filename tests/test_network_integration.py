import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hier_pg.network import HierPolicyNetwork
from scrp import (
    Container,
    SCRPConfig,
    SCRPEnvironment,
    SCRPInstance,
    SCRPRLAdapter,
    Scenario,
)


class OneDecisionSampler:
    def sample(self, instance, root_seed):
        return Scenario(root_seed, {1: 1, 2: 2}, {1: (1,), 2: (2, 3)}, "one-step")


def make_one_decision_adapter():
    instance = SCRPInstance(
        "network-one-decision",
        2,
        3,
        (Container(1, 1), Container(2, 2), Container(3, 2)),
        ((1, 2), (3,)),
        (1, 2),
    )
    core = SCRPEnvironment(SCRPConfig(2, 3), instance, OneDecisionSampler())
    return SCRPRLAdapter(core)


def make_network():
    torch.manual_seed(0)
    return HierPolicyNetwork(
        embed_dim=32,
        num_enc_layers=1,
        num_heads=4,
        ffn_dim=64,
        feature_scale=torch.ones(12),
    ).eval()


def low_log_probs(network, obs_tensor, forbidden):
    encoded = network.encode(obs_tensor)
    decoder = network.low_decoder
    query = decoder._build_query(encoded)
    context = decoder.norm(query + decoder.cross_attn(query, encoded, encoded))
    return decoder._pointer_scores(context, encoded, forbidden)


def test_o1_observation_low_forward_masking_and_greedy_legality():
    adapter = make_one_decision_adapter()
    obs, info = adapter.reset()
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    forbidden = torch.tensor(~info["action_mask"], dtype=torch.bool).unsqueeze(0)
    network = make_network()

    with torch.no_grad():
        action, log_prob = network.forward(
            obs_tensor, forbidden, greedy=True, mode="low"
        )
        log_probs = low_log_probs(network, obs_tensor, forbidden)

    assert action.shape == (1,)
    assert log_prob.shape == (1,)
    assert int(action.item()) == 1
    assert info["action_mask"][action.item()]
    assert torch.isneginf(log_probs[0, 0])
    assert torch.isfinite(log_probs[0, 1])
    assert torch.isfinite(log_prob).all()
    assert torch.allclose(log_probs.exp().sum(dim=-1), torch.ones(1))


def test_sampled_actions_can_never_select_masked_destination():
    adapter = make_one_decision_adapter()
    obs, info = adapter.reset()
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    forbidden = torch.tensor(~info["action_mask"], dtype=torch.bool).unsqueeze(0)
    network = make_network()
    with torch.no_grad():
        sampled = [
            int(network.forward(obs_tensor, forbidden, mode="low")[0].item())
            for _ in range(20)
        ]
    assert sampled == [1] * 20


def test_full_network_forward_step_smoke_episode_terminates():
    adapter = make_one_decision_adapter()
    network = make_network()
    obs, info = adapter.reset()
    trace = []
    while not info["terminated"]:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        forbidden = torch.tensor(~info["action_mask"], dtype=torch.bool).unsqueeze(0)
        with torch.no_grad():
            action, log_prob = network.forward(
                obs_tensor, forbidden, greedy=True, mode="low"
            )
        assert torch.isfinite(log_prob).all()
        selected = int(action.item())
        assert info["action_mask"][selected]
        obs, reward, terminated, truncated, info = adapter.step(selected)
        trace.append((selected, reward, terminated))
        assert not truncated

    assert trace == [(1, -1, True)]
    assert adapter.get_metrics()["relocation_count"] == 1
    assert adapter.get_metrics()["retrieval_count"] == 3
    assert np.isfinite(obs).all()
