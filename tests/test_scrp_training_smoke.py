import math

import torch

from scrp.training import (
    SCRPSanityTrainingConfig,
    make_scrp_o1_policy,
    make_scrp_training_tiny_env,
    run_scrp_low_episode,
    run_scrp_sanity_training,
)


def run_training(tmp_path):
    checkpoint = tmp_path / "scrp_phase_2_5.pt"
    config = SCRPSanityTrainingConfig(
        iterations=2,
        episodes_per_iteration=4,
        root_seed=31415,
        checkpoint_path=str(checkpoint),
    )
    result = run_scrp_sanity_training([make_scrp_training_tiny_env], config)
    return config, result, checkpoint


def test_low_only_training_loss_gradients_optimizer_and_pairing(tmp_path):
    config, result, checkpoint = run_training(tmp_path)
    assert result.optimizer_steps == config.iterations
    assert result.gradients_finite
    assert result.parameters_changed
    assert checkpoint.exists()
    assert len(result.paired_scenarios) == (
        config.iterations * config.episodes_per_iteration
    )
    assert all(record.match for record in result.paired_scenarios)

    for metrics in result.iteration_metrics:
        assert metrics.episodes == config.episodes_per_iteration
        assert metrics.high_decisions == 0
        assert metrics.low_decisions > 0
        assert metrics.invalid_action_count == 0
        assert metrics.truncated_count == 0
        assert metrics.scenario_mismatch_count == 0
        for value in (
            metrics.mean_relocations,
            metrics.mean_reward,
            metrics.mean_baseline_relocations,
            metrics.mean_advantage,
            metrics.policy_loss,
            metrics.entropy,
        ):
            assert math.isfinite(value)

    # No HIGH sample or HIGH loss was fabricated.
    assert all(parameter.grad is None for parameter in result.policy.high_decoder.parameters())
    assert any(parameter.grad is not None for parameter in result.policy.low_decoder.parameters())


def test_checkpoint_save_load_policy_optimizer_and_scrp_metadata(tmp_path):
    config, result, checkpoint_path = run_training(tmp_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert set(checkpoint) == {
        "policy_state_dict",
        "optimizer_state_dict",
        "training_config",
        "metadata",
    }
    assert checkpoint["metadata"] == {
        "problem_type": "SCRP",
        "observation_version": "O1",
        "num_stacks": 3,
        "max_tiers": 3,
        "feature_dim": 12,
        "decision_mode": "low",
        "phase": "2.5_sanity",
        "feature_scale": [1.0] * 12,
    }

    restored_policy = make_scrp_o1_policy(
        embed_dim=config.embed_dim,
        num_encoder_layers=config.num_encoder_layers,
        num_heads=config.num_heads,
        ffn_dim=config.ffn_dim,
        clip_constant=config.clip_constant,
    )
    restored_policy.load_state_dict(checkpoint["policy_state_dict"])
    restored_optimizer = torch.optim.Adam(
        restored_policy.parameters(), lr=config.learning_rate, eps=1e-5
    )
    restored_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    for expected, restored in zip(result.policy.parameters(), restored_policy.parameters()):
        assert torch.equal(expected.detach(), restored.detach())
    assert restored_optimizer.state_dict()["state"]


def test_trained_policy_still_completes_rollout(tmp_path):
    _, result, _ = run_training(tmp_path)
    trajectory = run_scrp_low_episode(
        make_scrp_training_tiny_env(),
        result.policy,
        123456,
        greedy=True,
    )
    assert trajectory.terminated
    assert not trajectory.truncated
    assert trajectory.invalid_action_count == 0
    assert trajectory.high_decisions == 0
    assert trajectory.episode_return == -trajectory.relocation_count
