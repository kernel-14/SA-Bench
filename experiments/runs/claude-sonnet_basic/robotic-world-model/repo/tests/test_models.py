"""
Unit tests for RWM and baseline models.

Tests model forward passes, autoregressive rollouts, and training steps.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from models import RoboticWorldModel, MLPWorldModel, RSSMWorldModel, TransformerWorldModel
from models import PolicyNetwork, ValueNetwork
from training import WorldModelTrainer, TeacherForcingTrainer
from utils import TrajectoryDataset, generate_synthetic_trajectories


# Test dimensions for ANYmal D
ANYMAL_OBS_SIZE = 45
ANYMAL_ACTION_SIZE = 12
ANYMAL_PRIV_SIZE = 8

# Test dimensions for Unitree G1
G1_OBS_SIZE = 96
G1_ACTION_SIZE = 29
G1_PRIV_SIZE = 30

BATCH_SIZE = 4
M = 8   # history horizon (reduced for testing)
N = 4   # forecast horizon (reduced for testing)


def test_rwm_forward_pass():
    """Test basic RWM forward pass."""
    model = RoboticWorldModel(
        obs_size=ANYMAL_OBS_SIZE,
        action_size=ANYMAL_ACTION_SIZE,
        priv_size=ANYMAL_PRIV_SIZE,
    )
    obs_hist = torch.randn(BATCH_SIZE, M, ANYMAL_OBS_SIZE)
    act_hist = torch.randn(BATCH_SIZE, M, ANYMAL_ACTION_SIZE)

    obs_mean, obs_std, priv_mean, priv_std, hidden = model(obs_hist, act_hist)

    assert obs_mean.shape == (BATCH_SIZE, M, ANYMAL_OBS_SIZE)
    assert obs_std.shape == (BATCH_SIZE, M, ANYMAL_OBS_SIZE)
    assert priv_mean.shape == (BATCH_SIZE, M, ANYMAL_PRIV_SIZE)
    assert (obs_std > 0).all()
    print("  test_rwm_forward_pass: PASSED")


def test_rwm_predict_step():
    """Test single-step prediction."""
    model = RoboticWorldModel(
        obs_size=ANYMAL_OBS_SIZE,
        action_size=ANYMAL_ACTION_SIZE,
        priv_size=ANYMAL_PRIV_SIZE,
    )
    obs = torch.randn(BATCH_SIZE, ANYMAL_OBS_SIZE)
    action = torch.randn(BATCH_SIZE, ANYMAL_ACTION_SIZE)

    obs_mean, obs_std, priv_mean, priv_std, hidden = model.predict_step(obs, action)

    assert obs_mean.shape == (BATCH_SIZE, ANYMAL_OBS_SIZE)
    assert obs_std.shape == (BATCH_SIZE, ANYMAL_OBS_SIZE)
    assert hidden.shape == (2, BATCH_SIZE, 256)
    print("  test_rwm_predict_step: PASSED")


def test_rwm_autoregressive_rollout():
    """Test full autoregressive rollout."""
    model = RoboticWorldModel(
        obs_size=ANYMAL_OBS_SIZE,
        action_size=ANYMAL_ACTION_SIZE,
        priv_size=ANYMAL_PRIV_SIZE,
    )
    obs_hist = torch.randn(BATCH_SIZE, M, ANYMAL_OBS_SIZE)
    act_hist = torch.randn(BATCH_SIZE, M, ANYMAL_ACTION_SIZE)
    future_acts = torch.randn(BATCH_SIZE, N, ANYMAL_ACTION_SIZE)

    pred_means, pred_stds, priv_means, priv_stds = model.autoregressive_rollout(
        obs_hist, act_hist, future_acts
    )

    assert pred_means.shape == (BATCH_SIZE, N, ANYMAL_OBS_SIZE)
    assert pred_stds.shape == (BATCH_SIZE, N, ANYMAL_OBS_SIZE)
    assert priv_means.shape == (BATCH_SIZE, N, ANYMAL_PRIV_SIZE)
    print("  test_rwm_autoregressive_rollout: PASSED")


def test_rwm_g1_dimensions():
    """Test with Unitree G1 dimensions."""
    model = RoboticWorldModel(
        obs_size=G1_OBS_SIZE,
        action_size=G1_ACTION_SIZE,
        priv_size=G1_PRIV_SIZE,
    )
    obs_hist = torch.randn(BATCH_SIZE, M, G1_OBS_SIZE)
    act_hist = torch.randn(BATCH_SIZE, M, G1_ACTION_SIZE)
    future_acts = torch.randn(BATCH_SIZE, N, G1_ACTION_SIZE)

    pred_means, pred_stds, priv_means, priv_stds = model.autoregressive_rollout(
        obs_hist, act_hist, future_acts
    )

    assert pred_means.shape == (BATCH_SIZE, N, G1_OBS_SIZE)
    print("  test_rwm_g1_dimensions: PASSED")


def test_mlp_baseline():
    """Test MLP baseline."""
    model = MLPWorldModel(
        obs_size=ANYMAL_OBS_SIZE,
        action_size=ANYMAL_ACTION_SIZE,
        priv_size=ANYMAL_PRIV_SIZE,
        history_horizon=M,
    )
    obs_hist = torch.randn(BATCH_SIZE, M, ANYMAL_OBS_SIZE)
    act_hist = torch.randn(BATCH_SIZE, M, ANYMAL_ACTION_SIZE)
    future_acts = torch.randn(BATCH_SIZE, N, ANYMAL_ACTION_SIZE)

    obs_mean, obs_std, priv_mean, priv_std = model(obs_hist, act_hist)
    assert obs_mean.shape == (BATCH_SIZE, ANYMAL_OBS_SIZE)

    pred_means, pred_stds, _, _ = model.autoregressive_rollout(obs_hist, act_hist, future_acts)
    assert pred_means.shape == (BATCH_SIZE, N, ANYMAL_OBS_SIZE)
    print("  test_mlp_baseline: PASSED")


def test_rssm_baseline():
    """Test RSSM baseline."""
    model = RSSMWorldModel(
        obs_size=ANYMAL_OBS_SIZE,
        action_size=ANYMAL_ACTION_SIZE,
        priv_size=ANYMAL_PRIV_SIZE,
    )
    obs_hist = torch.randn(BATCH_SIZE, M, ANYMAL_OBS_SIZE)
    act_hist = torch.randn(BATCH_SIZE, M, ANYMAL_ACTION_SIZE)
    future_acts = torch.randn(BATCH_SIZE, N, ANYMAL_ACTION_SIZE)

    pred_means, pred_stds, _, _ = model.autoregressive_rollout(obs_hist, act_hist, future_acts)
    assert pred_means.shape == (BATCH_SIZE, N, ANYMAL_OBS_SIZE)
    print("  test_rssm_baseline: PASSED")


def test_transformer_baseline():
    """Test Transformer baseline."""
    model = TransformerWorldModel(
        obs_size=ANYMAL_OBS_SIZE,
        action_size=ANYMAL_ACTION_SIZE,
        priv_size=ANYMAL_PRIV_SIZE,
        context_length=M,
    )
    obs_hist = torch.randn(BATCH_SIZE, M, ANYMAL_OBS_SIZE)
    act_hist = torch.randn(BATCH_SIZE, M, ANYMAL_ACTION_SIZE)
    future_acts = torch.randn(BATCH_SIZE, N, ANYMAL_ACTION_SIZE)

    pred_means, pred_stds, _, _ = model.autoregressive_rollout(obs_hist, act_hist, future_acts)
    assert pred_means.shape == (BATCH_SIZE, N, ANYMAL_OBS_SIZE)
    print("  test_transformer_baseline: PASSED")


def test_policy_network():
    """Test policy network."""
    policy = PolicyNetwork(obs_size=48, action_size=12)
    obs = torch.randn(BATCH_SIZE, 48)

    mean, std = policy(obs)
    assert mean.shape == (BATCH_SIZE, 12)
    assert std.shape == (BATCH_SIZE, 12)
    assert (std > 0).all()

    action, log_prob = policy.get_action(obs)
    assert action.shape == (BATCH_SIZE, 12)
    assert log_prob.shape == (BATCH_SIZE,)

    log_prob2, entropy, _ = policy.evaluate_actions(obs, action)
    assert log_prob2.shape == (BATCH_SIZE,)
    assert entropy.shape == (BATCH_SIZE,)
    print("  test_policy_network: PASSED")


def test_value_network():
    """Test value function network."""
    value_fn = ValueNetwork(obs_size=48)
    obs = torch.randn(BATCH_SIZE, 48)

    value = value_fn(obs)
    assert value.shape == (BATCH_SIZE, 1)
    print("  test_value_network: PASSED")


def test_autoregressive_training():
    """Test autoregressive training step."""
    model = RoboticWorldModel(
        obs_size=ANYMAL_OBS_SIZE,
        action_size=ANYMAL_ACTION_SIZE,
        priv_size=ANYMAL_PRIV_SIZE,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    trainer = WorldModelTrainer(
        model=model,
        optimizer=optimizer,
        history_horizon=M,
        forecast_horizon=N,
    )

    obs_hist = torch.randn(BATCH_SIZE, M, ANYMAL_OBS_SIZE)
    act_hist = torch.randn(BATCH_SIZE, M + N, ANYMAL_ACTION_SIZE)
    obs_tgt = torch.randn(BATCH_SIZE, N, ANYMAL_OBS_SIZE)
    priv_tgt = torch.randn(BATCH_SIZE, N, ANYMAL_PRIV_SIZE)

    metrics = trainer.train_step(obs_hist, act_hist, obs_tgt, priv_tgt)

    assert "loss" in metrics
    assert metrics["loss"] > 0
    print("  test_autoregressive_training: PASSED")


def test_teacher_forcing_training():
    """Test teacher forcing training step."""
    model = RoboticWorldModel(
        obs_size=ANYMAL_OBS_SIZE,
        action_size=ANYMAL_ACTION_SIZE,
        priv_size=ANYMAL_PRIV_SIZE,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    trainer = TeacherForcingTrainer(
        model=model,
        optimizer=optimizer,
        history_horizon=M,
    )

    obs_hist = torch.randn(BATCH_SIZE, M, ANYMAL_OBS_SIZE)
    act_hist = torch.randn(BATCH_SIZE, M + 1, ANYMAL_ACTION_SIZE)
    obs_tgt = torch.randn(BATCH_SIZE, 1, ANYMAL_OBS_SIZE)
    priv_tgt = torch.randn(BATCH_SIZE, 1, ANYMAL_PRIV_SIZE)

    metrics = trainer.train_step(obs_hist, act_hist, obs_tgt, priv_tgt)

    assert "loss" in metrics
    assert metrics["loss"] > 0
    print("  test_teacher_forcing_training: PASSED")


def test_dataset():
    """Test trajectory dataset creation."""
    observations, actions, privileged_info = generate_synthetic_trajectories(
        n_trajectories=5,
        trajectory_length=100,
        obs_size=ANYMAL_OBS_SIZE,
        action_size=ANYMAL_ACTION_SIZE,
        priv_size=ANYMAL_PRIV_SIZE,
    )

    dataset = TrajectoryDataset(observations, actions, M, N, privileged_info)

    assert len(dataset) > 0

    sample = dataset[0]
    assert len(sample) == 4

    obs_hist, act_hist, obs_tgt, priv_tgt = sample
    assert obs_hist.shape == (M, ANYMAL_OBS_SIZE)
    assert act_hist.shape == (M + N, ANYMAL_ACTION_SIZE)
    assert obs_tgt.shape == (N, ANYMAL_OBS_SIZE)
    assert priv_tgt.shape == (N, ANYMAL_PRIV_SIZE)
    print("  test_dataset: PASSED")


if __name__ == "__main__":
    print("=== RWM Tests ===")
    test_rwm_forward_pass()
    test_rwm_predict_step()
    test_rwm_autoregressive_rollout()
    test_rwm_g1_dimensions()

    print("\n=== Baseline Tests ===")
    test_mlp_baseline()
    test_rssm_baseline()
    test_transformer_baseline()

    print("\n=== Policy Tests ===")
    test_policy_network()
    test_value_network()

    print("\n=== Training Tests ===")
    test_autoregressive_training()
    test_teacher_forcing_training()
    test_dataset()

    print("\nAll tests passed!")
