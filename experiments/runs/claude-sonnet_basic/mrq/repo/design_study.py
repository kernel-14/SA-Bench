"""
Design study variants for MR.Q.

Implements the ablation variants from Table 2 of the paper:
"Towards General-Purpose Model-Free RL (MR.Q)" - Fujimoto et al., 2025

Usage:
    python design_study.py --variant linear_value --env_type gym --env_name HalfCheetah-v4 --seed 0
"""

import argparse
import os
import sys
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mrq.agent import MRQ
from mrq.networks import ZS_DIM, ZSA_DIM, StateEncoder, StateActionEncoder, ValueNetwork, PolicyNetwork
from utils.replay_buffer import ReplayBuffer
from utils.reward_encoding import get_reward_bins, two_hot_encode
from envs.wrappers import make_env
from train import evaluate, BENCHMARK_CONFIGS


class MRQLinearValue(MRQ):
    """
    Variant: Linear value function.
    Replace non-linear Q with linear weights w: Q(z_sa) = z_sa^T w
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace value networks with linear layers
        self.Q1 = nn.Linear(ZSA_DIM, 1).to(self.device)
        self.Q2 = nn.Linear(ZSA_DIM, 1).to(self.device)
        self.Q1_target = copy.deepcopy(self.Q1).to(self.device)
        self.Q2_target = copy.deepcopy(self.Q2).to(self.device)
        # Reinitialize value optimizer
        value_params = list(self.Q1.parameters()) + list(self.Q2.parameters())
        self.value_optimizer = torch.optim.AdamW(
            value_params, lr=3e-4, weight_decay=1e-4, eps=1e-8
        )


class MRQDynamicsTarget(MRQ):
    """
    Variant: Use state-action embedding as dynamics target.
    Instead of z_s' from target encoder, use z_sa' from target sa_encoder.
    """

    def update_encoder(self, seq_batch):
        states = seq_batch["states"]
        actions = seq_batch["actions"]
        rewards = seq_batch["rewards"]
        dones = seq_batch["dones"]

        total_loss = torch.tensor(0.0, device=self.device)
        zs_tilde = self.encoder(states[0])

        for t in range(self.enc_horizon):
            model_out, zsa = self.sa_encoder(zs_tilde, actions[t])
            zs_pred = model_out[:, :ZS_DIM]
            r_logits = model_out[:, ZS_DIM:ZS_DIM + self.reward_bins]
            d_pred = model_out[:, -1:]

            # Reward loss
            r_target = rewards[t + 1].squeeze(-1)
            two_hot = two_hot_encode(r_target, self.bins)
            reward_loss = F.cross_entropy(r_logits, two_hot)

            # Dynamics loss: use z_sa' from target sa_encoder (not z_s')
            with torch.no_grad():
                zs_next = self.encoder_target(states[t + 1])
                # Get target action for next state
                _, a_next = self.policy_target(zs_next)
                _, zsa_target = self.sa_encoder_target(zs_next, a_next)
            dynamics_loss = F.mse_loss(zs_pred, zsa_target)

            terminal_loss = F.mse_loss(d_pred, dones[t + 1])

            step_loss = (
                self.lambda_reward * reward_loss
                + self.lambda_dynamics * dynamics_loss
                + self.lambda_terminal_eff * terminal_loss
            )
            total_loss = total_loss + step_loss
            zs_tilde = zs_pred.detach()

        self.enc_optimizer.zero_grad()
        total_loss.backward()
        self.enc_optimizer.step()
        return total_loss.item()


class MRQNoTargetEncoder(MRQ):
    """
    Variant: No target encoder.
    Use current encoder for dynamics target (jointly optimized).
    """

    def update_encoder(self, seq_batch):
        states = seq_batch["states"]
        actions = seq_batch["actions"]
        rewards = seq_batch["rewards"]
        dones = seq_batch["dones"]

        total_loss = torch.tensor(0.0, device=self.device)
        zs_tilde = self.encoder(states[0])

        for t in range(self.enc_horizon):
            model_out, zsa = self.sa_encoder(zs_tilde, actions[t])
            zs_pred = model_out[:, :ZS_DIM]
            r_logits = model_out[:, ZS_DIM:ZS_DIM + self.reward_bins]
            d_pred = model_out[:, -1:]

            r_target = rewards[t + 1].squeeze(-1)
            two_hot = two_hot_encode(r_target, self.bins)
            reward_loss = F.cross_entropy(r_logits, two_hot)

            # Dynamics loss: use CURRENT encoder (no target network)
            zs_target = self.encoder(states[t + 1])  # No no_grad!
            dynamics_loss = F.mse_loss(zs_pred, zs_target)

            terminal_loss = F.mse_loss(d_pred, dones[t + 1])

            step_loss = (
                self.lambda_reward * reward_loss
                + self.lambda_dynamics * dynamics_loss
                + self.lambda_terminal_eff * terminal_loss
            )
            total_loss = total_loss + step_loss
            zs_tilde = zs_pred.detach()

        self.enc_optimizer.zero_grad()
        total_loss.backward()
        self.enc_optimizer.step()
        return total_loss.item()


class MRQMSEReward(MRQ):
    """
    Variant: MSE reward loss instead of categorical cross-entropy.
    """

    def update_encoder(self, seq_batch):
        states = seq_batch["states"]
        actions = seq_batch["actions"]
        rewards = seq_batch["rewards"]
        dones = seq_batch["dones"]

        total_loss = torch.tensor(0.0, device=self.device)
        zs_tilde = self.encoder(states[0])

        for t in range(self.enc_horizon):
            model_out, zsa = self.sa_encoder(zs_tilde, actions[t])
            zs_pred = model_out[:, :ZS_DIM]
            r_logits = model_out[:, ZS_DIM:ZS_DIM + self.reward_bins]
            d_pred = model_out[:, -1:]

            # MSE reward loss (instead of categorical)
            # Use mean of bins as scalar prediction
            bins = self.bins.to(self.device)
            r_pred_scalar = (torch.softmax(r_logits, dim=-1) * bins).sum(dim=-1, keepdim=True)
            reward_loss = F.mse_loss(r_pred_scalar, rewards[t + 1])

            with torch.no_grad():
                zs_target = self.encoder_target(states[t + 1])
            dynamics_loss = F.mse_loss(zs_pred, zs_target)
            terminal_loss = F.mse_loss(d_pred, dones[t + 1])

            step_loss = (
                self.lambda_reward * reward_loss
                + self.lambda_dynamics * dynamics_loss
                + self.lambda_terminal_eff * terminal_loss
            )
            total_loss = total_loss + step_loss
            zs_tilde = zs_pred.detach()

        self.enc_optimizer.zero_grad()
        total_loss.backward()
        self.enc_optimizer.step()
        return total_loss.item()


class MRQNoRewardScaling(MRQ):
    """
    Variant: No reward scaling (r_bar = r_bar' = 1).
    """

    def update_value(self, seq_batch, indices):
        # Override reward scale to 1
        orig_scale = self.reward_scale
        orig_scale_target = self.reward_scale_target
        self.reward_scale = 1.0
        self.reward_scale_target = 1.0
        result = super().update_value(seq_batch, indices)
        self.reward_scale = orig_scale
        self.reward_scale_target = orig_scale_target
        return result


class MRQNoMin(MRQ):
    """
    Variant: Use mean instead of min over twin critics.
    """

    def update_value(self, seq_batch, indices):
        states = seq_batch["states"]
        actions = seq_batch["actions"]
        rewards = seq_batch["rewards"]
        dones = seq_batch["dones"]

        batch_size = states.shape[1]

        with torch.no_grad():
            zs_HQ = self.encoder_target(states[self.q_horizon])
            _, a_target = self.policy_target(zs_HQ)

            if self.discrete:
                noise = torch.randn_like(a_target) * self.target_noise_std
                a_idx = (a_target + noise).argmax(dim=-1)
                a_target_noisy = F.one_hot(a_idx, self.action_dim).float()
            else:
                noise = torch.clamp(
                    torch.randn_like(a_target) * self.target_noise_std,
                    -self.target_noise_clip, self.target_noise_clip
                )
                a_target_noisy = torch.clamp(a_target + noise, -1.0, 1.0)

            _, zsa_target = self.sa_encoder_target(zs_HQ, a_target_noisy)
            Q1_target = self.Q1_target(zsa_target)
            Q2_target = self.Q2_target(zsa_target)
            # MEAN instead of MIN
            Q_target_mean = (Q1_target + Q2_target) / 2.0

            discounted_return = torch.zeros(batch_size, 1, device=self.device)
            not_done = torch.ones(batch_size, 1, device=self.device)
            for t in range(self.q_horizon):
                discounted_return = discounted_return + not_done * (self.gamma ** t) * rewards[t]
                not_done = not_done * (1.0 - dones[t])

            r_scale = max(self.reward_scale, 1e-8)
            target = (1.0 / r_scale) * (
                discounted_return + not_done * (self.gamma ** self.q_horizon) * self.reward_scale_target * Q_target_mean
            )

        with torch.no_grad():
            zs_0 = self.encoder(states[0])
            _, zsa_0 = self.sa_encoder(zs_0, actions[0])

        Q1_pred = self.Q1(zsa_0.detach())
        Q2_pred = self.Q2(zsa_0.detach())

        td_error1 = Q1_pred - target
        td_error2 = Q2_pred - target
        value_loss = F.huber_loss(Q1_pred, target) + F.huber_loss(Q2_pred, target)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.Q1.parameters()) + list(self.Q2.parameters()),
            self.grad_clip_value
        )
        self.value_optimizer.step()

        td_errors = ((td_error1 + td_error2) / 2.0).abs().detach().cpu().numpy().squeeze()
        return value_loss.item(), td_errors


class MRQNoLAP(MRQ):
    """
    Variant: No LAP (uniform sampling + MSE loss).
    """

    def update_value(self, seq_batch, indices):
        states = seq_batch["states"]
        actions = seq_batch["actions"]
        rewards = seq_batch["rewards"]
        dones = seq_batch["dones"]

        batch_size = states.shape[1]

        with torch.no_grad():
            zs_HQ = self.encoder_target(states[self.q_horizon])
            _, a_target = self.policy_target(zs_HQ)

            if self.discrete:
                noise = torch.randn_like(a_target) * self.target_noise_std
                a_idx = (a_target + noise).argmax(dim=-1)
                a_target_noisy = F.one_hot(a_idx, self.action_dim).float()
            else:
                noise = torch.clamp(
                    torch.randn_like(a_target) * self.target_noise_std,
                    -self.target_noise_clip, self.target_noise_clip
                )
                a_target_noisy = torch.clamp(a_target + noise, -1.0, 1.0)

            _, zsa_target = self.sa_encoder_target(zs_HQ, a_target_noisy)
            Q1_target = self.Q1_target(zsa_target)
            Q2_target = self.Q2_target(zsa_target)
            Q_target_min = torch.min(Q1_target, Q2_target)

            discounted_return = torch.zeros(batch_size, 1, device=self.device)
            not_done = torch.ones(batch_size, 1, device=self.device)
            for t in range(self.q_horizon):
                discounted_return = discounted_return + not_done * (self.gamma ** t) * rewards[t]
                not_done = not_done * (1.0 - dones[t])

            r_scale = max(self.reward_scale, 1e-8)
            target = (1.0 / r_scale) * (
                discounted_return + not_done * (self.gamma ** self.q_horizon) * self.reward_scale_target * Q_target_min
            )

        with torch.no_grad():
            zs_0 = self.encoder(states[0])
            _, zsa_0 = self.sa_encoder(zs_0, actions[0])

        Q1_pred = self.Q1(zsa_0.detach())
        Q2_pred = self.Q2(zsa_0.detach())

        # MSE loss instead of Huber
        td_error1 = Q1_pred - target
        td_error2 = Q2_pred - target
        value_loss = F.mse_loss(Q1_pred, target) + F.mse_loss(Q2_pred, target)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.Q1.parameters()) + list(self.Q2.parameters()),
            self.grad_clip_value
        )
        self.value_optimizer.step()

        td_errors = ((td_error1 + td_error2) / 2.0).abs().detach().cpu().numpy().squeeze()
        return value_loss.item(), td_errors


class MRQNoMR(MRQ):
    """
    Variant: No model-based representation learning.
    Train encoder end-to-end with value function only.
    """

    def update_encoder(self, seq_batch):
        # No encoder update - encoder is trained with value function
        return 0.0

    def update_value(self, seq_batch, indices):
        """Value update WITH encoder gradients (end-to-end training)."""
        states = seq_batch["states"]
        actions = seq_batch["actions"]
        rewards = seq_batch["rewards"]
        dones = seq_batch["dones"]

        batch_size = states.shape[1]

        with torch.no_grad():
            zs_HQ = self.encoder_target(states[self.q_horizon])
            _, a_target = self.policy_target(zs_HQ)

            if self.discrete:
                noise = torch.randn_like(a_target) * self.target_noise_std
                a_idx = (a_target + noise).argmax(dim=-1)
                a_target_noisy = F.one_hot(a_idx, self.action_dim).float()
            else:
                noise = torch.clamp(
                    torch.randn_like(a_target) * self.target_noise_std,
                    -self.target_noise_clip, self.target_noise_clip
                )
                a_target_noisy = torch.clamp(a_target + noise, -1.0, 1.0)

            _, zsa_target = self.sa_encoder_target(zs_HQ, a_target_noisy)
            Q1_target = self.Q1_target(zsa_target)
            Q2_target = self.Q2_target(zsa_target)
            Q_target_min = torch.min(Q1_target, Q2_target)

            discounted_return = torch.zeros(batch_size, 1, device=self.device)
            not_done = torch.ones(batch_size, 1, device=self.device)
            for t in range(self.q_horizon):
                discounted_return = discounted_return + not_done * (self.gamma ** t) * rewards[t]
                not_done = not_done * (1.0 - dones[t])

            r_scale = max(self.reward_scale, 1e-8)
            target = (1.0 / r_scale) * (
                discounted_return + not_done * (self.gamma ** self.q_horizon) * self.reward_scale_target * Q_target_min
            )

        # End-to-end: gradients flow through encoder
        zs_0 = self.encoder(states[0])
        _, zsa_0 = self.sa_encoder(zs_0, actions[0])

        Q1_pred = self.Q1(zsa_0)
        Q2_pred = self.Q2(zsa_0)

        td_error1 = Q1_pred - target
        td_error2 = Q2_pred - target
        value_loss = F.huber_loss(Q1_pred, target) + F.huber_loss(Q2_pred, target)

        # Update both encoder and value
        self.enc_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.Q1.parameters()) + list(self.Q2.parameters()),
            self.grad_clip_value
        )
        self.enc_optimizer.step()
        self.value_optimizer.step()

        td_errors = ((td_error1 + td_error2) / 2.0).abs().detach().cpu().numpy().squeeze()
        return value_loss.item(), td_errors


class MRQOneStepReturn(MRQ):
    """
    Variant: 1-step return (no multi-step).
    """

    def __init__(self, *args, **kwargs):
        kwargs["q_horizon"] = 1
        super().__init__(*args, **kwargs)


class MRQNoUnroll(MRQ):
    """
    Variant: No encoder unrolling (H_enc = 1).
    """

    def __init__(self, *args, **kwargs):
        kwargs["enc_horizon"] = 1
        super().__init__(*args, **kwargs)


VARIANTS = {
    "mrq": MRQ,
    "linear_value": MRQLinearValue,
    "dynamics_target": MRQDynamicsTarget,
    "no_target_encoder": MRQNoTargetEncoder,
    "mse_reward": MRQMSEReward,
    "no_reward_scaling": MRQNoRewardScaling,
    "no_min": MRQNoMin,
    "no_lap": MRQNoLAP,
    "no_mr": MRQNoMR,
    "one_step_return": MRQOneStepReturn,
    "no_unroll": MRQNoUnroll,
}


def train_variant(args):
    """Train a specific MR.Q variant."""
    import time

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    env = make_env(args.env_type, args.env_name, seed=args.seed)
    eval_env = make_env(args.env_type, args.env_name, seed=args.seed + 100)

    config = BENCHMARK_CONFIGS[args.env_type]
    total_steps = args.total_steps or config["total_steps"]
    eval_freq = args.eval_freq or config["eval_freq"]
    random_steps = 10_000

    enc_horizon = 5
    q_horizon = 3
    seq_len = max(enc_horizon, q_horizon) + 1

    buffer = ReplayBuffer(
        state_dim=env.state_dim or 1,
        action_dim=env.action_dim,
        max_size=1_000_000,
        batch_size=256,
        image_obs=env.image_obs,
        state_channels=env.state_channels,
        image_size=84,
        seq_len=seq_len,
        lap_alpha=0.4 if args.variant != "no_lap" else 0.0,
        min_priority=1.0,
        device=device,
    )

    AgentClass = VARIANTS[args.variant]
    agent = AgentClass(
        state_dim=env.state_dim or 1,
        action_dim=env.action_dim,
        discrete=env.discrete,
        image_obs=env.image_obs,
        state_channels=env.state_channels,
        action_scale=env.action_scale,
        enc_horizon=enc_horizon,
        q_horizon=q_horizon,
        device=device,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    results_file = os.path.join(
        args.output_dir,
        f"{args.variant}_{args.env_type}_{args.env_name}_seed{args.seed}.csv"
    )

    state = env.reset()
    episode_reward = 0.0
    eval_rewards = []
    eval_steps = []

    for t in range(1, total_steps + 1):
        if t <= random_steps:
            if env.discrete:
                action = np.random.randint(env.action_dim)
            else:
                action = np.random.uniform(-1.0, 1.0, env.action_dim).astype(np.float32)
        else:
            action = agent.select_action(state, explore=True)

        next_state, reward, done, terminated, info = env.step(action)
        episode_reward += reward

        if terminated:
            agent.update_terminal_weight(True)

        if env.discrete:
            action_stored = np.zeros(env.action_dim, dtype=np.float32)
            action_stored[action] = 1.0
        else:
            action_stored = action

        buffer.add(state, action_stored, reward, done)
        state = next_state

        if done:
            state = env.reset()
            episode_reward = 0.0

        if t > random_steps and len(buffer) >= 256:
            agent.train(buffer)

        if t % eval_freq == 0:
            eval_reward = evaluate(agent, eval_env, n_episodes=10)
            eval_rewards.append(eval_reward)
            eval_steps.append(t)
            print(f"[{args.variant}] Step {t}: {eval_reward:.2f}")

            with open(results_file, "w") as f:
                f.write("step,reward\n")
                for s, r in zip(eval_steps, eval_rewards):
                    f.write(f"{s},{r:.4f}\n")

    env.close()
    eval_env.close()
    return eval_rewards


def main():
    parser = argparse.ArgumentParser(description="MR.Q Design Study")
    parser.add_argument("--variant", type=str, default="mrq",
                        choices=list(VARIANTS.keys()))
    parser.add_argument("--env_type", type=str, required=True,
                        choices=["gym", "dmc_proprio", "dmc_visual", "atari"])
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--eval_freq", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="results/design_study")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    train_variant(args)


if __name__ == "__main__":
    main()
