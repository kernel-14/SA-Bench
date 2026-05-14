## trainer.py
"""
IMPALA-based Trainer for the DRC (Deep Repeated ConvLSTM) agent.

Implements on-policy V‑trace actor-critic training on a Sokoban environment
using a single‑process, batched rollout with multiple parallel environments.
All hyperparameters are read from a configuration object (config.yaml).

Classes:
    IMPALATrainer – main training class handling rollouts, loss computation,
                     optimizer steps, learning rate scheduling, and checkpointing.
"""

import os
import time
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project-level imports (avoiding circular dependencies)
# ---------------------------------------------------------------------------
from environment import SokobanEnv
from model import DRCNetwork
from utils import Config, set_seed


# ---------------------------------------------------------------------------
# Helper function for V‑trace computation
# ---------------------------------------------------------------------------
def _vtrace(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    bootstrap_value: torch.Tensor,
    logits: torch.Tensor,
    actions: torch.Tensor,
    behaviour_log_probs: Optional[torch.Tensor] = None,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
    discount: float = 0.97,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute V‑trace targets and importance weights for a batch of trajectories.

    Args:
        rewards: (T, N) tensor of rewards.
        values: (T, N) tensor of value predictions V(s_t).
        dones: (T, N) tensor of boolean episode terminations (1 if done).
        bootstrap_value: (N,) tensor of the value of the final state after the
                          unroll, i.e., V(s_{T}).
        logits: (T, N, num_actions) tensor of policy logits for each state.
        actions: (T, N) long tensor of the chosen actions.
        behaviour_log_probs: (T, N) tensor of log‑probabilities under the
                              behaviour policy. If None, they are computed
                              from logits directly (on‑policy setting).
        rho_bar: truncation level for the importance weight ρ.
        c_bar: truncation level for the importance weight c.
        discount: discount factor γ.

    Returns:
        vtrace_targets: (T, N) tensor of V‑trace targets.
        rho: (T, N) tensor of importance weights used in policy loss.
        advantages: (T, N) tensor of advantages (v_trace - V(s_t)) for policy gradient.
    """
    T, N = rewards.shape
    device = rewards.device

    # Compute target policy log‑probs and behaviour policy log‑probs
    target_probs = F.softmax(logits, dim=-1)
    pi_actions = target_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)  # (T, N)
    if behaviour_log_probs is None:
        mu_actions = pi_actions  # on‑policy
        behaviour_log_probs = torch.log(mu_actions + 1e-8)
    else:
        mu_actions = torch.exp(behaviour_log_probs)

    # Importance weights (clipped)
    rho = torch.min(rho_bar, pi_actions / (mu_actions + 1e-8))       # (T, N)
    c = torch.min(c_bar, pi_actions / (mu_actions + 1e-8))           # (T, N)

    # Work backwards to compute V‑trace targets
    vtrace_targets = torch.zeros_like(rewards)
    vs = bootstrap_value  # (N,)

    for t in reversed(range(T)):
        # Next state value and done flag
        done = dones[t].float()                     # (N,)
        next_v = vs                                 # v_{t+1} (for t=T-1 it's bootstrap)
        # V‑trace delta
        delta = rho[t] * (rewards[t] + discount * next_v * (1 - done) - values[t])
        # V‑trace value estimate v_s
        vs = values[t] + delta + discount * c[t] * (1 - done) * (vs - next_v)
        vtrace_targets[t] = vs

    # Advantages for policy gradient (typically v_trace - V(s_t))
    advantages = vtrace_targets - values

    return vtrace_targets, rho, advantages


# ---------------------------------------------------------------------------
# IMPALATrainer class
# ---------------------------------------------------------------------------
class IMPALATrainer:
    """
    IMPALA trainer for the DRC agent on Sokoban.

    Args:
        env_list: A list of SokobanEnv instances to be run in parallel.
        model: DRCNetwork instance.
        config: Config object containing all hyperparameters (see config.yaml).
    """
    def __init__(
        self,
        env_list: List[SokobanEnv],
        model: DRCNetwork,
        config: Config
    ):
        self.env_list = env_list
        self.num_envs = len(env_list)
        self.model = model
        self.config = config

        # Training hyperparameters from config
        train_cfg = config.training
        self.unroll_length = train_cfg.unroll_length
        self.batch_size = train_cfg.batch_size
        self.discount = train_cfg.discount
        self.vtrace_lambda = train_cfg.vtrace_lambda   # actually not λ but V‑trace λ? paper mentions vtrace λ = 0.97, but we use discount=γ; I'll keep both same.
        self.entropy_cost = train_cfg.entropy_cost
        self.policy_logit_l2 = train_cfg.policy_logit_l2
        self.head_l2 = train_cfg.head_l2
        self.initial_lr = train_cfg.learning_rate
        self.total_steps = train_cfg.total_steps
        self.checkpoint_interval = train_cfg.checkpoint_interval
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set device for model
        self.model.to(self.device)

        # Optimizer
        opt_name = train_cfg.optimizer.lower()
        if opt_name == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.initial_lr,
                betas=(train_cfg.adam_beta1, train_cfg.adam_beta2),
                eps=train_cfg.adam_epsilon
            )
        elif opt_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.initial_lr,
                betas=(train_cfg.adam_beta1, train_cfg.adam_beta2),
                eps=train_cfg.adam_epsilon
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

        # Training step counter (global)
        self.step = 0

        # For statistics logging (optional)
        self.episode_rewards = []
        self.episode_lengths = []

        print(f"Trainer initialised. Device: {self.device}, num_envs: {self.num_envs}, "
              f"unroll_length: {self.unroll_length}, total_steps: {self.total_steps}")

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------
    def _rollout(
        self,
        initial_hidden_states: List[Tuple[torch.Tensor, torch.Tensor]],
        initial_obs: np.ndarray
    ) -> Tuple[
        Dict[str, torch.Tensor],
        List[Tuple[torch.Tensor, torch.Tensor]],
        np.ndarray
    ]:
        """
        Perform an unroll of length self.unroll_length across all environments.

        Args:
            initial_hidden_states: list of (h,c) tuples per layer, each tensor
                                   shape (num_envs, channels, H, W).
            initial_obs: numpy array (num_envs, 8, 8, 7) of initial observations.

        Returns:
            trajectory: dict with keys:
                'observations'  (T, N, H, W, C) – stored as torch tensors (optional)
                'actions'       (T, N) long
                'rewards'       (T, N) float
                'dones'         (T, N) float (0/1)
                'logits'        (T, N, num_actions)
                'values'        (T, N)
                'behaviour_log_probs' (T, N)
                'bootstrap_value' (N,) detached
            final_hidden_states: updated hidden states after the unroll.
            final_obs: numpy array (num_envs, 8, 8, 7) after the last environment
                       step (used as initial obs for next rollout).
        """
        N = self.num_envs
        T = self.unroll_length

        hidden_states = initial_hidden_states
        obs = initial_obs

        # Storage lists
        obs_list = []          # store original numpy obs for debugging (optional)
        actions_list = []      # (N,) tensors
        rewards_list = []      # (N,) tensors
        dones_list = []        # (N,) tensors
        logits_list = []       # (N, num_actions)
        values_list = []       # (N,)
        log_probs_list = []    # (N,)

        for t in range(T):
            # Convert observation to tensor
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).to(self.device)  # (N,8,8,7)
            obs_tensor = obs_tensor.permute(0, 3, 1, 2)  # (N,7,8,8)

            # Forward pass through the agent (keep gradients)
            logits, values, new_hidden = self.model(obs_tensor, hidden_states,
                                                    num_ticks=self.config.agent.internal_ticks)

            # Action sampling
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            # Step all environments (list of envs)
            next_obs = np.empty_like(obs)
            reward = np.empty(N, dtype=np.float32)
            done_flags = np.zeros(N, dtype=np.bool_)
            for i in range(N):
                act = action[i].item()
                nobs, rew, terminated, truncated, _ = self.env_list[i].step(act)
                done = terminated or truncated
                next_obs[i] = nobs
                reward[i] = rew
                done_flags[i] = done
                if done:
                    # Reset this environment
                    next_obs[i] = self.env_list[i].reset()

            # Convert to tensors
            action_tensor = action.detach().cpu()  # keep on CPU for env stepping, but we'll move to device for storage
            reward_tensor = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
            done_tensor = torch.as_tensor(done_flags, dtype=torch.float32, device=self.device)

            # Mask hidden states for done environments (zero them)
            dones_mask = done_tensor.view(N, 1, 1, 1)
            new_hidden_masked = []
            for layer_idx, (h, c) in enumerate(new_hidden):
                h = h * (1 - dones_mask)
                c = c * (1 - dones_mask)
                new_hidden_masked.append((h, c))

            # Store tensors for later loss
            obs_list.append(obs)  # original numpy, keep for reference
            actions_list.append(action_tensor.to(self.device).long())
            rewards_list.append(reward_tensor)
            dones_list.append(done_tensor)
            logits_list.append(logits)            # requires grad
            values_list.append(values)            # requires grad
            log_probs_list.append(log_prob)        # requires grad

            # Update for next step
            obs = next_obs
            hidden_states = new_hidden_masked

        # After all steps, compute bootstrap value using final observations
        final_obs_tensor = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        final_obs_tensor = final_obs_tensor.permute(0, 3, 1, 2)  # (N,7,8,8)
        with torch.no_grad():  # no grad needed for bootstrap (it's a target)
            _, bootstrap_value, _ = self.model(final_obs_tensor, hidden_states,
                                               num_ticks=self.config.agent.internal_ticks)
            bootstrap_value = bootstrap_value.squeeze(-1).cpu()  # shape (N,)

        # Stack tensors along time dimension
        trajectory = {
            'observations': obs_list,  # keep as list of numpy arrays (maybe not sent to GPU)
            'actions': torch.stack(actions_list, dim=0),          # (T, N)
            'rewards': torch.stack(rewards_list, dim=0),          # (T, N)
            'dones': torch.stack(dones_list, dim=0),              # (T, N)
            'logits': torch.stack(logits_list, dim=0),            # (T, N, num_actions)
            'values': torch.stack(values_list, dim=0),            # (T, N)
            'behaviour_log_probs': torch.stack(log_probs_list, dim=0),  # (T, N)
            'bootstrap_value': bootstrap_value.to(self.device),   # (N,)
        }

        # Update global step counter
        self.step += N * T

        return trajectory, hidden_states, obs

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------
    def _compute_vtrace_loss(self, trajectory: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute V‑trace policy/value loss, entropy, and regularisation.

        Args:
            trajectory: dict from _rollout with keys:
                'logits'         (T, N, num_actions)
                'values'         (T, N)
                'actions'        (T, N) long
                'rewards'        (T, N)
                'dones'          (T, N) float 0/1
                'behaviour_log_probs' (T, N)
                'bootstrap_value' (N,)

        Returns:
            loss: scalar tensor, total loss with gradient.
            metrics: dict with individual loss components (for logging).
        """
        logits = trajectory['logits']                     # (T, N, A)
        values = trajectory['values']                     # (T, N)
        actions = trajectory['actions']                   # (T, N) long
        rewards = trajectory['rewards']                   # (T, N)
        dones = trajectory['dones']                       # (T, N) float
        behaviour_log_probs = trajectory['behaviour_log_probs'].detach()  # (T, N)
        bootstrap_value = trajectory['bootstrap_value']   # (N,)

        T, N, A = logits.shape
        device = logits.device

        # --- V‑trace targets and advantages ---
        vtrace_targets, rho, advantages = _vtrace(
            rewards=rewards,
            values=values,
            dones=dones.long(),         # _vtrace expects int/long for dones
            bootstrap_value=bootstrap_value,
            logits=logits,
            actions=actions,
            behaviour_log_probs=behaviour_log_probs,
            rho_bar=1.0,
            c_bar=1.0,
            discount=self.discount,
        )

        # --- Policy loss ---
        target_probs = F.softmax(logits, dim=-1)
        dist = Categorical(target_probs)
        action_log_probs = dist.log_prob(actions)  # (T, N)
        # Use rho weighting (Sutton et al. IMPALA: -rho * log π * (v_s - V))
        policy_loss = - (rho * action_log_probs * advantages.detach()).mean()  # scalar

        # --- Value loss ---
        value_loss = 0.5 * ((vtrace_targets - values) ** 2).mean()

        # --- Entropy bonus ---
        entropy = dist.entropy().mean()   # average over time and envs
        entropy_loss = -self.entropy_cost * entropy

        # --- Regularisation ---
        # L2 penalty on action logits (per step, average over T and N and actions perhaps)
        logit_l2_loss = self.policy_logit_l2 * (logits ** 2).mean()

        # L2 penalty on policy and value head weights (we add weight_decay via optimizer, but also explicit)
        # The paper mentions L2 regularisation of strength 1e-5 on policy and value heads.
        # We'll compute it manually to match paper exactly.
        head_l2_loss = 0.0
        for name, param in self.model.named_parameters():
            if 'policy_head' in name or 'value_head' in name:
                head_l2_loss += (param ** 2).sum()
        head_l2_loss = self.head_l2 * head_l2_loss

        # --- Total loss ---
        total_loss = policy_loss + value_loss + entropy_loss + logit_l2_loss + head_l2_loss

        metrics = {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'logit_l2': logit_l2_loss.item(),
            'head_l2': head_l2_loss.item(),
        }
        return total_loss, metrics

    # ------------------------------------------------------------------
    # Learning rate scheduler
    # ------------------------------------------------------------------
    def _update_learning_rate(self) -> None:
        """Linearly decay learning rate from initial_lr to 0 over total_steps."""
        lr = self.initial_lr * max(0.0, 1.0 - self.step / self.total_steps)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    def train(self, total_steps: Optional[int] = None) -> None:
        """
        Run the full training loop.

        Args:
            total_steps: if None, use config.training.total_steps.
        """
        if total_steps is None:
            total_steps = self.total_steps

        # Initialise hidden states and observations
        N = self.num_envs
        hidden_states = self.model.initial_state(batch_size=N)
        # Reset all environments and get first observations
        initial_obs = np.zeros((N, 8, 8, 7), dtype=np.float32)
        for i in range(N):
            initial_obs[i] = self.env_list[i].reset()

        # Progress bar
        pbar = tqdm(total=total_steps, desc="Training", initial=self.step)
        last_checkpoint_step = self.step

        while self.step < total_steps:
            # Rollout
            trajectory, hidden_states, initial_obs = self._rollout(hidden_states, initial_obs)

            # Compute loss and backprop
            loss, metrics = self._compute_vtrace_loss(trajectory)

            self.optimizer.zero_grad()
            loss.backward()
            # Optional gradient clipping (not mentioned in paper, but safe)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 40.0)
            self.optimizer.step()

            # Update learning rate
            self._update_learning_rate()

            # Logging
            pbar.set_postfix(
                loss=f"{metrics['total_loss']:.3f}",
                pol=f"{metrics['policy_loss']:.3f}",
                val=f"{metrics['value_loss']:.3f}",
                ent=f"{metrics['entropy']:.3f}",
                lr=f"{self.optimizer.param_groups[0]['lr']:.2e}"
            )
            pbar.update(self.unroll_length * N)  # steps advanced
            # We already updated self.step inside _rollout, but we can also update pbar manually

            # Periodic checkpointing
            if self.step - last_checkpoint_step >= self.checkpoint_interval:
                ckpt_path = os.path.join(
                    self.config.checkpoint_dir,
                    f"checkpoint_step_{self.step}.pt"
                )
                self.save_checkpoint(ckpt_path)
                last_checkpoint_step = self.step
                print(f"\nCheckpoint saved at step {self.step}")

        pbar.close()
        print("Training completed.")

    # ------------------------------------------------------------------
    # Checkpoint functions
    # ------------------------------------------------------------------
    def save_checkpoint(self, path: str) -> None:
        """Save model, optimizer, and step to a checkpoint file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        """Load model, optimizer, and step from a checkpoint file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step = checkpoint['step']
        print(f"Loaded checkpoint from {path} – step {self.step}")

