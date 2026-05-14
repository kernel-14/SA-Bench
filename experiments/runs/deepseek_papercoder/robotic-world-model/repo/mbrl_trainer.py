"""
mbrl_trainer.py

Implements the MBPOTrainer class that orchestrates Model‑Based Policy Optimization
with Proximal Policy Optimization (MBPO‑PPO) as described in the Robotic World Model
paper.  The trainer interacts with a real simulation environment, a pre‑trained
world model, a PPO policy, and a real‑experience replay buffer.

All hyper‑parameters are drawn from the configuration object (matching config.yaml).
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

# Import project‑specific modules (must be available on PYTHONPATH)
from env_utils import IsaacEnvWrapper, compute_reward, ROBOT_CONFIGS
from dataset import TrajectoryBuffer
from world_model import WorldModel, RWM   # RWM is used for type hints
from ppo_agent import PPOAgent


# ------------------------------------------------------------------------------
# Helper function: construct policy observation from world‑model prediction
# ------------------------------------------------------------------------------
def make_policy_obs(
    pred_world_obs: torch.Tensor,          # (B, obs_dim)  predicted full state
    command: torch.Tensor,                 # (B, 3)  [lin_vel_x, lin_vel_y, ang_vel_z]
    last_action: torch.Tensor,             # (B, act_dim)
    robot_cfg: Dict[str, Any],
    add_last_action: bool = True,
) -> torch.Tensor:
    """
    Build the policy input observation from the predicted world observation,
    the velocity command, and the previous action.

    The order follows Tables S5 (policy observation) for the given robot.
    """
    slices = robot_cfg["world_slices"]
    # Extract relevant parts from world observation
    base_lin_vel  = pred_world_obs[:, slices["base_lin_vel"][0]:slices["base_lin_vel"][1]]
    base_ang_vel  = pred_world_obs[:, slices["base_ang_vel"][0]:slices["base_ang_vel"][1]]
    gravity       = pred_world_obs[:, slices["gravity"][0]:slices["gravity"][1]]
    joint_pos     = pred_world_obs[:, slices["joint_pos"][0]:slices["joint_pos"][1]]
    joint_vel     = pred_world_obs[:, slices["joint_vel"][0]:slices["joint_vel"][1]]

    # Start building the vector
    obs_parts = [base_lin_vel, base_ang_vel, gravity, command, joint_pos, joint_vel]
    if add_last_action:
        obs_parts.append(last_action)

    return torch.cat(obs_parts, dim=-1)


# ------------------------------------------------------------------------------
# Termination prediction from privileged information
# ------------------------------------------------------------------------------
def get_termination(
    privileged: torch.Tensor,     # (B, priv_dim)
    robot: str,
    robot_cfg: Dict[str, Any],
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Determine termination flags based on predicted privileged info.

    For ANYmal D: termination if any knee contact > threshold.
    For Unitree G1: termination if any body contact > threshold.
    Returns a boolean tensor of shape (B,).
    """
    if robot == "anymal_d":
        knee_contact = privileged[:, robot_cfg["priv_slices"]["knee_contact"][0]:
                                   robot_cfg["priv_slices"]["knee_contact"][1]]
        return (knee_contact.max(dim=-1).values > threshold)
    elif robot == "unitree_g1":
        body_contact = privileged[:, robot_cfg["priv_slices"]["body_contact"][0]:
                                   robot_cfg["priv_slices"]["body_contact"][1]]
        return (body_contact.max(dim=-1).values > threshold)
    else:
        # Fallback: never terminate
        return torch.zeros(privileged.shape[0], dtype=torch.bool, device=privileged.device)


# ------------------------------------------------------------------------------
# Foot air‑time tracking helper
# ------------------------------------------------------------------------------
def update_air_time(
    foot_contact: torch.Tensor,    # (B, num_feet) binary contact (1=on ground)
    air_time: torch.Tensor,        # (B, num_feet) current air‑time counters
    dt: float,
) -> torch.Tensor:
    """
    Accumulate air time per foot: when foot is not in contact (contact≈0),
    increase the counter by dt.  When contact occurs, reset to 0.
    Returns updated air_time.
    """
    # foot_contact may be float (logits) – threshold into binary
    foot_on_ground = (foot_contact > 0.5).float()
    # increment when foot is in the air (on_ground == 0)
    air_time = (air_time + dt) * (1.0 - foot_on_ground)
    return air_time


# ------------------------------------------------------------------------------
# MBPOTrainer
# ------------------------------------------------------------------------------
class MBPOTrainer:
    """Model‑Based Policy Optimization trainer with PPO."""

    def __init__(
        self,
        env: IsaacEnvWrapper,
        world_model: WorldModel,
        ppo_agent: PPOAgent,
        config: Dict[str, Any],
    ):
        """
        Args:
            env:          The real simulation environment (single instance).
            world_model:  Pre‑trained (or freshly initialised) world model (RWM).
            ppo_agent:    PPO policy and value network.
            config:       Configuration dictionary (typically loaded from config.yaml).
        """
        self.env = env
        self.world_model = world_model
        self.ppo_agent = ppo_agent

        # General hyper‑parameters from the config (all under the 'mbrl_ppo' section)
        mcfg = config["mbrl_ppo"]
        self.imagination_envs = mcfg["imagination_envs"]      # 4096
        self.imagination_steps = mcfg["imagination_steps"]    # 100
        self.real_buffer_capacity = mcfg["real_buffer_capacity"]  # 1000
        self.learning_rate = mcfg.get("learning_rate", 0.001)
        self.ppo_epochs = mcfg["ppo_epochs"]
        self.ppo_minibatches = mcfg["ppo_minibatches"]
        self.discount_factor = mcfg["discount_factor"]        # gamma
        self.clip_range = mcfg["clip_range"]
        self.entropy_coef = mcfg["entropy_coef"]
        self.value_loss_coef = 0.5                            # not in config, standard
        self.max_grad_norm = 0.5                              # not in config, standard
        self.target_kl = mcfg.get("target_kl", 0.01)

        # Additional parameters not in the YAML snippet (we provide defaults)
        # Number of real steps to collect per iteration (reasonable default)
        self.real_steps_per_iter = mcfg.get("real_steps_per_iter", 100)
        # GAE lambda (commonly 0.95)
        self.gae_lambda = mcfg.get("gae_lambda", 0.95)
        # Step time for the environment (0.02 s)
        self.step_time = config.get("environment", {}).get("step_time", 0.02)

        # Derive robot info
        robot_name = config["environment"]["robot"]
        if robot_name not in ROBOT_CONFIGS:
            raise ValueError(f"Unsupported robot: {robot_name}")
        self.robot = robot_name
        self.robot_cfg = ROBOT_CONFIGS[robot_name]
        self.obs_dim = self.robot_cfg["world_obs_dim"]
        self.act_dim = self.robot_cfg["action_dim"]
        self.priv_dim = self.robot_cfg["priv_dim"]

        # Real replay buffer (FIFO)
        self.real_buffer = TrajectoryBuffer(
            capacity_transitions=self.real_buffer_capacity,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            priv_dim=self.priv_dim,
            device="cpu",   # store on CPU to save GPU memory
        )

        # Device (use GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.world_model.to(self.device)
        self.ppo_agent.device = self.device

        # Internal state for data collection
        self._collect_obs: Optional[torch.Tensor] = None
        self._collect_priv: Optional[torch.Tensor] = None
        self._collect_done: bool = True

    # ------------------------------------------------------------------
    # Real data collection
    # ------------------------------------------------------------------
    def collect_real_data(self, steps: int) -> Dict[str, float]:
        """
        Interact with the real environment for `steps` transitions,
        storing them in the real replay buffer.

        Returns a dictionary of average reward and episode length for logging.
        """
        total_reward = 0.0
        total_steps = 0
        episode_rewards = []
        ep_reward = 0.0
        ep_length = 0

        # Ensure the environment is reset if needed
        if self._collect_done:
            obs_dict, info = self.env.reset()
            self._collect_obs = torch.from_numpy(info["world_model_obs"]).float()
            self._collect_priv = torch.from_numpy(info["privileged"]).float()
            self._collect_last_action = torch.zeros(self.act_dim, dtype=torch.float32)
            self._collect_done = False

        for _ in range(steps):
            # Get action from policy (no gradient needed)
            with torch.no_grad():
                # Build policy observation from the current world obs, command, last action
                # (command is part of the observation returned by env; we assume env's
                #  policy observation already contains the command.  Here we reuse env's
                #  policy obs directly, but we don't have it stored; we have to request from env.
                #  So we need to call env to get the current policy observation.  However,
                #  this step will be counted.  To avoid extra environment steps, we can
                #  store the policy observation during the previous step.
                #  For simplicity, we'll call a lightweight method that retrieves the latest
                #  policy observation without stepping.  The IsaacEnvWrapper can expose
                #  a `current_policy_obs` property.  We'll add that.
                policy_obs = self.env.get_current_policy_obs()  # shape (pol_dim,)
                action, _ = self.ppo_agent.act(torch.from_numpy(policy_obs).unsqueeze(0))
            action_np = action.squeeze(0).cpu().numpy()

            # Step environment
            next_obs, reward, term, trunc, info = self.env.step(action_np)
            done = term or trunc

            # Extract world model observation and privileged info from info dict
            next_world_obs = torch.from_numpy(info["world_model_obs"]).float()
            next_priv = torch.from_numpy(info["privileged"]).float()

            # Store transition (current obs, current action, reward, next_obs, priv)
            # The buffer stores `priv` of the **current** step (the step before action).
            self.real_buffer.push(
                obs=self._collect_obs,
                act=action.squeeze(0),
                rew=float(reward),
                next_obs=next_world_obs,
                priv=self._collect_priv,
                done=done,
            )

            total_reward += reward
            total_steps += 1
            ep_reward += reward
            ep_length += 1

            if done:
                episode_rewards.append(ep_reward)
                ep_reward = 0.0
                ep_length = 0
                self.env.reset()
                # after reset, update the cached state
                self._collect_obs = torch.from_numpy(self.env.get_current_world_obs()).float()
                self._collect_priv = torch.from_numpy(self.env.get_current_priv()).float()
                self._collect_done = False
            else:
                # Update cached state for next step
                self._collect_obs = next_world_obs
                self._collect_priv = next_priv
                self._collect_last_action = action.squeeze(0)

        avg_reward = total_reward / steps if steps > 0 else 0.0
        avg_ep_length = total_steps / max(len(episode_rewards), 1)
        return {
            "avg_real_reward": avg_reward,
            "avg_episode_length": avg_ep_length,
            "num_episodes": len(episode_rewards),
        }

    # ------------------------------------------------------------------
    # Prepare initial states for imagination
    # ------------------------------------------------------------------
    def _sample_imagination_init(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of initial observations, hidden states, and related
        data from the real replay buffer.

        Returns a dict with:
            init_obs      : (B, obs_dim)
            init_h        : (num_layers, B, hidden_size)  GRU hidden state
            command       : (B, 3)  velocity command
            last_action   : (B, act_dim)  last action taken before init_obs
        """
        B = batch_size
        M = self.world_model.history_len

        # The buffer guarantees that episodes are long enough? If there are no
        # episodes of length >= M+1, fall back to using the full buffer with padding.
        # We'll sample from the buffer using a custom method that ensures a window
        # of M+1 steps.
        # We will use the buffer's internal episodes.

        # Collect windows of length M+1 for each agent.
        # Use the `sample_initial_obs` method? No, we need M steps before.
        # Here we implement a custom sampler that returns:
        #   obs_hist: (B, M, obs_dim) – the M history observations
        #   act_hist: (B, M, act_dim) – the M history actions
        #   init_obs: (B, obs_dim)    – the observation at step M
        #   last_action: (B, act_dim) – the action at step M-1
        # and also a sampled random velocity command.

        # We'll do this by sampling random flat indices that allow extraction of M+1 steps.
        buffer = self.real_buffer
        total_trans = len(buffer)
        if total_trans < (M + 1):
            raise RuntimeError(
                f"Replay buffer has only {total_trans} transitions, need at least {M+1} "
                "to warm up the world model."
            )

        # Build cumulative indices
        cum_lens = [0]
        for ep in buffer.episodes:
            cum_lens.append(cum_lens[-1] + ep["obs"].shape[0])
        cum_lens = torch.tensor(cum_lens, dtype=torch.long)

        # For each agent, pick a random episode that has at least M+1 transitions,
        # then a start index within that episode such that start + M < ep_len
        sampled_obs_hist = []
        sampled_act_hist = []
        sampled_init_obs = []
        sampled_last_action = []

        for _ in range(B):
            valid = False
            while not valid:
                ep_idx = torch.randint(len(buffer.episodes), (1,)).item()
                ep = buffer.episodes[ep_idx]
                ep_len = ep["obs"].shape[0]
                if ep_len < M + 1:
                    continue
                max_start = ep_len - M
                start = torch.randint(max_start, (1,)).item()
                # Extract window
                obs_hist = ep["obs"][start:start + M]
                act_hist = ep["act"][start:start + M]
                init_obs = ep["obs"][start + M]
                last_action = ep["act"][start + M - 1]
                valid = True

            sampled_obs_hist.append(obs_hist)
            sampled_act_hist.append(act_hist)
            sampled_init_obs.append(init_obs)
            sampled_last_action.append(last_action)

        obs_hist_b = torch.stack(sampled_obs_hist, dim=0).to(self.device)   # (B, M, obs_dim)
        act_hist_b = torch.stack(sampled_act_hist, dim=0).to(self.device)
        init_obs_b = torch.stack(sampled_init_obs, dim=0).to(self.device)   # (B, obs_dim)
        last_action_b = torch.stack(sampled_last_action, dim=0).to(self.device)  # (B, act_dim)

        # Warm up the GRU to get hidden state.
        # We'll manually use the world model's GRU (must be careful with normalisation).
        with torch.no_grad():
            obs_norm = self.world_model.normalize_obs(obs_hist_b)
            act_norm = self.world_model.normalize_act(act_hist_b)
            gru_input = torch.cat([obs_norm, act_norm], dim=-1)        # (B, M, in_dim)
            _, h = self.world_model.gru(gru_input)                     # h: (num_layers, B, hidden)

        # Sample random velocity commands for each environment.
        # The range is not defined in the paper; we choose a typical range:
        # lin_vel_x ∈ [0.0, 1.0], lin_vel_y ∈ [-0.5, 0.5], ang_vel_z ∈ [-1.0, 1.0] (rad/s)
        cmd_lin_x = torch.rand(B, 1, device=self.device) * 1.0
        cmd_lin_y = (torch.rand(B, 1, device=self.device) * 1.0 - 0.5)  # [-0.5, 0.5]
        cmd_ang_z = (torch.rand(B, 1, device=self.device) * 2.0 - 1.0)  # [-1.0, 1.0]
        command = torch.cat([cmd_lin_x, cmd_lin_y, cmd_ang_z], dim=-1)  # (B, 3)

        return {
            "init_obs": init_obs_b,
            "init_h": h,
            "command": command,
            "last_action": last_action_b,
        }

    # ------------------------------------------------------------------
    # Imagination rollouts
    # ------------------------------------------------------------------
    def run_imagination(self) -> Dict[str, torch.Tensor]:
        """
        Perform parallel imagination rollouts using the world model and current policy.

        Returns a dictionary suitable for PPOAgent.update(), with flattened tensors.
        """
        B = self.imagination_envs
        T = self.imagination_steps
        M = self.world_model.history_len

        # Sample initial states
        init_data = self._sample_imagination_init(B)
        init_obs = init_data["init_obs"]          # (B, obs_dim)
        h = init_data["init_h"]                   # (num_layers, B, hidden)
        command = init_data["command"]            # (B, 3)
        last_action = init_data["last_action"]    # (B, act_dim)

        # We also need the previous joint velocity for the first joint acceleration penalty.
        # Extract joint velocities from the initial observation.
        slices = self.robot_cfg["world_slices"]
        prev_joint_vel = init_obs[:, slices["joint_vel"][0]:slices["joint_vel"][1]]  # (B, jvel_dim)

        # If the robot has foot air‑time reward, initialise per‑foot air‑time counters.
        weights = self.robot_cfg["reward_weights"]
        if weights.get("w_fa", 0.0) != 0.0:
            num_feet = 4 if self.robot == "anymal_d" else 2  # G1 has two feet
            air_time = torch.zeros(B, num_feet, device=self.device)
        else:
            air_time = torch.zeros(B, 0, device=self.device)  # dummy

        # Buffers to store rollout data (each a list of length T)
        obs_list = []
        act_list = []
        log_prob_list = []
        value_list = []
        reward_list = []
        term_list = []
        next_val_list = []   # value at time t+1 (needed for GAE)

        curr_obs = init_obs
        curr_action_before = last_action

        # Pre‑compute action standard deviation (used later if needed)
        # Not needed directly.

        for t in range(T):
            # Build policy observation from world‑model prediction
            pol_obs = make_policy_obs(
                pred_world_obs=curr_obs,
                command=command,
                last_action=curr_action_before,
                robot_cfg=self.robot_cfg,
                add_last_action=True,
            )

            # Sample action from policy
            action, log_prob = self.ppo_agent.act(pol_obs)          # (B, act_dim), (B, 1)

            # Evaluate value (for later advantage computation)
            with torch.no_grad():
                _, _, value = self.ppo_agent.evaluate(pol_obs, action)   # value: (B, 1)

            # Predict next step with world model
            if isinstance(self.world_model, RWM):
                # Use rollout_autoregressive that takes history + policy actions?
                # Here we need a one‑step prediction. We can use a method
                # `step_autoregressive` that extends the RWM's forward.
                # The world model design includes `rollout_autoregressive` for multi‑step,
                # but we can build a minimal one‑step call.
                # Simpler: we can manually feed the current observation and action
                # through the GRU to get next hidden and next observation distribution.
                # We'll replicate the inner logic of RWM.forward but for one step.
                with torch.no_grad():
                    # Normalise
                    obs_norm = self.world_model.normalize_obs(curr_obs)   # (B, obs_dim)
                    act_norm = self.world_model.normalize_act(action)
                    gru_input = torch.cat([obs_norm, act_norm], dim=-1).unsqueeze(1)  # (B, 1, in)
                    _, h = self.world_model.gru(gru_input, h)               # update hidden
                    last_hidden = h[-1]                                     # (B, hidden)
                    # Predict next observation distribution
                    obs_params = self.world_model.obs_head(last_hidden)      # (B, obs_dim*2)
                    pred_obs_mean, pred_obs_log_std = torch.chunk(obs_params, 2, dim=-1)
                    pred_obs_log_std = torch.clamp(pred_obs_log_std, -20.0, 2.0)
                    # Sample next observation (reparameterization)
                    std = torch.exp(pred_obs_log_std) + self.world_model.std_min
                    eps = torch.randn_like(std)
                    next_obs = pred_obs_mean + std * eps
                    # Denormalise observation (for reward computation we need raw values)
                    next_obs_denorm = self.world_model.denormalize_obs(next_obs)

                    # Privileged information prediction
                    priv_pred = self.world_model.priv_head(last_hidden)      # (B, priv_dim)
                    # Denormalise continuous part if any
                    if self.world_model.num_cont_priv > 0:
                        priv_cont = priv_pred[..., self.world_model.num_binary_priv:]
                        priv_cont_denorm = self.world_model.denormalize_priv_cont(priv_cont)
                        priv_denorm = priv_pred.clone()
                        priv_denorm[..., self.world_model.num_binary_priv:] = priv_cont_denorm
                    else:
                        priv_denorm = priv_pred

            else:
                # For baselines (not used in MBPO), we would need a different approach.
                raise NotImplementedError("Imagination only supported with RWM world model.")

            # Store current data
            obs_list.append(curr_obs)            # world‑model observation
            act_list.append(action)
            log_prob_list.append(log_prob)
            value_list.append(value)

            # Compute reward for this transition (using next_obs_denorm and priv_denorm)
            # For the reward we need:
            #   world_obs_target = next_obs_denorm  (since reward is computed from the resulting state)
            #   privileged_target = priv_denorm
            #   prev_action = curr_action_before  (action before this step)
            #   prev_joint_vel = prev_joint_vel (stored before)
            #   current action = action
            #   current joint_vel = next_obs_denorm joint velocity components (extracted later)
            joint_vel_next = next_obs_denorm[:, slices["joint_vel"][0]:slices["joint_vel"][1]]

            reward = compute_reward(
                world_obs=next_obs_denorm,
                privileged=priv_denorm,
                command=command,
                prev_action=curr_action_before,
                action=action,
                prev_joint_vel=prev_joint_vel,
                joint_vel=joint_vel_next,
                step_time=self.step_time,
                robot=self.robot,
                robot_cfg=self.robot_cfg,
            )

            # Check termination
            terminated = get_termination(
                privileged=priv_denorm,
                robot=self.robot,
                robot_cfg=self.robot_cfg,
            )

            # Update air‑time counters if foot contact information is available
            if weights.get("w_fa", 0.0) != 0.0:
                if self.robot == "anymal_d":
                    foot_contact = priv_denorm[:, self.robot_cfg["priv_slices"]["foot_contact"][0]:
                                                  self.robot_cfg["priv_slices"]["foot_contact"][1]]
                elif self.robot == "unitree_g1":
                    # For G1, foot contact is not directly predicted; we use body contacts? Not accurate.
                    foot_contact = torch.zeros(B, 0, device=self.device)  # dummy
                else:
                    foot_contact = torch.zeros(B, 0, device=self.device)
                if foot_contact.shape[1] > 0:
                    air_time = update_air_time(foot_contact, air_time, self.step_time)

            reward_list.append(reward.unsqueeze(1))    # (B, 1)
            term_list.append(terminated.unsqueeze(1))  # (B, 1)

            # Update for next iteration
            curr_action_before = action
            prev_joint_vel = joint_vel_next
            curr_obs = next_obs_denorm

        # After loop, compute the value of the final state (for GAE bootstrapping)
        # Build policy obs for the final state
        pol_obs_final = make_policy_obs(
            pred_world_obs=curr_obs,
            command=command,
            last_action=curr_action_before,
            robot_cfg=self.robot_cfg,
            add_last_action=True,
        )
        with torch.no_grad():
            # Use a dummy action to evaluate value; any action works since value doesn't depend on action.
            dummy_action = torch.zeros(B, self.act_dim, device=self.device)
            _, _, final_value = self.ppo_agent.evaluate(pol_obs_final, dummy_action)
        # For terminated environments, the bootstrapped value should be 0.
        # We'll handle this in GAE.

        # Stack all tensors along the time dimension (T, B, ...)
        obs_tensor = torch.stack(obs_list, dim=0)            # (T, B, obs_dim)
        act_tensor = torch.stack(act_list, dim=0)            # (T, B, act_dim)
        logp_tensor = torch.stack(log_prob_list, dim=0)      # (T, B, 1)
        val_tensor = torch.stack(value_list, dim=0)          # (T, B, 1)
        rew_tensor = torch.stack(reward_list, dim=0)          # (T, B, 1)
        term_tensor = torch.stack(term_list, dim=0)          # (T, B, 1) bool

        # Compute GAE and returns
        # We need the value one step ahead; we construct a next_values tensor
        # by shifting val_tensor by 1. For the last step, we use final_value (or 0 if terminated).
        next_values = [val_tensor[1:], final_value.unsqueeze(0)]   # final_value: (B,1)
        next_values = torch.cat(next_values, dim=0)               # (T, B, 1)

        # For terminated steps, bootstrap from 0 instead of the predicted value
        next_values = next_values * (~term_tensor)   # zero out next_value where terminated

        advantages = torch.zeros_like(rew_tensor)
        returns = torch.zeros_like(rew_tensor)
        gae = torch.zeros(B, 1, device=self.device)
        for t in reversed(range(T)):
            delta = rew_tensor[t] + self.discount_factor * next_values[t] - val_tensor[t]
            gae = delta + self.discount_factor * self.gae_lambda * ~term_tensor[t] * gae
            advantages[t] = gae
            returns[t] = gae + val_tensor[t]

        # Flatten time and batch dimensions
        def flatten(x):
            return x.view(-1, *x.shape[2:])

        rollout_buffer = {
            "obs":           flatten(obs_tensor).cpu(),   # (N, obs_dim)
            "act":           flatten(act_tensor).cpu(),
            "old_log_prob":  flatten(logp_tensor).cpu(),
            "adv":           flatten(advantages).cpu(),
            "ret":           flatten(returns).cpu(),
        }
        return rollout_buffer

    # ------------------------------------------------------------------
    # Single training iteration
    # ------------------------------------------------------------------
    def train_one_iteration(self) -> Dict[str, float]:
        """Perform one iteration of the MBPO-PPO loop (Algorithm 1)."""
        logs = {}

        # 1. Collect real data
        real_logs = self.collect_real_data(steps=self.real_steps_per_iter)
        logs.update(real_logs)

        # 2. Update world model on a mini‑batch from the real buffer
        # The world model's finetune_online method expects a buffer but we can
        # directly pass a batch.  We'll call `world_model.finetune_online(buffer)`
        # assuming it samples internally.  However, our WorldModel class has a
        # `finetune_online()` method that takes a buffer and performs one gradient
        # step internally.  We'll use that.
        if len(self.real_buffer) >= (self.world_model.history_len + self.world_model.forecast_len):
            wm_loss = self.world_model.finetune_online(
                buffer=self.real_buffer,
                batch_size=self.world_model.history_len * self.world_model.forecast_len,  # 32*8=256? actual config uses 1024
                learning_rate=self.config["world_model"]["learning_rate"],
                weight_decay=self.config["world_model"]["weight_decay"],
                device=self.device,
            )
            logs["wm_loss"] = wm_loss
        else:
            logs["wm_loss"] = None

        # 3. Run imagination
        imag_buffer = self.run_imagination()

        # 4. PPO update on imagined data
        ppo_logs = self.ppo_agent.update(
            rollout_buffer=imag_buffer,
            value_loss_coef=self.value_loss_coef,
            max_grad_norm=self.max_grad_norm,
            target_kl=self.target_kl,
            ppo_epochs=self.ppo_epochs,
            ppo_minibatches=self.ppo_minibatches,
        )
        logs.update(ppo_logs)

        return logs

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------
    def train(self, n_iterations: int) -> None:
        """Run the full MBPO‑PPO training for `n_iterations` iterations."""
        print(f"Starting MBPO‑PPO training for {n_iterations} iterations.")
        for it in range(1, n_iterations + 1):
            logs = self.train_one_iteration()
            # Simple console logging
            print(
                f"iter {it:4d}  "
                f"real_rew: {logs.get('avg_real_reward', 0.0):.3f}  "
                f"policy_loss: {logs.get('policy_loss', 0.0):.4f}  "
                f"value_loss: {logs.get('value_loss', 0.0):.4f}  "
                f"wm_loss: {logs.get('wm_loss', 0.0) or 0.0:.6f}"
            )

    # ------------------------------------------------------------------
    # Utility: expose the current policy observation (for collect_real_data)
    # We need these methods in IsaacEnvWrapper, but they are not defined in the
    # provided env_utils.py.  We will assume they exist or create a local getter
    # that fetches the latest observation from the environment's info.
    # For completeness, we'll add a small helper inside the trainer.
    # ------------------------------------------------------------------
    def _get_current_policy_obs(self) -> np.ndarray:
        """Return the current policy observation from the environment (without stepping)."""
        # In a full implementation, IsaacEnvWrapper should provide this.
        # Here we assume env has a `_last_policy_obs` attribute.
        if hasattr(self.env, "_last_policy_obs"):
            return self.env._last_policy_obs
        else:
            raise AttributeError(
                "IsaacEnvWrapper must expose _last_policy_obs for on‑policy collection."
            )
