import torch
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
from collections import deque
from typing import List, Tuple, Dict, Any

from config import AgentConfig, TrainingConfig, SokobanEnvConfig
from model import DRC
from agent import DRCAgent
from data import SokobanEnv # Assuming a mock SokobanEnv for now

class IMPALALearner:
    """
    Implements the IMPALA (Importance Weighted Actor-Learner Architectures)
    training loop for the DRC agent.
    
    This is a simplified implementation focusing on the learner side,
    without explicit distributed actors. It simulates unroll collection
    and applies V-trace.
    """
    def __init__(self, agent_config: AgentConfig, training_config: TrainingConfig, log_dir: str = "./runs"):
        self.agent_config = agent_config
        self.training_config = training_config
        self.device = training_config.DEVICE

        self.model = DRC(agent_config).to(self.device)
        self.optimizer = Adam(self.model.parameters(), 
                              lr=training_config.LEARNING_RATE_MAX,
                              eps=1e-4) # IMPALA paper often uses smaller eps

        # Linear learning rate decay
        lr_lambda = lambda frame_idx: max(0.0, 
                                          1 - frame_idx / self.training_config.TOTAL_TRANSITIONS)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, datetime.now().strftime("%Y%m%d-%H%M%S")))
        
        print(f"Using device: {self.device}")

    def _compute_vtrace_returns(self, 
                                rewards: torch.Tensor, 
                                dones: torch.Tensor, 
                                values: torch.Tensor, 
                                bootstrap_value: torch.Tensor, 
                                behavior_policy_logits: torch.Tensor, 
                                target_policy_logits: torch.Tensor, 
                                actions: torch.Tensor):
        """
        Computes V-trace returns as described in the IMPALA paper.
        
        Args:
            rewards: Tensor of shape (unroll_length, batch_size)
            dones: Tensor of shape (unroll_length, batch_size)
            values: Tensor of shape (unroll_length + 1, batch_size) - includes bootstrap value V_T(s_T)
                    values[t] is V_T(s_t), values[unroll_length] is V_T(s_T)
            bootstrap_value: Tensor of shape (batch_size,) - used as V_T(s_T) if episode not done
            behavior_policy_logits: Tensor of shape (unroll_length, batch_size, num_actions)
            target_policy_logits: Tensor of shape (unroll_length, batch_size, num_actions)
            actions: Tensor of shape (unroll_length, batch_size)
        
        Returns:
            vs: V-trace values (unroll_length, batch_size) - target for value function
            pg_advantages: Policy gradient advantages (unroll_length, batch_size) - for policy loss
        """
        unroll_length, batch_size = rewards.shape
        gamma = self.training_config.DISCOUNT_RATE
        lambda_ = self.training_config.V_TRACE_LAMBDA
        
        # Calculate log-probabilities for behavior and target policies
        behavior_policy_lp = F.log_softmax(behavior_policy_logits, dim=-1)
        target_policy_lp = F.log_softmax(target_policy_logits, dim=-1)

        # Select log-probs for taken actions
        behavior_lp_actions = behavior_policy_lp.gather(-1, actions.unsqueeze(-1)).squeeze(-1) # (unroll_length, batch_size)
        target_lp_actions = target_policy_lp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)     # (unroll_length, batch_size)
        
        # Compute importance weights rho and clipping coefficients c
        # rho_s = pi(a|s) / mu(a|s)
        rho = torch.exp(target_lp_actions - behavior_lp_actions) 
        # c_s = min(rho_s, 1)
        c = torch.min(torch.ones_like(rho), rho) # Clip to 1
        
        # v-trace targets (Eq 7 from IMPALA paper)
        # Initialize v_s_plus_1 with bootstrap_value
        vs_t_plus_1 = bootstrap_value

        vs = [] # List to store v-trace values in reversed order
        
        for t in reversed(range(unroll_length)):
            reward_t = rewards[t]
            done_t = dones[t] # True if state s_{t+1} is terminal
            value_t = values[t] # V_T(s_t)
            value_t_plus_1 = values[t+1] # V_T(s_{t+1})

            # if s_{t+1} is terminal, the V-trace value for s_{t+1} is 0 in the future,
            # but V_T(s_{t+1}) is still the critic's prediction.
            # IMPALA uses (1-d_t) * vs_t_plus_1 for the next state's V-trace value,
            # effectively setting it to 0 if next state is terminal.
            
            # v_s: V-trace return for state s_t
            v_s_t = reward_t + gamma * ((1 - done_t) * vs_t_plus_1)
            # Apply V-trace correction:
            v_s_t = value_t + rho[t] * (v_s_t - value_t) + gamma * c[t] * ((1 - done_t) * (vs_t_plus_1 - value_t_plus_1))
            
            vs.insert(0, v_s_t) # Prepend to get correct order [v_0, ..., v_{T-1}]
            vs_t_plus_1 = v_s_t # Update for next iteration

        vs = torch.stack(vs, dim=0) # (unroll_length, batch_size)
        
        # Policy gradient advantages (Eq 9 from IMPALA paper)
        # Uses clipped rho (rho_bar = min(rho_s, clip_rho_threshold))
        # The paper specifies clip_rho_threshold=1 for pg_advantages, meaning no additional clipping beyond 1 if rho already clipped.
        # But for the rho in pg_advantages, it is typically capped at a higher value, e.g., 20.
        # The main paper does not specify this, so we will use the same clipping as for `c`.
        # pg_advantages = rho_t * (rewards[t] + gamma * (1-dones[t]) * vs[t+1] - values[t])
        # Where vs[t+1] here should be the *V-trace target* for the next state.
        
        # Let's re-calculate pg_advantages based on `vs` computed above
        # The paper (Appendix B) suggests: pg_advantages = rho_s * (r_s + gamma * v_{s+1} - V(s))
        # where v_{s+1} is the V-trace value for the next state (vs from above, shifted)
        
        # For policy gradient, we need (unroll_length, batch_size)
        pg_advantages = rho * (rewards + gamma * torch.cat([vs[1:], bootstrap_value.unsqueeze(0)], dim=0) * (1 - dones) - values[:-1])
        
        # The `rho` for policy gradient is also clipped, usually to a higher value (e.g. 20)
        # Let's assume the paper implies that rho for policy gradient is also capped at 1 for simplicity,
        # as a specific threshold isn't mentioned for policy gradient clipping.
        # This is a common simplification when rho is already used for clipping.
        
        # A_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        # v_trace_advantages = pi / mu (r + gamma * v_{t+1} - V(s_t))
        
        # Final `pg_advantages` for policy loss.
        # The `vs` are our targets for value function (unroll_length).
        # We need advantages for each state in the unroll (unroll_length).
        # General Advantage Estimation (GAE) usually uses critic value as baseline.
        
        # IMPALA's `pg_advantages` are also calculated iteratively.
        # It's (Eq 9) `rho_s * (r_s + gamma * (1-d_s) * v_{s+1} - V(s))`
        # Where `v_{s+1}` is `vs` shifted by 1.
        
        # Let's assume a simpler advantage: A = R_targ - V(s_t), where R_targ is V-trace.
        pg_advantages = vs - values[:-1] # Use vs as the return for policy gradient

        return vs, pg_advantages


    def train_step(self, 
                   observations: torch.Tensor, 
                   prev_h_states: torch.Tensor, 
                   prev_c_states: torch.Tensor, 
                   actions: torch.Tensor, 
                   rewards: torch.Tensor, 
                   dones: torch.Tensor, 
                   behavior_policy_logits: torch.Tensor):
        """
        Performs one training step.
        
        Args:
            observations: (unroll_length, batch_size, C_obs, H, W)
            prev_h_states: (batch_size, D, C, H, W) for initial h state of unroll
            prev_c_states: (batch_size, D, C, H, W) for initial c state of unroll
            actions: (unroll_length, batch_size)
            rewards: (unroll_length, batch_size)
            dones: (unroll_length, batch_size)
            behavior_policy_logits: (unroll_length, batch_size, num_actions) - from actor
        """
        unroll_length, batch_size, C_obs, H, W = observations.shape
        num_actions = self.agent_config.NUM_ACTIONS

        # Initialize recurrent states for the unroll (batch_size, D, C, H, W) -> list of D (h,c)
        initial_recurrent_states = [
            (prev_h_states[:, d, :, :, :], prev_c_states[:, d, :, :, :]) 
            for d in range(self.agent_config.D_CONVLSTM_LAYERS)
        ]
        
        # Forward pass through the model for the entire unroll
        # We need to collect policy logits and values for each step in the unroll.
        # This requires iterating through the unroll.
        
        all_policy_logits = []
        all_values = []
        current_recurrent_states = initial_recurrent_states

        for t in range(unroll_length):
            obs_t = observations[t] # (batch_size, C_obs, H, W)
            
            # The model's forward pass expects batch_size=1, so we need to process each element in batch_size individually.
            # But here `obs_t` already has batch_size.
            # The agent wrapper's `get_forward_pass_data` handles unsqueezing for batch_size=1.
            # For `train_step` in learner, the batch_size is the second dimension.
            # We need to adapt model.py's forward pass to accept actual batch_size (not just 1)
            # Or, flatten batch dimension for model pass. Let's assume model can handle batch_size > 1.
            
            # DRC model's forward expects `x_t` as (batch_size, C, H, W) and `prev_states` list of `(h,c)` where
            # each `h,c` are (batch_size, C, H, W).
            
            logits_t, value_t, new_states_t, _ = self.model(obs_t, current_recurrent_states)
            
            all_policy_logits.append(logits_t)
            all_values.append(value_t)
            current_recurrent_states = new_states_t # Update states for the next step in unroll
        
        # Also get value for the bootstrap state (V(s_T))
        # This is the value of the state AFTER the last action in the unroll.
        # The observation at unroll_length-1 (last observation in unroll) leads to s_T.
        # We need V_T(s_T) from the critic for the state reached after the last action in the unroll.
        # The `current_recurrent_states` at this point are the states after processing the LAST observation of the unroll.
        # So we pass `observations[-1]` and `current_recurrent_states` to get the bootstrap value.
        with torch.no_grad():
            _, bootstrap_value, _, _ = self.model(observations[-1], current_recurrent_states)
        
        # Stack results
        target_policy_logits = torch.stack(all_policy_logits, dim=0) # (unroll_length, batch_size, num_actions)
        values = torch.stack(all_values, dim=0) # (unroll_length, batch_size)
        
        # Add bootstrap value to values tensor for V-trace
        values_with_bootstrap = torch.cat([values, bootstrap_value.unsqueeze(0)], dim=0) # (unroll_length + 1, batch_size)

        # Compute V-trace returns and policy gradient advantages
        vs, pg_advantages = self._compute_vtrace_returns(
            rewards=rewards,
            dones=dones,
            values=values_with_bootstrap,
            bootstrap_value=bootstrap_value,
            behavior_policy_logits=behavior_policy_logits,
            target_policy_logits=target_policy_logits,
            actions=actions
        )

        # Reshape everything for loss computation: (unroll_length * batch_size, ...)
        flat_target_policy_logits = target_policy_logits.view(-1, num_actions)
        flat_behavior_policy_logits = behavior_policy_logits.view(-1, num_actions)
        flat_actions = actions.view(-1)
        flat_values = values.view(-1)
        flat_vs = vs.view(-1)
        flat_pg_advantages = pg_advantages.view(-1)

        # Policy Loss (IMPALA Eq 9)
        # log_pi(a_t|s_t) * pg_advantage
        target_policy_lp = F.log_softmax(flat_target_policy_logits, dim=-1)
        # Select log-probs for taken actions
        target_lp_actions = target_policy_lp.gather(-1, flat_actions.unsqueeze(-1)).squeeze(-1) # (unroll_length * batch_size)
        
        policy_loss = - (target_lp_actions * flat_pg_advantages).mean()
        
        # Value Loss (squared error between predicted value and V-trace target)
        value_loss = F.mse_loss(flat_values, flat_vs)
        
        # Entropy Loss (L2 penalty on action logits, Entropy penalty on policy)
        # L2 action logits penalty: sum of squares of logits
        l2_action_logits_loss = (flat_target_policy_logits ** 2).sum(dim=-1).mean() # As per paper Eq (12)
        
        # Entropy penalty: - sum (pi * log_pi)
        policy_probs = F.softmax(flat_target_policy_logits, dim=-1)
        entropy_loss = - (policy_probs * target_policy_lp).sum(dim=-1).mean() # As per paper Eq (10)

        # Total Loss
        total_loss = (
            policy_loss 
            + self.training_config.L2_ACTION_LOGITS_PENALTY * l2_action_logits_loss # Coefficient 1e-3
            + self.training_config.ENTROPY_PENALTY * entropy_loss                 # Coefficient 1e-2
            + 0.5 * value_loss # Typical scaling for value loss (from paper, if not explicitly 0.5, often implied for stability)
        )
        
        # L2 regularization for policy and value heads (weight decay handled by AdamW for overall params)
        # Paper specifies "L2 regularisation of strength 1e-5 on the policy and value heads"
        # and "Adam optimiser" (not AdamW).
        # If Adam, L2 regularization must be explicit.
        if self.training_config.OPTIMIZER == "Adam":
            l2_reg_policy_head = 0
            for param in self.model.policy_head.parameters():
                l2_reg_policy_head += torch.norm(param, p=2)
            l2_reg_value_head = 0
            for param in self.model.value_head.parameters():
                l2_reg_value_head += torch.norm(param, p=2)
            
            total_loss += self.training_config.L2_POLICY_VALUE_HEADS_REGULARIZATION * (l2_reg_policy_head + l2_reg_value_head)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.scheduler.step() # Update learning rate

        return {
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "l2_action_logits_loss": l2_action_logits_loss.item(),
            "learning_rate": self.optimizer.param_groups[0]['lr']
        }
    
    def run_training(self):
        env_config = SokobanEnvConfig()
        env = SokobanEnv(env_config)
        agent = DRCAgent(self.agent_config, self.model, self.device)

        current_total_transitions = 0
        episode_count = 0

        while current_total_transitions < self.training_config.TOTAL_TRANSITIONS:
            # Reset environment and agent states for a new episode
            current_obs, info = env.reset() # This is s_0
            agent.reset() # Reset agent's recurrent states for the start of episode

            # Initial recurrent states for the very first step of the unroll (s_0)
            # These are the states *before* the first observation (current_obs) is processed.
            H, W = self.agent_config.GRID_SIZE, self.agent_config.GRID_SIZE
            C = self.agent_config.CHANNELS
            D = self.agent_config.D_CONVLSTM_LAYERS
            initial_h_for_unroll_tensor = torch.zeros(1, D, C, H, W, device=self.device)
            initial_c_for_unroll_tensor = torch.zeros(1, D, C, H, W, device=self.device)
            
            unroll_observations_s_t = [] # Stores s_t
            unroll_actions_a_t = []      # Stores a_t
            unroll_rewards_r_t_plus_1 = [] # Stores r_{t+1}
            unroll_dones_d_t_plus_1 = []   # Stores d_{t+1} (True if s_{t+1} is terminal)
            unroll_behavior_policy_logits_at_t = [] # Stores policy_logits_t

            episode_done = False
            episode_truncated = False
            
            steps_in_unroll = 0
            while steps_in_unroll < self.training_config.UNROLL_LENGTH and not episode_done and not episode_truncated:
                # 1. Get policy_logits_t and value_t for current_obs (s_t)
                #    `agent.hidden_states` are the states *before* processing current_obs.
                policy_logits_t, value_t, new_states_t, _ = agent.get_forward_pass_data(current_obs, agent.hidden_states)
                
                # 2. Sample action_t from behavior policy
                action_t = torch.distributions.Categorical(logits=policy_logits_t).sample().item()
                
                # 3. Take action_t in environment to get (s_{t+1}, r_{t+1}, done_{t+1})
                next_obs, reward_t_plus_1, done_t_plus_1, truncated_t_plus_1, info = env.step(action_t)

                # Store sequence data
                unroll_observations_s_t.append(current_obs)
                unroll_actions_a_t.append(action_t)
                unroll_rewards_r_t_plus_1.append(reward_t_plus_1)
                unroll_dones_d_t_plus_1.append(done_t_plus_1)
                unroll_behavior_policy_logits_at_t.append(policy_logits_t.squeeze(0)) # Remove batch dim

                # Update agent's internal states for the next iteration (s_{t+1})
                agent.hidden_states = new_states_t 
                current_obs = next_obs
                episode_done = done_t_plus_1
                episode_truncated = truncated_t_plus_1
                steps_in_unroll += 1
            
            # --- After collecting one unroll of data ---
            
            # Ensure at least one step was taken in the unroll
            if len(unroll_observations_s_t) == 0:
                continue
            
            # Stack collected data into tensors
            batch_obs_tensor = torch.stack([torch.from_numpy(o).float().permute(2,0,1) for o in unroll_observations_s_t]).unsqueeze(1).to(self.device) # (unroll_len, 1, C, H, W)
            batch_actions_tensor = torch.tensor(unroll_actions_a_t, dtype=torch.long).unsqueeze(1).to(self.device)                          # (unroll_len, 1)
            batch_rewards_tensor = torch.tensor(unroll_rewards_r_t_plus_1, dtype=torch.float).unsqueeze(1).to(self.device)                   # (unroll_len, 1)
            batch_dones_tensor = torch.tensor(unroll_dones_d_t_plus_1, dtype=torch.bool).unsqueeze(1).to(self.device)                         # (unroll_len, 1)
            batch_behavior_logits_tensor = torch.stack(unroll_behavior_policy_logits_at_t).unsqueeze(1).to(self.device)                     # (unroll_len, 1, num_actions)
            
            # Perform training step
            metrics = self.train_step(
                observations=batch_obs_tensor,
                prev_h_states=initial_h_for_unroll_tensor,
                prev_c_states=initial_c_for_unroll_tensor,
                actions=batch_actions_tensor,
                rewards=batch_rewards_tensor,
                dones=batch_dones_tensor,
                behavior_policy_logits=batch_behavior_logits_tensor
            )

            current_total_transitions += len(unroll_observations_s_t)
            episode_count += 1

            if episode_count % 10 == 0: # Log more frequently for faster feedback
                print(f"Episode: {episode_count}, Transitions: {current_total_transitions}, "
                      f"Total Loss: {metrics['total_loss']:.4f}, LR: {metrics['learning_rate']:.6f}")
                for key, value in metrics.items():
                    if key != "learning_rate":
                        self.writer.add_scalar(f"Train/{key}", value, current_total_transitions)
                self.writer.add_scalar("Train/learning_rate", metrics["learning_rate"], current_total_transitions)
            
            # If the episode actually ended, reset agent's states for next episode
            if episode_done or episode_truncated:
                agent.reset()
        
        self.writer.close()
        print("Training finished.")

if __name__ == "__main__":
    agent_config = AgentConfig()
    training_config = TrainingConfig()
    
    # Example usage
    learner = IMPALALearner(agent_config, training_config)
    learner.run_training()
