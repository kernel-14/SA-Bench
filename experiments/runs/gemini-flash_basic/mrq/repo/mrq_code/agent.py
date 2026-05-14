import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import numpy as np

from mrq_code.models import StateEncoder, StateActionEncoder, ValueNetwork, PolicyNetwork
from mrq_code.losses import calculate_encoder_loss, calculate_value_loss, calculate_policy_loss
from mrq_code.replay_buffer import ReplayBuffer
from mrq_code.config import MRQConfig
from mrq_code.utils import gumbel_softmax, symexp

class MRQAgent:
    def __init__(self, state_dim, action_dim, is_discrete_action_space, image_observation_space=False, state_channels=3, device="cpu"):
        self.device = device
        self.action_dim = action_dim
        self.is_discrete_action_space = is_discrete_action_space
        self.image_observation_space = image_observation_space
        self.state_channels = state_channels
        
        # Initialize networks
        self.state_encoder = StateEncoder(state_dim, image_observation_space, state_channels).to(self.device)
        self.state_action_encoder = StateActionEncoder(action_dim, MRQConfig.ZS_DIM).to(self.device)
        
        self.q_network1 = ValueNetwork().to(self.device)
        self.q_network2 = ValueNetwork().to(self.device)
        self.policy_network = PolicyNetwork(action_dim, is_discrete=is_discrete_action_space).to(self.device)

        # Initialize target networks (hard copy initially)
        self.state_encoder_target = deepcopy(self.state_encoder).to(self.device)
        self.q_network1_target = deepcopy(self.q_network1).to(self.device)
        self.q_network2_target = deepcopy(self.q_network2).to(self.device)
        self.policy_network_target = deepcopy(self.policy_network).to(self.device)

        # Optimizers
        self.encoder_optimizer = optim.AdamW(
            list(self.state_encoder.parameters()) + list(self.state_action_encoder.parameters()),
            lr=MRQConfig.LEARNING_RATE_ENCODER_VALUE,
            weight_decay=MRQConfig.WEIGHT_DECAY
        )
        self.q_optimizer = optim.AdamW(
            list(self.q_network1.parameters()) + list(self.q_network2.parameters()),
            lr=MRQConfig.LEARNING_RATE_ENCODER_VALUE,
            weight_decay=MRQConfig.WEIGHT_DECAY
        )
        self.policy_optimizer = optim.AdamW(
            self.policy_network.parameters(),
            lr=MRQConfig.LEARNING_RATE_POLICY,
            weight_decay=MRQConfig.WEIGHT_DECAY
        )

        self.replay_buffer = ReplayBuffer()
        self.total_steps = 0
        self.running_average_reward = 1.0 # For reward scaling, initialized to 1 as per paper for unscaled rewards
        self.terminal_loss_active = False # Flag for activating terminal loss

    def act(self, state, evaluate=False):
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            zs = self.state_encoder(state_tensor)
            pre_activations = self.policy_network(zs)

            if self.is_discrete_action_space:
                if evaluate:
                    action = torch.argmax(pre_activations, dim=-1).cpu().numpy()
                else:
                    # Gumbel-Softmax for exploration during training
                    action_probs = gumbel_softmax(pre_activations, tau=MRQConfig.GUMBEL_SOFTMAX_TAU, hard=False)
                    action = torch.argmax(action_probs, dim=-1).cpu().numpy()
            else: # Continuous action space
                action = torch.tanh(pre_activations).cpu().numpy() # Actions typically [-1, 1]
                if not evaluate:
                    # Add Gaussian noise for exploration
                    noise = np.random.normal(0, MRQConfig.EXPLORATION_NOISE, size=self.action_dim)
                    action = np.clip(action + noise, -1, 1) # Assuming action range [-1, 1]
        return action.squeeze(0) # Return action without batch dimension

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.add(
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(action, dtype=torch.float32),
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(next_state, dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32)
        )

    def _update_target_networks(self):
        if self.total_steps % MRQConfig.TARGET_UPDATE_FREQUENCY == 0:
            self.state_encoder_target.load_state_dict(self.state_encoder.state_dict())
            self.q_network1_target.load_state_dict(self.q_network1.state_dict())
            self.q_network2_target.load_state_dict(self.q_network2.state_dict())
            self.policy_network_target.load_state_dict(self.policy_network.state_dict())
    
    def _unroll_dynamics(self, initial_zs, actions):
        # NOTE: This implementation simplifies the unrolling described in the paper (Equation 12).
        # The paper implies sampling subsequences of length H_Enc from the replay buffer.
        # Our current replay buffer implementation samples single transitions, so this unroll
        # is effectively a prediction for H_Enc steps *forward* based on the initial state-action,
        # rather than using true historical H_Enc steps.
        
        batch_size = initial_zs.shape[0]
        unrolled_reward_logits = []
        unrolled_next_zs = []
        unrolled_terminal_logits = []
        
        current_zs = initial_zs

        for t in range(MRQConfig.ENCODER_HORIZON):
            # For a proper unroll with a standard replay buffer, 'actions' here would need to be
            # a sequence of actions (batch_size, H_Enc, action_dim). Currently, 'actions' is only
            # for the first step (batch_size, action_dim). 
            # To match the paper, a specialized sequence-sampling replay buffer would be needed.
            action_t = actions[:, t, :] if actions.dim() == 3 else actions # Assuming actions can be (batch, H_Enc, action_dim) or (batch, action_dim)
            reward_l, next_z, terminal_l, _ = self.state_action_encoder(current_zs, action_t)
            unrolled_reward_logits.append(reward_l)
            unrolled_next_zs.append(next_z)
            unrolled_terminal_logits.append(terminal_l)
            current_zs = next_z # Use predicted next_z for next step

        return (
            torch.stack(unrolled_reward_logits, dim=1),
            torch.stack(unrolled_next_zs, dim=1),
            torch.stack(unrolled_terminal_logits, dim=1)
        )

    def _compute_target_q(self, next_states_batch, rewards_batch, dones_batch):
        with torch.no_grad():
            # Encode next states using target state encoder
            next_zs_target = self.state_encoder_target(next_states_batch)

            # Determine next actions using target policy network
            next_pre_activations_target = self.policy_network_target(next_zs_target)
            if self.is_discrete_action_space:
                # For discrete actions, use Gumbel-Softmax with noise and clip
                # Similar to original TD3, add noise to logits then select argmax or sample
                # Paper says 'add noise to each dimension', implying continuous noise to logits
                # And then take argmax (equivalent to Gumbel-Softmax with hard=True)
                noise = torch.randn_like(next_pre_activations_target) * MRQConfig.TARGET_POLICY_NOISE_SIGMA
                noisy_pre_activations = next_pre_activations_target + noise
                # Clip noise is described for continuous actions, but let's apply it conceptually here for discrete
                clipped_noisy_pre_activations = torch.clamp(noisy_pre_activations, 
                                                            -MRQConfig.TARGET_POLICY_NOISE_CLIP, 
                                                            MRQConfig.TARGET_POLICY_NOISE_CLIP)
                next_actions_target = gumbel_softmax(clipped_noisy_pre_activations, hard=True, tau=MRQConfig.GUMBEL_SOFTMAX_TAU)
                # If hard=True, this will be one-hot, suitable for discrete action input to state_action_encoder
            else:
                # For continuous actions, add clipped Gaussian noise
                next_actions_target = torch.tanh(next_pre_activations_target)
                noise = torch.randn_like(next_actions_target) * MRQConfig.TARGET_POLICY_NOISE_SIGMA
                noise = torch.clamp(noise, -MRQConfig.TARGET_POLICY_NOISE_CLIP, MRQConfig.TARGET_POLICY_NOISE_CLIP)
                next_actions_target = torch.clamp(next_actions_target + noise, -1, 1)

            # Get state-action embedding for next state and action
            _, _, _, next_zsa_target = self.state_action_encoder(next_zs_target, next_actions_target) # No grad on this

            # Get Q-values from target Q-networks and take the minimum
            q1_target = self.q_network1_target(next_zsa_target)
            q2_target = self.q_network2_target(next_zsa_target)
            min_q_target = torch.min(q1_target, q2_target)

            # Multi-step returns calculation
            # NOTE: This part of the code assumes that 'rewards_batch' corresponds to r_0
            # and only accounts for the discounted value of the *single* next state value.
            # A true multi-step return (sum_{t=0 to H_Q-1} gamma^t r_t + gamma^H_Q Q_target(s_H)) 
            # would require sampling H_Q consecutive (r,d) tuples from the replay buffer.
            # Current implementation is effectively a 1-step return with an adjusted Q_target for TD3.
            # To implement multi-step correctly, the replay buffer or the sampling mechanism 
            # needs to provide sequences of (reward, done) for H_Q steps.
            target_q_values = rewards_batch.unsqueeze(-1) + MRQConfig.DISCOUNT_FACTOR * min_q_target * (1 - dones_batch.unsqueeze(-1))

            # Reward scaling
            # Update running_average_reward
            self.running_average_reward = 0.995 * self.running_average_reward + 0.005 * rewards_batch.abs().mean()
            target_q_values = target_q_values / self.running_average_reward

            return target_q_values


    def update(self):
        if len(self.replay_buffer) < MRQConfig.MINI_BATCH_SIZE:
            return  # Not enough samples to train

        self.total_steps += 1
        
        states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(MRQConfig.MINI_BATCH_SIZE)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)

        # --- Encoder Update ---
        # The unrolling here is a simplified forward pass for H_Enc steps using only the initial action from the sampled transition.
        # A full implementation of unrolling as described in the paper (Equation 14) would require sampling 
        # a sequence of H_Enc (s,a,r,d,s') from the replay buffer to compare predicted unrolled values against actuals.
        # Current implementation only uses the dynamics model to predict next state embedding, reward, and terminal for the *first* step.
        zs = self.state_encoder(states)
        # If we were doing full unroll: 
        # unrolled_reward_logits, unrolled_predicted_next_zs, unrolled_terminal_logits = self._unroll_dynamics(zs, actions_sequence_for_unroll)
        # For now, treat H_Enc=1 effectively for training dynamics, as we only have (s,a,r,s',d) for a single step
        reward_logits, predicted_next_zs, terminal_logits, zsa = self.state_action_encoder(zs, actions)

        # Get target for dynamics loss from target state encoder
        with torch.no_grad():
            next_state_embeddings_target = self.state_encoder_target(next_states)
        
        # Check if terminal loss should be active
        if not self.terminal_loss_active and (dones.sum() > 0 or self.total_steps > MRQConfig.INITIAL_RANDOM_EXPLORATION_TIME_STEPS):
            self.terminal_loss_active = True
            print("Terminal loss activated.")

        encoder_loss, reward_loss, dynamics_loss, terminal_loss = calculate_encoder_loss(
            reward_logits, predicted_next_zs, terminal_logits, rewards, next_state_embeddings_target, dones, self.device
        )
        if self.terminal_loss_active:
            # Only add terminal loss if it's active as per paper (d=0 is seen or sufficient steps passed)
            encoder_loss += MRQConfig.LAMBDA_TERMINAL * terminal_loss

        self.encoder_optimizer.zero_grad()
        encoder_loss.backward()
        self.encoder_optimizer.step()

        # --- Value Network Update ---
        # Pass zsa through Q networks
        q1_pred = self.q_network1(zsa.detach()) # Detach zsa so Q-networks don't affect encoder
        q2_pred = self.q_network2(zsa.detach())

        # Compute target Q values
        target_q_values = self._compute_target_q(next_states, rewards, dones)
        
        q1_loss = calculate_value_loss(q1_pred, target_q_values, is_prioritized_sampling=True) * weights # Apply importance sampling weights
        q2_loss = calculate_value_loss(q2_pred, target_q_values, is_prioritized_sampling=True) * weights
        q_loss = (q1_loss + q2_loss).mean()

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update replay buffer priorities (using TD error from one of the Q networks)
        with torch.no_grad():
            # Compute TD error for priority update. Squeeze to remove extra dimension if present.
            # This error is used for the next sampling probabilities.
            td_error = torch.abs(q1_pred - target_q_values).squeeze().cpu().numpy() 
            self.replay_buffer.update_priorities(indices, td_error)

        # --- Policy Network Update ---
        # Delayed policy update (standard in TD3, often every 2 updates)
        if self.total_steps % MRQConfig.TARGET_UPDATE_FREQUENCY == 0: # Using same frequency as target updates for now
            # Get zs from current state encoder (detached)
            zs_policy = self.state_encoder(states.detach()) # Detach to prevent gradients flowing to encoder
            pre_activations_policy = self.policy_network(zs_policy)

            if self.is_discrete_action_space:
                # For discrete actions, use Gumbel-Softmax (hard=True for policy evaluation)
                actions_policy = gumbel_softmax(pre_activations_policy, tau=MRQConfig.GUMBEL_SOFTMAX_TAU, hard=True)
            else:
                # For continuous actions, use Tanh
                actions_policy = torch.tanh(pre_activations_policy)

            # Get zsa from state_action_encoder using actions chosen by current policy
            # Not propagating gradients to state_action_encoder for policy eval
            # We need to detach zs_policy and actions_policy to prevent gradients from flowing through
            # the state encoder and action selection process when calculating policy loss
            _, _, _, zsa_policy = self.state_action_encoder(zs_policy.detach(), actions_policy.detach()) 

            # Get Q-values from *current* Q-networks (not target) for policy gradient
            q1_policy = self.q_network1(zsa_policy)
            q2_policy = self.q_network2(zsa_policy)
            # Take the minimum of the two Q-values to reduce overestimation bias
            min_q_policy = torch.min(q1_policy, q2_policy)

            policy_loss = calculate_policy_loss(min_q_policy, pre_activations_policy, self.action_dim, self.is_discrete_action_space)
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            # Apply gradient clipping for policy network
            nn.utils.clip_grad_norm_(self.policy_network.parameters(), MRQConfig.GRADIENT_CLIP_NORM)
            self.policy_optimizer.step()
            
            # --- Target Networks Update ---
            self._update_target_networks()
