import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Union

# Assuming these are available in the project structure
from config import Config
from networks import Models
from metrics import RewardNormalizer
from utils import symlog, symexp, reward_to_categorical, categorical_to_reward, calculate_huber_loss


class MRQAgent:
    """Encapsulates the core learning logic of the MR.Q agent.

    This class handles action selection, loss computations for the encoder,
    value networks, and policy network, as well as the backpropagation steps.
    """

    def __init__(
        self,
        config: Config,
        models: Models,
        reward_normalizer: RewardNormalizer,
        device: torch.device,
        action_space_info: Dict[str, Any],
    ):
        """Initializes the MRQAgent.

        Args:
            config: Configuration object containing all hyperparameters.
            models: An instance of the Models class containing all neural networks and optimizers.
            reward_normalizer: An instance of RewardNormalizer for value target scaling.
            device: The PyTorch device (e.g., 'cuda', 'cpu') for tensor operations.
            action_space_info: Dictionary containing information about the action space.
        """
        self.config = config
        self.models = models
        self.reward_normalizer = reward_normalizer
        self.device = device
        self.action_space_info = action_space_info

        self.discrete_actions: bool = action_space_info["is_discrete"]
        self.action_dim: int = action_space_info["action_dim"]

        # Flag for dynamically enabling terminal loss
        self._terminal_loss_active: bool = False

    def act(self, obs: np.ndarray, add_noise: bool) -> np.ndarray:
        """Determines an action given an observation, optionally adding exploration noise.

        Args:
            obs: The current observation from the environment as a NumPy array.
            add_noise: A boolean indicating whether to add exploration noise to the action.

        Returns:
            The selected action as a NumPy array.
        """
        # Ensure obs is batched and on the correct device
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            # Get state embedding
            zs = self.models.state_encoder(obs_tensor)
            
            # Select action from policy network
            action_tensor = self.models.policy_net.act(
                zs,
                add_noise=add_noise,
                std=self.config.policy.exploration_noise_std,
                policy_noise_clip=None # policy_noise_clip is only for target policy noise
            )

        return action_tensor.squeeze(0).cpu().numpy()

    def compute_encoder_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculates the combined encoder loss (reward, dynamics, terminal) as per Equation 14.

        This involves unrolling the dynamics `H_Enc` steps using the `StateActionEncoder`
        and the `target_state_encoder`.

        Args:
            batch: A dictionary of batched transitions from the replay buffer, including:
                - s_0: Initial states of the sequences.
                - actions_seq: Sequence of actions (batch_size, H_Enc, action_dim).
                - rewards_seq: Sequence of rewards (batch_size, H_Enc).
                - dones_seq: Sequence of terminal flags (batch_size, H_Enc).
                - next_obs_seq: Sequence of next observations (batch_size, H_Enc, *obs_shape).

        Returns:
            The total encoder loss as a scalar PyTorch tensor.
        """
        s_0 = batch["s_0"].to(self.device)
        actions_seq = batch["actions_seq"].to(self.device)
        rewards_seq = batch["rewards_seq"].to(self.device)
        dones_seq = batch["dones_seq"].to(self.device)
        next_obs_seq = batch["next_obs_seq"].to(self.device)

        batch_size = s_0.shape[0]
        encoder_horizon = self.config.losses.encoder_horizon
        total_encoder_loss = torch.tensor(0.0, device=self.device)

        # Get initial state embedding
        zs_t = self.models.state_encoder(s_0)

        for t in range(encoder_horizon):
            a_t = actions_seq[:, t, :]
            r_t_true = rewards_seq[:, t]
            d_t_true = dones_seq[:, t]
            s_prime_t_true = next_obs_seq[:, t, :]

            # Predict next state embedding, reward logits, and terminal logits
            predicted_zs_prime_t, predicted_reward_logits_t, predicted_terminal_logits_t, _ = \
                self.models.state_action_encoder(zs_t, a_t)

            # --- Reward Loss (Equation 15) ---
            target_r_t_categorical = reward_to_categorical(r_t_true, self.config)
            
            # Cross-entropy loss for categorical reward prediction
            # -sum(p * log(softmax(q))) is equivalent to F.cross_entropy with target as one-hot distribution
            # Note: F.cross_entropy usually expects target indices. For target distributions, a manual
            # negative log-likelihood sum is used.
            reward_loss = -torch.sum(target_r_t_categorical * F.log_softmax(predicted_reward_logits_t, dim=-1), dim=-1).mean()
            total_encoder_loss += self.config.losses.lambda_reward * reward_loss

            # --- Dynamics Loss (Equation 16) ---
            with torch.no_grad(): # Target encoder uses detached parameters
                target_zs_prime_t = self.models.target_state_encoder(s_prime_t_true)
            dynamics_loss = F.mse_loss(predicted_zs_prime_t, target_zs_prime_t)
            total_encoder_loss += self.config.losses.lambda_dynamics * dynamics_loss

            # --- Terminal Loss (Equation 17) ---
            # Dynamically activate terminal loss if a terminal transition (d=True) is observed
            if not self._terminal_loss_active and d_t_true.any():
                self._terminal_loss_active = True

            if self._terminal_loss_active:
                terminal_loss = F.mse_loss(predicted_terminal_logits_t.squeeze(-1), d_t_true.float())
                total_encoder_loss += self.config.losses.lambda_terminal * terminal_loss
            
            # Update zs_t for the next unroll step (using predicted next state)
            zs_t = predicted_zs_prime_t

        return total_encoder_loss

    def compute_value_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates the TD3-style value loss, incorporating multi-step returns,
        double Q-learning, target policy noise, and reward scaling (Equation 19).

        Args:
            batch: A dictionary of batched transitions from the replay buffer, including:
                - s_0: Initial states (s_0).
                - actions_seq: Sequence of actions, where actions_seq[:, 0, :] is a_0.
                - rewards_seq: Sequence of rewards (batch_size, H_Q).
                - dones_seq: Sequence of terminal flags (batch_size, H_Q).
                - final_obs_k: Observation at the end of the H_Q sequence (s_H_Q).
                - is_weights: Importance sampling weights for PER.

        Returns:
            A tuple containing:
            - The total value loss as a scalar PyTorch tensor.
            - The absolute TD errors for updating PER priorities (batch_size,).
        """
        s_0 = batch["s_0"].to(self.device)
        actions_0 = batch["actions_seq"][:, 0, :].to(self.device) # Action taken from s_0
        rewards_seq = batch["rewards_seq"].to(self.device) # Sequence of rewards up to H_Q
        dones_seq = batch["dones_seq"].to(self.device)     # Sequence of done flags up to H_Q
        s_H_Q = batch["final_obs_k"].to(self.device)
        is_weights = batch["is_weights"].to(self.device)

        value_horizon = self.config.losses.value_horizon
        discount_factor = self.config.training.discount_factor

        # --- Current Q-values ---
        with torch.no_grad(): # Encoder is not updated by value loss
            zs_0 = self.models.state_encoder(s_0)
            # Use state_action_encoder to get zsa_0
            _, _, _, zsa_0 = self.models.state_action_encoder(zs_0, actions_0)

        q1 = self.models.value_net1(zsa_0) # (batch_size, 1)
        q2 = self.models.value_net2(zsa_0) # (batch_size, 1)

        # --- Target Q-value Calculation (y_target) ---
        with torch.no_grad():
            # Get target state embedding for s_H_Q
            target_zs_H_Q = self.models.target_state_encoder(s_H_Q)

            # Determine target actions with noise from target policy
            target_policy_noise_std = self.config.policy.policy_noise_std
            target_policy_noise_clip = self.config.policy.policy_noise_clip
            target_actions_H_Q = self.models.target_policy_net.act(
                target_zs_H_Q,
                add_noise=True,
                std=target_policy_noise_std,
                policy_noise_clip=target_policy_noise_clip,
            )
            # Use state_action_encoder to get target_zsa_H_Q
            _, _, _, target_zsa_H_Q = self.models.state_action_encoder(target_zs_H_Q, target_actions_H_Q)

            # Calculate target Q-values from target Q-networks
            target_q1_H_Q = self.models.target_value_net1(target_zsa_H_Q)
            target_q2_H_Q = self.models.target_value_net2(target_zsa_H_Q)

            # Take the minimum of the two target Q-values (TD3 trick)
            min_target_q_H_Q = torch.min(target_q1_H_Q, target_q2_H_Q) # (batch_size, 1)

            # Scale min_target_q_H_Q by the target average absolute reward (rho_prime)
            min_target_q_H_Q_scaled = min_target_q_H_Q * self.reward_normalizer.get_target_mean()

            # Calculate multi-step discounted rewards sum and check for any termination
            multistep_rewards_sum = torch.zeros_like(rewards_seq[:, 0], device=self.device)
            multistep_dones_any = torch.zeros_like(dones_seq[:, 0], dtype=torch.bool, device=self.device)
            current_gamma_power = 1.0
            
            for t in range(value_horizon):
                multistep_rewards_sum += current_gamma_power * rewards_seq[:, t]
                multistep_dones_any = multistep_dones_any | dones_seq[:, t]
                current_gamma_power *= discount_factor
            
            # Compute the multi-step target value (Bellman update)
            # If `multistep_dones_any` is True for a sequence, the Q-value term is zero.
            y_target = multistep_rewards_sum + \
                       (discount_factor ** value_horizon) * \
                       min_target_q_H_Q_scaled.squeeze(-1) * \
                       (~multistep_dones_any).float() # Squeeze for element-wise operation

            # Scale y_target by the current average absolute reward (rho)
            y_target_scaled = y_target / self.reward_normalizer.get_mean()
        
        # --- Value Losses (Huber Loss) ---
        # Note: q1, q2 are (batch_size, 1), y_target_scaled is (batch_size,)
        # Squeeze q1, q2 for element-wise comparison with y_target_scaled.
        loss_q1 = calculate_huber_loss(q1.squeeze(-1), y_target_scaled.detach())
        loss_q2 = calculate_huber_loss(q2.squeeze(-1), y_target_scaled.detach())

        # Apply importance sampling weights from PER
        loss_q1 = (loss_q1 * is_weights).mean()
        loss_q2 = (loss_q2 * is_weights).mean()
        
        value_loss = loss_q1 + loss_q2

        # --- TD Errors for PER update ---
        td_errors = (q1.squeeze(-1) - y_target_scaled.detach()).abs() # Use one Q-value for TD error

        return value_loss, td_errors

    def compute_policy_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculates the deterministic policy gradient loss (Equation 20),
        including pre-activation regularization.

        Args:
            batch: A dictionary of batched transitions from the replay buffer, containing:
                - s_0: Initial states of the sequences.

        Returns:
            The policy loss as a scalar PyTorch tensor.
        """
        s_0 = batch["s_0"].to(self.device)

        # Get state embedding
        with torch.no_grad(): # Encoder is not updated by policy loss
            zs_0 = self.models.state_encoder(s_0)

        # Get the raw outputs (pre-activations/logits) from the policy network
        policy_pre_activations = self.models.policy_net.forward(zs_0)
        
        # Get the actual actions determined by the current policy (no exploration noise for loss calculation)
        a_pi = self.models.policy_net.act(zs_0, add_noise=False)

        # Get state-action embedding for these policy actions
        with torch.no_grad(): # State-action encoder is not updated by policy loss
            _, _, _, zsa_pi = self.models.state_action_encoder(zs_0, a_pi)

        # Query both Q-networks for values of these actions
        # The Q-networks are NOT detached, as policy loss updates policy to maximize Q.
        q1_pi = self.models.value_net1(zsa_pi)
        q2_pi = self.models.value_net2(zsa_pi)

        # Take the minimum Q-value for policy improvement (TD3 trick)
        q_min_pi = torch.min(q1_pi, q2_pi)

        # Calculate the policy loss (maximizing Q-value is minimizing negative Q-value)
        policy_loss = -q_min_pi.mean()

        # Add pre-activation regularization (Equation 20, lambda_pre_activ * z_pi^2)
        policy_loss += self.config.losses.lambda_pre_activ * (policy_pre_activations ** 2).mean()

        return policy_loss

    def learn(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Union[float, np.ndarray]]:
        """Orchestrates a single learning step by computing and backpropagating all losses.

        Args:
            batch: A dictionary of batched transitions from the replay buffer.

        Returns:
            A dictionary containing scalar loss values and TD errors for PER update.
        """
        metrics: Dict[str, Union[float, np.ndarray]] = {}

        # --- Encoder Update ---
        self.models.encoder_optimizer.zero_grad()
        encoder_loss = self.compute_encoder_loss(batch)
        encoder_loss.backward()
        self.models.encoder_optimizer.step()
        metrics["encoder_loss"] = encoder_loss.item()

        # --- Value Update ---
        self.models.value_optimizer.zero_grad()
        value_loss, td_errors = self.compute_value_loss(batch)
        value_loss.backward()
        self.models.value_optimizer.step()
        metrics["value_loss"] = value_loss.item()
        metrics["td_errors"] = td_errors.detach().cpu().numpy() # For updating PER priorities

        # --- Policy Update ---
        self.models.policy_optimizer.zero_grad()
        policy_loss = self.compute_policy_loss(batch)
        policy_loss.backward()
        
        # Apply gradient clipping to the policy network parameters
        torch.nn.utils.clip_grad_norm_(
            self.models.policy_net.parameters(), self.config.optimizer.grad_clip_norm
        )
        self.models.policy_optimizer.step()
        metrics["policy_loss"] = policy_loss.item()

        return metrics

