import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, List, Tuple

# Local imports
from config import Config
from utils import compute_gae, gaussian_nll_loss
from models.rwm_model import RWMModel
from models.policy_value_model import PolicyModel, ValueModel
from environment import Environment
from data.replay_buffer import ReplayBuffer

class MBPOPPO_Trainer:
    """
    Implements the Model-Based Policy Optimization with PPO algorithm.
    Orchestrates real data collection, RWM fine-tuning, trajectory imagination,
    and policy/value network updates.
    """

    def __init__(
        self,
        policy_model: PolicyModel,
        value_model: ValueModel,
        rwm_model: RWMModel,
        policy_optimizer: torch.optim.Optimizer,
        value_optimizer: torch.optim.Optimizer,
        real_env: Environment,
        replay_buffer: ReplayBuffer,
        config: Config,
        rwm_trainer: "RWMTrainer", # Use string for forward reference to avoid circular import
        writer: SummaryWriter,
    ):
        """
        Initializes the MBPOPPO_Trainer.

        Args:
            policy_model: An instance of the PolicyModel (actor network).
            value_model: An instance of the ValueModel (critic network).
            rwm_model: An instance of the RWMModel (world model).
            policy_optimizer: PyTorch optimizer for the policy model.
            value_optimizer: PyTorch optimizer for the value model.
            real_env: An instance of the Environment for real interactions.
            replay_buffer: An instance of the ReplayBuffer.
            config: The global configuration object.
            rwm_trainer: An instance of the RWMTrainer for fine-tuning the RWM.
            writer: A TensorBoard SummaryWriter for logging.
        """
        self.policy_model = policy_model
        self.value_model = value_model
        self.rwm_model = rwm_model
        self.policy_optimizer = policy_optimizer
        self.value_optimizer = value_optimizer
        self.real_env = real_env
        self.replay_buffer = replay_buffer
        self.config = config
        self.rwm_trainer = rwm_trainer
        self.writer = writer
        self.device = config.global.device

        # Ensure models are on the correct device
        self.policy_model.to(self.device)
        self.value_model.to(self.device)
        self.rwm_model.to(self.device)

        # Extract MBPO-PPO hyperparameters
        self.imagination_environments: int = config.mbpo_ppo.training.imagination_environments
        self.imagination_steps_per_iteration: int = config.mbpo_ppo.training.imagination_steps_per_iteration
        self.gamma: float = config.mbpo_ppo.training.discount_factor_gamma
        self.gae_lambda: float = config.mbpo_ppo.training.gae_lambda # Added default 0.95 to config.yaml
        self.clip_range_epsilon: float = config.mbpo_ppo.training.clip_range_epsilon
        self.entropy_coefficient: float = config.mbpo_ppo.training.entropy_coefficient
        self.learning_epochs: int = config.mbpo_ppo.training.learning_epochs
        self.mini_batches: int = config.mbpo_ppo.training.mini_batches
        self.real_steps_per_iteration: int = config.mbpo_ppo.training.real_steps_per_iteration # Added to config.yaml
        self.rwm_finetune_iterations_per_mbpo_step: int = config.rwm_model.training.rwm_finetune_iterations_per_mbpo_step
        self.history_horizon_M: int = config.rwm_model.training.history_horizon_M
        self.contact_termination_threshold: float = config.environment.contact_termination_threshold # Added to config.yaml

        # Get observation/action dimensions from the environment
        env_dims = self.real_env.get_obs_dims()
        self.obs_wm_dim: int = env_dims["obs_wm_dim"]
        self.act_wm_dim: int = env_dims["action_dim"] # Action for WM is the same as policy action
        self.priv_dim: int = env_dims["priv_dim"]
        self.obs_policy_dim: int = env_dims["obs_policy_dim"]
        self.action_dim: int = env_dims["action_dim"]

        # Internal state for _collect_real_data (last observed values)
        # These will be updated by the reset() call before the first collection step
        self.current_obs_wm: np.ndarray = np.array([])
        self.current_obs_policy: np.ndarray = np.array([])
        self.current_priv_info: np.ndarray = np.array([])
        self.current_command_vel: np.ndarray = np.array([])
        self.last_action_policy: np.ndarray = np.zeros(self.action_dim, dtype=np.float32) # Default to zeros

        # Store robot type for specific logic
        self.robot_type: str = self.real_env.robot_type

    def _get_obs_wm_slice(self, component_name: str) -> slice:
        """Helper to get slice for a component within world model observation."""
        return self.real_env._obs_wm_slices.get(component_name)

    def _get_priv_info_slice(self, component_name: str) -> slice:
        """Helper to get slice for a component within privileged information."""
        return self.real_env._priv_info_slices.get(component_name)

    def _construct_policy_obs_from_wm_pred(
        self,
        predicted_obs_wm: torch.Tensor,  # (batch_size, obs_wm_dim)
        command_vel: torch.Tensor,       # (batch_size, 3)
        last_action: torch.Tensor,       # (batch_size, action_dim)
    ) -> torch.Tensor:
        """
        Constructs policy observations from RWM's predicted world model observation,
        command velocity, and the action taken to reach this state.
        This re-implements the logic from Environment._construct_obs_policy using tensors.
        """
        batch_size = predicted_obs_wm.shape[0]
        obs_policy_components: List[torch.Tensor] = []

        # Extract components from predicted_obs_wm using slices from real_env
        obs_policy_components.append(predicted_obs_wm[:, self._get_obs_wm_slice('base_lin_vel')])
        obs_policy_components.append(predicted_obs_wm[:, self._get_obs_wm_slice('base_ang_vel')])
        obs_policy_components.append(predicted_obs_wm[:, self._get_obs_wm_slice('projected_gravity')])
        obs_policy_components.append(command_vel)
        obs_policy_components.append(predicted_obs_wm[:, self._get_obs_wm_slice('joint_positions')])
        obs_policy_components.append(predicted_obs_wm[:, self._get_obs_wm_slice('joint_velocities')])
        obs_policy_components.append(last_action)

        return torch.cat(obs_policy_components, dim=-1)

    def _check_imagination_done(
        self, predicted_priv_info: torch.Tensor, predicted_obs_wm: torch.Tensor
    ) -> torch.Tensor:
        """
        Determines if an imagined trajectory segment is 'done' based on privileged information
        (e.g., contact signals from RWM prediction) and other criteria.

        Args:
            predicted_priv_info: Predicted privileged information (batch_size, priv_dim).
            predicted_obs_wm: Predicted world model observation (batch_size, obs_wm_dim).

        Returns:
            A boolean tensor (batch_size,) indicating if each imagination environment is done.
        """
        batch_size = predicted_priv_info.shape[0]
        # Initialize all as not done
        done_batch = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        # Check for contact-based termination
        if self.robot_type == "ANYmal D":
            knee_contact_slice = self._get_priv_info_slice('knee_contact')
            foot_contact_slice = self._get_priv_info_slice('foot_contact')
            if knee_contact_slice and foot_contact_slice:
                knee_contact_forces = predicted_priv_info[:, knee_contact_slice]
                foot_contact_forces = predicted_priv_info[:, foot_contact_slice]
                # A simple check: if any contact force is above a threshold
                undesired_contact = (torch.any(knee_contact_forces > self.contact_termination_threshold, dim=-1) |
                                     torch.any(foot_contact_forces > self.contact_termination_threshold, dim=-1))
                done_batch = done_batch | undesired_contact

        elif self.robot_type == "Unitree G1":
            body_contact_slice = self._get_priv_info_slice('body_contact')
            if body_contact_slice:
                body_contact_forces = predicted_priv_info[:, body_contact_slice]
                undesired_contact = torch.any(body_contact_forces > self.contact_termination_threshold, dim=-1)
                done_batch = done_batch | undesired_contact

        # Add any other termination conditions based on predicted_obs_wm if necessary
        # For instance, if base height goes below a threshold. This is not explicitly detailed
        # in the paper for RWM, but could be part of "predict such terminations"
        # However, paper mentions "ground contact by the base", which is more related to priv_info.
        
        return done_batch

    def _collect_real_data(self, num_steps: int) -> None:
        """
        Interacts with the real environment (simulator) to collect fresh experience
        and populate the replay buffer.

        Args:
            num_steps: The number of environment steps to collect.
        """
        self.policy_model.eval() # Policy in eval mode for data collection
        self.rwm_model.eval()    # RWM in eval mode (not used for real data collection directly)

        # Reset environment if it's the start or previous episode terminated
        if not self.current_obs_wm.size > 0: # Check if initial state is set
            self.current_obs_wm, self.current_obs_policy, self.current_priv_info, self.current_command_vel = self.real_env.reset()
            self.last_action_policy = np.zeros(self.action_dim, dtype=np.float32)

        for _ in range(num_steps):
            obs_policy_tensor = torch.tensor(self.current_obs_policy, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            with torch.no_grad():
                action_tensor = self.policy_model.sample_action(obs_policy_tensor)
            action_numpy = action_tensor.squeeze(0).cpu().numpy()

            # The `act_wm` argument in `add_transition` is the action *taken* at current_obs_wm/policy
            act_wm_to_buffer = action_numpy.copy() 
            # The `act_policy` argument in `add_transition` is the action that was part of current_obs_policy (i.e. last_action_policy)
            act_policy_to_buffer = self.last_action_policy.copy() 

            next_obs_wm, next_obs_policy, next_priv_info, env_reward, done, info = self.real_env.step(action_numpy)
            
            # The current command_vel is associated with the current observation and the action taken
            command_vel_to_buffer = self.current_command_vel.copy()

            self.replay_buffer.add_transition(
                self.current_obs_wm, act_wm_to_buffer, next_obs_wm, self.current_priv_info, next_priv_info,
                self.current_obs_policy, act_policy_to_buffer, next_obs_policy, env_reward, done, command_vel_to_buffer
            )

            # Update for next step
            self.current_obs_wm = next_obs_wm
            self.current_obs_policy = next_obs_policy
            self.current_priv_info = next_priv_info
            self.last_action_policy = action_numpy
            
            if done:
                self.current_obs_wm, self.current_obs_policy, self.current_priv_info, self.current_command_vel = self.real_env.reset()
                self.last_action_policy = np.zeros(self.action_dim, dtype=np.float32)
            
            # Log real environment rewards
            self.writer.add_scalar("MBPO-PPO/RealEnv/reward", env_reward, self.global_step)
            self.global_step += 1 # Increment global step for logging

    def _imagine_trajectories(self) -> Dict[str, torch.Tensor]:
        """
        Generates long-horizon imagined trajectories using the RWM and the policy.

        Returns:
            A dictionary of batched torch.Tensors containing imagined data for PPO update.
        """
        self.policy_model.eval() # Policy in eval mode for imagination
        self.rwm_model.eval()    # RWM in eval mode for imagination

        # Lists to store imagined data
        imagined_obs_policy_list: List[torch.Tensor] = []
        imagined_actions_list: List[torch.Tensor] = []
        imagined_log_probs_list: List[torch.Tensor] = []
        imagined_rewards_list: List[torch.Tensor] = []
        imagined_next_obs_policy_list: List[torch.Tensor] = []
        imagined_dones_list: List[torch.Tensor] = []
        imagined_value_preds_list: List[torch.Tensor] = []

        # 1. Sample initial states for imagination from the replay buffer
        # This will return a dict with initial_obs_policy_batch, initial_command_vel_batch,
        # rwm_initial_obs_hist_batch, rwm_initial_act_hist_batch.
        initial_data = self.replay_buffer.sample_policy_init_obs(self.imagination_environments)

        current_obs_policy_batch: torch.Tensor = initial_data['initial_obs_policy_batch']
        current_command_vel_batch: torch.Tensor = initial_data['initial_command_vel_batch']
        rwm_initial_obs_hist_batch: torch.Tensor = initial_data['initial_rwm_obs_hist_batch']
        rwm_initial_act_hist_batch: torch.Tensor = initial_data['initial_rwm_act_hist_batch']
        
        # Initialize GRU hidden state for RWM and process initial history
        rwm_hidden_state_batch: torch.Tensor = self.rwm_model.get_initial_hidden_state(self.imagination_environments)
        # Pass M real history steps through GRU to initialize its state
        # The outputs (mean/log_std) from this forward pass are discarded for imagination;
        # we only care about the updated rwm_hidden_state_batch.
        _, _, _, _, rwm_hidden_state_batch = self.rwm_model.forward(
            rwm_initial_obs_hist_batch, rwm_initial_act_hist_batch, rwm_hidden_state_batch
        )
        
        # The world model input for the first imagination step is the last observation from history
        current_obs_wm_for_rwm_input: torch.Tensor = rwm_initial_obs_hist_batch[:, -1, :].clone().detach()

        # Track which imagination environments are done
        imagination_dones = torch.zeros(self.imagination_environments, dtype=torch.bool, device=self.device)
        
        # 2. Imagination Loop
        for i in range(self.imagination_steps_per_iteration):
            # If an environment is already done, it should be reset or skipped.
            # For simplicity, we'll reset them by sampling new initial data if done.
            # This is done *after* collecting data for the current step,
            # so `done_batch` below will represent termination *at the end of the step*.
            
            # Policy action based on current imagined policy observation
            with torch.no_grad():
                value_pred_batch = self.value_model(current_obs_policy_batch).squeeze(-1) # (batch_size,)
                action_mean, action_log_std = self.policy_model(current_obs_policy_batch)
            
            normal_dist = distributions.Normal(action_mean, torch.exp(action_log_std))
            action_batch = normal_dist.rsample() # Sample action for this step (B, action_dim)
            log_probs_batch = normal_dist.log_prob(action_batch).sum(dim=-1) # (B,)

            # RWM prediction for next_obs_wm and next_priv_info
            with torch.no_grad(): # RWM is in eval mode
                mean_obs_wm, log_std_obs_wm, mean_priv_info, log_std_priv_info, next_rwm_hidden_state_batch = \
                    self.rwm_model.forward(
                        current_obs_wm_for_rwm_input.unsqueeze(1), # (B, 1, obs_wm_dim)
                        action_batch.unsqueeze(1),                 # (B, 1, act_wm_dim)
                        rwm_hidden_state_batch                     # (num_layers, B, hidden_dim)
                    )
            
            # Squeeze sequence dimension for mean/log_std, then sample
            mean_obs_wm = mean_obs_wm.squeeze(1)
            log_std_obs_wm = log_std_obs_wm.squeeze(1)
            mean_priv_info = mean_priv_info.squeeze(1)
            log_std_priv_info = log_std_priv_info.squeeze(1)

            # Sample next observation and privileged info using reparameterization trick
            std_obs_wm = torch.exp(log_std_obs_wm)
            sampled_next_obs_wm = mean_obs_wm + std_obs_wm * torch.randn_like(std_obs_wm)

            std_priv_info = torch.exp(log_std_priv_info)
            sampled_next_priv_info = mean_priv_info + std_priv_info * torch.randn_like(std_priv_info)

            # Construct policy observation for the next state from imagined RWM predictions
            next_obs_policy_imagined_batch = self._construct_policy_obs_from_wm_pred(
                predicted_obs_wm=sampled_next_obs_wm,
                command_vel=current_command_vel_batch,
                last_action=action_batch, # The action just taken
            )

            # Calculate imagined reward using the environment's reward function
            # Need to convert tensors back to numpy for environment's reward calculation
            imagined_reward_batch_np = self.real_env.calculate_reward_from_components(
                current_obs_policy_batch.cpu().numpy(),
                action_batch.cpu().numpy(),
                next_obs_policy_imagined_batch.cpu().numpy(),
                sampled_next_priv_info.cpu().numpy(),
                current_command_vel_batch.cpu().numpy(),
            )
            imagined_reward_batch = torch.tensor(imagined_reward_batch_np, dtype=torch.float32, device=self.device)

            # Determine imagination termination (done)
            done_batch = self._check_imagination_done(sampled_next_priv_info, sampled_next_obs_wm)
            
            # Store imagined transition
            imagined_obs_policy_list.append(current_obs_policy_batch)
            imagined_actions_list.append(action_batch)
            imagined_log_probs_list.append(log_probs_batch)
            imagined_rewards_list.append(imagined_reward_batch)
            imagined_next_obs_policy_list.append(next_obs_policy_imagined_batch)
            imagined_dones_list.append(done_batch)
            imagined_value_preds_list.append(value_pred_batch)

            # Update for next imagination step
            current_obs_policy_batch = next_obs_policy_imagined_batch
            current_obs_wm_for_rwm_input = sampled_next_obs_wm
            rwm_hidden_state_batch = next_rwm_hidden_state_batch

            # Reset environments that became "done" in imagination
            if done_batch.any():
                # For environments that are done, sample new initial states
                reset_indices = torch.where(done_batch)[0]
                num_resets = reset_indices.shape[0]

                if num_resets > 0:
                    reset_data = self.replay_buffer.sample_policy_init_obs(num_resets)
                    
                    current_obs_policy_batch[reset_indices] = reset_data['initial_obs_policy_batch']
                    current_command_vel_batch[reset_indices] = reset_data['initial_command_vel_batch']
                    
                    # For RWM history and hidden state, process the new history for reset environments
                    reset_rwm_initial_obs_hist = reset_data['initial_rwm_obs_hist_batch']
                    reset_rwm_initial_act_hist = reset_data['initial_rwm_act_hist_batch']

                    # Recompute hidden state for reset environments
                    new_rwm_hidden_state = self.rwm_model.get_initial_hidden_state(num_resets)
                    _, _, _, _, new_rwm_hidden_state = self.rwm_model.forward(
                        reset_rwm_initial_obs_hist, reset_rwm_initial_act_hist, new_rwm_hidden_state
                    )
                    rwm_hidden_state_batch[:, reset_indices, :] = new_rwm_hidden_state # Update specific indices

                    # Update current_obs_wm_for_rwm_input for reset environments
                    current_obs_wm_for_rwm_input[reset_indices] = reset_rwm_initial_obs_hist[:, -1, :].clone().detach()

        # Concatenate all imagined data
        imagined_data = {
            "obs_policy": torch.cat(imagined_obs_policy_list, dim=0),
            "actions": torch.cat(imagined_actions_list, dim=0),
            "log_probs_old": torch.cat(imagined_log_probs_list, dim=0),
            "rewards": torch.cat(imagined_rewards_list, dim=0),
            "next_obs_policy": torch.cat(imagined_next_obs_policy_list, dim=0),
            "dones": torch.cat(imagined_dones_list, dim=0),
            "values": torch.cat(imagined_value_preds_list, dim=0),
        }
        
        return imagined_data

    def _update_policy(self, imagined_data: Dict[str, torch.Tensor]) -> Tuple[float, float]:
        """
        Updates the policy and value networks using the collected imagined data via PPO.

        Args:
            imagined_data: A dictionary of torch.Tensors containing imagined experiences.

        Returns:
            A tuple of (policy_loss_item, value_loss_item) from the last mini-batch.
        """
        self.policy_model.train() # Policy in train mode for updates
        self.value_model.train()  # Value in train mode for updates

        obs_policy_batch = imagined_data["obs_policy"]
        actions_batch = imagined_data["actions"]
        log_probs_old_batch = imagined_data["log_probs_old"]
        rewards_batch = imagined_data["rewards"]
        dones_batch = imagined_data["dones"]
        values_batch = imagined_data["values"] # Value predictions *at time of imagination*

        # Calculate next_values from next_obs_policy
        with torch.no_grad():
            # For the last step of a trajectory, if it's done, next_value should be 0.
            # Otherwise, it's the value of the next state.
            # However, compute_gae handles this using next_values and dones.
            # The imagined_next_obs_policy_list in `_imagine_trajectories` stores
            # the next_obs for the last step too, so we need to pass it.
            # For terminal states, value function should return 0.
            # The GAE calculation needs V(s_t+1) for the TD-error.
            # For this reason, we calculate V(s_t+1) from `imagined_data["next_obs_policy"]`.
            next_values_batch = self.value_model(imagined_data["next_obs_policy"]).squeeze(-1)


        # Compute Generalized Advantage Estimation and Returns
        advantages_batch, returns_batch = compute_gae(
            rewards_batch, values_batch, next_values_batch, dones_batch, self.gamma, self.gae_lambda
        )

        # Normalize advantages for stability (standard practice in PPO)
        advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

        # Reshape for easier indexing if needed (e.g. for creating mini-batches)
        total_samples = obs_policy_batch.shape[0]
        indices = np.arange(total_samples)

        last_policy_loss, last_value_loss = 0.0, 0.0

        for _ in range(self.learning_epochs):
            np.random.shuffle(indices)
            for start_idx in range(0, total_samples, total_samples // self.mini_batches):
                end_idx = start_idx + total_samples // self.mini_batches
                batch_indices = indices[start_idx:end_idx]

                # Extract mini-batch
                mb_obs_policy = obs_policy_batch[batch_indices]
                mb_actions = actions_batch[batch_indices]
                mb_log_probs_old = log_probs_old_batch[batch_indices]
                mb_advantages = advantages_batch[batch_indices]
                mb_returns = returns_batch[batch_indices]
                
                # Policy Loss
                log_probs_new, entropy_batch, _ = self.policy_model.evaluate_actions(mb_obs_policy, mb_actions)
                ratio = torch.exp(log_probs_new - mb_log_probs_old)
                
                surrogate1 = ratio * mb_advantages
                surrogate2 = torch.clamp(ratio, 1 - self.clip_range_epsilon, 1 + self.clip_range_epsilon) * mb_advantages
                
                policy_loss = -torch.min(surrogate1, surrogate2).mean() - self.entropy_coefficient * entropy_batch.mean()

                # Value Loss
                values_new = self.value_model(mb_obs_policy).squeeze(-1)
                value_loss = (values_new - mb_returns).pow(2).mean() # MSE loss

                # Update policy network
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

                # Update value network
                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()
                
                last_policy_loss = policy_loss.item()
                last_value_loss = value_loss.item()

        return last_policy_loss, last_value_loss

    def run_training_loop(self, total_iterations: int, evaluator: "Evaluator") -> None:
        """
        Executes the main MBPO-PPO training loop.

        Args:
            total_iterations: The total number of learning iterations.
            evaluator: An instance of the Evaluator class for periodic evaluations.
        """
        self.global_step = 0 # Track total steps in real environment
        
        # Initial reset for the environment state for data collection
        self.current_obs_wm, self.current_obs_policy, self.current_priv_info, self.current_command_vel = self.real_env.reset()
        self.last_action_policy = np.zeros(self.action_dim, dtype=np.float32)

        print(f"Starting MBPO-PPO training for {total_iterations} iterations...")
        for iteration in range(total_iterations):
            # 1. Collect real data
            self._collect_real_data(self.real_steps_per_iteration)
            self.writer.add_scalar("MBPO-PPO/ReplayBuffer_size", self.replay_buffer.size(), iteration)

            # 2. Fine-tune RWM
            self.rwm_trainer.finetune_rwm(self.rwm_finetune_iterations_per_mbpo_step, iteration)

            # 3. Imagine trajectories
            imagined_data = self._imagine_trajectories()

            # 4. Update policy
            policy_loss, value_loss = self._update_policy(imagined_data)
            
            self.writer.add_scalar("MBPO-PPO/Policy_Loss", policy_loss, iteration)
            self.writer.add_scalar("MBPO-PPO/Value_Loss", value_loss, iteration)

            if (iteration + 1) % self.config.global.eval_freq == 0:
                avg_reward, avg_episode_len = evaluator.evaluate_policy_in_env(
                    num_episodes=10, # Evaluate over 10 episodes
                    render=False # Render only for specific manual checks
                )
                self.writer.add_scalar("MBPO-PPO/Eval/Average_Reward", avg_reward, iteration)
                self.writer.add_scalar("MBPO-PPO/Eval/Average_Episode_Length", avg_episode_len, iteration)
                print(f"Iteration {iteration+1}/{total_iterations} | Policy Loss: {policy_loss:.4f} | Value Loss: {value_loss:.4f} | Avg Eval Reward: {avg_reward:.2f}")

            if (iteration + 1) % self.config.global.save_freq == 0:
                model_path = f"{self.config.global.model_dir}/policy_model_iter_{iteration+1}.pth"
                torch.save(self.policy_model.state_dict(), model_path)
                model_path = f"{self.config.global.model_dir}/value_model_iter_{iteration+1}.pth"
                torch.save(self.value_model.state_dict(), model_path)
                model_path = f"{self.config.global.model_dir}/rwm_model_iter_{iteration+1}.pth"
                torch.save(self.rwm_model.state_dict(), model_path)
                print(f"Models saved at iteration {iteration+1}")

        print("MBPO-PPO training complete.")

