```python
import numpy as np
import torch
import math
from typing import List, Tuple, Dict, Any, TYPE_CHECKING

# Local imports
from config import Config
from environment import Environment

# To avoid circular imports for type hinting without full imports
if TYPE_CHECKING:
    from models.rwm_model import RWMModel
    from models.policy_value_model import PolicyModel
    # ReplayBuffer is not directly used by Evaluator, but its output data format
    # from _collect_rollout_data needs to be consistent.


def _compute_rmse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculates the Root Mean Squared Error (RMSE) between predictions and targets.
    
    Args:
        predictions: Predicted tensor.
        targets: Ground truth tensor.
        
    Returns:
        The RMSE as a float.
    """
    if predictions.shape != targets.shape:
        raise ValueError(f"Shape mismatch: predictions {predictions.shape}, targets {targets.shape}")
    return torch.sqrt(torch.mean((predictions - targets)**2)).item()

def _compute_nrmse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculates the Normalized Root Mean Squared Error (NRMSE) between predictions and targets.
    Normalized by the standard deviation of the targets. Adds a small epsilon to avoid division by zero.
    
    Args:
        predictions: Predicted tensor.
        targets: Ground truth tensor.
        
    Returns:
        The NRMSE as a float.
    """
    rmse = _compute_rmse(predictions, targets)
    target_std = targets.std()
    if target_std < 1e-8: # If targets are nearly constant
        return rmse # If std is tiny, RMSE is effectively the absolute error, no meaningful relative error
    return rmse / (target_std + 1e-8) # Add epsilon for numerical stability


class Evaluator:
    """
    Evaluates the performance of the trained Robotic World Model (RWM) and
    the optimized policy. Provides methods for:
    - Autoregressive prediction accuracy of RWM.
    - Robustness of RWM under noisy conditions.
    - Policy performance in the simulation environment.
    """

    def __init__(
        self,
        rwm_model: "RWMModel",
        policy_model: "PolicyModel",
        real_env: Environment,
        config: Config,
    ):
        """
        Initializes the Evaluator.

        Args:
            rwm_model: An instance of the RWMModel.
            policy_model: An instance of the PolicyModel.
            real_env: An instance of the Environment for real interactions.
            config: The global configuration object.
        """
        self.rwm_model = rwm_model
        self.policy_model = policy_model
        self.real_env = real_env
        self.config = config
        self.device = torch.device(config.global.device)

        self.M: int = self.config.rwm_model.training.history_horizon_M
        self.N: int = self.config.rwm_model.training.forecast_horizon_N # Training forecast horizon

        # Cache for collected ground truth trajectories. This allows multiple evaluations
        # without re-running environment interactions.
        self._trajectories_cache: List[Dict[str, torch.Tensor]] = []

        # Ensure models are in evaluation mode
        self.rwm_model.eval()
        self.policy_model.eval()

        print("Evaluator initialized.")

    def _collect_rollout_data(
        self, num_trajectories: int, max_steps_per_trajectory: int = 1000
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Collects a specified number of ground-truth trajectories from the real_env
        using the current policy.

        Args:
            num_trajectories: The number of complete episodes/trajectories to collect.
            max_steps_per_trajectory: Maximum steps for an episode before truncation.

        Returns:
            A list of dictionaries, where each dict represents a collected trajectory
            with sequences of tensors for 'obs_wm', 'act_wm', 'priv_info', 'obs_policy',
            and 'command_vel'. All tensors are on the configured device.
        """
        print(f"Collecting {num_trajectories} ground-truth trajectories from environment...")
        collected_trajectories: List[Dict[str, torch.Tensor]] = []

        self.policy_model.eval() # Ensure policy is in eval mode during data collection

        for i in range(num_trajectories):
            # Reset environment and get initial observations/command
            obs_wm_np, obs_policy_np, priv_info_np, command_vel_np = self.real_env.reset()
            current_obs_wm_t = torch.tensor(obs_wm_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            current_obs_policy_t = torch.tensor(obs_policy_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            current_priv_info_t = torch.tensor(priv_info_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            current_command_vel_t = torch.tensor(command_vel_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            # Store sequences for the current trajectory
            # obs_wm_seq_t will store O_0, O_1, ..., O_L
            # act_wm_seq_t will store A_0, A_1, ..., A_{L-1} (action taken at O_t to get O_{t+1})
            # priv_info_seq_t will store PI_0, PI_1, ..., PI_L
            # obs_policy_seq_t will store OP_0, OP_1, ..., OP_L
            # command_vel_seq_t will store CV_0, CV_1, ..., CV_L
            obs_wm_seq_t: List[torch.Tensor] = [current_obs_wm_t.squeeze(0)]
            act_wm_seq_t: List[torch.Tensor] = []
            priv_info_seq_t: List[torch.Tensor] = [current_priv_info_t.squeeze(0)]
            obs_policy_seq_t: List[torch.Tensor] = [current_obs_policy_t.squeeze(0)]
            command_vel_seq_t: List[torch.Tensor] = [current_command_vel_t.squeeze(0)]

            episode_length = 0
            done = False

            while not done and episode_length < max_steps_per_trajectory:
                with torch.no_grad():
                    action_t = self.policy_model.sample_action(current_obs_policy_t)
                action_np = action_t.squeeze(0).cpu().numpy()

                # Step the environment
                # Note: `reward` is for environment's own reward function, not relevant for RWM evaluation
                next_obs_wm_np, next_obs_policy_np, next_priv_info_np, _, done, _ = self.real_env.step(action_np)

                # Convert next states to tensors
                next_obs_wm_t = torch.tensor(next_obs_wm_np, dtype=torch.float32, device=self.device).unsqueeze(0)
                next_obs_policy_t = torch.tensor(next_obs_policy_np, dtype=torch.float32, device=self.device).unsqueeze(0)
                next_priv_info_t = torch.tensor(next_priv_info_np, dtype=torch.float32, device=self.device).unsqueeze(0)
                
                # Append the action taken at current_obs_wm_t
                act_wm_seq_t.append(action_t.squeeze(0)) # Store as (action_dim,)

                # Update current states for the next loop iteration
                current_obs_wm_t = next_obs_wm_t
                current_obs_policy_t = next_obs_policy_t
                current_priv_info_t = next_priv_info_t
                # command_vel_t remains the same throughout an episode unless specified otherwise by env

                # Append the next observations and command velocity (associated with that state)
                obs_wm_seq_t.append(current_obs_wm_t.squeeze(0))
                priv_info_seq_t.append(current_priv_info_t.squeeze(0))
                obs_policy_seq_t.append(current_obs_policy_t.squeeze(0))
                command_vel_seq_t.append(current_command_vel_t.squeeze(0))

                episode_length += 1
            
            # Stack all collected tensors into single batched tensors for the trajectory
            # A trajectory with length L has L observations and L-1 actions
            if len(obs_wm_seq_t) > self.M: # Ensure enough data for at least history
                trajectory = {
                    "obs_wm_seq": torch.stack(obs_wm_seq_t),       # (L, obs_wm_dim)
                    "act_wm_seq": torch.stack(act_wm_seq_t),       # (L-1, act_wm_dim) - Actions are for transitions
                    "priv_info_seq": torch.stack(priv_info_seq_t), # (L, priv_dim)
                    "obs_policy_seq": torch.stack(obs_policy_seq_t), # (L, obs_policy_dim)
                    "command_vel_seq": torch.stack(command_vel_seq_t) # (L, command_dim)
                }
                collected_trajectories.append(trajectory)
            else:
                print(f"Skipping short trajectory {i} (length {episode_length}). Need >M steps.")

        # Cache collected trajectories for subsequent evaluations
        self._trajectories_cache = collected_trajectories 
        print(f"Collected {len(collected_trajectories)} trajectories for evaluation cache.")
        return collected_trajectories

    def _run_rwm_ground_truth_actions_autoregression(
        self,
        initial_obs_hist: torch.Tensor,    # (1, M, obs_wm_dim)
        initial_act_hist: torch.Tensor,    # (1, M, act_wm_dim)
        ground_truth_future_act_wm_seq: torch.Tensor, # (1, N_eval, act_wm_dim)
        forecast_steps_eval: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs RWM in autoregressive mode, feeding it ground truth actions for the forecast steps.
        This is used for evaluating RWM's prediction accuracy (error accumulation) independent of policy errors.

        Args:
            initial_obs_hist: M historical observations (batch_size=1).
            initial_act_hist: M historical actions (batch_size=1).
            ground_truth_future_act_wm_seq: Sequence of N_eval ground truth actions for the forecast.
            forecast_steps_eval: Number of steps to forecast (N_eval).

        Returns:
            A tuple containing:
            - predicted_obs_seq: Sequence of predicted observations (1, N_eval, obs_wm_dim).
            - predicted_priv_seq: Sequence of predicted privileged information (1, N_eval, priv_dim).
        """
        self.rwm_model.eval() # Ensure RWM is in evaluation mode
        
        batch_size: int = initial_obs_hist.shape[0] # Should be 1 for evaluation
        
        # Initialize GRU hidden state for the history processing
        initial_gru_hidden_state = self.rwm_model.get_initial_hidden_state(batch_size).to(self.device)

        # Process the initial M historical steps to get the GRU's hidden state after history.
        # The outputs (mean/log_std) from this forward pass are not directly used for loss here,
        # but the updated hidden state is crucial for the forecast.
        # Note: initial_act_hist covers actions A_0 to A_{M-1} (M actions).
        _, _, _, _, current_rwm_hidden_state = self.rwm_model.forward(
            obs_hist_batch=initial_obs_hist, # O_0 to O_{M-1}
            act_hist_batch=initial_act_hist, # A_0 to A_{M-1}
            initial_hidden_state=initial_gru_hidden_state,
        )

        # The observation that "leads" into the first forecast step (at index M in gt sequence)
        # is the last observation from the history (O_{M-1}).
        current_obs_wm_input = initial_obs_hist[:, -1, :].clone().detach() # (1, obs_wm_dim)

        predicted_obs_seq: List[torch.Tensor] = []
        predicted_priv_seq: List[torch.Tensor] = []

        with torch.no_grad():
            for k in range(forecast_steps_eval): # k from 0 to N_eval-1
                # Action for the k-th forecast step (this is A_{M+k} in gt sequence)
                # This action is taken from predicted_obs_seq[k] to get predicted_obs_seq[k+1]
                action_k = ground_truth_future_act_wm_seq[:, k, :] # (1, act_wm_dim)

                # RWM forward pass for one step: (batch_size, 1, dim) for sequence input
                mean_obs, log_std_obs, mean_priv, log_std_priv, next_rwm_hidden_state = self.rwm_model.forward(
                    obs_hist_batch=current_obs_wm_input.unsqueeze(1),  # (1, 1, obs_wm_dim)
                    act_hist_batch=action_k.unsqueeze(1),              # (1, 1, act_wm_dim)
                    initial_hidden_state=current_rwm_hidden_state,     # (num_gru_layers, 1, hidden_state_dim)
                )

                # Sample next observation and privileged info using reparameterization trick
                std_obs = torch.exp(log_std_obs.squeeze(1))
                sampled_obs_wm = mean_obs.squeeze(1) + std_obs * torch.randn_like(std_obs)

                std_priv = torch.exp(log_std_priv.squeeze(1))
                sampled_priv_info = mean_priv.squeeze(1) + std_priv * torch.randn_like(std_priv)
                
                predicted_obs_seq.append(sampled_obs_wm)
                predicted_priv_seq.append(sampled_priv_info)

                # Update current_obs_wm_input and hidden state for the next iteration
                current_obs_wm_input = sampled_obs_wm
                current_rwm_hidden_state = next_rwm_hidden_state

        return torch.stack(predicted_obs_seq, dim=1), torch.stack(predicted_priv_seq, dim=1)


    def evaluate_rwm_autoregressive(self, num_trajectories: int, forecast_steps_max: int) -> Dict[str, Any]:
        """
        Evaluates the RWM's autoregressive prediction accuracy.
        The RWM is initialized with M historical steps, then predicts N_eval future steps,
        using ground truth actions for the forecast.

        Args:
            num_trajectories: The number of ground-truth trajectories to use.
            forecast_steps_max: The maximum number of steps to forecast for evaluation.

        Returns:
            A dictionary containing evaluation metrics (e.g., mean NRMSE over forecast steps).
            The 'forecast_steps' key holds a list of steps (1-indexed).
        """
        print(f"Evaluating RWM autoregressive prediction for {num_trajectories} trajectories "
              f"with forecast up to {forecast_steps_max} steps (using GT actions)...")
        
        # Minimum trajectory length required: M observations for history + forecast_steps_max observations for targets.
        # This means the trajectory must have at least M + forecast_steps_max data points.
        # An L-step trajectory has L observations (index 0 to L-1) and L-1 actions (index 0 to L-2).
        # History is O_0..O_{M-1} (M observations) and A_0..A_{M-1} (M actions).
        # Targets are O_M..O_{M+N_eval-1} (N_eval observations)
        # Future actions are A_M..A_{M+N_eval-1} (N_eval actions)
        # So total observation length = M + N_eval. Total action length = M + N_eval.
        # Thus, required length of obs_wm_seq must be at least M + forecast_steps_max.
        # Required length of act_wm_seq must be at least M + forecast_steps_max.
        min_trajectory_obs_len = self.M + forecast_steps_max
        min_trajectory_act_len = self.M + forecast_steps_max
        
        # Collect ground truth data if not already cached, ensuring enough length
        if len(self._trajectories_cache) < num_trajectories:
            self._trajectories_cache = self._collect_rollout_data(num_trajectories, max_steps_per_trajectory=min_trajectory_obs_len * 2) # collect more than min needed
            # Filter for trajectories long enough for history + forecast_steps_max
            self._trajectories_cache = [
                traj for traj in self._trajectories_cache 
                if traj['obs_wm_seq'].shape[0] >= min_trajectory_obs_len and 
                   traj['act_wm_seq'].shape[0] >= min_trajectory_act_len
            ]
            print(f"Filtered to {len(self._trajectories_cache)} trajectories of sufficient length for RWM eval.")
            if len(self._trajectories_cache) < num_trajectories:
                print(f"Warning: Not enough sufficiently long trajectories collected ({len(self._trajectories_cache)} < {num_trajectories}).")

        
        # Store NRMSE values for each forecast step
        all_obs_nrmses: List[List[float]] = [[] for _ in range(forecast_steps_max)]
        all_priv_nrmses: List[List[float]] = [[] for _ in range(forecast_steps_max)]

        for traj_idx, gt_trajectory in enumerate(self._trajectories_cache):
            if traj_idx >= num_trajectories: # Limit to requested number of trajectories
                break

            gt_obs_wm_seq = gt_trajectory["obs_wm_seq"]
            gt_