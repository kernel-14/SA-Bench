import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any

# Local imports
from config import Config
from utils import gaussian_nll_loss
from models.rwm_model import RWMModel
from data.replay_buffer import ReplayBuffer

class RWMTrainer:
    """
    Handles the training logic for the Robotic World Model (RWM),
    including pretraining and fine-tuning. Implements the dual-autoregressive
    mechanism and the multi-step prediction error objective.
    """

    def __init__(
        self,
        rwm_model: RWMModel,
        optimizer: torch.optim.Optimizer,
        replay_buffer: ReplayBuffer,
        config: Config,
        writer: SummaryWriter,
    ):
        """
        Initializes the RWMTrainer.

        Args:
            rwm_model: An instance of the RWMModel class.
            optimizer: A PyTorch optimizer configured for rwm_model's parameters.
            replay_buffer: An instance of the ReplayBuffer class.
            config: The global configuration object.
            writer: A TensorBoard SummaryWriter for logging training metrics.
        """
        self.rwm_model = rwm_model
        self.optimizer = optimizer
        self.replay_buffer = replay_buffer
        self.config = config
        self.writer = writer
        self.device = self.config.global.device

        # Hyperparameters for RWM training
        self.batch_size: int = self.config.rwm_model.training.batch_size
        self.history_horizon_M: int = self.config.rwm_model.training.history_horizon_M
        self.forecast_horizon_N: int = self.config.rwm_model.training.forecast_horizon_N
        self.forecast_decay_alpha: float = self.config.rwm_model.training.forecast_decay_alpha

        # Ensure model is on the correct device
        self.rwm_model.to(self.device)

    def _train_step(self) -> float:
        """
        Executes a single forward and backward pass for the RWM, performing
        dual-autoregressive prediction and loss calculation.

        Returns:
            The scalar value of the calculated total loss for this step.
        """
        self.rwm_model.train() # Ensure model is in training mode

        # 1. Data Sampling
        # replay_buffer.sample_rwm_batch is expected to return 'target_act_wm' as well
        batch_data: Dict[str, torch.Tensor] = self.replay_buffer.sample_rwm_batch(
            self.batch_size, self.history_horizon_M, self.forecast_horizon_N
        )

        # 2. Device Transfer (already handled by ReplayBuffer)
        history_obs_wm = batch_data["history_obs_wm"]
        history_act_wm = batch_data["history_act_wm"]
        target_obs_wm = batch_data["target_obs_wm"]
        target_priv_info = batch_data["target_priv_info"]
        target_act_wm = batch_data["target_act_wm"] # Actions corresponding to forecast steps
        initial_hidden_states = batch_data["initial_hidden_states"]

        # 3. Optimizer Zero_grad
        self.optimizer.zero_grad()

        # 4. Initial RWM Forward Pass (processing the full history sequence)
        # The RWMModel.forward processes the entire M-step history to yield the
        # GRU state and predictions for the step immediately following the history.
        # However, for autoregressive prediction, we only need the *final hidden state*
        # after processing history to start forecasting.
        _, _, _, _, current_rwm_hidden_state = self.rwm_model.forward(
            obs_hist_batch=history_obs_wm,
            act_hist_batch=history_act_wm,
            initial_hidden_state=initial_hidden_states,
        )

        # The observation that "leads" into the first forecast step (t+M)
        # is the last observation from the history.
        current_obs_wm_input = history_obs_wm[:, -1, :].clone().detach() # (batch_size, obs_wm_dim)
                                                                       # Detach to prevent gradients from flowing into history inputs

        # 5. Forecast Prediction & Loss Calculation (Outer Autoregression)
        total_loss = torch.tensor(0.0, device=self.device)

        for k in range(self.forecast_horizon_N): # k from 0 to N-1
            # Target for the k-th forecast step (t+M+k+1)
            target_obs_k = target_obs_wm[:, k, :]
            target_priv_k = target_priv_info[:, k, :]
            
            # Action for the k-th forecast step (a_{t+M+k})
            action_k = target_act_wm[:, k, :]

            # RWM forward pass for one step: (batch_size, 1, dim) for sequence input
            mean_obs, log_std_obs, mean_priv, log_std_priv, next_rwm_hidden_state = self.rwm_model.forward(
                obs_hist_batch=current_obs_wm_input.unsqueeze(1),  # (batch_size, 1, obs_wm_dim)
                act_hist_batch=action_k.unsqueeze(1),              # (batch_size, 1, act_wm_dim)
                initial_hidden_state=current_rwm_hidden_state,     # (num_gru_layers, batch_size, hidden_state_dim)
            )

            # Squeeze the sequence dimension for loss calculation: (batch_size, dim)
            mean_obs = mean_obs.squeeze(1)
            log_std_obs = log_std_obs.squeeze(1)
            mean_priv = mean_priv.squeeze(1)
            log_std_priv = log_std_priv.squeeze(1)

            # Calculate Observation Loss
            obs_loss = gaussian_nll_loss(mean_obs, log_std_obs, target_obs_k).mean() # Mean over batch

            # Calculate Privileged Information Loss
            priv_loss = gaussian_nll_loss(mean_priv, log_std_priv, target_priv_k).mean() # Mean over batch

            # Calculate Decay Factor: paper's sum index k starts from 1, so our 0-indexed k becomes k+1
            alpha_k = self.forecast_decay_alpha ** (k + 1)

            # Accumulate Total Loss
            total_loss += alpha_k * (obs_loss + priv_loss)

            # Autoregressive Feedback: Sample next observation using reparameterization trick
            # This sampled observation becomes the input for the next forecast step
            std_obs = torch.exp(log_std_obs)
            sampled_obs = mean_obs + std_obs * torch.randn_like(std_obs)
            current_obs_wm_input = sampled_obs # Update for the next iteration

            # Update hidden state for the next step
            current_rwm_hidden_state = next_rwm_hidden_state

        # Normalize Total Loss as per Equation 2 in the paper
        total_loss = total_loss / self.forecast_horizon_N

        # 6. Backpropagation
        total_loss.backward()

        # 7. Optimizer Step
        self.optimizer.step()

        return total_loss.item()

    def pretrain_rwm(self, num_iterations: int):
        """
        Drives the RWM pretraining phase for a specified number of iterations.

        Args:
            num_iterations: The total number of training steps for pretraining.
        """
        print(f"Starting RWM pretraining for {num_iterations} iterations...")
        for i in range(num_iterations):
            loss = self._train_step()
            self.writer.add_scalar("RWM/pretrain_loss", loss, i)

            if (i + 1) % 100 == 0 or i == 0:
                print(f"RWM Pretrain Iteration {i+1}/{num_iterations}, Loss: {loss:.4f}")
        
        self.rwm_model.eval() # Set model to evaluation mode after pretraining
        print("RWM pretraining complete.")

    def finetune_rwm(self, num_iterations: int, mbpo_ppo_iteration: int):
        """
        Fine-tunes the RWM on newly collected data during the MBPO-PPO training loop.

        Args:
            num_iterations: The number of training steps for fine-tuning.
            mbpo_ppo_iteration: The current iteration of the main MBPO-PPO training loop,
                                used for logging purposes.
        """
        # print(f"Starting RWM fine-tuning for {num_iterations} iterations at MBPO-PPO iter {mbpo_ppo_iteration}...")
        for i in range(num_iterations):
            loss = self._train_step()
            # Use mbpo_ppo_iteration as the global step for fine-tuning logs
            self.writer.add_scalar("RWM/finetune_loss", loss, mbpo_ppo_iteration)

            # Optional: print current loss and iteration to the console
            # if (i + 1) % (num_iterations // 5 if num_iterations > 5 else 1) == 0:
            #     print(f"  RWM Finetune Step {i+1}/{num_iterations}, Loss: {loss:.4f}")
        
        self.rwm_model.eval() # Set model to evaluation mode after fine-tuning
        # print("RWM fine-tuning complete.")

