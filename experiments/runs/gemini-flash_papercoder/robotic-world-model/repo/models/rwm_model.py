import torch
import torch.nn as nn
import torch.distributions as distributions
from typing import List, Tuple, Any, TYPE_CHECKING

# Local imports
from config import Config
from utils import gaussian_nll_loss

# For type hinting without circular import issues
if TYPE_CHECKING:
    from .policy_value_model import PolicyModel


def _build_mlp_layers(
    input_dim: int,
    hidden_dims: List[int] | int, # Accepts list or single int for hidden_dims
    output_dim: int,
    activation_type: str,
    output_activation: bool = False,
) -> nn.Sequential:
    """
    Helper function to build a multi-layer perceptron (MLP).

    Args:
        input_dim: The dimension of the input layer.
        hidden_dims: A list of integers, where each integer represents the number of neurons
                     in a hidden layer, or a single integer for one hidden layer.
        output_dim: The dimension of the output layer.
        activation_type: A string specifying the activation function for hidden layers (e.g., "ReLU", "ELU").
        output_activation: If True, applies the activation function to the output layer.

    Returns:
        A torch.nn.Sequential model representing the MLP.
    """
    layers: List[nn.Module] = []
    current_dim = input_dim

    # Get activation function module
    if activation_type == "ReLU":
        activation_fn = nn.ReLU()
    elif activation_type == "ELU":
        activation_fn = nn.ELU()
    elif activation_type == "Tanh":
        activation_fn = nn.Tanh()
    else:
        raise ValueError(f"Unsupported activation type: {activation_type}")

    # Ensure hidden_dims is a list for iteration
    if isinstance(hidden_dims, int):
        hidden_dims = [hidden_dims]

    # Hidden layers
    for h_dim in hidden_dims:
        layers.append(nn.Linear(current_dim, h_dim))
        layers.append(activation_fn)
        current_dim = h_dim

    # Output layer
    layers.append(nn.Linear(current_dim, output_dim))
    if output_activation:
        layers.append(activation_fn)

    return nn.Sequential(*layers)


class RWMModel(nn.Module):
    """
    The Robotic World Model (RWM) architecture.
    Consists of a GRU base for processing historical observation-action sequences
    and MLP heads for predicting the mean and standard deviation of the
    next observation and privileged information.
    """

    def __init__(self, obs_wm_dim: int, act_wm_dim: int, priv_dim: int, config: Config):
        """
        Initializes the RWM network components.

        Args:
            obs_wm_dim: Dimension of the world model observation space.
            act_wm_dim: Dimension of the action space.
            priv_dim: Dimension of the privileged information space.
            config: Configuration object containing model hyperparameters.
        """
        super().__init__()
        self.obs_wm_dim = obs_wm_dim
        self.act_wm_dim = act_wm_dim
        self.priv_dim = priv_dim
        self.config = config
        self.device = config.global.device

        # --- GRU Base Initialization ---
        gru_input_dim = obs_wm_dim + act_wm_dim
        
        # Retrieve GRU architecture from config
        base_hidden_shape = self.config.rwm_model.architecture.base_hidden_shape
        # According to design: [256, 256] implies two GRU layers, each with 256 hidden units.
        self.hidden_state_dim = base_hidden_shape[0]  # e.g., 256
        self.num_gru_layers = len(base_hidden_shape) # e.g., 2

        self.gru_base = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=self.hidden_state_dim,
            num_layers=self.num_gru_layers,
            batch_first=True,  # Input/output tensors are (batch, seq, feature)
        ).to(self.device)

        # --- MLP Head Initialization ---
        heads_hidden_shape = self.config.rwm_model.architecture.heads_hidden_shape
        heads_activation_type = self.config.rwm_model.architecture.heads_activation

        # Observation Head: predicts mean and log_std for obs_wm
        self.obs_head = _build_mlp_layers(
            input_dim=self.hidden_state_dim,
            hidden_dims=heads_hidden_shape,
            output_dim=2 * self.obs_wm_dim, # 2 * dim for mean and log_std
            activation_type=heads_activation_type,
            output_activation=False,
        ).to(self.device)

        # Privileged Information Head: predicts mean and log_std for priv_info
        self.priv_head = _build_mlp_layers(
            input_dim=self.hidden_state_dim,
            hidden_dims=heads_hidden_shape,
            output_dim=2 * self.priv_dim, # 2 * dim for mean and log_std
            activation_type=heads_activation_type,
            output_activation=False,
        ).to(self.device)

    def forward(
        self,
        obs_hist_batch: torch.Tensor,
        act_hist_batch: torch.Tensor,
        initial_hidden_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Processes a sequence of historical observation-action pairs through the GRU
        and predicts the distribution parameters for the *next* observation and
        privileged information.

        Args:
            obs_hist_batch: Batch of historical observations. Shape (batch_size, M, obs_wm_dim).
            act_hist_batch: Batch of historical actions. Shape (batch_size, M, act_wm_dim).
            initial_hidden_state: The hidden state of the GRU to start the processing
                                  of the history. Shape (num_gru_layers, batch_size, hidden_state_dim).

        Returns:
            A tuple containing:
            - mean_obs: Predicted mean for the next observation (batch_size, obs_wm_dim).
            - log_std_obs: Predicted log standard deviation for the next observation (batch_size, obs_wm_dim).
            - mean_priv: Predicted mean for the next privileged information (batch_size, priv_dim).
            - log_std_priv: Predicted log standard deviation for the next privileged information (batch_size, priv_dim).
            - final_hidden_state: The GRU hidden state after processing the entire history
                                  (num_gru_layers, batch_size, hidden_state_dim).
        """
        # Ensure inputs are on the correct device
        obs_hist_batch = obs_hist_batch.to(self.device)
        act_hist_batch = act_hist_batch.to(self.device)
        initial_hidden_state = initial_hidden_state.to(self.device)

        # Concatenate observation and action for GRU input at each time step
        gru_input = torch.cat([obs_hist_batch, act_hist_batch], dim=-1) # (batch_size, M, gru_input_dim)

        # GRU forward pass
        # output: (batch_size, M, hidden_state_dim) - output features from the last layer for each t
        # final_hidden_state: (num_layers, batch_size, hidden_state_dim) - hidden state for each layer at the last t
        _, final_hidden_state = self.gru_base(gru_input, initial_hidden_state)

        # The input to the heads is the hidden state of the LAST GRU layer
        # after processing the entire history sequence.
        head_input = final_hidden_state[-1, :, :] # (batch_size, hidden_state_dim)

        # --- Observation Head Prediction ---
        obs_pred_raw = self.obs_head(head_input)
        mean_obs = obs_pred_raw[..., :self.obs_wm_dim]
        log_std_obs = obs_pred_raw[..., self.obs_wm_dim:]

        # --- Privileged Information Head Prediction ---
        priv_pred_raw = self.priv_head(head_input)
        mean_priv = priv_pred_raw[..., :self.priv_dim]
        log_std_priv = priv_pred_raw[..., self.priv_dim:]

        return mean_obs, log_std_obs, mean_priv, log_std_priv, final_hidden_state

    def predict_autoregressive(
        self,
        initial_obs_hist: List[torch.Tensor],
        initial_act_hist: List[torch.Tensor],
        forecast_steps: int,
        policy_model: 'PolicyModel', # Type hinted, but functionally unused inside this method due to design constraints
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs `forecast_steps` of autoregressive prediction using the RWM.
        This method is designed for RWM self-evaluation where policy actions might be
        pre-recorded or fixed, as the full policy-driven imagination with command_vel
        is handled by the MBPOPPO_Trainer.

        Args:
            initial_obs_hist: A list of M historical observations, each (batch_size, obs_wm_dim).
            initial_act_hist: A list of M historical actions, each (batch_size, act_wm_dim).
            forecast_steps: The number of future steps (N) to predict autoregressively.
            policy_model: The policy model. Unused internally in this method to generate actions
                          due to the specified interface constraints (lack of command_vel/obs_policy).

        Returns:
            A tuple containing:
            - predicted_obs_seq: Sequence of predicted observations (batch_size, N, obs_wm_dim).
            - predicted_priv_seq: Sequence of predicted privileged information (batch_size, N, priv_dim).
        """
        batch_size = initial_obs_hist[0].shape[0]
        
        # Prepare history for the initial GRU forward pass
        obs_hist_batch = torch.stack(initial_obs_hist, dim=1).to(self.device) # (batch_size, M, obs_wm_dim)
        act_hist_batch = torch.stack(initial_act_hist, dim=1).to(self.device) # (batch_size, M, act_wm_dim)

        # Initialize GRU hidden state for the history processing
        initial_gru_hidden_state = self.get_initial_hidden_state(batch_size).to(self.device)

        # Process the initial M historical steps to get the first prediction (t+1)
        # and the GRU's hidden state after processing history.
        mean_obs_t_plus_1, log_std_obs_t_plus_1, mean_priv_t_plus_1, log_std_priv_t_plus_1, current_rwm_hidden_state = \
            self.forward(obs_hist_batch, act_hist_batch, initial_gru_hidden_state)

        # Sample the first predicted observation and privileged info using reparameterization trick
        # z = mu + sigma * epsilon, where epsilon ~ N(0,1)
        std_obs_t_plus_1 = torch.exp(log_std_obs_t_plus_1)
        current_obs_wm = mean_obs_t_plus_1 + std_obs_t_plus_1 * torch.randn_like(std_obs_t_plus_1)

        std_priv_t_plus_1 = torch.exp(log_std_priv_t_plus_1)
        current_priv_info = mean_priv_t_plus_1 + std_priv_t_plus_1 * torch.randn_like(std_priv_t_plus_1)

        predicted_obs_seq: List[torch.Tensor] = [current_obs_wm]
        predicted_priv_seq: List[torch.Tensor] = [current_priv_info]

        # Action for forecasting: Use the last action from history repeatedly
        # This is a simplification based on the design's constraints for this specific method
        current_act_wm = initial_act_hist[-1].to(self.device) # (batch_size, act_wm_dim)

        # --- Autoregressive Prediction Loop ---
        for _ in range(forecast_steps - 1): # Already predicted one step
            # Unsqueeze to add a sequence dimension of 1 for the GRU forward pass
            # (batch_size, 1, obs_wm_dim) and (batch_size, 1, act_wm_dim)
            obs_input_for_gru = current_obs_wm.unsqueeze(1)
            act_input_for_gru = current_act_wm.unsqueeze(1)

            # Predict the next observation and privileged information
            mean_obs, log_std_obs, mean_priv, log_std_priv, next_rwm_hidden_state = \
                self.forward(obs_input_for_gru, act_input_for_gru, current_rwm_hidden_state)

            # Sample next observation and privileged info
            std_obs = torch.exp(log_std_obs)
            current_obs_wm = mean_obs + std_obs * torch.randn_like(std_obs)

            std_priv = torch.exp(log_std_priv)
            current_priv_info = mean_priv + std_priv * torch.randn_like(std_priv)
            
            current_rwm_hidden_state = next_rwm_hidden_state # Update hidden state

            predicted_obs_seq.append(current_obs_wm)
            predicted_priv_seq.append(current_priv_info)

        # Stack list of tensors into a single tensor (batch_size, N, dim)
        return torch.stack(predicted_obs_seq, dim=1), torch.stack(predicted_priv_seq, dim=1)


    def get_initial_hidden_state(self, batch_size: int) -> torch.Tensor:
        """
        Returns a zero-initialized hidden state for the GRU.

        Args:
            batch_size: The batch size for which to create the hidden state.

        Returns:
            A torch.Tensor representing the initial hidden state
            (num_gru_layers, batch_size, hidden_state_dim).
        """
        return torch.zeros(
            self.num_gru_layers, batch_size, self.hidden_state_dim, device=self.device
        )

