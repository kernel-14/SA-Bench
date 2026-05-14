import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from utils.common_utils import MLPBlock, init_weights, get_optimizer


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Sinusoidal positional embedding for time encoding in diffusion models.
    """
    def __init__(self, dim: int):
        """
        Initializes the SinusoidalPositionalEmbedding module.

        Args:
            dim (int): The dimension of the output embeddings.
        """
        super().__init__()
        self.dim = dim

        # Compute the sinusoidal embeddings once
        # Maximum expected timestep is self.diffusion_steps, so create a buffer up to that.
        # This will be initialized in DenoisingDiffusionModel.
        # For now, it's a placeholder. The actual embeddings will be computed and stored
        # as a non-trainable buffer.
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Generates sinusoidal embeddings for a batch of timesteps.

        Args:
            t (torch.Tensor): A tensor of timesteps (batch_size,).

        Returns:
            torch.Tensor: Sinusoidal embeddings of shape (batch_size, self.dim).
        """
        # Ensure t is a float tensor and add a dimension for broadcasting with inv_freq
        t_float = t.float().unsqueeze(-1)  # (batch_size, 1)

        # Compute sin and cos components
        sin_input = t_float * self.inv_freq
        emb = torch.cat((sin_input.sin(), sin_input.cos()), dim=-1) # (batch_size, dim)

        # Handle odd dimensions if any
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1), mode='constant', value=0.0)
        return emb


class DenoisingDiffusionModel(nn.Module):
    """
    Implements a conditional Denoising Diffusion Probabilistic Model (DDPM).
    This model learns to predict the noise added to data (x_0) at a given timestep (t),
    conditioned on a relevance score (c). It supports Classifier-Free Guidance (CFG).
    """

    def __init__(self, config: Config, state_dim: Tuple[int, ...], action_dim: int,
                 reward_dim: int, condition_dim: int):
        """
        Initializes the DenoisingDiffusionModel.

        Args:
            config (Config): Configuration object.
            state_dim (Tuple[int, ...]): Dimension of the state observation.
                                         (e.g., (state_features,) for state-based, (latent_dim,) for pixel-latent).
            action_dim (int): Dimension of the action space.
            reward_dim (int): Dimension of the reward (typically 1).
            condition_dim (int): Dimension of the relevance function output (condition for generation).
        """
        super().__init__()
        self.config: Config = config
        self.device: torch.device = self.config.get_hyperparam('experiment.device')

        # Dimensions of transition components
        self.state_feature_dim: int = state_dim[0] # Assuming state_dim is already flattened (e.g. (256,) or (512,))
        self.action_dim: int = action_dim
        self.reward_dim: int = reward_dim
        self.condition_dim: int = condition_dim

        # The full flattened dimension of a transition (s, a, s', r)
        # s + a + s' + r
        self.transition_dim: int = self.state_feature_dim * 2 + self.action_dim + self.reward_dim

        # Time embedding
        self.time_embedding_dim: int = 64  # A common choice for time embeddings
        self.time_embedding = SinusoidalPositionalEmbedding(self.time_embedding_dim)

        # Noise prediction network (epsilon_theta)
        # Input to MLP: x_t (transition_dim) + time_embedding (time_embedding_dim) + condition (condition_dim)
        # Output: predicted noise (transition_dim)
        denoise_input_dim: int = self.transition_dim + self.time_embedding_dim + self.condition_dim
        denoise_output_dim: int = self.transition_dim

        # Using RL agent policy network hidden units/layers as a heuristic for the diffusion MLP
        # since specific diffusion network config is not provided in config.yaml.
        denoise_hidden_units: int = self.config.get_hyperparam('rl_agent.policy_hidden_units')
        denoise_hidden_layers: int = self.config.get_hyperparam('rl_agent.policy_hidden_layers')

        self.denoise_model = MLPBlock(
            input_dim=denoise_input_dim,
            output_dim=denoise_output_dim,
            hidden_units=denoise_hidden_units,
            num_hidden_layers=denoise_hidden_layers,
            activation_fn_name="ReLU",
            output_activation_fn_name=None # Output is noise, not bounded.
        )

        # Diffusion Schedule Parameters (DDPM-style linear schedule)
        self.diffusion_steps: int = self.config.get_hyperparam('generative_model.diffusion_steps')

        # Betas: linear schedule from 1e-4 to 0.02 is typical
        betas = torch.linspace(1e-4, 0.02, self.diffusion_steps, device=self.device)
        self.register_buffer('betas', betas)

        # Pre-compute useful values for the diffusion process
        alphas = 1.0 - betas
        self.register_buffer('alphas', alphas)

        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer('alpha_bars', alpha_bars)

        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        self.register_buffer('sqrt_alpha_bars', torch.sqrt(alpha_bars))
        self.register_buffer('sqrt_one_minus_alpha_bars', torch.sqrt(1.0 - alpha_bars))

        # Optimizer for the denoise model
        gen_lr: float = self.config.get_hyperparam('generative_model.learning_rate')
        optimizer_name: str = self.config.get_hyperparam('generative_model.optimizer')
        self.optimizer = get_optimizer(optimizer_name, self.denoise_model.parameters(), lr=gen_lr)

        self.to(self.device)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Performs the forward pass of the noise prediction network.
        Predicts the noise added to x_0 given x_t, timestep t, and an optional condition.

        Args:
            x_t (torch.Tensor): Noisy data at timestep t. Shape (batch_size, transition_dim).
            t (torch.Tensor): Timesteps. Shape (batch_size,).
            condition (Optional[torch.Tensor]): Relevance score c. Shape (batch_size, condition_dim).
                                              Can be None for unconditional generation (CFG).

        Returns:
            torch.Tensor: The predicted noise epsilon. Shape (batch_size, transition_dim).
        """
        # Ensure all inputs are on the correct device
        x_t = x_t.to(self.device)
        t = t.to(self.device)

        # Process timestep embedding
        t_emb = self.time_embedding(t).to(self.device) # (batch_size, time_embedding_dim)

        # Concatenate inputs for the denoise model
        input_tensor_parts: List[torch.Tensor] = [x_t, t_emb]

        # Handle condition: if None, use a zero tensor for the null condition Ø
        if condition is not None:
            condition = condition.to(self.device)
            input_tensor_parts.append(condition)
        else:
            # Create a zero tensor for the null condition Ø, broadcastable to batch_size
            null_condition = torch.zeros(x_t.shape[0], self.condition_dim, device=self.device)
            input_tensor_parts.append(null_condition)

        # Concatenate all parts
        model_input = torch.cat(input_tensor_parts, dim=-1) # (batch_size, denoise_input_dim)

        # Predict noise
        predicted_noise = self.denoise_model(model_input) # (batch_size, transition_dim)
        return predicted_noise

    def train_step(self, batch: Dict[str, torch.Tensor],
                   condition_scores: torch.Tensor, uncond_prob: float) -> float:
        """
        Performs one training step for the diffusion model.

        Args:
            batch (Dict[str, torch.Tensor]): A dictionary of transition components ('state', 'action', 'reward', 'next_state').
            condition_scores (torch.Tensor): Relevance scores for the transitions in the batch.
                                             Shape (batch_size, condition_dim).
            uncond_prob (float): Probability of dropping the condition during training for CFG.

        Returns:
            float: The loss value for this training step.
        """
        self.optimizer.zero_grad()

        # Prepare x_0 (original data) by concatenating transition components
        # Ensure they are on the correct device
        state = batch['state'].to(self.device)
        action = batch['action'].to(self.device)
        reward = batch['reward'].to(self.device)
        next_state = batch['next_state'].to(self.device)

        # Flatten state and next_state if they are images (i.e. latents from CNN encoder)
        if state.ndim > 2: # Assuming (batch, C, H, W) for pixels, convert to (batch, C*H*W) or (batch, latent_dim)
            # This should not happen if `state_dim` is already (latent_dim,)
            # If `state_dim` represents raw pixel shape, then this would require a CNN encoder here,
            # but the design specifies `state_dim` as output of visual encoder.
            # So, we expect state.shape to be (batch_size, self.state_feature_dim)
            if state.shape[1] != self.state_feature_dim:
                raise ValueError(f"State tensor unexpected shape for flattening: {state.shape}. Expected (batch_size, {self.state_feature_dim})")
            # If state is already (batch, latent_dim), no flattening needed
        
        x_0 = torch.cat([state, action, reward, next_state], dim=-1) # (batch_size, transition_dim)

        # Sample a random timestep t
        t = torch.randint(0, self.diffusion_steps, (x_0.shape[0],), device=self.device, dtype=torch.long)

        # Sample noise
        epsilon = torch.randn_like(x_0, device=self.device)

        # Forward diffusion process: x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * epsilon
        sqrt_alpha_bar_t = self.sqrt_alpha_bars[t].unsqueeze(-1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t].unsqueeze(-1)
        x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * epsilon

        # Classifier-Free Guidance (CFG) Training - Apply condition dropout
        # p_uncond from config
        uncond_drop_prob: float = self.config.get_hyperparam('generative_model.unconditional_drop_prob')
        
        # Create a mask for samples where condition is NOT dropped
        cond_mask = (torch.rand(x_0.shape[0], device=self.device) > uncond_drop_prob).float() # (batch_size,)

        # Prepare conditional input for the forward pass
        # For samples where condition is dropped, use a zero tensor (null condition Ø)
        # For samples where condition is kept, use the actual condition_scores
        # The `forward` method handles `None` condition as a zero tensor internally,
        # so we can simply pass `condition_scores` directly with `cond_mask` applied,
        # and the forward method will handle the null token based on its existence.
        # Alternatively, we can construct the conditional input here explicitly as per the logic analysis.
        
        # Method 1: Construct conditional input (explicitly follow logic analysis for CFG training objective)
        # Equation 2: (1 - p) * y + p * Ø
        # where p is Bernoulli(p_uncond).
        # We model this by using actual `condition_scores` for (1-p) cases and `zero_like` for p cases.
        cond_input = condition_scores.clone() # Start with all conditions
        # Identify which conditions to zero out (where cond_mask is 0, i.e., dropped)
        cond_input[cond_mask == 0] = 0.0 # Set conditions to zero for dropped samples

        # Predict noise using the denoise model
        epsilon_pred = self.forward(x_t, t, cond_input)

        # Compute loss
        loss = F.mse_loss(epsilon_pred, epsilon)

        # Backpropagate and optimize
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def sample(self, num_samples: int, condition_scores: torch.Tensor,
               guidance_scale: float, timesteps: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Generates synthetic transitions using the trained diffusion model with Classifier-Free Guidance.

        Args:
            num_samples (int): The number of synthetic transitions to generate.
            condition_scores (torch.Tensor): The relevance scores (conditions) for generation.
                                             Shape (num_samples, condition_dim).
            guidance_scale (float): The Classifier-Free Guidance scale (ω).
            timesteps (Optional[int]): Number of reverse diffusion steps to perform.
                                       If None, uses self.diffusion_steps.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing generated 'state', 'action', 'reward', 'next_state' tensors.
        """
        # Ensure conditions are on the correct device
        condition_scores = condition_scores.to(self.device)

        # Initialize x_T with random Gaussian noise
        x_t = torch.randn(num_samples, self.transition_dim, device=self.device)

        sampling_timesteps = timesteps if timesteps is not None else self.diffusion_steps
        
        # Adjust for using a subset of diffusion steps for faster sampling
        # We need to map `sampling_timesteps` to the actual `diffusion_steps` indices.
        # Example: if diffusion_steps=1000, sampling_timesteps=100, then we use t = 999, 989, ..., 9.
        step_indices = torch.linspace(self.diffusion_steps - 1, 0, sampling_timesteps, dtype=torch.long).to(self.device)

        # Reverse diffusion loop
        for i, t_idx in enumerate(step_indices):
            t_tensor = torch.full((num_samples,), t_idx, device=self.device, dtype=torch.long)

            # Predict noise with CFG (Equation from paper's background section)
            # ε_pred = ω * ε_θ(x_n, n, y) + (1 - ω) * ε_θ(x_n, n, Ø)
            
            # Conditional prediction (y is condition_scores)
            epsilon_pred_cond = self.forward(x_t, t_tensor, condition_scores)
            
            # Unconditional prediction (Ø is None)
            epsilon_pred_uncond = self.forward(x_t, t_tensor, None) # None implies zero tensor internally

            # Combine using guidance scale
            epsilon_pred = guidance_scale * epsilon_pred_cond + (1.0 - guidance_scale) * epsilon_pred_uncond

            # Calculate terms for one reverse step of DDPM
            alpha_t = self.alphas[t_idx]
            sqrt_recip_alpha_t = self.sqrt_recip_alphas[t_idx]
            sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t_idx]
            beta_t = self.betas[t_idx]

            # Estimate x_0 from x_t and predicted noise
            # x_0_pred = (x_t - sqrt_one_minus_alpha_bar_t * epsilon_pred) / sqrt_alpha_bar_t
            # This estimate can be used, but the direct DDPM reverse step is more common.

            # Calculate mean of posterior q(x_{t-1} | x_t, x_0)
            mean_t = sqrt_recip_alpha_t * (x_t - (beta_t / sqrt_one_minus_alpha_bar_t) * epsilon_pred)

            if t_idx > 0:
                # Sample z ~ N(0, I) for re-introducing noise
                z = torch.randn_like(x_t, device=self.device)
                # Variance of posterior
                variance = beta_t
                x_t = mean_t + torch.sqrt(variance) * z
            else:
                x_t = mean_t # No noise added at the last step (t=0)

        # x_t now represents the generated x_0
        generated_x_0 = x_t

        # Deconstruct the generated transition into its components
        current_idx = 0
        gen_state = generated_x_0[:, current_idx:current_idx + self.state_feature_dim]
        current_idx += self.state_feature_dim
        gen_action = generated_x_0[:, current_idx:current_idx + self.action_dim]
        current_idx += self.action_dim
        gen_reward = generated_x_0[:, current_idx:current_idx + self.reward_dim]
        current_idx += self.reward_dim
        gen_next_state = generated_x_0[:, current_idx:current_idx + self.state_feature_dim]

        return {
            'state': gen_state,
            'action': gen_action,
            'reward': gen_reward,
            'next_state': gen_next_state,
            'done': torch.zeros(num_samples, 1, dtype=torch.bool, device=self.device) # Synthetic data are usually not 'done'
        }

