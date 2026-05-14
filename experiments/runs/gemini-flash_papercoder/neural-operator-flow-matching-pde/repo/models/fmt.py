import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional

# Assuming Config is in the project root or accessible via sys.path
from config import Config

# Import components from models/components.py
from models.components import (
    RMSNorm,
    SwiGLU,
    Attention,  # Encapsulates FlashAttention v2 logic
    AdaLNZero,
    TransformerBlock,
    CrossAttention, # Custom cross-attention module used for condition token
    SinusoidalPositionalEmbedding # Re-using for scalar time embedding if desired, though linear layers also work.
                                # The design asks for nn.Sequential(nn.Linear, SwiGLU, nn.Linear) for time_embedding.
                                # Let's follow the design.
)

# --- Minimal PatchEmbed for FMTModel input (ideally in models/components.py) ---
# This class transforms image-like features (B, C, H, W) into sequences of tokens (B, N_patches, D_embed).
# For the latent temporal pyramid, patch_size=1 means each "pixel" of the latent grid becomes a token.
class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    Converts 2D feature maps (like latent codes) into a sequence of flat tokens.
    """
    def __init__(self, in_channels: int, patch_size: int, embedding_dim: int):
        super().__init__()
        self.patch_size = patch_size
        # Convolution to project channels to embedding_dim
        # If patch_size=1, this is a 1x1 convolution
        self.proj = nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
        Returns:
            torch.Tensor: Output tokens of shape (B, N_patches, embedding_dim).
        """
        x = self.proj(x)  # (B, embedding_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)  # (B, embedding_dim, N_patches)
        x = x.transpose(1, 2) # (B, N_patches, embedding_dim)
        return x
# --- End PatchEmbed ---


class FMTModel(nn.Module):
    """
    Implements the Flow Marching Transformer (FMT) model.
    This model predicts the velocity field for interpolated latent states,
    conditioned on a temporal pyramid of past latent states and a history hidden state.
    """
    def __init__(self, config: Config):
        """
        Initializes the FMTModel.

        Args:
            config (Config): Configuration object containing model hyperparameters.
        """
        super().__init__()
        self.config = config

        # Retrieve global data type for model parameters and computations
        self.dtype: torch.dtype = getattr(torch, self.config.get('global.dtype', 'float16'))

        # Retrieve model dimensions and parameters from config
        self.embedding_dim: int = self.config.get('fmt_model.embedding_dim', 512)
        self.num_heads: int = self.config.get('fmt_model.num_heads', 8)
        self.head_dim: int = self.config.get('fmt_model.head_dim', 64)
        self.num_layers: int = self.config.get('fmt_model.num_layers', 12)
        self.latent_channels: int = self.config.get('p2vae_model.latent_channels', 16)
        self.latent_resolution: Tuple[int, int] = tuple(self.config.get('p2vae_model.latent_resolution', [16, 16]))
        self.gru_hidden_size: int = self.config.get('fmt_model.gru.hidden_size', 512)
        self.pyramid_factors: Dict[str, int] = self.config.get('fmt_model.temporal_pyramid', {
            'x0_downsample_factor': 8,
            'x1_downsample_factor': 4,
            'x2_downsample_factor': 2,
            'x3_downsample_factor': 1
        })

        # The total conditioning dimension for TransformerBlocks (t_step_embedding + h_history)
        self.total_conditioning_dim: int = self.embedding_dim + self.gru_hidden_size

        # 1. Time Embedding for t_step
        # Maps a scalar 't' (0 to 1) to an 'embedding_dim'-sized vector.
        # This embedded vector will be used for conditioning in AdaLN-Zero.
        # Design specifies: nn.Sequential(nn.Linear, SwiGLU, nn.Linear)
        self.time_embedding = nn.Sequential(
            nn.Linear(1, self.embedding_dim),
            SwiGLU(dim=self.embedding_dim, hidden_dim=self.embedding_dim * 2), # SwiGLU expects dim and hidden_dim
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )

        # 2. Patch Embeddings for Latent Temporal Pyramid inputs
        # An ModuleDict to hold separate PatchEmbed layers for each unique downsampling factor.
        # Each PatchEmbed converts a (B, C, H_res, W_res) latent tensor into (B, N_patches, embedding_dim) tokens.
        # Assuming 1x1 patches, each spatial location in the downsampled latent grid becomes a token.
        self.latent_patch_embeddings = nn.ModuleDict()
        # Collect unique factors from the pyramid_factors config
        unique_factors = set(self.pyramid_factors.values())
        for factor_value in sorted(list(unique_factors)):
            # PatchEmbed takes (in_channels, patch_size, embedding_dim)
            self.latent_patch_embeddings[str(factor_value)] = PatchEmbed(
                in_channels=self.latent_channels,
                patch_size=1, # Treat each pixel of the latent grid as a patch
                embedding_dim=self.embedding_dim
            )

        # 3. Transformer (SiT-like architecture)
        # This stack of TransformerBlocks processes the sequence of tokens from the latent pyramid.
        self.transformer = self._build_transformer()

        # 4. GRU for Diffusion Forcing (history state update)
        # Responsible for evolving the latent history state 'h'.
        self.gru = self._build_gru()

        # 5. Cross-attention for condition token projection
        # This module compresses the tokens of the current interpolated state (y_{s,t_s}^{k_s})
        # into a single representative token for the GRU input.
        self.condition_token_proj = self._build_condition_token_proj()

        # 6. Velocity Output Head
        # Projects the Transformer's output tokens (corresponding to the target latent state)
        # back to the latent channel dimension, forming the predicted velocity field.
        # Assumes the Transformer outputs 'embedding_dim' features per token, and we need 'latent_channels'.
        self.velocity_head = nn.Linear(self.embedding_dim, self.latent_channels)
        
        # Move model to specified data type
        self.to(self.dtype)


    def _build_transformer(self) -> nn.Module:
        """
        Constructs the Transformer stack (sequence of TransformerBlocks).
        Each block includes self-attention, FFN, RMSNorm, and AdaLN-Zero conditioning.
        """
        transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim=self.embedding_dim,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                conditioning_dim=self.total_conditioning_dim, # Pass combined conditioning dimension
                attn_class=Attention, # Use the Attention class from components.py
                dtype=self.dtype
            )
            for _ in range(self.num_layers)
        ])
        return nn.Sequential(*transformer_blocks)


    def _build_gru(self) -> nn.Module:
        """
        Constructs the Gated Recurrent Unit (GRU) for updating the history state 'h'.
        The GRU's input consists of the concatenated condition token and time embedding.
        """
        # GRU input size is the concatenation of the condition token (embedding_dim)
        # and the time embedding (embedding_dim).
        gru_input_size = self.embedding_dim + self.embedding_dim
        # The GRU config in YAML implies the input_size will be dynamically determined.
        # The hidden_size is retrieved from config.
        
        return nn.GRU(
            input_size=gru_input_size,
            hidden_size=self.gru_hidden_size,
            batch_first=True # Input/output tensors are (batch, seq_len, features)
        )


    def _build_condition_token_proj(self) -> CrossAttention:
        """
        Builds the cross-attention module from `models/components.py` used to condense
        the tokens of `current_latent_y_tk` into a single `embedding_dim` token.
        """
        return CrossAttention(
            query_dim=self.embedding_dim, # The learned query token will be this size
            context_dim=self.embedding_dim, # The tokens from latent_y_tk are also this size
            output_dim=self.embedding_dim, # The compressed token will be this size
            heads=self.num_heads,
            dim_head=self.head_dim,
            dtype=self.dtype
        )


    def downsample_latent(self, latent_y: torch.Tensor, factor: int) -> torch.Tensor:
        """
        Spatially downsamples a latent tensor using average pooling.

        Args:
            latent_y (torch.Tensor): Input latent tensor of shape (B, C, H, W).
            factor (int): Downsampling factor.

        Returns:
            torch.Tensor: Downsampled latent tensor of shape (B, C, H/factor, W/factor).
        """
        if factor == 1:
            return latent_y
        # Use average pooling for robust downsampling as per common practice and implicitly for efficiency.
        # Antialiasing is often beneficial for downsampling, but F.avg_pool2d does not directly support it.
        # If higher quality downsampling is needed, `skimage.transform.resize` or `F.interpolate(mode='area')`
        # would be done earlier in `DatasetProcessor`. Here, it's for efficiency of latent grids.
        return F.avg_pool2d(latent_y, kernel_size=factor, stride=factor)


    def encode_condition_token(self, latent_y_tk: torch.Tensor) -> torch.Tensor:
        """
        Encodes the `latent_y_tk` (interpolated latent state) into a single
        `embedding_dim` token using the `CrossAttention` module.

        Args:
            latent_y_tk (torch.Tensor): The interpolated latent state `y_{s,t_s}^{k_s}`,
                                        shape (B, latent_channels, H, W).

        Returns:
            torch.Tensor: A single token representing `latent_y_tk`, shape (B, embedding_dim).
        """
        # Ensure input is of the correct dtype
        latent_y_tk = latent_y_tk.to(self.dtype)

        # 1. Patch embed `latent_y_tk` (full resolution, factor=1)
        # This converts (B, C, H, W) to (B, H*W, embedding_dim)
        current_y_tokens = self.latent_patch_embeddings[str(1)](latent_y_tk)

        # 2. Use the CrossAttention module to condense these tokens
        # The CrossAttention module internally manages its own learnable query.
        cond_token = self.condition_token_proj(current_y_tokens)
        
        return cond_token


    def update_history_h(self, h_prev: torch.Tensor, cond_token: torch.Tensor, t_step_embedding: torch.Tensor) -> torch.Tensor:
        """
        Updates the GRU hidden state (history 'h') based on the previous state,
        the encoded condition token, and the current time embedding.

        Args:
            h_prev (torch.Tensor): Previous GRU hidden state `h_{s-1}`, shape (B, gru_hidden_size).
            cond_token (torch.Tensor): Encoded single token from `y_{s,t_s}^{k_s}`, shape (B, embedding_dim).
            t_step_embedding (torch.Tensor): Embedded `t_s` for the current step, shape (B, embedding_dim).

        Returns:
            torch.Tensor: The new GRU hidden state `h_s`, shape (B, gru_hidden_size).
        """
        # Ensure all inputs are of the correct dtype
        h_prev = h_prev.to(self.dtype)
        cond_token = cond_token.to(self.dtype)
        t_step_embedding = t_step_embedding.to(self.dtype)

        # Concatenate the condition token and time embedding to form the GRU input features.
        # Unsqueeze to create a sequence of length 1: (B, 1, embedding_dim * 2).
        gru_input = torch.cat([cond_token, t_step_embedding], dim=-1).unsqueeze(1) # (B, 1, gru_input_size)

        # Unsqueeze h_prev to match GRU's expected hidden state format (num_layers * num_directions, B, hidden_size).
        # Assuming single-layer, unidirectional GRU, so num_layers * num_directions = 1.
        h_prev_unsqueeze = h_prev.unsqueeze(0) # (1, B, gru_hidden_size)

        # Pass input and previous hidden state to the GRU.
        _, h_new = self.gru(gru_input, h_prev_unsqueeze)

        # Squeeze the num_layers dimension back to (B, gru_hidden_size)
        return h_new.squeeze(0)


    def forward(self, interpolated_latent_states: List[torch.Tensor], current_t_step: torch.Tensor, h_history: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the FMT model to predict the velocity field.

        Args:
            interpolated_latent_states (List[torch.Tensor]): A list of 4 interpolated latent states
                                        `[y_{0,t_0}^{k_0}, y_{1,t_1}^{k_1}, y_{2,t_2}^{k_2}, y_{3,t_3}^{k_3}]`.
                                        Each tensor is of shape (B, latent_channels, H, W).
            current_t_step (torch.Tensor): The `t` parameter (e.g., `t_3`) corresponding to the last
                                        interpolated state, shape (B, 1).
            h_history (torch.Tensor): The GRU's hidden state `h_{s-1}` (e.g., `h_2`),
                                        conditioned on past frames, shape (B, gru_hidden_size).

        Returns:
            torch.Tensor: The predicted velocity field `g`, shape (B, latent_channels, H, W).
        """
        batch_size = interpolated_latent_states[0].shape[0]

        # Ensure inputs are on the correct device and dtype
        current_t_step = current_t_step.to(self.dtype)
        h_history = h_history.to(self.dtype)

        # 1. Embed `current_t_step`
        # The time_embedding expects a tensor of shape (B, 1) as input to the first Linear layer.
        # current_t_step is already (B, 1).
        t_step_embedding = self.time_embedding(current_t_step) # (B, embedding_dim)

        # Combine t_step_embedding and h_history for global conditioning of TransformerBlocks
        global_conditioning_vector = torch.cat([t_step_embedding, h_history], dim=-1) # (B, total_conditioning_dim)


        # 2. Process `interpolated_latent_states` for temporal pyramid and tokenize
        tokens_list = []
        pyramid_factors_keys = ['x0_downsample_factor', 'x1_downsample_factor', 'x2_downsample_factor', 'x3_downsample_factor']
        
        # Iterate through the list of 4 interpolated latent states and their corresponding factors
        for i, y_i_tk in enumerate(interpolated_latent_states):
            factor_key = pyramid_factors_keys[i]
            factor = self.pyramid_factors.get(factor_key, 1) # Default to 1 if key not found

            # Ensure latent state is of the correct dtype
            y_i_tk = y_i_tk.to(self.dtype)
            
            # Downsample the latent state
            downsampled_y = self.downsample_latent(y_i_tk, factor) # (B, C, H_factor, W_factor)
            
            # Patch embed the downsampled latent state into tokens
            patch_embedding_module = self.latent_patch_embeddings[str(factor)]
            tokens = patch_embedding_module(downsampled_y) # (B, N_patches_i, embedding_dim)
            tokens_list.append(tokens)

        # Concatenate all token sequences to form the single input sequence for the Transformer
        transformer_input_tokens = torch.cat(tokens_list, dim=1) # (B, total_N_patches, embedding_dim)

        # 3. Transformer Pass with AdaLN-Zero conditioning
        # Each TransformerBlock uses `global_conditioning_vector` for its AdaLN-Zero mechanism.
        transformer_output = transformer_input_tokens
        for block in self.transformer:
            transformer_output = block(transformer_output, conditioning_vector=global_conditioning_vector)

        # 4. Velocity Head - Predict velocity for the last state (y_{3,t_3}^{k_3} / y_{s,t_s}^{k_s})
        # The last segment of tokens in `transformer_output` corresponds to the full-resolution
        # latent state `y_{3,t_3}^{k_3}` (or `y_{s,t_s}^{k_s}`).
        num_tokens_last_state = self.latent_resolution[0] * self.latent_resolution[1] # 16 * 16 = 256
        
        # Check if the output has enough tokens for the last state
        if transformer_output.shape[1] < num_tokens_last_state:
            raise ValueError(f"Transformer output has {transformer_output.shape[1]} tokens, "
                             f"but expected at least {num_tokens_last_state} for the last state.")

        predicted_velocity_tokens = transformer_output[:, -num_tokens_last_state:, :] # (B, 256, embedding_dim)

        # Apply the velocity head to project each token's `embedding_dim` features to `latent_channels`.
        # Output: (B, 256, latent_channels)
        velocity_tokens_projected = self.velocity_head(predicted_velocity_tokens)

        # Reshape the projected tokens back into the 2D latent grid format (B, C, H, W).
        predicted_velocity = velocity_tokens_projected.transpose(1, 2).reshape(
            batch_size, self.latent_channels, self.latent_resolution[0], self.latent_resolution[1]
        )

        return predicted_velocity
