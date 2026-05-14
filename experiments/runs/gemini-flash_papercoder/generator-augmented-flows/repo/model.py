import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Dict, Any

# --- Helper Modules for SongUNet ---

class SinusoidalPositionalEmbedding(nn.Module):
    """
    Sinusoidal Positional Embedding for time/sigma values.
    Transforms a scalar into a high-dimensional vector using sine and cosine functions
    at various frequencies.
    """
    def __init__(self, dim: int):
        """
        Initializes the SinusoidalPositionalEmbedding.

        Args:
            dim (int): The output dimension of the embedding.
        """
        super().__init__()
        self.dim = dim
        # Create a range of frequencies, spaced exponentially.
        # These are precomputed, will be broadcast across batch dim later.
        self.inv_freq = 1.0 / (10000**(torch.arange(0, dim, 2).float() / dim))

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Generates sinusoidal embeddings for a batch of timesteps.

        Args:
            timesteps (torch.Tensor): A tensor of scalar timesteps (or sigma values),
                                      shape (batch_size,).

        Returns:
            torch.Tensor: The sinusoidal embedding for each timestep,
                          shape (batch_size, dim).
        """
        # Ensure timesteps is at least 1D
        if timesteps.ndim == 0:
            timesteps = timesteps.unsqueeze(0)
        
        # (batch_size, 1) * (1, dim/2) -> (batch_size, dim/2)
        outer_product = timesteps.unsqueeze(-1) * self.inv_freq.to(timesteps.device)
        emb = torch.cat([outer_product.sin(), outer_product.cos()], dim=-1)
        return emb

class GroupNorm(nn.Module):
    """
    Group Normalization layer.
    """
    def __init__(self, num_channels: int, num_groups: int = 32, eps: float = 1e-5):
        """
        Initializes the GroupNorm layer.

        Args:
            num_channels (int): Number of channels in the input.
            num_groups (int): Number of groups to separate the channels into.
            eps (float): A value added to the denominator for numerical stability.
        """
        super().__init__()
        # Standard implementation of GroupNorm in PyTorch
        self.gn = nn.GroupNorm(num_groups, num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Group Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, H, W).

        Returns:
            torch.Tensor: Normalized tensor.
        """
        return self.gn(x)

class Swish(nn.Module):
    """
    Swish activation function (SiLU).
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

class ResidualBlock(nn.Module):
    """
    A standard Residual Block for U-Net architectures, incorporating time embeddings.
    """
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, dropout: float):
        """
        Initializes the ResidualBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            time_emb_dim (int): Dimension of the time embedding.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.norm1 = GroupNorm(in_channels)
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Time embedding projection: MLP for time conditioning
        self.time_proj = nn.Sequential(
            Swish(),
            nn.Linear(time_emb_dim, out_channels)
        )

        self.norm2 = GroupNorm(out_channels)
        self.act2 = Swish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.dropout = nn.Dropout(dropout)

        # Skip connection: 1x1 convolution if channels change, else identity
        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_connection = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the ResidualBlock.

        Args:
            x (torch.Tensor): Input feature map, shape (B, C_in, H, W).
            t_emb (torch.Tensor): Time embedding, shape (B, time_emb_dim).

        Returns:
            torch.Tensor: Output feature map, shape (B, C_out, H, W).
        """
        # First convolution block
        h = self.conv1(self.act1(self.norm1(x)))
        # Add time embedding (broadcasted)
        h = h + self.time_proj(t_emb)[:, :, None, None] 
        # Second convolution block
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        # Residual connection
        return h + self.skip_connection(x)

class AttentionBlock(nn.Module):
    """
    A self-attention block for feature maps, using scaled dot-product attention.
    """
    def __init__(self, channels: int):
        """
        Initializes the AttentionBlock.

        Args:
            channels (int): Number of input/output channels.
        """
        super().__init__()
        self.norm = GroupNorm(channels)
        # Query, Key, Value projections
        self.query = nn.Conv2d(channels, channels, kernel_size=1)
        self.key = nn.Conv2d(channels, channels, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the AttentionBlock.

        Args:
            x (torch.Tensor): Input feature map, shape (B, C, H, W).

        Returns:
            torch.Tensor: Output feature map, shape (B, C, H, W).
        """
        shortcut = x
        h = self.norm(x)
        
        q = self.query(h)
        k = self.key(h)
        v = self.value(h)

        B, C, H, W = q.shape
        # Reshape from (B, C, H, W) to (B, H*W, C) for attention computation
        q = q.view(B, C, H * W).permute(0, 2, 1) 
        k = k.view(B, C, H * W).permute(0, 2, 1) 
        v = v.view(B, C, H * W).permute(0, 2, 1) 

        # Scaled Dot-Product Attention: (Q @ K_T) @ V
        # (B, H*W, C) @ (B, C, H*W) -> (B, H*W, H*W)
        attention_scores = torch.baddbmm(q, k.transpose(-2, -1), alpha=C**-0.5)
        attention_scores = attention_scores.softmax(dim=-1) 

        # (B, H*W, H*W) @ (B, H*W, C) -> (B, H*W, C)
        h = torch.bmm(attention_scores, v)

        h = h.permute(0, 2, 1).view(B, C, H, W) # Reshape back to (B, C, H, W)
        h = self.proj_out(h)
        return h + shortcut

class Downsample(nn.Module):
    """
    Downsampling layer (2x spatial resolution reduction) using Conv2d with stride 2.
    """
    def __init__(self, channels: int):
        """
        Initializes the Downsample layer.

        Args:
            channels (int): Number of input/output channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the Downsample layer.
        """
        return self.conv(x)

class Upsample(nn.Module):
    """
    Upsampling layer (2x spatial resolution increase) using nearest interpolation
    followed by Conv2d.
    """
    def __init__(self, channels: int):
        """
        Initializes the Upsample layer.

        Args:
            channels (int): Number of input/output channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the Upsample layer.
        """
        # Nearest-neighbor upsampling followed by convolution
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)

# --- Main SongUNet Architecture ---

class SongUNet(nn.Module):
    """
    NCSN++ (SongUNet) architecture, serving as the backbone F_theta network
    for the ConsistencyModel, aligning with Karras et al. (2022) implementation style.
    """
    def __init__(
        self,
        img_resolution: int,
        in_channels: int,
        out_channels: int,
        model_channels: int,
        num_blocks: List[int], # e.g., [3, 3, 3] for 3 stages
        channel_mult: List[int], # e.g., [1, 2, 2] for 3 stages
        attn_resolutions: List[int],
        dropout_rate: float,
        embedding_type: str = 'positional'
    ):
        """
        Initializes the SongUNet model.

        Args:
            img_resolution (int): Spatial resolution of input images.
            in_channels (int): Number of input image channels.
            out_channels (int): Number of output channels (typically same as in_channels).
            model_channels (int): Base number of channels in the U-Net.
            num_blocks (List[int]): Number of residual blocks at each resolution level.
                                    Its length should match that of channel_mult.
            channel_mult (List[int]): Multiplicative factors for channels at each stage.
                                      Determines the number of down/up-sampling stages.
            attn_resolutions (List[int]): Spatial resolutions where attention blocks are applied.
            dropout_rate (float): Dropout probability.
            embedding_type (str): Type of time embedding ('positional' or 'fourier').
                                  Only 'positional' is implemented here.
        """
        super().__init__()

        # Store initialization parameters for potential use (e.g., by EMA for model copying)
        self._unet_init_params = { 
            "img_resolution": img_resolution,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "model_channels": model_channels,
            "num_blocks": num_blocks,
            "channel_mult": channel_mult,
            "attn_resolutions": attn_resolutions,
            "dropout_rate": dropout_rate,
            "embedding_type": embedding_type
        }

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embedding_type = embedding_type
        # Time embedding dimension; commonly `model_channels * 4` in EDM/NCSN++ models
        time_emb_dim = model_channels * 4 

        # Time embedding MLP processes sinusoidal features
        self.time_embed_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(time_emb_dim // 2), # Outputs dim/2 sin + dim/2 cos = dim features
            nn.Linear(time_emb_dim, time_emb_dim), # Linear layer to map to target dim
            Swish(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Input convolution block
        self.input_blocks = nn.ModuleList([
            nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        ])
        
        # Track current feature map channels and resolution
        curr_channels = model_channels
        curr_resolution = img_resolution
        
        # List to store features for skip connections
        self.skip_connection_channels = [model_channels] 

        # Downsampling path construction
        for i, mult in enumerate(channel_mult):
            out_channels_mult = model_channels * mult
            for _ in range(num_blocks[i]):
                self.input_blocks.append(ResidualBlock(curr_channels, out_channels_mult, time_emb_dim, dropout_rate))
                curr_channels = out_channels_mult
                if curr_resolution in attn_resolutions:
                    self.input_blocks.append(AttentionBlock(curr_channels))
                self.skip_connection_channels.append(curr_channels) # Store output of residual/attention block
            
            # Add downsample layer if not the last stage
            if i != len(channel_mult) - 1:
                self.input_blocks.append(Downsample(curr_channels))
                curr_resolution //= 2
                self.skip_connection_channels.append(curr_channels) # Store output of downsample layer

        # Bottleneck (middle) blocks
        self.middle_blocks = nn.ModuleList([
            ResidualBlock(curr_channels, curr_channels, time_emb_dim, dropout_rate),
            AttentionBlock(curr_channels),
            ResidualBlock(curr_channels, curr_channels, time_emb_dim, dropout_rate)
        ])

        # Upsampling path construction
        self.output_blocks = nn.ModuleList([])
        for i, mult in reversed(list(enumerate(channel_mult))):
            out_channels_mult = model_channels * mult
            for _ in range(num_blocks[i] + 1): # +1 to account for skip connection concatenation
                # Get channels from the corresponding skip connection
                skip_ch = self.skip_connection_channels.pop() 
                input_channels_res_block = curr_channels + skip_ch # Concatenated channels
                
                self.output_blocks.append(ResidualBlock(input_channels_res_block, out_channels_mult, time_emb_dim, dropout_rate))
                curr_channels = out_channels_mult
                if curr_resolution in attn_resolutions:
                    self.output_blocks.append(AttentionBlock(curr_channels))
            
            # Add upsample layer if not the first stage (highest resolution)
            if i != 0:
                self.output_blocks.append(Upsample(curr_channels))
                curr_resolution *= 2
        
        # Final output layer
        self.output_layer = nn.Sequential(
            GroupNorm(curr_channels),
            Swish(),
            nn.Conv2d(curr_channels, out_channels, kernel_size=3, padding=1)
        )

    def _get_time_embedding(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Generates time embedding for the given sigma values.
        Uses a sinusoidal embedding followed by an MLP.

        Args:
            sigma (torch.Tensor): A tensor of sigma values, shape (batch_size,).

        Returns:
            torch.Tensor: The processed time embedding, shape (batch_size, time_emb_dim).
        """
        # The `time_embed_mlp` expects a 1D tensor of scalar values as input to
        # `SinusoidalPositionalEmbedding`. This is handled inside `SinusoidalPositionalEmbedding`.
        return self.time_embed_mlp(sigma)

    def forward(self, x: torch.Tensor, sigma_t: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the SongUNet.

        Args:
            x (torch.Tensor): Input noisy image, shape (batch_size, in_channels, H, W).
            sigma_t (torch.Tensor): Noise level for x, shape (batch_size,).

        Returns:
            torch.Tensor: Output of the U-Net, F_theta(x, sigma_t),
                          shape (batch_size, out_channels, H, W).
        """
        t_emb = self._get_time_embedding(sigma_t)

        hs = [] # To store intermediate outputs for skip connections
        
        # Initial convolution
        h = self.input_blocks[0](x) 
        hs.append(h)

        # Downsampling path
        block_idx = 1
        for i, mult in enumerate(self.channel_mult):
            for _ in range(self.num_blocks[i]):
                h = self.input_blocks[block_idx](h, t_emb)
                block_idx += 1
                if h.shape[-1] in self.attn_resolutions:
                    h = self.input_blocks[block_idx](h)
                    block_idx += 1
                hs.append(h)
            if i != len(self.channel_mult) - 1:
                h = self.input_blocks[block_idx](h) # Downsample layer
                block_idx += 1
                hs.append(h) # Store for next downsample stage input, not directly for skip
                
        # The last element in hs after downsampling path is the input to the bottleneck,
        # but it shouldn't be used as a skip connection in the upsampling path.
        # This implementation stores all intermediate outputs. A more explicit list of
        # skip connection points might be needed if `hs` gets too large or is misused.
        # For a standard U-Net, hs.pop() will retrieve in reverse order of being added.
        # The construction means: initial_conv_out, then (res_block_out, attn_block_out)*, then downsample_out.
        # The skips for upsampling should be the *output of the blocks at each level before downsampling*,
        # and the bottleneck output.

        # Bottleneck
        for block in self.middle_blocks:
            if isinstance(block, ResidualBlock):
                h = block(h, t_emb)
            else: # AttentionBlock
                h = block(h)

        # Upsampling path
        upsample_block_idx = 0
        for i, mult in reversed(list(enumerate(self.channel_mult))):
            for _ in range(self.num_blocks[i] + 1):
                # Pop the corresponding skip connection feature map
                h_skip = hs.pop() 
                h = torch.cat([h, h_skip], dim=1) # Concatenate skip connection
                
                # Apply residual/attention block
                block = self.output_blocks[upsample_block_idx]
                if isinstance(block, ResidualBlock):
                    h = block(h, t_emb)
                else: # AttentionBlock
                    h = block(h)
                upsample_block_idx += 1
            
            if i != 0: # Apply upsample layer if not the first upsampling stage (highest resolution)
                h = self.output_blocks[upsample_block_idx](h)
                upsample_block_idx += 1
        
        return self.output_layer(h)


# --- Consistency Model Wrapper ---

class ConsistencyModel(nn.Module):
    """
    Encapsulates the SongUNet backbone and applies the consistency model
    parametrization as defined in the paper.
    """
    def __init__(
        self,
        unet_params: Dict[str, Any],
        sigma_0: float,
        sigma_d_sq: float,
        data_dim: int,
        device: torch.device
    ):
        """
        Initializes the ConsistencyModel.

        Args:
            unet_params (Dict[str, Any]): Dictionary of parameters to initialize SongUNet.
            sigma_0 (float): Smallest noise level.
            sigma_d_sq (float): Empirical variance of the data distribution.
            data_dim (int): Dimensionality of the data (e.g., C*H*W).
            device (torch.device): The device to run the model on.
        """
        super().__init__()
        # Store unet_params to allow EMA to correctly create a copy
        self._unet_init_params = unet_params 
        self.unet = SongUNet(**unet_params).to(device)
        self.sigma_0 = sigma_0
        self.sigma_d_sq = sigma_d_sq
        self.data_dim = float(data_dim) # Ensure float for calculations in c_out denominator
        self.device = device
        
    def get_output_coefficients(self, sigma_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the c_skip and c_out coefficients for the consistency model parametrization.

        Args:
            sigma_t (torch.Tensor): Current noise levels, shape (batch_size,).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple (c_skip_val, c_out_val),
                                               each of shape (batch_size,).
        """
        # Ensure sigma_t is on the correct device
        sigma_t = sigma_t.to(self.device)

        # c_skip(sigma) = sigma_d^2 / (sigma_d^2 + (sigma - sigma_0)^2)
        # Ensure sigma_d_sq is a tensor for element-wise operations
        sigma_d_sq_tensor = torch.tensor(self.sigma_d_sq, device=self.device)
        c_skip_val = sigma_d_sq_tensor / (sigma_d_sq_tensor + (sigma_t - self.sigma_0).pow(2))

        # c_out(sigma) = sigma / (sqrt(d) * (sigma - sigma_0) * sqrt(sigma_d^2 + sigma^2))
        # Add a small epsilon to avoid division by zero when sigma_t is exactly sigma_0
        epsilon = 1e-5 
        denominator_term_1 = torch.sqrt(torch.tensor(self.data_dim, device=self.device))
        denominator_term_2 = (sigma_t - self.sigma_0)
        denominator_term_3 = torch.sqrt(sigma_d_sq_tensor + sigma_t.pow(2))
        
        denominator = denominator_term_1 * denominator_term_2 * denominator_term_3
        
        c_out_val = sigma_t / (denominator + epsilon)

        return c_skip_val, c_out_val

    def forward(self, x_t: torch.Tensor, sigma_t: torch.Tensor) -> torch.Tensor:
        """
        Computes the output of the consistency model f_theta(x_t, sigma_t).

        Args:
            x_t (torch.Tensor): Noisy input data, shape (batch_size, C, H, W).
            sigma_t (torch.Tensor): Current noise levels, shape (batch_size,).

        Returns:
            torch.Tensor: Predicted clean data point (f_theta output),
                          shape (batch_size, C, H, W).
        """
        # Ensure inputs are on the correct device
        x_t = x_t.to(self.device)
        sigma_t = sigma_t.to(self.device)

        # Get F_theta output from the U-Net backbone
        F_theta_output = self.unet(x_t, sigma_t)

        # Get the coefficients
        c_skip, c_out = self.get_output_coefficients(sigma_t)

        # Reshape coefficients for broadcasting (batch_size, 1, 1, 1) to match (B, C, H, W)
        c_skip = c_skip.view(-1, 1, 1, 1)
        c_out = c_out.view(-1, 1, 1, 1)

        # Apply the consistency model parametrization formula
        f_theta_output = c_skip * x_t + c_out * F_theta_output
        return f_theta_output

# --- EMA Utility ---

class EMA:
    """
    Exponential Moving Average (EMA) utility to maintain a copy of model weights
    for stable inference and target network creation.
    """
    def __init__(self, model: ConsistencyModel, decay: float):
        """
        Initializes the EMA.

        Args:
            model (ConsistencyModel): The model whose EMA weights will be tracked.
            decay (float): The EMA decay rate.
        """
        self.decay = decay
        # Create a new instance of the same model type and load state dict from the original model.
        # This ensures the EMA model is a distinct entity with its own weights.
        self.ema_model = ConsistencyModel(
            unet_params=model._unet_init_params, # Use stored params from ConsistencyModel's init
            sigma_0=model.sigma_0,
            sigma_d_sq=model.sigma_d_sq,
            data_dim=model.data_dim,
            device=model.device
        )
        self.ema_model.load_state_dict(model.state_dict())
        for p in self.ema_model.parameters():
            p.requires_grad_(False) # EMA model parameters should not be updated by gradients
        self.ema_model.eval() # Set to eval mode; important to disable dropout and batch norm updates

    def update(self, new_model: nn.Module):
        """
        Updates the EMA weights based on the current weights of the new_model.

        Args:
            new_model (nn.Module): The training model whose weights will be used for update.
        """
        with torch.no_grad(): # Ensure no gradient tracking during EMA update process
            for ema_param, new_param in zip(self.ema_model.parameters(), new_model.parameters()):
                # In-place update: ema_param = decay * ema_param + (1 - decay) * new_param
                ema_param.data.mul_(self.decay).add_(new_param.data, alpha=1 - self.decay)

    def get_model(self) -> ConsistencyModel:
        """
        Returns the EMA model.

        Returns:
            ConsistencyModel: The EMA model with its current weights.
        """
        return self.ema_model

