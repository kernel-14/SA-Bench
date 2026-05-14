
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from typing import Optional, Tuple

from layers import RMSNorm, SwiGLU, SelfAttention, CrossAttention
from config import P2VAEConfig, FMTConfig

class TimestepEmbedder(nn.Module):
    """
    Embeds integer timestamps into a continuous vector representation.
    Similar to positional embeddings, but for time.
    """
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        Args:
            t (torch.Tensor): Timesteps to embed, shape `(batch_size,)`.
            dim (int): Embedding dimension.
            max_period (float): Controls the maximum period of the sinusoidal waves.
        Returns:
            torch.Tensor: Sinusoidal embeddings, shape `(batch_size, dim)`.
        """
        # (d_model / 2)
        half = dim // 2
        # (d_model / 2)
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half)
        # (batch_size, d_model / 2)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        # Pad to `dim` if necessary
        if dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)

class ResBlock(nn.Module):
    """
    A standard residual block with convolutional layers and group normalization.
    Used in VAE encoder/decoder.
    """
    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: Optional[int] = None):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.silu = nn.SiLU()

        self.time_proj = None
        if time_embed_dim is not None:
            self.time_proj = nn.Linear(time_embed_dim, 2 * out_channels)

        self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.silu(self.norm1(x))
        h = self.conv1(h)

        if self.time_proj is not None and time_embed is not None:
            time_scale, time_shift = self.time_proj(self.silu(time_embed)).chunk(2, dim=1)
            h = h * (1 + time_scale[:, :, None, None]) + time_shift[:, :, None, None]

        h = self.silu(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip_connection(x)

class Downsample(nn.Module):
    """Downsampling block for VAE."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Upsample(nn.Module):
    """Upsampling block for VAE."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)

class P2VAEEncoder(nn.Module):
    """
    Encoder part of the P2VAE, based on a standard VAE encoder architecture.
    Compresses spatial features.
    """
    def __init__(self,
                 in_channels: int = P2VAEConfig.IN_CHANNELS,
                 latent_channels: int = P2VAEConfig.LATENT_CHANNELS,
                 base_channels: int = P2VAEConfig.BASE_DIM_16M, # Can be 64 or 128
                 num_res_blocks: int = 2,
                 channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4),
                 attn_resolutions: Tuple[int, ...] = (8, 16) # Resolutions where self-attention is applied
                ):
        super().__init__()
        self.num_resolutions = len(channel_multipliers)
        curr_channels = base_channels

        self.conv_in = nn.Conv2d(in_channels, curr_channels, kernel_size=3, padding=1)

        blocks = []
        for i, mult in enumerate(channel_multipliers):
            out_channels = base_channels * mult
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(curr_channels, out_channels))
                curr_channels = out_channels
            if i < self.num_resolutions - 1: # No downsample after the last block
                blocks.append(Downsample(curr_channels))
        self.down_blocks = nn.ModuleList(blocks)

        self.mid_block = ResBlock(curr_channels, curr_channels) # Mid block for consistency

        self.norm_out = nn.GroupNorm(32, curr_channels)
        self.conv_out = nn.Conv2d(curr_channels, latent_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)

        for block in self.down_blocks:
            x = block(x)

        x = self.mid_block(x)
        
        x = self.norm_out(x)
        x = self.conv_out(x)
        return x

class P2VAEDecoder(nn.Module):
    """
    Decoder part of the P2VAE. Reconstructs spatial features from latent representation.
    """
    def __init__(self,
                 out_channels: int = P2VAEConfig.OUT_CHANNELS,
                 latent_channels: int = P2VAEConfig.LATENT_CHANNELS,
                 base_channels: int = P2VAEConfig.BASE_DIM_16M,
                 num_res_blocks: int = 2,
                 channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4)
                ):
        super().__init__()
        self.num_resolutions = len(channel_multipliers)
        
        # Reverse the channel multipliers for the decoder
        channel_multipliers = tuple(reversed(channel_multipliers))
        
        curr_channels = base_channels * channel_multipliers[0]

        self.conv_in = nn.Conv2d(latent_channels, curr_channels, kernel_size=3, padding=1)
        self.mid_block = ResBlock(curr_channels, curr_channels)

        blocks = []
        for i, mult in enumerate(channel_multipliers):
            out_channels = base_channels * mult
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(curr_channels, out_channels))
                curr_channels = out_channels
            if i < self.num_resolutions - 1: # No upsample after the last block
                blocks.append(Upsample(curr_channels))
        self.up_blocks = nn.ModuleList(blocks)

        self.norm_out = nn.GroupNorm(32, curr_channels)
        self.conv_out = nn.Conv2d(curr_channels, out_channels, kernel_size=3, padding=1) # Note: out_channels here is the last 'curr_channels' from loop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        x = self.mid_block(x)

        for block in self.up_blocks:
            x = block(x)

        x = self.norm_out(x)
        x = self.conv_out(x)
        return x


class TransformerBlock(nn.Module):
    """
    A Transformer block using AdaLN-Zero for conditioning, RMSNorm, SelfAttention, and SwiGLU.
    """
    def __init__(self, embed_dim: int, num_heads: int, head_dim: int, cond_dim: int, dropout: float = 0.0):
        super().__init__()
        self.adaln_zero = AdaLNZero(embed_dim, cond_dim)
        self.self_attn = SelfAttention(embed_dim, num_heads, head_dim, dropout=dropout)
        self.mlp = SwiGLU(embed_dim, embed_dim) # SwiGLU uses embed_dim as its internal dimension, outputting embed_dim

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # Self-attention block
        norm_x_attn, scale_attn, shift_attn, \
        scale_mlp, shift_mlp, \
        alpha_attn, alpha_mlp = self.adaln_zero(x, cond)
        
        # Apply scale and shift before self-attention
        norm_x_attn = norm_x_attn * (1 + scale_attn) + shift_attn
        attn_output = self.self_attn(norm_x_attn)
        x = x + alpha_attn * attn_output # Residual connection with alpha scaling

        # MLP block
        norm_x_mlp = self.adaln_zero.norm(x) # Reuse RMSNorm from AdaLNZero
        norm_x_mlp = norm_x_mlp * (1 + scale_mlp) + shift_mlp
        mlp_output = self.mlp(norm_x_mlp)
        x = x + alpha_mlp * mlp_output # Residual connection with alpha scaling
        return x

class DiffusionForcingGRU(nn.Module):
    """
    GRU-based RNN for the diffusion forcing scheme, evolving the latent state `h`.
    It takes the previous latent state `h_s-1`, a compressed representation of `x_s,t_s^k_s`,
    and `t_s` as input to update `h_s`.
    The current state `x_t_k` is first compressed onto a single token by cross attention.
    """
    def __init__(self,
                 input_dim: int, # Dimension of the compressed x_t_k
                 hidden_dim: int, # Same as embedding dimension in SiT (FMT)
                 time_embed_dim: int = 256,
                 num_heads: int = 8,
                 head_dim: int = FMTConfig.HEAD_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Project t_s into a suitable dimension
        self.time_embedder = TimestepEmbedder(hidden_dim, frequency_embedding_size=time_embed_dim)

        # Cross-attention to compress x_t_k into a single token vector
        # Query: A learnable token that represents the summary of x_t_k
        # Context: x_t_k tokens themselves
        self.cross_attn = CrossAttention(query_dim=hidden_dim, context_dim=input_dim, num_heads=num_heads, head_dim=head_dim)
        self.learnable_query_token = nn.Parameter(torch.randn(1, 1, hidden_dim)) # A single token to query the state

        # GRU input: compressed_x_t_k + time_embed + h_s-1
        # The paper states "RNN parametrized by phi like original DF paper to evolve latent state h_s"
        # and "shared the same internal dimension as the embedding dimension in SiT".
        # So, input to GRU should be the concatenated features.
        self.gru = nn.GRU(input_size=hidden_dim * 2, # compressed_x_t_k + time_embed
                          hidden_size=hidden_dim,
                          batch_first=True)

    def forward(self, h_prev: torch.Tensor, x_t_k: torch.Tensor, t_s: torch.Tensor) -> torch.Tensor:
        # h_prev: [batch_size, hidden_dim]
        # x_t_k: [batch_size, num_tokens, input_dim] (e.g., latent grid tokens from P2VAE)
        # t_s: [batch_size,] (scalar timestep for the physical step s)

        batch_size = x_t_k.shape[0]

        # 1. Embed t_s
        t_embed = self.time_embedder(t_s) # [batch_size, hidden_dim]

        # 2. Compress x_t_k using cross-attention
        # Repeat the learnable query token for the batch
        query_token = self.learnable_query_token.repeat(batch_size, 1, 1) # [batch_size, 1, hidden_dim]
        compressed_x_t_k = self.cross_attn(query_token, x_t_k).squeeze(1) # [batch_size, hidden_dim]

        # 3. Concatenate inputs for GRU
        gru_input = torch.cat([compressed_x_t_k, t_embed], dim=-1).unsqueeze(1) # [batch_size, 1, hidden_dim * 2]

        # 4. Evolve latent state h
        # GRU expects h_prev as [num_layers, batch_size, hidden_size]
        h_prev_unsqueeze = h_prev.unsqueeze(0) # [1, batch_size, hidden_dim]
        output, h_s = self.gru(gru_input, h_prev_unsqueeze) # h_s is [1, batch_size, hidden_dim]
        
        return h_s.squeeze(0) # [batch_size, hidden_dim]

