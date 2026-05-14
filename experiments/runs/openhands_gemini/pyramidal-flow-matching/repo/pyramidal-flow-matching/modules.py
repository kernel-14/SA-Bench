
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from layers import Conv3DBlock, Upsample3D, Downsample3D, TransformerBlock, RotaryPositionEmbedding

class VAEEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=2, hidden_dims=[64, 128, 256, 512],
                 downsample_factors=[(2,2,2), (2,2,2), (2,2,2)], causal_conv=True):
        super().__init__()
        self.initial_conv = Conv3DBlock(in_channels, hidden_dims[0], kernel_size=(3,3,3), stride=1, padding=(1,1,1), causal=causal_conv)

        self.down_blocks = nn.ModuleList()
        current_channels = hidden_dims[0]
        for i in range(len(hidden_dims) - 1):
            block = nn.Sequential(
                Conv3DBlock(current_channels, hidden_dims[i+1], kernel_size=(3,3,3), stride=1, padding=(1,1,1), causal=causal_conv),
                Downsample3D(downsample_factors[i], mode='nearest')
            )
            self.down_blocks.append(block)
            current_channels = hidden_dims[i+1]

        self.final_conv = nn.Conv3d(current_channels, out_channels * 2, kernel_size=1) # For mean and logvar

    def forward(self, x):
        x = self.initial_conv(x)
        for block in self.down_blocks:
            x = block(x)
        mu_logvar = self.final_conv(x)
        mu, logvar = mu_logvar.chunk(2, dim=1)
        return mu, logvar

class VAEDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=2, hidden_dims=[512, 256, 128, 64],
                 upsample_factors=[(2,2,2), (2,2,2), (2,2,2)], causal_conv=True):
        super().__init__()
        self.initial_conv = nn.Conv3d(in_channels, hidden_dims[0], kernel_size=1)

        self.up_blocks = nn.ModuleList()
        current_channels = hidden_dims[0]
        for i in range(len(hidden_dims) - 1):
            block = nn.Sequential(
                Upsample3D(upsample_factors[i], mode='nearest'),
                Conv3DBlock(current_channels, hidden_dims[i+1], kernel_size=(3,3,3), stride=1, padding=(1,1,1), causal=causal_conv)
            )
            self.up_blocks.append(block)
            current_channels = hidden_dims[i+1]

        self.final_conv = Conv3DBlock(current_channels, out_channels, kernel_size=(3,3,3), stride=1, padding=(1,1,1), causal=causal_conv, activation=nn.Identity)

    def forward(self, x):
        x = self.initial_conv(x)
        for block in self.up_blocks:
            x = block(x)
        x = self.final_conv(x)
        return x

class VAE(nn.Module):
    def __init__(self, in_channels, latent_dim, hidden_dims=[64, 128, 256, 512],
                 downsample_factors=[(2,2,2), (2,2,2), (2,2,2)], causal_conv=True):
        super().__init__()
        self.encoder = VAEEncoder(in_channels, latent_dim, hidden_dims=hidden_dims,
                                  downsample_factors=downsample_factors, causal_conv=causal_conv)
        self.decoder = VAEDecoder(latent_dim, in_channels, hidden_dims=list(reversed(hidden_dims)),
                                  upsample_factors=list(reversed(downsample_factors)), causal_conv=causal_conv)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

class DiTBlock(nn.Module):
    """
    A Diffusion Transformer (DiT) block, composed of adaptive layer norm, attention, and MLP.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, ada_norm_dim=1024):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = TransformerBlock(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, act_layer, norm_layer)
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ada_norm_1 = AdaLayerNorm(dim, ada_norm_dim)
        self.ada_norm_2 = AdaLayerNorm(dim, ada_norm_dim)

    def forward(self, x, t_embed, mask=None, rotary_pos_emb=None):
        x = x + self.attn(self.ada_norm_1(self.norm1(x), t_embed), mask=mask, rotary_pos_emb=rotary_pos_emb)
        x = x + self.mlp(self.ada_norm_2(self.norm2(x), t_embed))
        return x

class AdaLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization.
    Used in DiT to condition on timestep embeddings.
    """
    def __init__(self, dim, ada_norm_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.linear = nn.Linear(ada_norm_dim, 2 * dim)

    def forward(self, x, t_embed):
        gamma, beta = self.linear(t_embed).chunk(2, dim=-1)
        x = self.norm(x)
        x = x * (1 + gamma) + beta
        return x

class TimestepEmbedder(nn.Module):
    """
    Embeds timestep t into a continuous representation.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)

    def timestep_embedding(self, timesteps, dim, max_period=10000):
        # From OpenAI's original DiT implementation
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

class LabelEmbedder(nn.Module):
    """
    Embeds class labels (or text embeddings) into a continuous representation.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes, hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_size, hidden_size, bias=True)
        )

    def forward(self, labels):
        emb = self.embedding_table(labels)
        return self.mlp(emb)

import math
