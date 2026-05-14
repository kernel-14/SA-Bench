import torch
import torch.nn as nn
import math
from utils.modules import RMSNorm, SwiGLU, FlashAttention, AdaLNZero, SiTBlock, DiffusionForcingGRU

class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear1 = nn.Linear(dim, dim * 4)
        self.act = SwiGLU()
        self.linear2 = nn.Linear(dim * 2, dim) # SwiGLU doubles the dim

    def forward(self, t):
        # Sinusoidal positional embeddings
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        emb = self.linear1(emb)
        emb = self.act(emb)
        emb = self.linear2(emb)
        return emb


class DiffusionForcingModule(nn.Module):
    def __init__(self, latent_channels, embedding_dim, hidden_dim_gru):
        super().__init__()
        self.gru = DiffusionForcingGRU(hidden_dim_gru, hidden_dim_gru) # input and hidden dim are the same
        # Linear projection to convert latent_channels to gru_input_dim for cross-attention
        # The paper mentions 'x_t_k is first compressed onto a single token by cross attention'
        # This would require an attention mechanism to produce a single token from the latent grid.
        # For simplicity and given the GRU input expects a single token, we will use a global average pooling + linear for compression.
        # A proper cross-attention would be more complex and require a query vector.
        self.compress_latent = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), # c16p16 -> c16p1
            nn.Conv2d(latent_channels, hidden_dim_gru, kernel_size=1) # Project channels to gru input dim
        )

        self.initial_h = nn.Parameter(torch.randn(1, hidden_dim_gru)) # Initial hidden state for GRU

    def forward(self, x_t_k_latent, h_prev):
        # x_t_k_latent: (batch_size, latent_channels, H, W)
        # h_prev: (batch_size, hidden_dim_gru)

        # Compress latent state to a single token
        compressed_x = self.compress_latent(x_t_k_latent).squeeze(-1).squeeze(-1) # (batch_size, hidden_dim_gru)

        # Update GRU state
        h_curr = self.gru(compressed_x, h_prev) # GRU expects (seq_len, batch, input_size) but custom GRU handles batch dim already
        return h_curr # (batch_size, hidden_dim_gru)


class FMT(nn.Module):
    def __init__(self, 
                 latent_channels=16, 
                 latent_resolution=16, 
                 embedding_dim=256, # For FMT-S
                 num_layers=12, 
                 num_heads=8, 
                 dim_head=64,
                 mlp_ratio=4,
                ):
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_resolution = latent_resolution
        self.embedding_dim = embedding_dim

        # Input projection for latent states
        # Flatten (latent_channels, H, W) to (H*W, latent_channels) and then project to embedding_dim
        self.token_embedding = nn.Linear(latent_channels, embedding_dim)

        # Positional embedding for spatial tokens
        self.spatial_pos_embedding = nn.Parameter(torch.randn(1, latent_resolution * latent_resolution, embedding_dim))

        # Timestep embedding
        self.time_embedding = TimestepEmbedding(embedding_dim)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            SiTBlock(embedding_dim, num_heads, dim_head, mlp_ratio)
            for _ in range(num_layers)
        ])

        # Final projection to output velocity field (same shape as input latent)
        self.output_projection = nn.Linear(embedding_dim, latent_channels)

    def forward(self, y_t_k, t, h_cond):
        # y_t_k: (batch_size, latent_channels, latent_resolution, latent_resolution)
        # t: (batch_size,)
        # h_cond: (batch_size, embedding_dim) - assuming h_cond is already projected to embedding_dim or compatible

        b, c, res_h, res_w = y_t_k.shape

        # Tokenize latent state (flatten spatial and project channels)
        y_tokens = y_t_k.view(b, c, -1).permute(0, 2, 1) # (B, H*W, C)
        y_tokens = self.token_embedding(y_tokens) # (B, H*W, embedding_dim)

        # Add spatial positional embedding (assuming a fixed resolution for simplicity, or re-interpolate)
        # For variable resolutions from temporal pyramids, this would need to be handled more flexibly.
        # For now, let's assume a fixed resolution input to FMT, and downsampling happens outside.
        if res_h * res_w != self.latent_resolution * self.latent_resolution:
            raise ValueError(
                f"Input latent resolution ({res_h}x{res_w}) does not match model's latent_resolution ({self.latent_resolution}x{self.latent_resolution}). "
                "Temporal pyramids require flexible spatial embeddings or separate models."
            )

        y_tokens = y_tokens + self.spatial_pos_embedding

        # Get time embedding
        t_emb = self.time_embedding(t) # (B, embedding_dim)

        # Combine time embedding and conditional history for AdaLN-Zero
        # Assuming h_cond also has  for simplicity.
        combined_cond = t_emb + h_cond # Simple addition for now, assuming dimensions are compatible

        for block in self.transformer_blocks:
            y_tokens = block(y_tokens, combined_cond)

        # Project back to latent channels and reshape
        velocity_tokens = self.output_projection(y_tokens) # (B, H*W, latent_channels)
        velocity_field = velocity_tokens.permute(0, 2, 1).view(b, c, res_h, res_w) # (B, C, H, W)

        return velocity_field


# Example usage (for verification, not part of the submission)
if __name__ == "__main__":
    # Test DiffusionForcingModule
    latent_channels = 16
    embedding_dim = 256 # Assuming GRU hidden_dim matches embedding_dim for simplicity
    df_module = DiffusionForcingModule(latent_channels, embedding_dim, embedding_dim)

    dummy_latent_input = torch.randn(2, latent_channels, 16, 16)
    dummy_h_prev = torch.randn(2, embedding_dim)

    h_curr = df_module(dummy_latent_input, dummy_h_prev)
    print("DiffusionForcingModule output h_curr shape: " + str(h_curr.shape)) # Expected (2, 256)

    # Test FMT
    fmt_s = FMT(embedding_dim=256, num_layers=6) # Small version
    fmt_b = FMT(embedding_dim=512, num_layers=12) # Base version
    fmt_l = FMT(embedding_dim=768, num_layers=24) # Large version

    print("FMT-S Parameters: " + str(sum(p.numel() for p in fmt_s.parameters())))
    print("FMT-B Parameters: " + str(sum(p.numel() for p in fmt_b.parameters())))
    print("FMT-L Parameters: " + str(sum(p.numel() for p in fmt_l.parameters())))

    dummy_y_t_k = torch.randn(2, latent_channels, 16, 16)
    dummy_t = torch.rand(2) # time t between 0 and 1
    dummy_h_cond = torch.randn(2, embedding_dim) # Condition from GRU/history

    velocity_output = fmt_s(dummy_y_t_k, dummy_t, dummy_h_cond)
    print("FMT output velocity shape: " + str(velocity_output.shape)) # Expected (2, 16, 16, 16)

