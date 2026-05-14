
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

from layers import SpectralConv1d, MLP, SelfAttention, CrossAttention

class FNOBlock(nn.Module):
    def __init__(self, in_channels, out_channels, modes, activation=nn.GELU):
        super().__init__()
        self.conv = SpectralConv1d(in_channels, out_channels, modes)
        self.mlp = MLP(in_channels, out_channels, out_channels, 2, activation) # Simple 2-layer MLP

    def forward(self, x):
        # x: (batch_size, channels, spatial_dim)
        x_mlp = rearrange(x, 'b c s -> b s c') # MLP expects (batch, spatial, channels)
        x_mlp = self.mlp(x_mlp)
        x_mlp = rearrange(x_mlp, 'b s c -> b c s')

        return self.conv(x) + x_mlp # Fourier transform + point-wise linear transformation

# Simplified Mamba-like SSM block based on recurrence and convolution concept
# This is a conceptual implementation following the description "learnable convolution kernels K_tau"
# and "causal recurrence". A full Mamba SSM is complex and beyond what can be inferred
# purely from the paper's abstract description without direct implementation details.
class MambaSSM(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv1d = nn.Conv1d(in_channels=self.d_inner, out_channels=self.d_inner,
                                kernel_size=d_conv, groups=self.d_inner,
                                padding=d_conv - 1) # Causal padding, simplified
        self.act = nn.SiLU()

        # Simplified A and B for SSM, typically learnable
        # For simplicity, using a linear layer to generate B and C from input
        self.x_proj = nn.Linear(self.d_inner, d_state * 2) # projects to (B, C)
        self.dt_proj = nn.Linear(self.d_inner, d_model) # projects to Delta

        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.shape[1]

        # 1. Linear projection and split for gate
        x_and_res = self.in_proj(x) # (batch, seq_len, d_inner * 2)
        x_and_res = rearrange(x_and_res, 'b l (d_inner_2) -> b (d_inner_2) l') # (batch, d_inner * 2, seq_len)

        # Apply causal convolution to x
        # Only take the first d_inner channels for convolution input
        x_conv = self.conv1d(x_and_res[:, :self.d_inner, :])[:, :, :seq_len]
        x_conv = self.act(x_conv)

        # Split into x and residual connection
        x_main = x_conv
        res = x_and_res[:, self.d_inner:, :seq_len] # This is meant to be a gated output part

        x_main = rearrange(x_main, 'b c l -> b l c') # Back to (batch, seq_len, d_inner)
        res = rearrange(res, 'b c l -> b l c')

        # Simplified SSM core (not full Mamba SSM, just conceptual recurrence)
        # This part simulates the selective state space effect conceptually
        # In a real Mamba, A, B, C, D, and Delta are learned and interact dynamically
        # Here we just apply an MLP-like transform with activation
        A = -torch.exp(torch.rand(self.d_state, self.d_state, device=x.device)) # Conceptual A matrix
        B = self.x_proj(x_main)[:, :, :self.d_state] # Conceptual B
        C = self.x_proj(x_main)[:, :, self.d_state:] # Conceptual C
        dt = F.softplus(self.dt_proj(x_main)) # Conceptual Delta t

        # A very simplified recurrence (not true Mamba scan, just element-wise ops)
        # Real Mamba uses a parallel scan
        y = dt * B + C # This is a placeholder for actual state update
        y = self.act(y)

        # Project out
        y = self.out_proj(y) # (batch, seq_len, d_model)

        return y * torch.sigmoid(res) # Apply gating with residual, simplified.


class PerceiverIOBlock(nn.Module):
    def __init__(self, dim, latent_dim, heads, dim_head, num_latents):
        super().__init__()
        self.cross_attn_input_to_latent = CrossAttention(latent_dim, dim, heads=heads, dim_head=dim_head)
        self.latent_self_attn = SelfAttention(latent_dim, heads=heads, dim_head=dim_head)
        self.feed_forward_latent = MLP(latent_dim, latent_dim, latent_dim * 2, 2) # 2-layer MLP

    def forward(self, x, latents):
        # x: (batch, seq_len, dim) - input from FNO, etc.
        # latents: (batch, num_latents, latent_dim)

        # Cross-attention: input to latents
        # Queries from latents, keys/values from input (from FNO-based mapping)
        # The paper states: "keys and values are obtained from FNO-based mapping from the inputs"
        # For this block, we assume `x` is already the result of an FNO-based mapping.
        latents = self.cross_attn_input_to_latent(latents, x) + latents

        # Self-attention on latents
        latents = self.latent_self_attn(latents) + latents

        # Feed-forward for latents
        latents = self.feed_forward_latent(latents) + latents
        return latents

class CodomainAttention(nn.Module):
    """
    Codomain Attention Mechanism as described in [13].
    Similarity is detected between features (codomain dimensions), not between samples/tokens.
    This implies operating on the channel dimension.
    """
    def __init__(self, in_channels, out_channels, heads=8, dim_head=None):
        super().__init__()
        self.heads = heads
        self.out_channels = out_channels # Output channels will be the new 'codomain' dimension
        self.dim_head = dim_head if dim_head is not None else in_channels // heads
        self.scale = self.dim_head ** -0.5

        inner_dim = self.dim_head * heads

        # Linear layers to project features for Q, K, V
        # Applied channel-wise (e.g., Conv1d with kernel size 1 across spatial dim)
        self.to_q = nn.Conv1d(in_channels, inner_dim, 1, bias=False)
        self.to_k = nn.Conv1d(in_channels, inner_dim, 1, bias=False)
        self.to_v = nn.Conv1d(in_channels, inner_dim, 1, bias=False)
        self.to_out = nn.Conv1d(inner_dim, out_channels, 1)

    def forward(self, x):
        # x: (batch, in_channels, spatial_dim)
        h = self.heads
        batch, c, s = x.shape

        q = self.to_q(x) # (batch, inner_dim, spatial_dim)
        k = self.to_k(x)
        v = self.to_v(x)

        # Reshape for multi-head attention: (batch, heads, dim_head, spatial_dim)
        q = rearrange(q, 'b (h d) s -> b h d s', h=h)
        k = rearrange(k, 'b (h d) s -> b h d s', h=h)
        v = rearrange(v, 'b (h d) s -> b h d s', h=h)

        # Compute similarity between features (codomain dimensions), not spatial tokens
        # We want attention over the 'd' dimension, where 'd' now represents the feature dimension within a head
        # This implies a dot product across the 'd' dimension, summing over 's' (spatial) if needed.
        # However, the description implies feature-wise attention.
        # A common interpretation for channel-wise attention would be treating each spatial location
        # as a 'token' and channels as 'features'.
        # If we interpret "similarity not between samples, but between features" as attention *over channels*:
        
        # Reshape to treat channels as sequence elements, spatial as batch
        q_attn = rearrange(q, 'b h d s -> (b s) h d') # (batch*spatial, heads, dim_head)
        k_attn = rearrange(k, 'b h d s -> (b s) h d')
        v_attn = rearrange(v, 'b h d s -> (b s) h d')

        # Dot product for attention score (batch*spatial, heads, d, d)
        sim = torch.einsum('B h i, B h j -> B h i j', q_attn, k_attn) * self.scale
        attn = sim.softmax(dim=-1)

        # Apply attention to values
        out = torch.einsum('B h i j, B h j -> B h i', attn, v_attn) # (batch*spatial, heads, d)
        out = rearrange(out, '(b s) h d -> b (h d) s', b=batch) # (batch, inner_dim, spatial_dim)

        return self.to_out(out) # (batch, out_channels, spatial_dim)
