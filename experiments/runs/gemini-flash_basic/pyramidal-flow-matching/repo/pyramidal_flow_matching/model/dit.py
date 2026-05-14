import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Helper functions for DiT (conceptual)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TimestepEmbedder(nn.Module):
    """Embeds timestep data.
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
        t_freq = torch.arange(start=0, end=self.frequency_embedding_size // 2, dtype=torch.float32, device=t.device)
        t_freq = t_freq * -(math.log(10000.0) / (self.frequency_embedding_size // 2 - 1))
        t_freq = torch.exp(t_freq)
        t_embed = torch.cat([torch.sin(t[..., None] * t_freq), torch.cos(t[..., None] * t_freq)], dim=-1)
        return self.mlp(t_embed)

class LabelEmbedder(nn.Module):
    """Embeds class labels and other conditions.
    """
    def __init__(self, num_classes, hidden_size, drop_rate=0.):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes, hidden_size)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return self.drop(embeddings)

class RotaryPositionalEmbeddings(nn.Module):
    """ 1D Rotary Positional Embeddings (RoPE) as described in Su et al., 2024 """
    def __init__(self, dim, seq_len=None):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached = seq_len
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_caches(self, x, seq_len):
        # x: (batch, num_heads, seq_len, head_dim)
        # seq_len: int
        if seq_len > self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]
        return self._cos_cached[:, :, :seq_len, ...], self._sin_cached[:, :, :seq_len, ...]

    def rotate_half(self, x):
        x1, x2 = x[..., :self.dim//2], x[..., self.dim//2:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope(self, q, k, seq_len):
        cos, sin = self._update_cos_sin_caches(q, seq_len)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed

    def forward(self, q, k):
        # q, k: (batch, num_heads, seq_len, head_dim)
        seq_len = q.shape[2]
        return self.apply_rope(q, k, seq_len)


class TransformerBlock(nn.Module):
    """ A single Transformer block with adaptive layer norm (adaLN-ZERO) """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, adaLN_channels=None,
                 use_rope=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        # AdaLN-ZERO parameters (for timestep and text conditioning)
        self.adaLN_modulation = nn.Linear(adaLN_channels, 6 * dim) if adaLN_channels is not None else None
        self.use_rope = use_rope
        if self.use_rope:
            # RoPE is applied to the features of each head, so dim is head_dim
            self.rope = RotaryPositionalEmbeddings(dim=dim // num_heads)

    def forward(self, x, y=None):
        # y is the conditioning information (time + text embedding)
        if self.adaLN_modulation is not None:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(6, dim=1)
            # Unsqueeze for broadcasting across sequence length
            shift_msa, scale_msa, gate_msa = shift_msa.unsqueeze(1), scale_msa.unsqueeze(1), gate_msa.unsqueeze(1)
            shift_mlp, scale_mlp, gate_mlp = shift_mlp.unsqueeze(1), scale_mlp.unsqueeze(1), gate_mlp.unsqueeze(1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = 0, 0, 0, 0, 0, 0

        # Attention block
        h = self.norm1(x)
        h = h * (1 + scale_msa) + shift_msa
        
        if self.use_rope:
            # Manual attention to apply RoPE
            B, N, C = h.shape # N is sequence length (num_patches)
            qkv = self.attn.qkv(h).reshape(B, N, 3, self.attn.num_heads, C // self.attn.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            q, k = self.rope(q, k) # Apply RoPE to query and key
            
            attn = (q @ k.transpose(-2, -1)) * self.attn.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn.attn_drop(attn)

            x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = x + gate_msa * self.attn.proj(x_attn)
        else:
            x = x + gate_msa * self.attn(h)
        
        # MLP block
        h = self.norm2(x)
        h = h * (1 + scale_mlp) + shift_mlp
        x = x + gate_mlp * self.mlp(h)
        return x


class DiT(nn.Module):
    """Diffusion Transformer (DiT) model with a 3D-aware input. 
    Processes video latents (B, C, T, H, W)."""
    def __init__(self, 
                 input_size: tuple[int, int, int], # (T, H, W) of latent frames
                 patch_size: tuple[int, int, int] = (1, 2, 2), # (t, h, w) of patches
                 in_channels: int = 4, 
                 hidden_size: int = 1152, 
                 depth: int = 28, 
                 num_heads: int = 16, 
                 mlp_ratio: float = 4.0, 
                 class_dropout_prob: float = 0.1, 
                 num_classes: int = 1000, # Example for ImageNet, can be adjusted or removed for text-only
                 learn_sigma: bool = True, # Predicts both mean and variance of the noise
                 condition_on_text: bool = True,
                 text_embedding_dim: int = 768, # For CLIP or T5 text embeddings
                 use_rope_for_temporal: bool = True,
                ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.input_size = input_size
        self.condition_on_text = condition_on_text

        self.x_embedder = PatchEmbed3D(input_size, patch_size, in_channels, hidden_size)
        self.num_patches = self.x_embedder.num_patches
        
        # Spatial positional embedding. Paper mentions sinusoidal for spatial, but for simplicity,
        # and common practice in DiT variants, a learnable positional embedding is used here.
        # If a true sinusoidal PE is needed, it would be added at each forward pass based on patch coordinates.
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=True)

        self.t_embedder = TimestepEmbedder(hidden_size)
        if condition_on_text:
            assert text_embedding_dim is not None, "text_embedding_dim must be provided for text conditioning."
            self.y_embedder = nn.Linear(text_embedding_dim, hidden_size) # Map text embedding to hidden_size
        else:
            self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, 
                             norm_layer=nn.LayerNorm, adaLN_channels=hidden_size,
                             use_rope=use_rope_for_temporal
                            ) for _ in range(depth)
        ])

        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, self.out_channels * self.patch_size[0] * self.patch_size[1] * self.patch_size[2])
        )

        # Initialize weights (conceptual, actual init will be more complex)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, t, y=None):
        # x: (B, C, T, H, W) - latent video
        # t: (B,) - timestep
        # y: (B, text_embedding_dim) or (B,) - text/class condition

        # Patchify and add positional embedding
        x = self.x_embedder(x) + self.pos_embed # (B, N, hidden_size)

        # Get conditioning embeddings
        t_embed = self.t_embedder(t) # (B, hidden_size)
        if self.condition_on_text:
            assert y is not None, "Text condition 'y' must be provided for text-conditioned DiT."
            y_embed = self.y_embedder(y) # (B, hidden_size)
            c = t_embed + y_embed # (B, hidden_size) - combined conditioning
        else:
            # Handle class dropout for classifier-free guidance during inference
            # For training, y would typically be class labels
            if y is None: # Classifier-free guidance, y is dropped
                y_embed = self.y_embedder(torch.full((x.shape[0],), self.num_classes, device=x.device)) # Placeholder for null label
            else:
                y_embed = self.y_embedder(y)
            c = t_embed + y_embed # (B, hidden_size)

        for block in self.blocks:
            x = block(x, c) # Pass conditioning to each transformer block

        # Final layer to unpatchify and output to original latent space shape
        x = self.final_layer(x) # (B, N, self.out_channels * patch_size_prod)
        x = self.unpatchify(x) # (B, self.out_channels, T, H, W)
        return x

    def unpatchify(self, x):
        # Rearrange to (B, out_channels, T, H, W)
        pt, ph, pw = self.patch_size
        T, H, W = self.input_size
        
        # Calculate original dimensions based on patchification
        out_channels = self.out_channels
        Nt, Nh, Nw = self.x_embedder.num_patches_t, self.x_embedder.num_patches_h, self.x_embedder.num_patches_w
        
        # x: (B, Nt * Nh * Nw, out_channels * pt * ph * pw)
        x = x.reshape(x.shape[0], Nt, Nh, Nw, out_channels, pt, ph, pw)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous() # (B, out_channels, Nt, pt, Nh, ph, Nw, pw)
        x = x.reshape(x.shape[0], out_channels, T, H, W)
        return x


class PatchEmbed3D(nn.Module):
    """ Video to Patch Embedding with 3D convolution """
    def __init__(self, input_size=(16, 32, 32), patch_size=(1, 2, 2), in_channels=4, embed_dim=768):
        super().__init__()
        self.input_size = input_size # (T, H, W) of latent video
        self.patch_size = patch_size # (t, h, w) of patches
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # Calculate number of patches
        self.num_patches_t = input_size[0] // patch_size[0]
        self.num_patches_h = input_size[1] // patch_size[1]
        self.num_patches_w = input_size[2] // patch_size[2]
        assert self.num_patches_t * patch_size[0] == input_size[0], "Input temporal dimension must be divisible by patch temporal dimension"
        assert self.num_patches_h * patch_size[1] == input_size[1], "Input height must be divisible by patch height"
        assert self.num_patches_w * patch_size[2] == input_size[2], "Input width must be divisible by patch width"
        
        self.num_patches = self.num_patches_t * self.num_patches_h * self.num_patches_w

        self.proj = nn.Conv3d(in_channels,
                              embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):
        # x: (B, C, T, H, W)
        # Ensure input dimensions match the expected input_size for patchification
        assert x.shape[2] == self.input_size[0] and \
               x.shape[3] == self.input_size[1] and \
               x.shape[4] == self.input_size[2], \
               f"Input tensor temporal, height, or width size {x.shape[2:]} does not match expected input size {self.input_size}."

        x = self.proj(x) # (B, embed_dim, Nt, Nh, Nw)
        x = x.flatten(2).transpose(1, 2) # (B, N, embed_dim)
        return x
