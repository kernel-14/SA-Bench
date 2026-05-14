
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

from nfig.layers import fft_2d, ifft_2d, FrequencyMask, SpatialResampler

# --- VQ-GAN Components (Encoder, Decoder, Vector Quantizer) ---

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + self.shortcut(residual))

class Encoder(nn.Module):
    def __init__(self, in_channels=3, nf=128, num_res_blocks=2, ch_mult=(1,2,4)):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, nf, 3, padding=1)
        self.down_blocks = nn.ModuleList()
        curr_nf = nf
        for i, mult in enumerate(ch_mult):
            out_nf = nf * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResidualBlock(curr_nf, out_nf))
                curr_nf = out_nf
            if i < len(ch_mult) - 1:
                self.down_blocks.append(nn.Conv2d(curr_nf, curr_nf, 4, stride=2, padding=1)) # Downsample

        self.conv_out = nn.Conv2d(curr_nf, curr_nf, 3, padding=1) # Last conv to get C channel latent

    def forward(self, x):
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        x = self.conv_out(x)
        return x # B, C, H', W'

class Decoder(nn.Module):
    def __init__(self, out_channels=3, nf=128, num_res_blocks=2, ch_mult=(1,2,4)):
        super().__init__()
        # Reverse the channel multiplier logic from encoder
        ch_mult = ch_mult[::-1]
        self.conv_in = nn.Conv2d(nf * ch_mult[0], nf * ch_mult[0], 3, padding=1) # Initial conv for latent input
        
        self.up_blocks = nn.ModuleList()
        curr_nf = nf * ch_mult[0]
        for i, mult in enumerate(ch_mult):
            out_nf = nf * mult # This is the target feature map size after upsampling
            for _ in range(num_res_blocks):
                self.up_blocks.append(ResidualBlock(curr_nf, out_nf))
                curr_nf = out_nf
            if i < len(ch_mult) - 1:
                self.up_blocks.append(nn.ConvTranspose2d(curr_nf, curr_nf, 4, stride=2, padding=1)) # Upsample
        
        self.conv_out = nn.Conv2d(curr_nf, out_channels, 3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        for block in self.up_blocks:
            x = block(x)
        x = self.conv_out(x)
        return x # B, C, H, W

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings # K
        self.embedding_dim = embedding_dim # C
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, z):
        # z: (B, C, H, W)
        # Permute to (B, H, W, C)
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.embedding_dim) # (B*H*W, C)

        # Calculate distances to embeddings
        distances = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
                    torch.sum(self.embedding.weight**2, dim=1) - \
                    2 * torch.matmul(z_flattened, self.embedding.weight.T)

        # Get the closest embedding indices
        min_distances_indices = torch.argmin(distances, dim=1)
        z_q_flattened = self.embedding(min_distances_indices) # (B*H*W, C)
        z_q = rearrange(z_q_flattened, '(b h w) c -> b h w c', b=z.shape[0], h=z.shape[1], w=z.shape[2])

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        # Compute commitment loss
        loss = self.commitment_cost * torch.mean((z_q.detach() - z)**2) + \
               torch.mean((z_q - z.detach())**2)

        # Permute back to (B, C, H, W)
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        
        return z_q, loss, min_distances_indices.view(z.shape[0], z.shape[1], z.shape[2]) # Return tokens as well

# --- Frequency-guided Residual Quantization ---

class FrequencyGuidedResidualQuantizer(nn.Module):
    def __init__(self, codebook_size, embedding_dim, freq_scaling_factors, base_H, base_W, commitment_cost=0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.freq_scaling_factors = freq_scaling_factors
        self.num_frequency_bands = len(freq_scaling_factors)
        self.base_H = base_H # H'
        self.base_W = base_W # W'

        self.quantizers = nn.ModuleList([
            VectorQuantizer(codebook_size, embedding_dim, commitment_cost)
            for _ in range(self.num_frequency_bands) # One quantizer per band
        ])
        self.spatial_resampler = SpatialResampler()

        # Determine the (h_i, w_i) for each band
        # The paper implies these dimensions are derived from the scaling factors
        self.band_dims = []
        for scale in freq_scaling_factors:
            h_i = math.ceil(self.base_H / scale) # Simplified assumption: scaling factor applies to H and W
            w_i = math.ceil(self.base_W / scale)
            self.band_dims.append((h_i, w_i))

    def forward(self, hat_f_bands):
        # hat_f_bands: List of (B, C, H', W') for each frequency band
        # (output of FrequencyGuidedDecomposer)
        
        quantized_hat_f_bands = []
        quantization_losses = []
        token_indices_bands = []
        
        R_prev = torch.zeros_like(hat_f_bands[0]) # R_{i-1} in the paper, initialized to zero for i=0

        for i in range(self.num_frequency_bands):
            current_hat_f = hat_f_bands[i] # This is hat_f_i in the paper
            
            # (R_{i-1} + hat_f_i) for i >= 1, and hat_f_i for i=0
            input_for_quantizer = current_hat_f if i == 0 else (R_prev + current_hat_f)

            # Downsample input_for_quantizer to (h_i, w_i) for v_i extraction
            h_i, w_i = self.band_dims[i]
            v_i_downsampled = self.spatial_resampler(input_for_quantizer, h_i, w_i) # v_i

            # Quantize v_i
            v_i_quantized_downsampled, q_loss, token_indices = self.quantizers[i](v_i_downsampled)
            quantization_losses.append(q_loss)
            token_indices_bands.append(token_indices) # Store discrete tokens

            # Upsample v_i^q back to original feature map size (H', W')
            v_i_quantized_upsampled = self.spatial_resampler(v_i_quantized_downsampled, self.base_H, self.base_W)
            
            # Compute residual R_i
            R_curr = current_hat_f - v_i_quantized_upsampled
            if i > 0:
                R_curr = R_prev + R_curr # R_i = hat_R_{i-1} + (hat_f_i - Z(v_i, H', W')) (as per paper, slightly simplified)
            R_prev = R_curr.detach() # Detach for residual accumulation

            quantized_hat_f_bands.append(v_i_quantized_upsampled)
        
        return quantized_hat_f_bands, quantization_losses, token_indices_bands

# --- Transformer (VAR Transformer backbone) ---

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, mask=None):
        h = self.heads

        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1) # B, N, inner_dim

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (sim.shape[-1] - mask.shape[-1], 0), value=False)
            mask = rearrange(mask, 'b i -> b 1 i 1') * rearrange(mask, 'b j -> b 1 1 j')
            sim.masked_fill_(~mask, -torch.inf)

        # Causal mask for decoder-only transformer
        i, j = sim.shape[-2:]
        causal_mask = torch.ones((i, j), device=x.device, dtype=torch.bool).triu(j - i + 1)
        sim.masked_fill_(causal_mask, -torch.inf)

        attn = sim.softmax(dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(self.norm(x))

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.attn = SelfAttention(dim, heads, dim_head)
        self.ff = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x, mask=None):
        x = self.attn(x, mask=mask) + x
        x = self.ff(x) + x
        return x

class NFIGTransformer(nn.Module):
    def __init__(self, num_tokens, max_seq_len, dim, depth, heads, dim_head, mlp_dim, dropout=0., num_classes=1000):
        super().__init__()
        self.num_classes = num_classes
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim) # Positional embeddings for sequence of tokens
        self.class_emb = nn.Embedding(num_classes + 1, dim) # +1 for null class for CFG

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(TransformerBlock(dim, heads, dim_head, mlp_dim, dropout))

        self.norm = LayerNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens)

    def forward(self, x, class_label=None, mask=None):
        # x: (B, N) where N is sequence length of tokens
        b, n = x.shape
        
        token_embeddings = self.token_emb(x)
        positions = torch.arange(n, device=x.device)
        position_embeddings = self.pos_emb(positions)

        x = token_embeddings + position_embeddings

        if class_label is not None:
            class_embeddings = self.class_emb(class_label)
            # Add class embedding to the first token, or broadcast across all tokens
            # Following common practice in some VAR implementations, add to all tokens.
            x = x + class_embeddings.unsqueeze(1) # Unsqueeze to (B, 1, D) for broadcasting

        for block in self.layers:
            x = block(x, mask=mask)

        x = self.norm(x)
        logits = self.to_logits(x)
        return logits

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, nf=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, nf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf * 2, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf * 2, nf * 4, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf * 4, nf * 8, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf * 8, 1, 4, stride=1, padding=0) # Output a single logit
        )

    def forward(self, x):
        return self.model(x)
