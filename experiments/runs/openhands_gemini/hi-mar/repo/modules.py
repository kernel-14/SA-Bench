import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import DiTBlock, get_2d_sincos_pos_embed
import math

class TimestepEmbedder(nn.Module):
    """
    Embeds timestep t into a a vector of specified dimension.
    Borrowed from DiT implementation.
    """
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t can be 1D (B,) or 2D (B, 1)
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        
        # Sinusoidal positional embedding
        # Based on https://github.com/openai/guided-diffusion/blob/27c20a8d9ad98399f51614cd8a0edf68dd78d283/guided_diffusion/unet.py#L471
        half_dim = self.frequency_embedding_size // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.float() * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.frequency_embedding_size % 2 == 1:  # last dim element set to 0
            emb = torch.nn.functional.pad(emb, (0, 1))
        
        emb = self.mlp(emb)
        return emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into a a vector of specified dimension.
    Borrowed from DiT implementation.
    """
    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes, hidden_size)
        self.num_classes = num_classes
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Dropout(p=dropout_prob)
        )
    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding_table(labels)
        emb = self.mlp(embeddings)
        return emb

class HiMARTransformer(nn.Module):
    """
    The Hierarchical Masked Autoregressive Transformer backbone.
    Consists of a stack of DiTBlocks with AdaLNZero.
    This Transformer is used for both phases as per the paper, with scale-aware blocks.
    """
    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: int = 4, # As per DiTBlock (4*dim)
        cond_dim: int = None, # Combined dim of timestep, label, and scale embeddings
        patch_size: int = None, # Not directly used in Transformer, but for pos_embed calculation
        img_size: int = None, # Size of the latent grid (e.g., 32x32 for 256x256 image with VAE downsample 8)
        low_res_img_size: int = None, # Size of the low-res latent grid (e.g., 16x16 for 128x128 image with VAE downsample 8)
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Positional embeddings for both high-res and low-res tokens
        if img_size is not None:
            self.pos_embed_high_res = nn.Parameter(
                torch.zeros(1, img_size * img_size, hidden_size), requires_grad=False
            )
            self.register_buffer(
                "pos_embed_high_res_precomputed",
                get_2d_sincos_pos_embed(hidden_size, img_size, cls_token=False).float().unsqueeze(0)
            )
            
        if low_res_img_size is not None:
            self.pos_embed_low_res = nn.Parameter(
                torch.zeros(1, low_res_img_size * low_res_img_size, hidden_size), requires_grad=False
            )
            self.register_buffer(
                "pos_embed_low_res_precomputed",
                get_2d_sincos_pos_embed(hidden_size, low_res_img_size, cls_token=False).float().unsqueeze(0)
            )
        
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, cond_dim, ada_type="adaln_zero")
            for _ in range(num_layers)
        ])
        
        self.final_layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.adaLN_mlp = nn.Sequential(
            nn.Linear(cond_dim, 2 * hidden_size),
            nn.SiLU()
        )
        nn.init.constant_(self.adaLN_mlp[-1].weight, 0)
        nn.init.constant_(self.adaLN_mlp[-1].bias, 0)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, t_cond: torch.Tensor, c_cond: torch.Tensor = None, s_cond: torch.Tensor = None) -> torch.Tensor:
        """
        x: input tokens (B, N, hidden_size)
        t_cond: timestep embedding (B, hidden_size)
        c_cond: class/text embedding (B, hidden_size)
        s_cond: scale embedding (B, hidden_size) - not explicitly mentioned as separate, but combined into DiT cond
        """
        # Combine conditioning information
        # The paper mentions 'scale vector v' from sinusoidal embedding and MLP,
        # then injected via AdaLN-Zero. DiT's conditioning is (t_embed + y_embed).
        # We'll combine t_cond and c_cond for the main conditioning input.
        # Scale information is usually implicit in the conditioning.
        
        # If s_cond is provided, it means we are using scale-aware transformer
        if s_cond is not None:
            cond = t_cond + c_cond + s_cond
        else: # For original DiT or if scale embedding is not explicitly separated
            cond = t_cond + c_cond 

        # Add positional embedding based on resolution. This needs to be handled externally based on phase.
        # For simplicity, we assume x already has the correct positional embedding added before being passed to this module.
        # In the paper, it implicitly states "the scale vector v is injected ... via AdaLN-Zero operations".
        # This implies that `cond` should include scale information.
        # For Hi-MAR, the same Transformer is used for both phases, but with different input token sequences (low-res vs high-res)
        # and potentially different conditioning, e.g., low-res tokens as pivots for high-res phase.

        for block in self.blocks:
            x = block(x, cond) # DiTBlock handles LN internally

        # Final LN and projection
        # This part is simplified for Hi-MAR as the output of the final block acts as conditional tokens.
        # The final projection for AdaLNZero usually involves scale and shift for a final layer norm,
        # followed by linear projection.
        # Following DiT, we apply final layer norm and a linear layer before prediction head.
        
        final_norm_cond = self.adaLN_mlp(cond)
        shift_final, scale_final = final_norm_cond.chunk(2, dim=1)
        shift_final = shift_final.unsqueeze(1)
        scale_final = scale_final.unsqueeze(1)
        x = self.final_layer_norm(x) * (1 + scale_final) + shift_final

        return x

class MLPDiffusionHead(nn.Module):
    """
    MLP-based Diffusion Head as described in MAR and Phase 1 of Hi-MAR.
    It models masked token probability distribution individually.
    """
    def __init__(self, in_channels: int, hidden_size: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.adaln = AdaLN(hidden_size, cond_dim) # Use AdaLN, not AdaLNZero
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.silu = nn.SiLU()
        self.linear2 = nn.Linear(hidden_size, out_channels) # Output should be in_channels for VAE latent

    def forward(self, x: torch.Tensor, t_cond: torch.Tensor) -> torch.Tensor:
        """
        x: conditional tokens from Transformer (B, N, hidden_size)
        t_cond: timestep embedding (B, hidden_size)
        """
        # cond in AdaLN from paper: context vector 'c' obtained by summing timestep embedding and conditional tokens.
        # However, the paper states MLP-based diffusion head only takes conditional tokens of masked tokens as conditions.
        # "an additional diffusion head conditioned on Z^s is adopted for the small scale optimization as in MAR"
        # and "MLP-based diffusion head that treats each token independently"
        # This means the output `x` (conditional tokens from Transformer) is processed individually.
        
        # For MLP-based, the adaptive normalization parameters are typically derived from time (and potentially class)
        # We need to refine the usage of AdaLN for MLPDiffusionHead to match the paper's description
        # Equation (3) describes the DiTBlock, but the MLP-based head is described differently:
        # "MLP-based Diffusion head blocks include adaLN, layernorm, and feed-forward layers." (Fig 2d)
        # This implies: LN(x) + FFN(LN(x)) with AdaLN controlling the LN parameters.
        
        # Let's re-interpret Figure 2(d) for MLP-based Diffusion head.
        # It shows `adaLN -> layernorm -> FFN`
        # Assuming `c` in Figure 2(e) refers to the combined conditioning (time + conditional tokens `x` in some form)
        # For MLP-based, it's typically just a function of time.
        
        # Re-interpreting for an MLP-based diffusion head based on standard practice and the paper's description
        # "model the probability distribution of x_i conditioned on the output of the masked autoregressive Transformer z_i"
        # and "takes the conditional tokens of masked tokens as conditions"
        # This means `x` itself is the condition, and time is also a condition.
        # The equation from Fig 2d: y_a = y^i + gamma_1 * Attention(alpha_1 * LN(y^i) + beta_1) is for Diffusion Transformer Head.
        # For MLP-based: it is likely a simpler structure, where AdaLN acts on `x` directly.
        # From paper: `y^i` in Fig 2d seems to be the input to the current diffusion head block.
        # The structure looks like `input -> AdaLN(input, cond) -> FFN`.
        # For this, I will use `t_cond` as the `cond` for AdaLN.
        
        # The typical structure for an MLP based diffusion head with AdaLN (similar to how it's used in U-Net based diff models):
        # x_norm = self.adaln.norm(x) * (1 + scale) + shift (where scale, shift from t_cond)
        # return self.mlp(x_norm)
        
        # Let's align with "adaLN, layernorm, and feed-forward layers." from Fig 2(d)
        # It suggests an AdaLN applied before a simple MLP.
        
        # The `AdaLN` module returns (alpha1, beta1, gamma1, alpha2, beta2, gamma2)
        # For a simple MLP with AdaLN, it typically applies scaling and shifting to the input features.
        # The paper's Figure 2(d) diagram is a bit ambiguous for MLP-based head, but usually implies
        # applying adaptive normalization directly to the input before feeding it into linear layers.
        
        # Let's use the first set of alpha/beta/gamma for a single adaptive normalization.
        alpha, beta, gamma = self.adaln(t_cond)[0:3] # Only need first 3 for single block-like op
        
        # Apply normalization and then an MLP
        x = F.layer_norm(x, (x.shape[-1],)) * (1 + alpha) + beta # Layer norm, then scale and shift
        
        # The paper's description for Diff. Head 1/2 hidden sizes indicates these are the dimensions
        # of the MLP inside the diffusion head, not necessarily the input/output of the whole head.
        # For simplicity, we assume these are internal dimensions.
        x = self.linear1(x)
        x = self.silu(x)
        x = self.linear2(x)
        return x

class DiffusionTransformerHead(nn.Module):
    """
    Diffusion Transformer Head as described in Phase 2 of Hi-MAR.
    Uses a stack of DiTBlocks with AdaLN.
    """
    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_heads: int,
        out_channels: int, # Output features for VAE latent (e.g., 4)
        mlp_ratio: int = 4,
        cond_dim: int = None, # Combined dim of timestep and conditional tokens
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, cond_dim, ada_type="adaln")
            for _ in range(num_layers)
        ])
        
        self.final_layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.adaln_mlp = nn.Sequential(
            nn.Linear(cond_dim, 2 * hidden_size),
            nn.SiLU()
        )
        self.linear_out = nn.Linear(hidden_size, out_channels)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, c_tokens: torch.Tensor) -> torch.Tensor:
        """
        x: noise-corrupted visual tokens (B, N, hidden_size) - input of the first block
        t_emb: timestep embedding (B, hidden_size)
        c_tokens: conditional tokens (e.g., predicted low-res tokens from phase 1) (B, M, hidden_size)
                  or combined context tokens (class/text + low-res pivots)
        
        The paper states `c` (context vector) is obtained by summing time step embedding and conditional tokens.
        It also says "considers all the masked and unmasked conditional tokens".
        This implies `c_tokens` should already include any context and potentially the full set of (masked/unmasked) tokens,
        or that `cond_dim` is designed to take combined features.
        
        Based on Figure 2(e), the Diffusion Transformer Head takes (noise_corrupted_vector, c).
        `c` is shown as `time embedding + conditional tokens`.
        This implies `c` is a single vector per batch element, not a sequence.
        So, `c_tokens` must be aggregated/pooled if it's a sequence.
        However, the `cond_dim` for AdaLN is the dimension of this `c`.
        
        Let's assume `c_tokens` here refers to the final conditional vector (after processing, e.g., pooling/attention).
        If `c_tokens` is indeed a sequence of tokens, then `cond_dim` would need to match its feature dim,
        and `adaln` would need to accept a sequence and apply pooling itself or be applied per token.
        Given the typical DiT usage, `cond` for AdaLN is a `(B, D)` vector.
        
        So, we'll assume `c_tokens` (e.g., low-res pivots) are processed to form a fixed-size context vector,
        which is then combined with `t_emb`.
        
        However, the phrase "considers all the masked and unmasked conditional tokens" implies these tokens are part of the attention mechanism.
        This would mean `x` should be concatenated with `c_tokens` *before* the DiTBlocks, and `cond` for AdaLN is just `t_emb`.
        
        Let's follow Figure 2(e) text: "c denotes the context vector obtained by summating the time step embedding and the conditional tokens".
        This `c` is fed to AdaLN.
        "The Diffusion Transformer head contains a stack of Transformer blocks, and the i-th block with input y_i is computed as:"
        This `y_i` is the sequence being processed (noise_corrupted_vector).
        So, `cond` for AdaLN should be `t_emb + (pooled)c_tokens`.
        
        For `c_tokens` (B, M, hidden_size), we need to pool it. Mean pooling is a common choice.
        """
        # Aggregate c_tokens if it's a sequence
        if c_tokens.ndim == 3: # If c_tokens is a sequence of conditional tokens
            # Simple mean pooling or a learned aggregation could be used.
            # For now, let's use mean pooling.
            c_tokens_pooled = c_tokens.mean(dim=1) # (B, hidden_size)
        else: # Assumed to be (B, hidden_size) already
            c_tokens_pooled = c_tokens
        
        cond = t_emb + c_tokens_pooled # (B, hidden_size)

        for block in self.blocks:
            x = block(x, cond)

        # Final LN and projection
        final_norm_cond = self.adaln_mlp(cond)
        shift_final, scale_final = final_norm_cond.chunk(2, dim=1)
        shift_final = shift_final.unsqueeze(1)
        scale_final = scale_final.unsqueeze(1)
        x = self.final_layer_norm(x) * (1 + scale_final) + shift_final
        x = self.linear_out(x)
        return x