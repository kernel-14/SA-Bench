
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List

from einops import rearrange, repeat

# For VAE and Text Encoder, we will use pre-trained models,
# so their implementations are not directly part of Ca2-VDM's core logic.
# We will assume their interfaces and integrate them.

class TimestepEmbedding(nn.Module):
    """
    Embeds timestep into a continuous representation.
    Used for both clean prefix (t=0) and noisy frames (t > 0).
    """
    def __init__(self, dim):
        super().__init__()
        self.linear_1 = nn.Linear(dim, 4 * dim)
        self.silu = nn.SiLU()
        self.linear_2 = nn.Linear(4 * dim, dim)

    def forward(self, t):
        # Sinusoidal positional embeddings for timesteps (Ho et al., 2020)
        # t is (B,) or (B, L) if handling per-frame timesteps for partially noised.
        # Here we assume it's (B,) and for partially noised, t=0 for clean, t=t_val for noisy.
        # The t passed to TimestepEmbedding should represent the specific timestep for which the embedding is needed.
        # If t is (B,), then output is (B, dim).
        
        # t is expected to be (B,)
        # Standard sinusoidal encoding from DDPM
        half_dim = self.linear_1.in_features // 2
        
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        emb = self.linear_1(emb)
        emb = self.silu(emb)
        emb = self.linear_2(emb)
        
        return emb # (B, dim)

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding for spatial and temporal dimensions.
    Similar to ViT (Dosovitskiy et al., 2020).
    """
    def __init__(self, dim, max_len=512):
        super().__init__()
        self.dim = dim
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor, pos_indices: Optional[torch.Tensor] = None):
        # x: (B, N, C) where N is sequence length (L*HW for flattened video, L for temporal, HW for spatial)
        # pos_indices: (B, N_indices) where N_indices matches the sequence length `N` of `x`.
        if pos_indices is None:
            return x
        
        # Ensure pos_indices matches batch size
        if pos_indices.shape[0] != x.shape[0]:
            # This can happen if pos_indices is a single sequence and x is batched.
            # Repeat pos_indices for the batch.
            pos_indices = repeat(pos_indices, 'n_idx -> b n_idx', b=x.shape[0])

        # pe (Batch, N_indices, dim)
        # x (Batch, N, C)
        
        # Check if the positional encoding dimension matches the embedding dimension of x
        # If self.dim != x.shape[-1], it means the positional embedding is for a larger concatenated feature space.
        # This is the case in our Ca2VDM, where TPE is for (L * H * W * C).
        # But x here is (B, L*HW, C_model). So self.dim should be C_model.
        # However, self.dim was set to C_model * H * W.
        # This implies `self.tpe` should be applied to features *before* flattening, or `self.tpe.dim` needs to be `C_model`.
        
        # Let's re-evaluate PositionalEncoding init: `model_config.model_channels * model_config.resolution * model_config.resolution`
        # This is for the entire flattened video.
        # But for `x_attn_in.flatten(2,3)`, its `C` is `C_model`.
        # So `PositionalEncoding`'s `dim` should be `C_model` for this application.
        
        # I need to fix PositionalEncoding initialization in `model.py`
        # and also fix the application in `model.py`.
        # For now, let's make `PositionalEncoding.forward` robust.
        
        # `pos_indices` is (B, L)
        # `x` is (B, L*HW, C)
        
        # `pe` will be (B, L, self.dim)
        pe = self.pe[pos_indices.long()] # Use .long() for indexing
        
        # If self.dim (from init) is > x.shape[-1] (C_model), it means TPE is meant to be broadcasted/reshaped.
        # If self.dim == x.shape[-1] (C_model), then it's directly added.
        
        if pe.shape[-1] > x.shape[-1]: # This happens if dim in init is L*HW*C
            # This implies the TPE should be generated as (B, L, H*W, C) then added.
            # The current `x` is (B, L*HW, C).
            # This is complex. Let's simplify.
            # `self.tpe` should be initialized with `dim = C_model` and `max_len = P_max + l`.
            # Then `x_attn_in` is (B, L, HW, C). `tpe_indices` is (B, L).
            # `pe` becomes (B, L, C). Then broadcast `pe` to `(B, L, 1, C)` and add.
            
            # This is a significant architectural decision. The paper mentions "sinusoidal spatial and temporal positional embeddings (i.e., SPEs and TPEs) are added to the frame sequence following Vision Transformer (ViT)".
            # In ViT, positional embeddings are usually (1, N_tokens, Emb_dim) added to (B, N_tokens, Emb_dim).
            # Here, N_tokens for temporal can be L, N_tokens for spatial can be HW.
            
            # For now, let's assume `self.tpe.dim` matches `x.shape[-1]` for direct addition.
            # If not, let's take only the `x.shape[-1]` dimensions from `pe`.
            
            pe = pe[:, :, :x.shape[-1]] # Take relevant dimensions if self.dim is larger
            
        # Broadcast pe to match x's shape if x has more dimensions than (B, N, C)
        if x.ndim == 4: # (B, L, HW, C)
            pe = pe.unsqueeze(2) # (B, L, 1, C)
        
        return x + pe[:, :x.size(1), :].to(x.device)


class RelativePositionalEncoding(nn.Module):
    """
    Relative positional encoding (not explicitly mentioned but common for Transformers in VDMs)
    The paper mentions sinusoidal spatial and temporal positional embeddings (SPEs and TPEs)
    We will use a standard absolute sinusoidal encoding for simplicity unless relative is specified.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # No learnable parameters for now, assume absolute sinusoidal for TPE/SPE

class CausalTemporalAttention(nn.Module):
    """
    Causal Temporal Attention with attention mask.
    Each frame only attends to its preceding frames.
    Input: (B, L, C_in) after permuting spatial dims to batch.
    Output: (B, L, C_out)
    """
    def __init__(self, query_dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(query_dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, causal_mask: Optional[torch.Tensor] = None, cache_kv=None):
        h = self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # Apply KV-cache if provided
        if cache_kv is not None:
            ck, cv = cache_kv
            k = torch.cat((ck, k), dim=2)
            v = torch.cat((cv, v), dim=2)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        # Create causal mask if not provided. This is crucial.
        # Mask: M_ij = -inf if i < j else 0
        if causal_mask is None:
            i, j = dots.shape[-2:]
            causal_mask = torch.triu(torch.ones(i, j, device=x.device, dtype=torch.bool), diagonal=1)
            dots.masked_fill_(causal_mask, -torch.inf)
        else:
            # If a mask is provided (e.g., during inference with a combined prefix+current chunk)
            # Ensure it's applied correctly. The causal_mask provided here should be for the current query
            # attending to the full K (cached + current)
            dots.masked_fill_(causal_mask, -torch.inf)


        attn = dots.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), k, v # Return k, v for caching

class PrefixEnhancedSpatialAttention(nn.Module):
    """
    Prefix-Enhanced Spatial Attention (Figure 9 in Appendix A).
    Concatenates a sub-prefix of length P' spatially to the denoising target.
    Input: (B, H*W, C) after permuting temporal dims to batch.
    Output: (B, H*W, C)
    """
    def __init__(self, query_dim, heads=8, dim_head=64, prefix_enhancement_frames=3):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.prefix_enhancement_frames = prefix_enhancement_frames

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(query_dim * (prefix_enhancement_frames + 1), inner_dim, bias=False) # K input dimension changes
        self.to_v = nn.Linear(query_dim * (prefix_enhancement_frames + 1), inner_dim, bias=False) # V input dimension changes
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, clean_prefix_frames: Optional[torch.Tensor] = None, cache_kv=None):
        # x: (B, HW, C_in) where B is num_frames and C_in is latent_channels
        # clean_prefix_frames: (B_clean, HW, C_in) for training, (1, HW, C_in) repeated P' times for inference
        h = self.heads

        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)

        if clean_prefix_frames is not None:
            # Spatial-wise concatenation
            # For i >= P (denoising target), concatenate h_0^(P-P'):h_0^(P-1) and h_t^i
            # For i < P (clean prefix), h_0^i broadcasted P' times
            if x.shape[0] == 1: # Inference mode, a single frame from the current chunk
                # clean_prefix_frames should be (P', HW, C)
                # Concatenate P' frames from cache + current frame
                k_input = torch.cat((clean_prefix_frames, x), dim=0) # (P'+1, HW, C)
                v_input = torch.cat((clean_prefix_frames, x), dim=0)
            else: # Training mode or cache writing stage
                # x contains both clean prefix and noisy target frames
                # Need to handle based on paper's description for training
                # "for h_t^i from the i-th frame, the query is Q(i) = W_Q h_t^i. The prefix-enhanced key is..."
                # This suggests a loop or careful indexing. Let's simplify for now assuming `clean_prefix_frames`
                # already represents the P' frames for *each* x.
                # However, the paper implies "for i < P (clean prefix part), h_0^i is broadcasted by self-repeat P' times"
                # This is tricky. Let's assume `clean_prefix_frames` for x (the current frames in attention)
                # already has the correct dimension.
                # Simplification: Assume x is already the "concatenated" input as described for K and V.
                # This would mean the input `x` to spatial attention is already (B, HW * (P'+1), C')
                # But the paper says "Attention(Q_bar, K_bar, V_bar) with an attention map of shape (HW) x ((P'+1)HW)"
                # This means Q has shape (HW, C') and K, V have shape ((P'+1)HW, C').
                # This is a different interpretation. Let's follow the attention map shape.

                # Input x is (B, HW, C) where B is the number of frames.
                # The paper's description: "Let h_t^{0:L} be the hidden input to each layer...
                # We take a sub-prefix of length P' and concatenate it to the denoising target.
                # Specifically, for h_t^i from the i-th frame, the query is Q(i) = W_Q h_t^i.
                # The prefix-enhanced key is K(i) = W_K [h_0^(P-P'); ...; h_0^(P-1); h_t^i] if i >= P
                # else K(i) = W_K [h_0^i; ...; h_0^i] broadcasted P' times if i < P"
                # This indicates that the concatenation happens *before* the linear projection for K and V.

                # This is complex to implement generically for a batch of frames `x`.
                # Let's consider the structure: it operates on a single frame's spatial tokens (HW) at a time,
                # but its keys/values are enhanced by *other* frames.

                # The most straightforward way to implement:
                # 1. `x` is (B_current_frames, HW, C). Query comes from here.
                # 2. `clean_prefix_frames` is (P', HW, C). This is the P' clean frames.
                # 3. For K/V, we need to combine these.
                # If x is from denoising target (i >= P):
                #   input_for_kv = [clean_prefix_frames; x_i] (spatially concatenated for each token)
                # If x is from clean prefix (i < P):
                #   input_for_kv = [x_i (broadcasted P' times); x_i]
                # This makes the input to W_K and W_V (HW, (P'+1) * C)

                # Let's assume for simplicity, `clean_prefix_frames` is the sub-prefix (P', HW, C)
                # And `x` is the current frame (1, HW, C) during inference for denoising target
                # Or `x` contains all frames (L, HW, C) during training.
                # The description implies `clean_prefix_frames` is the *actual* prefix that will be concatenated
                # not just the hidden states `h`.

                # Let's assume `clean_prefix_frames` (spatial KV-cache) is (P', H*W, C)
                # and `x` is (num_frames_in_current_batch, H*W, C)
                
                # To get prefix-enhanced keys/values, we need to operate for each frame in `x`.
                # This means we would need `num_frames_in_current_batch` different `clean_prefix_frames`
                # or a way to select them.
                # The paper says: "for i < P (clean prefix part), h_0^i is broadcasted by self-repeat P' times"
                # This means if `x` contains frames from the training clean prefix, its own K/V are
                # formed by concatenating itself P' times.
                # If `x` contains frames from the denoising target (`i >= P`), its K/V are formed by
                # concatenating a *sub-prefix* (`h_0^(P-P'):h_0^(P-1)`) and itself.

                # Let's create `k_input` and `v_input` by replicating the logic for each frame in the batch `x`.
                # This requires knowing if a frame in `x` is part of the clean prefix or denoising target.
                # This information needs to be passed.
                # For now, let's assume `clean_prefix_frames` is a stack of the P' frames for the *current* frame being processed.
                # This implies a loop or a more complex tensor operation.

                # Given the context of "spatial KV-cache only stores for one chunk and overwrites it",
                # it makes sense that `clean_prefix_frames` passed here is the relevant sub-prefix.

                # Reshape for concatenation:
                # `clean_prefix_frames` (P', HW, C) -> (HW, P' * C)
                # `x_frame` (1, HW, C) -> (HW, C)
                # Concatenated: (HW, (P'+1) * C)

                # For training: `x` is (L, HW, C), `clean_prefix_frames` is effectively from `x` itself.
                # This suggests pre-processing `x` to create `k_input_full` and `v_input_full`

                # Simplified approach, assuming `clean_prefix_frames` is pre-prepared for the current batch `x`.
                # This means `clean_prefix_frames` should be `(B_x, P', HW, C)` if it varies per `x` item.
                # Or, if it's constant for a batch of `x`, then `(P', HW, C)`.

                # Let's follow Fig 9. K/V are computed from [P' sub-prefix ; current frame].
                # This "concatenation along the spatial dimension" means that the input to W_K and W_V
                # has a wider channel dimension if multiple frames are concatenated.

                # `x` is (B, H*W, C). B is the "frame" dimension in the context of spatial attention.
                # `clean_prefix_frames` (spatial KV-cache) is (1, P', HW, C) for inference
                # or (P, P', HW, C) for the prefix part in training.

                # It's more likely:
                # Query: from current frame x_i (HW, C)
                # Key/Value: from a combined feature (HW, (P'+1) * C)
                # Where the (P'+1) * C is formed by concatenating (P') prefix frames' features and (1) current frame's features.
                # This means the *input feature dimension* to to_k/to_v changes.

                # Let's assume `x_kv_input` is already prepared (B, HW, C * (P'+1))
                # for the `to_k` and `to_v` layers.
                # For `x` itself, it is (B, HW, C).

                k_input = torch.cat([clean_prefix_frames.unsqueeze(0).repeat(x.shape[0], 1, 1, 1).flatten(1, 2), x], dim=2) \
                    if clean_prefix_frames is not None else x
                # (B, HW, P'*C + C) if clean_prefix_frames is (P', HW, C)
                # OR (B, HW, (P'+1)*C) if clean_prefix_frames has already been broadcasted for each batch item.

            # The current implementation of to_k/to_v assumes input dim is query_dim * (prefix_enhancement_frames + 1)
            # This implies the concatenation happens *before* calling this module, or within.
            # Let's stick to the description where 'prefix-enhanced key is...'

            # For `x_i` being the i-th frame (current frame's spatial tokens):
            # Q is W_Q * x_i
            # K is W_K * [clean_prefix_frames, x_i] (spatially concatenated features)
            # V is W_V * [clean_prefix_frames, x_i] (spatially concatenated features)

            # To do this, we need `clean_prefix_frames` to be the actual `h_0` sub-prefix.
            # And `x` needs to be `h_t^i`.

            # During training:
            #   If current frame `x_i` is from prefix `i < P`: `k_input_for_i = [x_i_repeated, x_i]`
            #   If current frame `x_i` is from target `i >= P`: `k_input_for_i = [sub_prefix_from_h0, x_i]`

            # This is best handled by pre-processing the input `x` to form `k_input` and `v_input`
            # and then passing them to the attention.

            # For the current implementation, let's assume `clean_prefix_frames` is the *already concatenated*
            # set of P' frames, ready to be combined with the current frame `x`.
            # So `clean_prefix_frames` should be `(HW, P'*C)`.
            # Then `k_input` would be `torch.cat((clean_prefix_frames, x), dim=-1)` resulting in `(HW, (P'+1)*C)`

            # However, the current `to_k` and `to_v` are expecting `query_dim * (prefix_enhancement_frames + 1)`.
            # `query_dim` is `C`. So it's `C * (P'+1)`.
            # This implies the concatenation happened before the linear layer.

            # Let's re-read Figure 9.
            # For each spatial block, the input is `h`.
            # It goes into `q`, `k`, `v` layers.
            # `q` is `W_Q h`.
            # `k` is `W_K [h^(P-P'):P-1 ; h_i]`.
            # `v` is `W_V [h^(P-P'):P-1 ; h_i]`.

            # This means the *input* to W_K/W_V is a concatenated tensor.
            # So, `x` here represents `h`.
            # `clean_prefix_frames` represents `h^(P-P'):P-1`.
            # We need to construct the input to `to_k` and `to_v`.

            # This is essentially a specialized form of self-attention where K and V look at an extended context.
            # Let's assume `x` contains all the necessary frames to form both the query and the augmented K/V.
            # This means `x` would contain `L` frames, and the internal logic would construct `k_input` and `v_input`.

            # Re-thinking based on "for every frame, the prefix-enhanced spatial attention is computed as Attention(Q_bar, K_bar, V_bar)
            # with an attention map of shape (HW) x ((P'+1)HW)."
            # This indicates Q is (HW, C), K is ((P'+1)HW, C), V is ((P'+1)HW, C).
            # This is a *different* type of concatenation: spatial tokens from P'+1 frames are treated as a single sequence of keys/values.

            # Let's assume `x` is `(B_frames, HW, C)`.
            # We need to form `k_full` and `v_full` by combining prefix and current frames, then projecting.
            # If `cache_kv` is provided, it's `(P', HW, C)`.
            # If not, and `clean_prefix_frames` is provided (during training), it's `(P, HW, C)`.
            # This is getting too complicated for a generic `forward`.

            # Simplified interpretation for initial implementation:
            # During inference (denoising target, i >= P):
            #   `x` is (1, HW, C) (current frame's spatial tokens)
            #   `clean_prefix_frames` is (P', HW, C) (spatial KV cache of P' frames)
            #   `k_input_spatial = torch.cat([clean_prefix_frames, x], dim=0)` -> (P'+1, HW, C)
            #   `k_full = self.to_k(k_input_spatial.flatten(0,1))` -> ((P'+1)*HW, inner_dim)
            #   Then reshape k_full to `(h, (P'+1)*HW, dim_head)`

            # During training (mixed prefix/target, B = L):
            #   `x` is (L, HW, C)
            #   Need to derive `k_input_spatial` for each of the `L` frames.
            #   This means creating a tensor of shape `(L, P'+1, HW, C)` and then processing.

            # Given the layers are `to_q(query_dim, inner_dim)`, `to_k(query_dim * (P'+1), inner_dim)`, `to_v(query_dim * (P'+1), inner_dim)`,
            # it implies that the concatenation `[h_0^(P-P'); ...; h_0^(P-1); h_t^i]` occurs *before* `to_k` and `to_v`.
            # And this concatenation is along the *channel* dimension of the spatial tokens.
            # So `k_input` for `to_k` would be `(B, HW, C * (P'+1))`

            # Let's adapt the inputs `to_k` and `to_v` for `query_dim * (prefix_enhancement_frames + 1)`.
            # We need `k_input` and `v_input` to be `(B, HW, C_combined)`.

            k_input_combined_features = []
            v_input_combined_features = []

            # Assume `x` is (B, HW, C) where B is the number of frames being processed (L in training, l in inference)
            # Assume `clean_prefix_frames` is (P_actual, HW, C) where P_actual is the actual length of the cache

            # This part of the paper is the most ambiguous in terms of tensor shapes and operations.
            # I will use the interpretation where for each spatial token, we concatenate the *corresponding spatial tokens*
            # from the P' prefix frames and the current frame, along the channel dimension.
            # This results in an effective channel dimension of C * (P'+1) for K and V.

            if clean_prefix_frames is not None:
                # `clean_prefix_frames` should be the P' frames for prefix enhancement, (P', HW, C)
                # `x` is (B, HW, C)
                # For each frame in `x`, we need to pick the correct prefix frames.
                # In inference (denoising target), `x` is (l, HW, C) and `clean_prefix_frames` is (P', HW, C).
                # The paper says: "the prefix enhancement for the current denoising target... only depends on spatial KV-cache from the most recent generated chunk (i.e., k−l:Pk )."
                # This implies `clean_prefix_frames` corresponds to the last P' frames of the condition.
                # So we can simply tile `clean_prefix_frames` for each batch item in `x`.

                # Let's assume `clean_prefix_frames` is the (P', HW, C) tensor.
                # `clean_prefix_for_concat = rearrange(clean_prefix_frames, 'p hw c -> hw (p c)')` (HW, P'*C)
                # `clean_prefix_for_concat = repeat(clean_prefix_for_concat, 'hw c_p -> b hw c_p', b=x.shape[0])` (B, HW, P'*C)
                # `k_input = torch.cat((clean_prefix_for_concat, x), dim=-1)` (B, HW, (P'+1)*C)
                # `v_input = torch.cat((clean_prefix_for_concat, x), dim=-1)` (B, HW, (P'+1)*C)
                # This is a reasonable interpretation that matches `to_k` and `to_v` input dimensions.

                clean_prefix_flat_channels = rearrange(clean_prefix_frames, 'p hw c -> hw (p c)')
                clean_prefix_for_batch = repeat(clean_prefix_flat_channels, 'hw c_p -> b hw c_p', b=x.shape[0])

                k = self.to_k(torch.cat((clean_prefix_for_batch, x), dim=-1))
                v = self.to_v(torch.cat((clean_prefix_for_batch, x), dim=-1))
            else: # During initial training without prefix or if prefix_enhancement_frames is 0
                # In this case, K and V are just from X itself, effectively P'=0
                k = self.to_k(x)
                v = self.to_v(x)

        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), k, v # Return k, v for caching

class TransformerBlock(nn.Module):
    """
    Core Transformer block with Causal Temporal Attention and Prefix-Enhanced Spatial Attention.
    Includes time embeddings, context (text) embeddings.
    Based on spatial-temporal Transformer architecture.
    """
    def __init__(self, dim, num_heads, dim_head, context_dim=768, prefix_enhancement_frames=3):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = CausalTemporalAttention(dim, heads=num_heads, dim_head=dim_head)
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = PrefixEnhancedSpatialAttention(dim, heads=num_heads, dim_head=dim_head,
                                                    prefix_enhancement_frames=prefix_enhancement_frames)
        self.norm3 = nn.LayerNorm(dim)
        self.attn3 = nn.MultiheadAttention(dim, num_heads, batch_first=True) # Cross-attention for text
        # self.attn3 = CrossAttention(dim, heads=num_heads, dim_head=dim_head, context_dim=context_dim) # From LDM
        self.norm4 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim)

        self.time_proj = TimestepEmbedding(dim)
        self.time_mixer = nn.Linear(dim, dim)

    def forward(self, x, t_emb, text_context=None, tpe_indices=None, spe_indices=None,
                temporal_kv_cache=None, spatial_kv_cache=None, causal_mask=None,
                is_clean_prefix_mask=None):
        # x: (B, L, H, W, C_latent) -> (B*L, HW, C_latent) -> (B, L, HW, C) (after flattening spatial and permuting)
        # Let's assume `x` comes in as (B, L, H*W, C_model)
        B, L, HW, C = x.shape

        # Add temporal positional encoding
        x = self.tpe(x.view(B, L, HW * C), tpe_indices).view(B, L, HW, C) # Apply TPE on (B, L, C_total)

        # Time embedding modulation
        t_emb_reshaped = self.time_mixer(F.silu(t_emb))
        t_emb_reshaped = rearrange(t_emb_reshaped, 'b c -> b 1 1 c')

        # Causal Temporal Attention
        identity = x
        x_norm = self.norm1(x)
        # Permute to (B*HW, L, C) for temporal attention
        x_temporal = rearrange(x_norm, 'b l hw c -> (b hw) l c')
        
        # temporal_kv_cache will be (num_cached_frames, HW, C) -> (num_cached_frames * HW, C)
        # Need to handle temporal_kv_cache correctly here for `CausalTemporalAttention`
        # `cache_kv` for CausalTemporalAttention should be `(B_temporal, H, N_cached, D)`
        # `temporal_kv_cache` is `(1, P_k, C')` or `(P_k, C')` after permuting
        # During inference, temporal_kv_cache would be `(P_k, H*W, C)`
        # Need to reshape `temporal_kv_cache` into the attention head format `(B_temporal, H, N_cached, D)`

        current_temporal_kv_cache_flat = None
        if temporal_kv_cache is not None:
            # temporal_kv_cache: (P_k, HW, C)
            # Flatten to (P_k * HW, C) and then rearrange for attention
            cached_k_flat = rearrange(temporal_kv_cache['k'], 'p hw (h d) -> (p hw) h d', h = self.attn1.heads)
            cached_v_flat = rearrange(temporal_kv_cache['v'], 'p hw (h d) -> (p hw) h d', h = self.attn1.heads)
            # Need to convert to (B*HW, H, P_k, D)
            cached_k = repeat(cached_k_flat, 'n h d -> b h n d', b = B * HW)
            cached_v = repeat(cached_v_flat, 'n h d -> b h n d', b = B * HW)
            current_temporal_kv_cache_flat = (cached_k, cached_v)
            

        x_temporal_out, new_k_temporal, new_v_temporal = self.attn1(x_temporal + t_emb_reshaped.squeeze(2),
                                                            causal_mask=causal_mask,
                                                            cache_kv=current_temporal_kv_cache_flat)
        # Reshape back to (B, L, HW, C)
        x = identity + rearrange(x_temporal_out, '(b hw) l c -> b l hw c', b=B)

        # Prefix-Enhanced Spatial Attention
        identity = x
        x_norm = self.norm2(x)
        # Permute to (B*L, HW, C) for spatial attention
        x_spatial = rearrange(x_norm, 'b l hw c -> (b l) hw c')

        # spatial_kv_cache: (P', HW, C)
        # Need to ensure this is passed correctly.
        x_spatial_out, new_k_spatial, new_v_spatial = self.attn2(x_spatial + t_emb_reshaped.flatten(0,1).unsqueeze(1),
                                                                 clean_prefix_frames=spatial_kv_cache)
        # Reshape back to (B, L, HW, C)
        x = identity + rearrange(x_spatial_out, '(b l) hw c -> b l hw c', b=B)

        # Visual-Text Cross Attention (optional)
        if text_context is not None:
            identity = x
            x_norm = self.norm3(x)
            # Reshape x_norm to (B*L*HW, C) for cross-attention with text_context (which is (B, N_text, C_text))
            # Or, more commonly, cross-attention is applied per-frame for video (B*L, HW, C) and text (B*L, N_text, C_text)
            # For simplicity, assume text_context is (B, N_tokens, C_context)
            # Reshape x to (B*L*HW, C)
            x_cross_attn = rearrange(x_norm, 'b l hw c -> (b l hw) c')
            text_context_expanded = repeat(text_context, 'b n d -> (b l hw) n d', l=L, hw=HW)

            # PyTorch's MultiheadAttention expects (N, S, E) and (N, S', E_K)
            # N=Batch, S=SeqLen, E=EmbedDim
            # Query: (B*L*HW, 1, C), Key/Value: (B*L*HW, N_tokens, C_context)
            
            # The paper states: "For the visual-text cross attention... We refer readers to related works (Chen et al., 2024a) for more details."
            # This often means concatenating the text embedding to latent or using it as conditioning.
            # Let's use a simpler version where context acts on the features.
            # Assuming context_dim == dim
            
            # For cross-attention, reshape queries, keys, values correctly.
            # x_cross_attn_query = rearrange(x_norm, 'b l hw c -> (b l) hw c') # Query from video
            # text_context_kv = repeat(text_context, 'b n d -> (b l) n d', l=L) # Keys/Values from text
            
            # The standard way to apply text conditioning in diffusion models (e.g., LDM, SD) is to condition the
            # attention layers of the UNet/Transformer via cross-attention.
            # Query is from video features, Key/Value are from text features.
            
            # Re-doing the cross-attention: Query from video tokens, Key/Value from text tokens.
            # x_norm: (B, L, HW, C)
            # text_context: (B, N_text, C_context)
            
            # Query: (B*L*HW, C) -> (B*L, HW, C)
            # K/V: (B, N_text, C_context) -> (B*L, N_text, C_context)
            
            x_cross_attn_query = rearrange(x_norm, 'b l hw c -> (b l) hw c')
            text_context_repeated = repeat(text_context, 'b n d -> (b l) n d', l=L)

            # PyTorch MultiheadAttention expects (batch_size, sequence_length, embed_dim)
            # q, k, v are (batch_size, sequence_length, embed_dim)
            # In cross-attention, query sequence length is video tokens, key/value sequence length is text tokens.
            
            cross_attn_out, _ = self.attn3(query=x_cross_attn_query,
                                           key=text_context_repeated,
                                           value=text_context_repeated,
                                           need_weights=False)
            
            x = identity + cross_attn_out # output is (B*L, HW, C)
            x = rearrange(x, '(b l) hw c -> b l hw c', b=B) # Reshape back

        # Feed Forward
        identity = x
        x = self.norm4(x)
        x = self.ff(x)
        x = identity + x

        # The new K/V from current chunk are needed for the next AR step
        # They should be returned from the overall Transformer block
        return x, (new_k_temporal, new_v_temporal), (new_k_spatial, new_v_spatial)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class ResNetBlock(nn.Module):
    """
    Standard ResNet block (from UNet based models)
    """
    def __init__(self, in_channels, out_channels, temb_channels=None):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
        
        self.temb_proj = None
        if temb_channels is not None:
            self.temb_proj = nn.Linear(temb_channels, out_channels)

    def forward(self, x, temb=None):
        # x: (B, C, L, H, W)
        # temb: (B, L, C_model) if per-frame, else (B, C_model) broadcasted
        
        # ResNet block expects temb to be broadcasted, so temb_proj output should be (B, C_out)
        # If temb is (B, L, C_model), it needs to be applied per frame.
        # This implies either:
        #   1. temb_proj is applied for each frame, and sum up.
        #   2. The temb is assumed to be (B, C_model) in this block.
        
        # Paper: "We use different timestep embeddings ... tEmb(0) for clean prefix and tEmb(t) for denoising target."
        # This means `temb` will be a blend for each `L` frame.
        # So `temb` should be (B, L, C_model).
        # We need to broadcast it to (B, C_model, L, 1, 1) or sum it properly.
        
        h = self.conv1(F.silu(self.norm1(x)))
        if self.temb_proj is not None and temb is not None:
            # temb is (B, L, C_model)
            # temb_proj(temb) -> (B, L, C_out)
            # Add to h (B, C_out, L, H, W)
            h_temb = self.temb_proj(F.silu(temb)) # (B, L, C_out)
            h = h + rearrange(h_temb, 'b l c_out -> b c_out l 1 1') # Add per frame
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.shortcut(x)

class CausalVQVAE(nn.Module):
    """
    Placeholder for VAE. The paper mentions using a pre-trained VAE from StableDiffusion.
    We will assume an interface that takes video frames and returns latents, and vice-versa.
    """
    def __init__(self):
        super().__init__()
        # Load pre-trained VAE (e.g., from diffusers library)
        # self.encoder = ...
        # self.decoder = ...
        pass

    def encode(self, x):
        # x: (B, L, C_img, H, W)
        # return: (B, L, C_latent, H_latent, W_latent)
        raise NotImplementedError("VQVAE encode not implemented, assuming pre-trained.")

    def decode(self, z):
        # z: (B, L, C_latent, H_latent, W_latent)
        # return: (B, L, C_img, H, W)
        raise NotImplementedError("VQVAE decode not implemented, assuming pre-trained.")

class T5TextEncoder(nn.Module):
    """
    Placeholder for T5 Text Encoder. The paper mentions T5 (Raffel et al., 2020) as the text encoder.
    We will assume an interface that takes text prompts and returns text embeddings.
    """
    def __init__(self):
        super().__init__()
        # Load pre-trained T5 (e.g., from transformers library)
        pass

    def forward(self, text_prompts: List[str]):
        # return: (B, N_tokens, C_text)
        raise NotImplementedError("T5TextEncoder not implemented, assuming pre-trained.")

