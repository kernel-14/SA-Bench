import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalTemporalAttention(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.head_dim = head_dim

    def forward(self, q, k, v, mask=None):
        # q, k, v are of shape (batch_size, num_heads, sequence_length, head_dim)
        
        # Causal attention mask: Each position can only attend to itself and previous positions.
        # This is a lower triangular mask.
        if mask is None:
            seq_len = q.shape[-2]
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device))
            causal_mask = causal_mask.bool()
        else:
            causal_mask = mask # If a mask is provided, use it

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply causal mask
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        return out

class MultiHeadCausalTemporalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.causal_attn = CausalTemporalAttention(self.head_dim)

    def forward(self, x, mask=None):
        # x shape: (batch_size, sequence_length, embed_dim)
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output = self.causal_attn(q, k, v, mask=mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        output = self.out_proj(attn_output)
        return output


class TemporalKVQueue:
    def __init__(self, max_length, embed_dim, num_heads, head_dim):
        self.max_length = max_length
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.queue_k = None
        self.queue_v = None
        self.current_length = 0

    def enqueue(self, k_new, v_new):
        # k_new, v_new are expected to be (batch_size, num_heads, chunk_length, head_dim)
        if self.queue_k is None:
            self.queue_k = k_new
            self.queue_v = v_new
        else:
            self.queue_k = torch.cat((self.queue_k, k_new), dim=2)
            self.queue_v = torch.cat((self.queue_v, v_new), dim=2)
        self.current_length = self.queue_k.shape[2]

        if self.current_length > self.max_length:
            dequeue_len = self.current_length - self.max_length
            self.queue_k = self.queue_k[:, :, dequeue_len:, :]
            self.queue_v = self.queue_v[:, :, dequeue_len:, :]
            self.current_length = self.max_length

    def get_cache(self):
        return self.queue_k, self.queue_v

    def reset(self):
        self.queue_k = None
        self.queue_v = None
        self.current_length = 0


class CyclicTemporalPositionalEmbedding(nn.Module):
    def __init__(self, max_len, embed_dim):
        super().__init__()
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.pe = self._get_positional_embedding(max_len, embed_dim)

    def _get_positional_embedding(self, seq_len, embed_dim):
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(seq_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0) # Add batch dimension

    def forward(self, seq_len, offset=0, batch_size=1, device='cpu'):
        # seq_len: length of the current chunk to be embedded
        # offset: current position in the sequence, used for cyclic shift
        
        # Ensure positional embeddings are on the correct device
        if self.pe.device != device:
            self.pe = self.pe.to(device)

        # Cyclic shift mechanism
        # The TPEs are assigned chunk-by-chunk as the autoregression progresses.
        # If the cumulatively generated video exceeds the training length (max_len),
        # TPEs are assigned from the beginning cyclically.
        
        # Select the relevant part of the positional embedding based on offset and seq_len
        # The indices will wrap around 
        indices = torch.arange(seq_len, device=device) + offset
        indices = indices % self.max_len
        
        # Retrieve the embeddings and expand for the batch size
        return self.pe[:, indices, :].expand(batch_size, -1, -1)



class TemporalKVQueue:
    def __init__(self, max_length, embed_dim, num_heads, head_dim):
        self.max_length = max_length
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.queue_k = None
        self.queue_v = None
        self.current_length = 0

    def enqueue(self, k_new, v_new):
        # k_new, v_new are expected to be (batch_size, num_heads, chunk_length, head_dim)
        if self.queue_k is None:
            self.queue_k = k_new
            self.queue_v = v_new
        else:
            self.queue_k = torch.cat((self.queue_k, k_new), dim=2)
            self.queue_v = torch.cat((self.queue_v, v_new), dim=2)
        self.current_length = self.queue_k.shape[2]

        if self.current_length > self.max_length:
            dequeue_len = self.current_length - self.max_length
            self.queue_k = self.queue_k[:, :, dequeue_len:, :]
            self.queue_v = self.queue_v[:, :, dequeue_len:, :]
            self.current_length = self.max_length

    def get_cache(self):
        return self.queue_k, self.queue_v

    def reset(self):
        self.queue_k = None
        self.queue_v = None
        self.current_length = 0


class CyclicTemporalPositionalEmbedding(nn.Module):
    def __init__(self, max_len, embed_dim):
        super().__init__()
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.pe = self._get_positional_embedding(max_len, embed_dim)

    def _get_positional_embedding(self, seq_len, embed_dim):
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(seq_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0) # Add batch dimension

    def forward(self, seq_len, offset=0, batch_size=1, device='cpu'):
        # seq_len: length of the current chunk to be embedded
        # offset: current position in the sequence, used for cyclic shift
        
        # Ensure positional embeddings are on the correct device
        if self.pe.device != device:
            self.pe = self.pe.to(device)

        # Cyclic shift mechanism
        # The TPEs are assigned chunk-by-chunk as the autoregression progresses.
        # If the cumulatively generated video exceeds the training length (max_len),
        # TPEs are assigned from the beginning cyclically.
        
        # Select the relevant part of the positional embedding based on offset and seq_len
        # The indices will wrap around 
        indices = torch.arange(seq_len, device=device) + offset
        indices = indices % self.max_len
        
        # Retrieve the embeddings and expand for the batch size
        return self.pe[:, indices, :].expand(batch_size, -1, -1)



class PrefixEnhancedSpatialAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, P_prime):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.P_prime = P_prime # Sub-prefix length

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x_current, h0_prefix=None):
        # x_current: (batch_size * H * W, L_chunk, embed_dim) - current frames to be denoised
        # h0_prefix: (batch_size * H * W, P_prime, embed_dim) - clean prefix frames (spatial KV-cache)

        batch_spatial_size, l_chunk, _ = x_current.shape

        q = self.q_proj(x_current).view(batch_spatial_size, l_chunk, self.num_heads, self.head_dim).transpose(1, 2)
        
        if h0_prefix is not None:
            # Concatenate prefix_enhanced key and value
            # The paper says: "W^K [h0_prefix; h_t^i]"
            # h0_prefix has shape (batch_spatial_size, P_prime, embed_dim)
            # x_current has shape (batch_spatial_size, L_chunk, embed_dim)
            # For k and v, we concatenate the prefix frames with the current frames
            
            # For the denoising target (i >= P), keys are from [h_0^{P-P'}, ..., h_0^{P-1}, h_t^i]
            # For the clean prefix part (i < P), keys are from [h_0^i, ..., h_0^i] (broadcasted P' times)
            # Given that this module processes a *current chunk* (x_current) which is *denoising target*,
            # we consider the first case where h0_prefix is the sub-prefix (spatial KV-cache)
            
            # The paper states: "for h_t^i from the i-th frame, the query is Q_bar(i) = W^Q h_t^i.
            # The prefix-enhanced key is W^K [h_0^{P-P'}; ...; h_0^{P-1}; h_t^i] if i >= P"
            # This implies spatial-wise concatenation. Assuming h0_prefix is already 'flattened' spatially
            # or handled within the Attention computation.
            
            # When concatenating features, they should be along the spatial dimension.
            # Here, the input x_current is already (B*H*W, L_chunk, C'), where B*H*W is effectively batch.
            # If h0_prefix is (B*H*W, P_prime, C'), then we concatenate along the P_prime / L_chunk dimension.
            # The paper states "concatenation along the spatial dimension" in Eq 4, but the tensor shapes
            # in the definition of the attention (HW) x ((P' + 1)HW) implies flattening HxW and then concatenating
            # the full (P'+1) frames' flattened spatial features. This implies a different structure for attention.

            # Re-interpreting "concatenation along the spatial dimension" (Figure 9 in Appendix, which is out of scope)
            # and the formula for prefix-enhanced key: "W^K [h_0^{P-P'}; ...; h_0^{P-1}; h_t^i]".
            # This implies for *each* frame h_t^i (from x_current), its keys and values are formed by concatenating
            # the features of P_prime prefix frames with itself.

            # Let's assume h_current is (B, L_chunk, H, W, C) and h0_prefix is (B, P_prime, H, W, C)
            # For each frame in L_chunk, we'd form a new sequence of (P_prime + 1) frames.
            # Then apply spatial attention. This would mean the effective sequence length for spatial attention
            # becomes (P_prime + 1) * H * W.

            # Given the current input format (batch_spatial_size, L_chunk, embed_dim),
            # where batch_spatial_size = batch * H * W, we would need to reshape to get individual frames.
            # This is complex and might not align with the provided `x_current` directly.

            # Let's assume the "spatial-wise concatenation" is handled implicitly for the K/V projections,
            # meaning the `h0_prefix` acts as additional key/value context.
            # This means `k` and `v` will have a longer sequence length.

            k_current = self.k_proj(x_current).view(batch_spatial_size, l_chunk, self.num_heads, self.head_dim).transpose(1, 2)
            v_current = self.v_proj(x_current).view(batch_spatial_size, l_chunk, self.num_heads, self.head_dim).transpose(1, 2)
            
            k_prefix = self.k_proj(h0_prefix).view(batch_spatial_size, self.P_prime, self.num_heads, self.head_dim).transpose(1, 2)
            v_prefix = self.v_proj(h0_prefix).view(batch_spatial_size, self.P_prime, self.num_heads, self.head_dim).transpose(1, 2)

            k = torch.cat((k_prefix, k_current), dim=2)
            v = torch.cat((v_prefix, v_current), dim=2)
        else:
            k = self.k_proj(x_current).view(batch_spatial_size, l_chunk, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x_current).view(batch_spatial_size, l_chunk, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_spatial_size, l_chunk, self.embed_dim)
        output = self.out_proj(attn_output)
        return output



class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear_1 = nn.Linear(dim, dim * 4)
        self.silu = nn.SiLU()
        self.linear_2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.silu(x)
        x = self.linear_2(x)
        return x


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


class Ca2VDMBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, P_prime, max_temporal_len, use_text_cond=False, text_embed_dim=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn_temporal = MultiHeadCausalTemporalAttention(embed_dim, num_heads)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn_spatial = PrefixEnhancedSpatialAttention(embed_dim, num_heads, P_prime)

        if use_text_cond:
            if text_embed_dim is None:
                raise ValueError("text_embed_dim must be provided if use_text_cond is True")
            self.norm_cross = nn.LayerNorm(embed_dim)
            self.attn_cross = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True) # Standard Cross-attention
        self.use_text_cond = use_text_cond

        self.norm3 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim)

        self.temporal_pos_embed = CyclicTemporalPositionalEmbedding(max_len=max_temporal_len, embed_dim=embed_dim)

    def forward(
        self, 
        x, # (batch_size, num_frames_chunk, H, W, embed_dim)
        timestep_emb, # (batch_size, embed_dim)
        text_embed=None, # (batch_size, seq_len_text, text_embed_dim)
        temporal_kv_cache=None, # (batch_size, num_heads, P_k, head_dim)
        spatial_kv_cache=None, # (batch_size * H * W, P_prime, embed_dim)
        tpe_offset=0
    ):
        # Add timestep embedding to input (simple addition for now, typically done more elaborately)
        # The paper mentions different timestep embeddings for clean prefix and denoising target.
        # This block processes a *chunk* of frames, which can be part of clean prefix or denoising target.
        # We assume `timestep_emb` already contains this distinction for the current chunk frames.
        
        # Reshape x for temporal attention: (B, L, H, W, C) -> (B*H*W, L, C) for spatial attention
        # Or (B, L, H, W, C) -> (B*H*W, L, C) -> (B, L, C) for temporal if we average H, W?
        # Paper says: "input to each layer is first permuted by treating the spatial resolution H x W as the batch dimension"
        # This suggests temporal attention acts on (B*H*W, L, C)
        batch_size, num_frames_chunk, H, W, C = x.shape
        x = x.view(batch_size * H * W, num_frames_chunk, C)

        # 1. Temporal Attention
        identity = x
        x = self.norm1(x)
        
        # Apply TPEs before temporal attention
        tpe = self.temporal_pos_embed(num_frames_chunk, offset=tpe_offset, batch_size=batch_size * H * W, device=x.device)
        x_with_tpe = x + tpe

        # Prepare K and V for causal temporal attention, potentially including cached KVs
        # In CausalTemporalAttention, q, k, v are (batch_size*H*W, num_heads, seq_len, head_dim)
        # The `x_with_tpe` is already (batch_size*H*W, num_frames_chunk, C)

        # The internal q,k,v projections in MultiHeadCausalTemporalAttention will handle the splitting into heads
        # We need to pass temporal_kv_cache to MultiHeadCausalTemporalAttention to concatenate for K, V if it exists.
        # This requires modifying MultiHeadCausalTemporalAttention to accept explicit k_cache, v_cache.
        # For now, I'll pass the full sequence and let it handle causality. The KV-cache logic needs to be outside this block for now.
        # The paper says: "The model reads the clean key and value caches as K_0^{0:Pk}, V_0^{0:Pk}. Then, they are concatenated to the noisy ones as ..."
        # This means the concatenation happens *before* the attention module.

        # Let's adjust for temporal_kv_cache handling:
        # Current MultiHeadCausalTemporalAttention takes x and mask.
        # We need to construct the full K and V sequence by concatenating the cache with the current chunk's K/V.

        # Temporarily, I'll assume MultiHeadCausalTemporalAttention can internally handle if temporal_kv_cache is passed, 
        # but this is not how it's implemented right now. For simplicity, let's keep the current attention module as is for now
        # and assume full sequence input for training (no cache yet) and then modify for inference (with cache).
        # For the block definition, I'll assume `x` contains *all* frames (prefix + current chunk) during training for simplicity.
        # During inference, the KV-cache is external and concatenated.

        # For this block, let's consider `x` to be the full sequence being processed in this step (current chunk + prefix if any).
        # The KV-cache will be handled *before* calling this block in the main VDM loop.
        # So, the `x` input to `attn_temporal` will already be `[prefix_frames, current_chunk_frames]`
        # and the `tpe` would also be for this combined sequence.

        # Re-evaluating based on Figure 4(b) and section 3.3 Temporal KV-Cache:
        # `CausalAttn(Q_t^{Pk:Pk+l}, K_tilde(k,t), V_tilde(k,t))`
        # where `K_tilde = [K_0^{0:Pk}, K_t^{Pk:Pk+l}]`
        # This means Q is only for the current chunk, while K and V are for the concatenated sequence (cache + current chunk).
        # So, `MultiHeadCausalTemporalAttention` needs to take separate Q, K, V.
        # I'll modify MultiHeadCausalTemporalAttention to accept q, k, v separately rather than just x.

        # Re-defining MultiHeadCausalTemporalAttention to take Q, K, V, mask
        # This will be `self.attn_temporal(q_current, k_full, v_full, mask)`

        # First, let's extract Q, K, V from x for the current chunk
        q_current = self.attn_temporal.q_proj(x_with_tpe).view(batch_size * H * W, num_frames_chunk, self.attn_temporal.num_heads, self.attn_temporal.head_dim).transpose(1, 2)
        k_current_for_attn = self.attn_temporal.k_proj(x_with_tpe).view(batch_size * H * W, num_frames_chunk, self.attn_temporal.num_heads, self.attn_temporal.head_dim).transpose(1, 2)
        v_current_for_attn = self.attn_temporal.v_proj(x_with_tpe).view(batch_size * H * W, num_frames_chunk, self.attn_temporal.num_heads, self.attn_temporal.head_dim).transpose(1, 2)

        # Concatenate with temporal_kv_cache if provided
        if temporal_kv_cache is not None:
            cached_k, cached_v = temporal_kv_cache # temporal_kv_cache should be (num_heads, P_k, head_dim) * per batch element
            # cached_k, cached_v need to be expanded for H*W spatial dimensions for temporal attention
            # This means temporal_kv_cache should be (batch_size, num_heads, P_k, head_dim)
            # and then expanded for H*W like: cached_k.unsqueeze(0).expand(H*W, -1, -1, -1) and then concatenated to each batch element
            # No, the paper describes temporal attention after reshaping where spatial resolution is batch dimension.
            # So `x` is already `(B*H*W, L, C)`. If `temporal_kv_cache` is `(B, num_heads, P_k, head_dim)` then it needs reshaping.

            # Let's assume `temporal_kv_cache` is provided as `(batch_size * H * W, num_heads, P_k, head_dim)`
            k_full = torch.cat((cached_k, k_current_for_attn), dim=2)
            v_full = torch.cat((cached_v, v_current_for_attn), dim=2)
            mask_len = cached_k.shape[2] + num_frames_chunk
            causal_mask = torch.tril(torch.ones(num_frames_chunk, mask_len, device=x.device), diagonal=cached_k.shape[2]-1).bool()
        else:
            k_full = k_current_for_attn
            v_full = v_current_for_attn
            causal_mask = None # MultiHeadCausalTemporalAttention will create a default mask
        
        # Ensure the mask is (B*H*W, 1, L_chunk, L_full_sequence)
        if causal_mask is not None: # Expand mask for batch and heads
            causal_mask = causal_mask.unsqueeze(1).unsqueeze(0).expand(batch_size * H * W, self.attn_temporal.num_heads, -1, -1)

        attn_output = self.attn_temporal.causal_attn(q_current, k_full, v_full, mask=causal_mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size * H * W, num_frames_chunk, C)
        x = identity + self.attn_temporal.out_proj(attn_output)

        # 2. Spatial Attention
        # Reshape x back to (B, L, H, W, C) for conceptual clarity if needed, or keep (B*H*W, L, C)
        # For spatial attention, 'L' is batch_dim, so we want (L_chunk, B*H*W, C) if we follow text, or
        # (B*L_chunk, H*W, C) which is how PrefixEnhancedSpatialAttention is expecting.
        # My PrefixEnhancedSpatialAttention expects x_current of (batch_size * H * W, L_chunk, embed_dim)
        # and h0_prefix of (batch_size * H * W, P_prime, embed_dim).

        # The x coming out of temporal attention is (batch_size * H * W, num_frames_chunk, C)
        # This directly matches x_current for spatial attention if we interpret L_chunk as the sequence length for spatial attention for a single (H*W) location.
        # This interpretation is wrong. Spatial attention operates *within* a frame. L is frame dimension.

        # Correct interpretation for spatial attention: 
        # Input `h_t^{0:L}` (B, L, H, W, C). For spatial attention, `L` is treated as batch dimension.
        # So, for each frame `i`, `h_t^i` is `(B, H, W, C)`. Flatten `H*W` to be sequence length.
        # So, input to spatial attention would be `(B * L, H * W, C)`.
        # The prefix `h_0^{P-P':P-1}` would also be `(B * P_prime, H * W, C)`.

        # Let's adjust the input `x` to `Ca2VDMBlock` to be `(B, L, C, H, W)` or `(B, L, H, W, C)`
        # Assume `x` input is `(batch_size, num_frames_chunk, H, W, embed_dim)`

        x_spatial_input = x.view(batch_size * num_frames_chunk, H * W, C) # (B*L_chunk, H*W, C)
        identity_spatial = x_spatial_input
        x_spatial_input = self.norm2(x_spatial_input)

        # Prepare spatial_kv_cache for PrefixEnhancedSpatialAttention
        # spatial_kv_cache is (batch_size * H * W, P_prime, embed_dim) as defined in PrefixEnhancedSpatialAttention
        # This means spatial_kv_cache needs to be (B*P_prime, H*W, C) to match. Let's adjust.

        # Let's assume spatial_kv_cache is (batch_size, P_prime, H, W, embed_dim)
        # Reshape it to (batch_size * P_prime, H * W, embed_dim)
        if spatial_kv_cache is not None:
            # (batch_size, P_prime, H, W, embed_dim) -> (batch_size * P_prime, H * W, embed_dim)
            reshaped_spatial_kv_cache = spatial_kv_cache.view(batch_size * self.attn_spatial.P_prime, H * W, C)
        else:
            reshaped_spatial_kv_cache = None

        spatial_attn_output = self.attn_spatial(x_spatial_input, h0_prefix=reshaped_spatial_kv_cache)
        x_spatial_output = identity_spatial + self.attn_spatial.out_proj(spatial_attn_output)

        # Reshape back to (batch_size, num_frames_chunk, H, W, embed_dim)
        x = x_spatial_output.view(batch_size, num_frames_chunk, H, W, C)
        
        # 3. Cross Attention (Optional)
        if self.use_text_cond and text_embed is not None:
            identity_cross = x
            # Reshape x for cross-attention: (B, L, H, W, C) -> (B, L*H*W, C)
            x_cross = x.view(batch_size, num_frames_chunk * H * W, C)
            x_cross = self.norm_cross(x_cross)
            # MultiheadAttention expects (query, key, value) as (seq_len, batch_size, embed_dim)
            # or (batch_size, seq_len, embed_dim) with batch_first=True
            # text_embed is (batch_size, seq_len_text, text_embed_dim)
            # For now, assuming embed_dim == text_embed_dim. If not, a projection is needed.
            attn_output, _ = self.attn_cross(query=x_cross, key=text_embed, value=text_embed)
            x = identity_cross + attn_output.view(batch_size, num_frames_chunk, H, W, C)
        
        # 4. Feed Forward
        identity_ff = x
        # Reshape x for FF: (B, L, H, W, C) -> (B*L*H*W, C)
        x_ff = x.view(batch_size * num_frames_chunk * H * W, C)
        x_ff = self.norm3(x_ff)
        x_ff = self.ff(x_ff)
        x = identity_ff + x_ff.view(batch_size, num_frames_chunk, H, W, C)

        return x

