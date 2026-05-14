
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple

from einops import rearrange, repeat

from .config import ModelConfig
from .modules import (
    TimestepEmbedding,
    PositionalEncoding,
    CausalTemporalAttention,
    PrefixEnhancedSpatialAttention,
    FeedForward,
    ResNetBlock,
    T5TextEncoder,
    CausalVQVAE, # Placeholder
)

class CausalAttentionBlock(nn.Module):
    """
    A block combining Causal Temporal Attention and Prefix-Enhanced Spatial Attention
    with optional cross-attention, similar to a Transformer layer in a UNet.
    """
    def __init__(self, dim, num_heads, dim_head, context_dim=768, prefix_enhancement_frames=3,
                 use_cross_attention=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = CausalTemporalAttention(dim, heads=num_heads, dim_head=dim_head)
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = PrefixEnhancedSpatialAttention(dim, heads=num_heads, dim_head=dim_head,
                                                    prefix_enhancement_frames=prefix_enhancement_frames)
        
        self.use_cross_attention = use_cross_attention
        if self.use_cross_attention:
            self.norm3 = nn.LayerNorm(dim)
            self.attn3 = nn.MultiheadAttention(dim, num_heads, batch_first=True) # Cross-attention for text
        
        self.norm4 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim)

    def forward(self, x, t_emb: torch.Tensor, text_context=None,
                temporal_kv_cache=None, spatial_kv_cache=None, causal_mask=None):
        # x: (B, L, H*W, C)
        # t_emb: (B, L, C_model) for per-frame modulation
        B, L, HW, C = x.shape
        
        # Reshape t_emb for broadcasting to attention layers
        # t_emb_temporal: (B, L, C_model) -> (B*HW, L, C_model) by repeating
        t_emb_temporal_broadcast = repeat(t_emb, 'b l c_t -> (b hw) l c_t', hw=HW)

        # t_emb_spatial: (B, L, C_model) -> (B*L, C_model)
        t_emb_spatial_flat = rearrange(t_emb, 'b l c_t -> (b l) c_t')


        # Causal Temporal Attention
        identity = x
        x_norm = self.norm1(x)
        x_temporal = rearrange(x_norm, 'b l hw c -> (b hw) l c') # (B*HW, L, C)
        
        # Add t_emb to x_temporal input
        x_temporal = x_temporal + t_emb_temporal_broadcast

        current_temporal_kv_cache_flat = None
        if temporal_kv_cache is not None:
            # temporal_kv_cache: (P_k, HW, C)
            # Need to convert to (B*HW, H, P_k, D) for attention
            num_cached_frames = temporal_kv_cache.shape[0]
            
            # Reshape cached K, V from (P_k, HW, C) to (P_k*HW, H, D_head)
            # This is tricky: CausalTemporalAttention's K,V output is (B_attn, H_heads, N, D_head)
            # If temporal_kv_cache is K, V from a previous step, it should already be processed by the same to_qkv layer.
            # Let's assume temporal_kv_cache is a dict {'k': tensor, 'v': tensor} where tensor is (P_k, HW, C)
            
            # temporal_kv_cache['k'] : (P_k, HW, C)
            cached_k = rearrange(temporal_kv_cache['k'], 'p hw (h d) -> (p hw) h d', h = self.attn1.heads)
            cached_v = rearrange(temporal_kv_cache['v'], 'p hw (h d) -> (p hw) h d', h = self.attn1.heads)
            
            # Then repeat for the current batch of (B*HW) queries
            cached_k = repeat(cached_k, 'n h d -> b h n d', b = B * HW)
            cached_v = repeat(cached_v, 'n h d -> b h n d', b = B * HW)

            current_temporal_kv_cache_flat = (cached_k, cached_v)
            

        x_temporal_out, new_k_temporal, new_v_temporal = self.attn1(x_temporal,
                                                            causal_mask=causal_mask, # (L, P_k+L)
                                                            cache_kv=current_temporal_kv_cache_flat)
        x = identity + rearrange(x_temporal_out, '(b hw) l c -> b l hw c', b=B)

        # Prefix-Enhanced Spatial Attention
        identity = x
        x_norm = self.norm2(x)
        x_spatial = rearrange(x_norm, 'b l hw c -> (b l) hw c') # (B*L, HW, C)
        
        # Add t_emb to x_spatial input
        x_spatial = x_spatial + t_emb_spatial_flat.unsqueeze(1) # Unsqueeze to broadcast across HW tokens

        # spatial_kv_cache: (P', HW, C)
        # This will be passed to PrefixEnhancedSpatialAttention, which will use it for K/V concatenation
        x_spatial_out, new_k_spatial, new_v_spatial = self.attn2(x_spatial,
                                                                 clean_prefix_frames=spatial_kv_cache) # (P', HW, C)
        x = identity + rearrange(x_spatial_out, '(b l) hw c -> b l hw c', b=B)

        # Visual-Text Cross Attention (optional)
        if self.use_cross_attention and text_context is not None:
            identity = x
            x_norm = self.norm3(x)
            
            x_cross_attn_query = rearrange(x_norm, 'b l hw c -> (b l) hw c') # (B*L, HW, C)
            text_context_repeated = repeat(text_context, 'b n d -> (b l) n d', l=L) # (B*L, N_text, C_context)
            
            cross_attn_out, _ = self.attn3(query=x_cross_attn_query,
                                           key=text_context_repeated,
                                           value=text_context_repeated,
                                           need_weights=False)
            
            x = identity + cross_attn_out
            x = rearrange(x, '(b l) hw c -> b l hw c', b=B)

        # Feed Forward
        identity = x
        x = self.norm4(x)
        x = self.ff(x)
        x = identity + x

        # Return new K, V for current chunk (processed by `self.attn1.to_qkv` or similar)
        # These are the keys/values *computed from the current input `x`*
        # `new_k_temporal` and `new_v_temporal` are (B*HW, H, L, D_head)
        # We need them as (L, HW, C) for the next AR step's cache.
        new_k_temporal_rearranged = rearrange(new_k_temporal[:, :, -L:, :], '(b hw) h l d -> l hw (h d)', b=B)
        new_v_temporal_rearranged = rearrange(new_v_temporal[:, :, -L:, :], '(b hw) h l d -> l hw (h d)', b=B)

        # `new_k_spatial` and `new_v_spatial` are (B*L, H, HW, D_head)
        # We need the *last P'* frames for spatial cache from the denoised output `x`.
        # This should come from the *actual frames* in `x`, not the attention outputs.
        # The paper states: "In the cache writing stage, the denoised latent frames are first enhanced via self-repeat and then computed to obtain the clean spatial keys and values."
        # This means the output `x` after the spatial attention has the clean (denoised) features.
        # We should use these features to compute the spatial KV for the next step.
        # This means we need to run a *partial forward* (up to spatial attention KV computation)
        # on the *denoised frames* in the cache writing stage.
        # For now, let's assume `new_k_spatial, new_v_spatial` are from the actual denoised output `x` in the cache writing stage.
        # This requires a separate forward pass or specific handling.
        # For simplicity, during the *denoising pass*, we don't compute the new spatial KV.
        # In the cache writing stage, we would pass the denoised `z_0^{P_k:P_k+l}` to a partial model.

        # For the `cache_writing_stage`, the logic will be different.
        # For the current forward pass (denoising stage), we return the temporal KVs.
        
        return x, {'k': new_k_temporal_rearranged, 'v': new_v_temporal_rearranged}

class Ca2VDM(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.model_config = model_config

        self.time_embed = TimestepEmbedding(model_config.model_channels)
        # TPE should have dim = C_model and max_len = P_max + l
        self.tpe = PositionalEncoding(model_config.model_channels, # Embedding dimension is C_model
                                      max_len=model_config.max_condition_frames + model_config.chunk_length) # Max total length
        
        # UNet-like structure: Encoder, Middle, Decoder
        # Initial convolution
        self.conv_in = nn.Conv3d(model_config.latent_channels, model_config.model_channels, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        num_resolutions = len(model_config.channel_mult)
        channels = [model_config.model_channels * mult for mult in model_config.channel_mult]
        
        # Encoder
        in_channels = model_config.model_channels
        for i in range(num_resolutions):
            out_channels = channels[i]
            for _ in range(model_config.num_res_blocks):
                self.down_blocks.append(nn.ModuleList([
                    ResNetBlock(in_channels, out_channels, model_config.model_channels),
                    CausalAttentionBlock(
                        out_channels, model_config.num_heads, model_config.model_channels // model_config.num_heads,
                        context_dim=model_config.context_dim,
                        prefix_enhancement_frames=model_config.prefix_enhancement_frames,
                        use_cross_attention=True if model_config.context_dim > 0 else False
                    )
                ]))
                in_channels = out_channels
            if i < num_resolutions - 1:
                self.down_blocks.append(nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)) # Downsample

        # Middle block
        self.mid_block1 = ResNetBlock(in_channels, in_channels, model_config.model_channels)
        self.mid_attn = CausalAttentionBlock(
            in_channels, model_config.num_heads, model_config.model_channels // model_config.num_heads,
            context_dim=model_config.context_dim,
            prefix_enhancement_frames=model_config.prefix_enhancement_frames,
            use_cross_attention=True if model_config.context_dim > 0 else False
        )
        self.mid_block2 = ResNetBlock(in_channels, in_channels, model_config.model_channels)

        # Decoder
        for i in reversed(range(num_resolutions)):
            out_channels = channels[i]
            for _ in range(model_config.num_res_blocks + 1): # +1 for skip connection from encoder
                self.up_blocks.append(nn.ModuleList([
                    ResNetBlock(in_channels + out_channels, out_channels, model_config.model_channels), # Skip connection
                    CausalAttentionBlock(
                        out_channels, model_config.num_heads, model_config.model_channels // model_config.num_heads,
                        context_dim=model_config.context_dim,
                        prefix_enhancement_frames=model_config.prefix_enhancement_frames,
                        use_cross_attention=True if model_config.context_dim > 0 else False
                    )
                ]))
                in_channels = out_channels
            if i > 0:
                self.up_blocks.append(nn.ConvTranspose3d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1)) # Upsample

        # Final convolution
        self.norm_out = nn.GroupNorm(32, in_channels)
        self.conv_out = nn.Conv3d(in_channels, model_config.latent_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor,
                clean_prefix_frames_len: torch.Tensor, # (B,) specifying P for each item in batch
                text_context: Optional[torch.Tensor] = None,
                tpe_indices: Optional[torch.Tensor] = None,
                temporal_kv_caches: Optional[List[Dict[str, torch.Tensor]]] = None,
                spatial_kv_caches: Optional[List[torch.Tensor]] = None,
                causal_mask: Optional[torch.Tensor] = None,
                is_cache_writing_stage: bool = False # Flag for cache writing stage
                ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """
        x: (B, L, C_latent, H, W)
        timesteps: (B,) -> This is the `t_val` for the noisy part.
        clean_prefix_frames_len: (B,) -> Number of frames (P) in the clean prefix for each batch item.
        text_context: (B, N_tokens, C_text)
        tpe_indices: (B, L)
        temporal_kv_caches: List of {'k': tensor (P_k, HW, C), 'v': tensor (P_k, HW, C)} for each attention layer
        spatial_kv_caches: List of (P', HW, C) for each attention layer
        causal_mask: (L_current_chunk, P_k + L_current_chunk)
        """
        B, L, C_latent, H, W = x.shape

        # Construct per-frame timestep embeddings (B, L, C_model)
        t_emb_per_frame = torch.zeros(B, L, self.model_config.model_channels, device=x.device)
        
        t_0_emb = self.time_embed(torch.zeros(B, dtype=torch.long, device=x.device)) # (B, C_model)
        
        # Only compute t_val_emb if there are noisy frames (P < L)
        noisy_batch_indices = (clean_prefix_frames_len < L).nonzero(as_tuple=True)[0]
        if noisy_batch_indices.numel() > 0:
            t_vals = timesteps[noisy_batch_indices] # (num_noisy_batches,)
            t_val_emb = self.time_embed(t_vals) # (num_noisy_batches, C_model)

        for b_idx in range(B):
            P_b = clean_prefix_frames_len[b_idx].item()
            t_emb_per_frame[b_idx, :P_b] = t_0_emb[b_idx]
            if P_b < L:
                # Find the index of this batch item in noisy_batch_indices
                idx_in_noisy = (noisy_batch_indices == b_idx).nonzero(as_tuple=True)[0]
                if idx_in_noisy.numel() > 0:
                    t_emb_per_frame[b_idx, P_b:] = t_val_emb[idx_in_noisy]

        # t_emb for ResNet blocks: (B, L, C_model)
        # t_emb for Attention blocks: (B, L, C_model)

        # Convert video to (B, C, L, H, W) for 3D conv and then back for attention
        x_3d = rearrange(x, 'b l c h w -> b c l h w')
        x_3d = self.conv_in(x_3d) # (B, C_model, L, H, W)

        # Store outputs for skip connections
        h_skip = []
        
        current_temporal_kv_cache_idx = 0
        current_spatial_kv_cache_idx = 0

        # Encoder
        for block_pair in self.down_blocks:
            if isinstance(block_pair, nn.Conv3d): # Downsampling
                x_3d = block_pair(x_3d)
            else: # ResNet and Attention Block
                resnet_block, attn_block = block_pair
                x_3d = resnet_block(x_3d, t_emb_per_frame) # Pass per-frame temb
                h_skip.append(x_3d)
                
                # Reshape for attention block: (B, L, C, H, W) -> (B, L, H*W, C)
                x_attn_in = rearrange(x_3d, 'b c l h w -> b l (h w) c')

                # Apply TPE (Cyclic-TPEs)
                # The tpe is applied on the full sequence of (B, L, HW*C)
                # We need to compute TPE *for the current L frames* based on `tpe_indices`.
                # Then reshape and add.
                if tpe_indices is not None:
                    # Apply TPE. `x_attn_in` is (B, L, HW, C). `tpe_indices` is (B, L).
                    # `self.tpe` expects `x` as (B, N, C) for direct addition.
                    # Here, `N` corresponds to `L`. So `x_attn_in` becomes `(B, L, C)` for TPE.
                    # We need to create a dummy `x` of shape `(B, L, C)` to pass to `self.tpe`.
                    pe_values = self.tpe(torch.zeros_like(x_attn_in[:,:,0,:]), tpe_indices) # (B, L, C_model)
                    x_attn_in = x_attn_in + pe_values.unsqueeze(2) # (B, L, 1, C_model) broadcast over HW


                # Prepare KV caches for the current attention block
                current_temp_cache = temporal_kv_caches[current_temporal_kv_cache_idx] if temporal_kv_caches else None
                current_spatial_cache = spatial_kv_caches[current_spatial_kv_cache_idx] if spatial_kv_caches else None

                # Pass through Attention Block
                x_attn_out, new_temporal_kv = attn_block(x_attn_in, t_emb_per_frame, text_context,
                                                      temporal_kv_cache=current_temp_cache,
                                                      spatial_kv_cache=current_spatial_cache,
                                                      causal_mask=causal_mask)
                
                # Update KV cache index for next layer
                current_temporal_kv_cache_idx += 1
                if spatial_kv_caches: # Spatial cache only exists if prefix enhancement is enabled
                    current_spatial_kv_cache_idx += 1

                x_3d = rearrange(x_attn_out, 'b l (h w) c -> b c l h w', h=H, w=W) # Back to (B, C, L, H, W)
        
        # Middle Block
        x_3d = self.mid_block1(x_3d, t_emb_per_frame)
        x_attn_in = rearrange(x_3d, 'b c l h w -> b l (h w) c')
        
        if tpe_indices is not None:
            pe_values = self.tpe(torch.zeros_like(x_attn_in[:,:,0,:]), tpe_indices) # (B, L, C_model)
            x_attn_in = x_attn_in + pe_values.unsqueeze(2) # (B, L, 1, C_model) broadcast over HW

        current_temp_cache = temporal_kv_caches[current_temporal_kv_cache_idx] if temporal_kv_caches else None
        current_spatial_cache = spatial_kv_caches[current_spatial_kv_cache_idx] if spatial_kv_caches else None
        
        x_attn_out, new_temporal_kv = self.mid_attn(x_attn_in, t_emb_per_frame, text_context,
                                                  temporal_kv_cache=current_temp_cache,
                                                  spatial_kv_cache=current_spatial_cache,
                                                  causal_mask=causal_mask)
        current_temporal_kv_cache_idx += 1
        if spatial_kv_caches:
            current_spatial_kv_cache_idx += 1
        
        x_3d = rearrange(x_attn_out, 'b l (h w) c -> b c l h w', h=H, w=W)
        x_3d = self.mid_block2(x_3d, t_emb)


        # Decoder
        for block_pair in self.up_blocks:
            if isinstance(block_pair, nn.ConvTranspose3d): # Upsampling
                x_3d = block_pair(x_3d)
            else: # ResNet and Attention Block
                resnet_block, attn_block = block_pair
                # Pop from skip connections
                skip_x = h_skip.pop()
                x_3d = torch.cat((x_3d, skip_x), dim=1) # Concatenate along channel dim
                x_3d = resnet_block(x_3d, t_emb_per_frame) # Pass per-frame temb
                
                x_attn_in = rearrange(x_3d, 'b c l h w -> b l (h w) c')

                if tpe_indices is not None:
                    x_attn_in = self.tpe(x_attn_in.flatten(2,3), tpe_indices).view(B, L, HW, C)

                current_temp_cache = temporal_kv_caches[current_temporal_kv_cache_idx] if temporal_kv_caches else None
                current_spatial_cache = spatial_kv_caches[current_spatial_kv_cache_idx] if spatial_kv_caches else None

                x_attn_out, new_temporal_kv = attn_block(x_attn_in, t_emb_per_frame, text_context, # Pass per-frame temb
                                                      temporal_kv_cache=current_temp_cache,
                                                      spatial_kv_cache=current_spatial_cache,
                                                      causal_mask=causal_mask)
                current_temporal_kv_cache_idx += 1
                if spatial_kv_caches:
                    current_spatial_kv_cache_idx += 1

                x_3d = rearrange(x_attn_out, 'b l (h w) c -> b c l h w', h=H, w=W)
        
        x_3d = F.silu(self.norm_out(x_3d))
        x_3d = self.conv_out(x_3d) # (B, C_latent, L, H, W)
        
        # Revert to (B, L, C_latent, H, W)
        output_noise_pred = rearrange(x_3d, 'b c l h w -> b l c h w')

        # When `is_cache_writing_stage` is False (denoising stage), we return the noise prediction.
        # The new_temporal_kv is not collected in this path.
        # It's only collected during the separate `get_kv_caches` call.
        
        return output_noise_pred, [] # Returning empty list for new_temporal_kv for now (as it's for get_kv_caches)

    def get_kv_caches(self, x: torch.Tensor, timesteps: torch.Tensor, # timesteps should be (B,) and all 0s
                      text_context: Optional[torch.Tensor] = None,
                      tpe_indices: Optional[torch.Tensor] = None
                      ) -> Tuple[List[Dict[str, torch.Tensor]], Optional[torch.Tensor]]:
        """
        Performs a partial forward pass on denoised frames (t=0) to compute
        temporal and spatial KV-caches for all attention layers.
        This is used in the "cache writing stage".
        """
        B, L, C_latent, H, W = x.shape
        # For cache writing stage, all frames are clean (t=0)
        t_emb_per_frame = self.time_embed(torch.zeros(B, L, dtype=torch.long, device=x.device)).view(B, L, -1) # (B, L, C_model)

        x_3d = rearrange(x, 'b l c h w -> b c l h w')
        x_3d = self.conv_in(x_3d)

        temp_kv_caches_out = []
        
        # Spatial KV cache is not collected per-layer, but as the features of the last P' frames.
        # It's an output of the *entire* cache writing pass.
        # We will collect the last `P_prime` frames from the input `x` *at the end*.

        h_skip = []

        for block_pair in self.down_blocks:
            if isinstance(block_pair, nn.Conv3d):
                x_3d = block_pair(x_3d)
            else:
                resnet_block, attn_block = block_pair
                x_3d = resnet_block(x_3d, t_emb_per_frame)
                h_skip.append(x_3d)
                
                x_attn_in = rearrange(x_3d, 'b c l h w -> b l (h w) c')

                if tpe_indices is not None:
                    x_attn_in = self.tpe(x_attn_in.flatten(2,3), tpe_indices).view(B, L, HW, C)

                x_attn_out, new_temporal_kv = attn_block(x_attn_in, t_emb_per_frame, text_context,
                                                      temporal_kv_cache=None,
                                                      spatial_kv_cache=None,
                                                      causal_mask=None) # Standard causal mask for this L frames
                
                temp_kv_caches_out.append(new_temporal_kv)
        
        # Middle Block
        x_3d = self.mid_block1(x_3d, t_emb_per_frame)
        x_attn_in = rearrange(x_3d, 'b c l h w -> b l (h w) c')
        
        if tpe_indices is not None:
            x_attn_in = self.tpe(x_attn_in.flatten(2,3), tpe_indices).view(B, L, HW, C)

        x_attn_out, new_temporal_kv = self.mid_attn(x_attn_in, t_emb_per_frame, text_context,
                                                  temporal_kv_cache=None,
                                                  spatial_kv_cache=None,
                                                  causal_mask=None)
        temp_kv_caches_out.append(new_temporal_kv)
        x_3d = rearrange(x_attn_out, 'b l (h w) c -> b c l h w', h=H, w=W)
        x_3d = self.mid_block2(x_3d, t_emb_per_frame)


        # Decoder (similar logic, but also handles skip connections)
        for block_pair in self.up_blocks:
            if isinstance(block_pair, nn.ConvTranspose3d):
                x_3d = block_pair(x_3d)
            else:
                resnet_block, attn_block = block_pair
                skip_x = h_skip.pop()
                x_3d = torch.cat((x_3d, skip_x), dim=1)
                x_3d = resnet_block(x_3d, t_emb_per_frame)

                x_attn_in = rearrange(x_3d, 'b c l h w -> b l (h w) c')

                if tpe_indices is not None:
                    x_attn_in = self.tpe(x_attn_in.flatten(2,3), tpe_indices).view(B, L, HW, C)

                x_attn_out, new_temporal_kv = attn_block(x_attn_in, t_emb_per_frame, text_context,
                                                      temporal_kv_cache=None,
                                                      spatial_kv_cache=None,
                                                      causal_mask=None)
                temp_kv_caches_out.append(new_temporal_kv)
                x_3d = rearrange(x_attn_out, 'b l (h w) c -> b c l h w', h=H, w=W)
        
        spatial_kv_features_out = None
        if self.model_config.prefix_enhancement_frames > 0:
            # We want the last P' frames from the *input* `x` (which are the clean denoised frames).
            # This is the `P'` features of shape (P', C_latent, H, W)
            # Input `x` is (B, L, C_latent, H, W)
            # We need the last `P_prime` frames from each batch item.
            # If `L` (current chunk length) is less than `P_prime`, we use `L`.
            frames_for_spatial_cache = min(L, self.model_config.prefix_enhancement_frames)
            
            # This needs to handle a batch.
            # The spatial KV cache is a single tensor (P', HW, C) as per paper: "only store the spatial KV-cache for one chunk"
            # So, this should be taken from `x` of batch item 0 (or aggregated).
            # For simplicity, during inference, B=1.
            # For training (cache writing stage), we are computing KVs for the current L frames.
            # The spatial KV cache is the *features* of the last P' frames of the *denoised video*.
            
            # The paper says: "In the cache writing stage, the denoised latent frames are first enhanced via self-repeat and then
            # computed to obtain the clean spatial keys and values."
            # The `get_kv_caches` function is specifically for cache writing.
            # So `x` here is `z_0^{P_k:P_k+l}` (the denoised chunk).
            # We need to extract its last `P_prime` frames.
            
            spatial_kv_features_out = x[0, -frames_for_spatial_cache:, :, :, :] # (frames_for_spatial_cache, C_latent, H, W)
            spatial_kv_features_out = rearrange(spatial_kv_features_out, 'p c h w -> p (h w) c') # (frames_for_spatial_cache, HW, C_latent)

        return temp_kv_caches_out, spatial_kv_features_out

