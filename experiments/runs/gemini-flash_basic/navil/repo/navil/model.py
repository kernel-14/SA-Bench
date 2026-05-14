
# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from navil.config import NaViLConfig
from navil.utils import RotaryEmbedding1D, RotaryEmbedding2D, PixelShuffle

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (batch_size, C, H, W)
        x = self.proj(x) # (batch_size, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2) # (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2) # (batch_size, num_patches, embed_dim)
        return x

class ModalitySpecificAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, is_visual_expert=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.is_visual_expert = is_visual_expert
        # RoPE will be applied externally or integrated into query/key projection

    def _shape(self, tensor, seq_len, bsz):
        return tensor.view(bsz, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states, attention_mask=None, rope=None):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states) * self.scaling
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = self._shape(query_states, q_len, bsz)
        key_states = self._shape(key_states, q_len, bsz)
        value_states = self._shape(value_states, q_len, bsz)

        if rope is not None:
            # Assuming rope is a function that applies RoPE to query and key
            # For now, this is a placeholder and real RoPE would modify query_states and key_states
            query_states = rope(query_states)
            key_states = rope(key_states)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output


class ModalitySpecificFFN(nn.Module):
    def __init__(self, hidden_size, intermediate_size, is_visual_expert=False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoELayer(nn.Module):
    def __init__(self, config: NaViLConfig.ModelConfig, is_llm_layer: bool):
        super().__init__()
        self.is_llm_layer = is_llm_layer

        self.visual_attention_expert = ModalitySpecificAttention(
            config.llm_hidden_size, config.llm_num_attention_heads, is_visual_expert=True
        ) if config.moe_enabled else None

        self.linguistic_attention_expert = ModalitySpecificAttention(
            config.llm_hidden_size, config.llm_num_attention_heads, is_visual_expert=False
        ) if is_llm_layer else None

        self.visual_ffn_expert = ModalitySpecificFFN(
            config.llm_hidden_size, config.llm_intermediate_size, is_visual_expert=True
        ) if config.moe_enabled else None

        self.linguistic_ffn_expert = ModalitySpecificFFN(
            config.llm_hidden_size, config.llm_intermediate_size, is_visual_expert=False
        ) if is_llm_layer else None

        self.norm1 = nn.LayerNorm(config.llm_hidden_size)
        self.norm2 = nn.LayerNorm(config.llm_hidden_size)

    def forward(self, hidden_states, attention_mask=None, visual_mask=None, rope_1d=None, rope_2d=None):
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)

        attn_output = torch.zeros_like(hidden_states)

        # Simplified MoE routing: apply experts based on visual_mask
        # A proper MoE would have a gating network to select and combine expert outputs
        if visual_mask is not None and self.visual_attention_expert and self.linguistic_attention_expert:
            # Apply visual expert to visual tokens
            if visual_mask.any():
                attn_output[visual_mask] = self.visual_attention_expert(hidden_states[visual_mask], attention_mask, rope_2d)
            # Apply linguistic expert to linguistic tokens
            if (~visual_mask).any():
                attn_output[~visual_mask] = self.linguistic_attention_expert(hidden_states[~visual_mask], attention_mask, rope_1d)
        elif self.linguistic_attention_expert: # Pure linguistic processing
            attn_output = self.linguistic_attention_expert(hidden_states, attention_mask, rope_1d)
        elif self.visual_attention_expert: # Pure visual processing
             attn_output = self.visual_attention_expert(hidden_states, attention_mask, rope_2d)
        else:
            attn_output = hidden_states # No MoE or experts, just pass through (should not happen if moe_enabled is True)

        hidden_states = residual + attn_output

        # FFN layer
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)

        ffn_output = torch.zeros_like(hidden_states)
        if visual_mask is not None and self.visual_ffn_expert and self.linguistic_ffn_expert:
            if visual_mask.any():
                ffn_output[visual_mask] = self.visual_ffn_expert(hidden_states[visual_mask])
            if (~visual_mask).any():
                ffn_output[~visual_mask] = self.linguistic_ffn_expert(hidden_states[~visual_mask])
        elif self.linguistic_ffn_expert:
            ffn_output = self.linguistic_ffn_expert(hidden_states)
        elif self.visual_ffn_expert:
            ffn_output = self.visual_ffn_expert(hidden_states)
        else:
            ffn_output = hidden_states

        hidden_states = residual + ffn_output

        return hidden_states


class VisualEncoderLayer(nn.Module):
    def __init__(self, config: NaViLConfig.ModelConfig):
        super().__init__()
        self.attn = ModalitySpecificAttention(config.visual_encoder_width, config.llm_num_attention_heads, is_visual_expert=True)
        self.ffn = ModalitySpecificFFN(config.visual_encoder_width, config.llm_intermediate_size, is_visual_expert=True)
        self.norm1 = nn.LayerNorm(config.visual_encoder_width)
        self.norm2 = nn.LayerNorm(config.visual_encoder_width)

    def forward(self, hidden_states, attention_mask=None, rope_2d=None):
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attn(hidden_states, attention_mask, rope=rope_2d)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class VisualEncoder(nn.Module):
    def __init__(self, config: NaViLConfig.ModelConfig):
        super().__init__()
        self.patch_embedding = PatchEmbedding(
            config.visual_patch_size, in_channels=3, embed_dim=config.visual_encoder_width
        )
        self.layers = nn.ModuleList([
            VisualEncoderLayer(config) for _ in range(config.visual_encoder_depth)
        ])
        self.norm = nn.LayerNorm(config.visual_encoder_width)
        # RoPE for visual encoder is applied per layer
        self.rope_2d = RotaryEmbedding2D(config.visual_encoder_width // config.llm_num_attention_heads, config.visual_patch_size, config.visual_image_size)

    def forward(self, pixel_values, attention_mask=None):
        hidden_states = self.patch_embedding(pixel_values)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, rope_2d=self.rope_2d)

        return self.norm(hidden_states)


class Connector(nn.Module):
    def __init__(self, config: NaViLConfig.ModelConfig):
        super().__init__()
        # C is the connector which downsamples the encoded image embeddings through pixel shuffle [15]
        # and projects them to the LLM’s feature space by a MLP.

        # Assuming the visual encoder output width might be different from LLM hidden size
        # The paper implies pixel shuffle happens before MLP projection.
        # The exact upscale_factor for pixel shuffle from a Transformer output is not clearly specified.
        # For now, we will simply use an MLP to project from visual_encoder_width to llm_hidden_size.
        # If pixel shuffle is truly meant, the visual_encoder_width would need to be a multiple of (upscale_factor^2)
        # and the input to MLP would be (H*W*C_after_pixel_shuffle).
        self.mlp = nn.Linear(config.visual_encoder_width, config.llm_hidden_size)

    def forward(self, visual_features):
        # visual_features: (batch_size, num_patches, visual_encoder_width)
        # Simplified: just an MLP projection
        return self.mlp(visual_features)


class LLMModule(nn.Module):
    def __init__(self, config: NaViLConfig.ModelConfig):
        super().__init__()
        # Placeholder for a pre-trained LLM (e.g., InternLM2-Base).
        # This will contain multiple transformer layers, each with MoE if enabled.

        self.embedding = nn.Embedding(32000, config.llm_hidden_size) # Example vocab size

        self.layers = nn.ModuleList([
            MoELayer(config, is_llm_layer=True) for _ in range(config.llm_num_hidden_layers)
        ])
        self.norm = nn.LayerNorm(config.llm_hidden_size)
        self.output_proj = nn.Linear(config.llm_hidden_size, 32000) # Example vocab size

        self.rope_1d = RotaryEmbedding1D(config.llm_hidden_size // config.llm_num_attention_heads)

    def forward(self, input_ids, visual_features, attention_mask=None, visual_mask=None):
        text_embeddings = self.embedding(input_ids)

        # Concatenate visual and text embeddings
        # The paper states: "These images are first encoded into visual tokens by the visual encoder and the MLP projector,
        # and then concatenated with the textual tokens to formulate the multimodal token sequence and fed into the LLM."

        # Combined sequence for LLM processing
        hidden_states = torch.cat((visual_features, text_embeddings), dim=1)

        if visual_mask is None:
            # Create a visual mask for the concatenated sequence
            visual_token_count = visual_features.shape[1]
            text_token_count = text_embeddings.shape[1]
            combined_visual_mask = torch.cat([
                torch.ones(visual_token_count, dtype=torch.bool, device=hidden_states.device),
                torch.zeros(text_token_count, dtype=torch.bool, device=hidden_states.device)
            ]).unsqueeze(0).expand(hidden_states.shape[0], -1)
        else:
            combined_visual_mask = visual_mask # If already provided for the combined sequence

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, visual_mask=combined_visual_mask, rope_1d=self.rope_1d)

        hidden_states = self.norm(hidden_states)
        logits = self.output_proj(hidden_states)
        return logits


class NaViL(nn.Module):
    def __init__(self, config: NaViLConfig.ModelConfig):
        super().__init__()
        self.config = config

        self.visual_encoder = VisualEncoder(config)
        self.connector = Connector(config)
        self.llm = LLMModule(config)

    def forward_one_scale(self, pixel_values, input_ids, attention_mask=None, visual_mask=None):
        visual_features_raw = self.visual_encoder(pixel_values)
        visual_features_projected = self.connector(visual_features_raw)

        logits = self.llm(input_ids, visual_features_projected, attention_mask, visual_mask)
        return logits

    def forward(self, pixel_values_list, input_ids, attention_mask=None, visual_mask=None):
        all_visual_features_projected = []
        # The paper mentions <end_of_scale> token, which would be handled by the tokenizer
        # This current implementation just concatenates the visual features.
        for pixel_values in pixel_values_list:
            visual_features_raw = self.visual_encoder(pixel_values)
            visual_features_projected = self.connector(visual_features_raw)
            all_visual_features_projected.append(visual_features_projected)

        combined_visual_features = torch.cat(all_visual_features_projected, dim=1)

        logits = self.llm(input_ids, combined_visual_features, attention_mask, visual_mask)
        return logits

