"""
NaViL: Native Multimodal Large Language Model

Architecture:
- Visual Encoder V_{d,w}: bidirectional transformer with 2D-RoPE
- Connector C: pixel shuffle + MLP projection
- MoE-extended LLM: LLM with modality-specific MHA-MMoE and FFN-MMoE layers
- Visual Multi-scale Packing for any-resolution input

Special tokens:
- <begin_of_image>: marks start of image token subsequence
- <end_of_image>: marks end of image token subsequence
- <end_of_line>: marks end of each row of image tokens (spatial position)
- <end_of_scale>: marks end of each scale in multi-scale packing
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .visual_encoder import VisualEncoder, VisualEncoderConfig
from .moe import ModalitySpecificMoE, MMoEConfig, VISUAL_MODALITY, LINGUISTIC_MODALITY
from .norm import get_rms_norm as RMSNorm


@dataclass
class NaViLConfig:
    """Configuration for NaViL model."""

    # LLM backbone config (InternLM2-1.8B defaults)
    llm_hidden_size: int = 2048
    llm_num_layers: int = 24
    llm_num_heads: int = 16
    llm_num_kv_heads: int = 8          # GQA
    llm_head_dim: int = 128
    llm_intermediate_size: int = 8192
    llm_vocab_size: int = 92544        # InternLM2 vocab size
    llm_max_position_embeddings: int = 32768
    llm_rope_theta: float = 1000000.0
    llm_norm_eps: float = 1e-5

    # Visual encoder config
    visual_encoder_depth: int = 24
    visual_encoder_width: int = 1472
    visual_encoder_num_heads: int = 16
    visual_encoder_patch_size: int = 16
    visual_encoder_pixel_shuffle_factor: int = 2

    # MoE config - which layers use MMoE
    # By default, all LLM layers use MMoE
    use_moe: bool = True
    moe_layer_indices: Optional[List[int]] = None  # None = all layers

    # Multi-scale packing
    use_multiscale: bool = True
    multiscale_downsample_rate: float = 0.5 * math.sqrt(2)  # tau = sqrt(2)/2
    multiscale_min_area: int = 32 * 32  # minimum area threshold

    # Special token IDs (will be set from tokenizer)
    begin_of_image_token_id: int = -1
    end_of_image_token_id: int = -1
    end_of_line_token_id: int = -1
    end_of_scale_token_id: int = -1

    # Training
    pad_token_id: int = 0
    ignore_index: int = -100

    # Model variant
    model_name: str = "NaViL-2B"

    @classmethod
    def navil_2b(cls) -> "NaViLConfig":
        """NaViL-2B config: InternLM2-1.8B + 600M visual encoder."""
        return cls(
            llm_hidden_size=2048,
            llm_num_layers=24,
            llm_num_heads=16,
            llm_num_kv_heads=8,
            llm_head_dim=128,
            llm_intermediate_size=8192,
            llm_vocab_size=92544,
            visual_encoder_depth=24,
            visual_encoder_width=1472,
            visual_encoder_num_heads=16,
            model_name="NaViL-2B",
        )

    @classmethod
    def navil_9b(cls) -> "NaViLConfig":
        """NaViL-9B config: Qwen3-8B + 1.2B visual encoder."""
        return cls(
            llm_hidden_size=4096,
            llm_num_layers=36,
            llm_num_heads=32,
            llm_num_kv_heads=8,
            llm_head_dim=128,
            llm_intermediate_size=22016,
            llm_vocab_size=151936,  # Qwen3 vocab size
            visual_encoder_depth=28,
            visual_encoder_width=1792,
            visual_encoder_num_heads=16,
            model_name="NaViL-9B",
        )


class RoPE1D(nn.Module):
    """1D Rotary Position Embedding for LLM (causal attention)."""

    def __init__(self, head_dim: int, max_seq_len: int = 32768, theta: float = 1000000.0):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta

        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: (B, num_heads, seq_len, head_dim)
            k: (B, num_kv_heads, seq_len, head_dim)
            position_ids: (B, seq_len)
        """
        # position_ids: (B, seq_len) -> (B, seq_len, head_dim/2)
        inv_freq = self.inv_freq.unsqueeze(0).unsqueeze(0)  # (1, 1, head_dim/2)
        pos = position_ids.unsqueeze(-1).float()  # (B, seq_len, 1)
        freqs = pos * inv_freq  # (B, seq_len, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (B, seq_len, head_dim)

        cos = emb.cos().unsqueeze(1)  # (B, 1, seq_len, head_dim)
        sin = emb.sin().unsqueeze(1)

        def rotate_half(x):
            x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
            return torch.cat([-x2, x1], dim=-1)

        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        return q, k


class LLMAttention(nn.Module):
    """
    Causal multi-head attention for LLM with GQA support.
    Uses 1D-RoPE.
    """

    def __init__(self, config: NaViLConfig):
        super().__init__()
        self.num_heads = config.llm_num_heads
        self.num_kv_heads = config.llm_num_kv_heads
        self.head_dim = config.llm_head_dim
        self.hidden_size = config.llm_hidden_size
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.llm_hidden_size, config.llm_num_heads * config.llm_head_dim, bias=False)
        self.k_proj = nn.Linear(config.llm_hidden_size, config.llm_num_kv_heads * config.llm_head_dim, bias=False)
        self.v_proj = nn.Linear(config.llm_hidden_size, config.llm_num_kv_heads * config.llm_head_dim, bias=False)
        self.o_proj = nn.Linear(config.llm_num_heads * config.llm_head_dim, config.llm_hidden_size, bias=False)

        self.rope = RoPE1D(config.llm_head_dim, theta=config.llm_rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, seq_len, _ = x.shape

        q = self.q_proj(x).reshape(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(B, -1)

        q, k = self.rope(q, k, position_ids)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        new_pkv = (k, v)

        # GQA: repeat KV heads
        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        kv_len = k.shape[2]
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).reshape(B, seq_len, self.num_heads * self.head_dim)
        out = self.o_proj(out)
        return out, new_pkv


class LLMFFN(nn.Module):
    """SwiGLU FFN for LLM."""

    def __init__(self, config: NaViLConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.llm_hidden_size, config.llm_intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.llm_hidden_size, config.llm_intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.llm_intermediate_size, config.llm_hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class NaViLLLMLayer(nn.Module):
    """
    Single LLM layer in NaViL.

    If use_moe=True, uses ModalitySpecificMoE (MHA-MMoE + FFN-MMoE).
    Otherwise, uses standard LLM attention + FFN.
    """

    def __init__(self, config: NaViLConfig, use_moe: bool = True):
        super().__init__()
        self.use_moe = use_moe
        self.hidden_size = config.llm_hidden_size

        if use_moe:
            moe_config = MMoEConfig(
                hidden_size=config.llm_hidden_size,
                num_heads=config.llm_num_heads,
                head_dim=config.llm_head_dim,
                intermediate_size=config.llm_intermediate_size,
                num_modalities=2,
                norm_eps=config.llm_norm_eps,
            )
            self.moe = ModalitySpecificMoE(moe_config)
            # Store rope for use in moe
            self.rope = RoPE1D(config.llm_head_dim, theta=config.llm_rope_theta)
        else:
            self.norm1 = RMSNorm(config.llm_hidden_size, eps=config.llm_norm_eps)
            self.attn = LLMAttention(config)
            self.norm2 = RMSNorm(config.llm_hidden_size, eps=config.llm_norm_eps)
            self.ffn = LLMFFN(config)

    def forward(
        self,
        x: torch.Tensor,
        modality_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        if self.use_moe:
            assert modality_ids is not None, "modality_ids required for MoE layers"

            def rope_fn(q, k, pos_ids):
                return self.rope(q, k, pos_ids)

            return self.moe(
                x,
                modality_ids=modality_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                rope_fn=rope_fn,
            )
        else:
            residual = x
            x_norm = self.norm1(x)
            attn_out, new_pkv = self.attn(
                x_norm,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
            )
            x = residual + attn_out

            residual = x
            x = residual + self.ffn(self.norm2(x))
            return x, new_pkv


class VisualMultiScalePacking(nn.Module):
    """
    Visual Multi-scale Packing for any-resolution input.

    Given input image I_0 ∈ R^{H_0 × W_0 × 3} and downsampling rate τ,
    produces multi-scale sequence {I_i} where:
      H_i = τ^i * H_0, W_i = τ^i * W_0
    until area < threshold.

    τ = sqrt(2)/2 ≈ 0.707 (set in paper)

    Special tokens inserted:
    - <end_of_line> after each row of image tokens
    - <end_of_scale> after each scale
    """

    def __init__(self, config: NaViLConfig):
        super().__init__()
        self.tau = config.multiscale_downsample_rate
        self.min_area = config.multiscale_min_area
        self.patch_size = config.visual_encoder_patch_size
        self.pixel_shuffle_factor = config.visual_encoder_pixel_shuffle_factor

    def get_scale_sizes(self, H: int, W: int) -> List[Tuple[int, int]]:
        """
        Get list of (H_i, W_i) for each scale.
        Sizes are rounded to multiples of patch_size.
        """
        sizes = [(H, W)]
        h, w = H, W
        while True:
            h_new = int(h * self.tau)
            w_new = int(w * self.tau)
            # Round to multiples of patch_size
            h_new = max(self.patch_size, (h_new // self.patch_size) * self.patch_size)
            w_new = max(self.patch_size, (w_new // self.patch_size) * self.patch_size)
            if h_new * w_new < self.min_area:
                break
            if h_new == h and w_new == w:
                break
            sizes.append((h_new, w_new))
            h, w = h_new, w_new
        return sizes

    def forward(
        self,
        image: torch.Tensor,
        visual_encoder: "VisualEncoder",
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """
        Args:
            image: (1, C, H, W) single image
            visual_encoder: the visual encoder module
        Returns:
            all_tokens: (1, total_tokens, llm_hidden_size)
            grid_sizes: list of (H_i', W_i') after connector for each scale
        """
        _, C, H, W = image.shape
        scale_sizes = self.get_scale_sizes(H, W)

        all_tokens = []
        grid_sizes = []

        for h_i, w_i in scale_sizes:
            # Resize image to this scale
            img_i = F.interpolate(
                image,
                size=(h_i, w_i),
                mode="bilinear",
                align_corners=False,
            )
            # Encode
            tokens_i, grid_i = visual_encoder(img_i)
            all_tokens.append(tokens_i)
            grid_sizes.append(grid_i)

        # Concatenate all scales
        all_tokens = torch.cat(all_tokens, dim=1)  # (1, sum_tokens, llm_hidden_size)
        return all_tokens, grid_sizes


class NaViLModel(nn.Module):
    """
    NaViL: Native Multimodal Large Language Model.

    Architecture:
    1. Visual Encoder: bidirectional transformer with 2D-RoPE
    2. Connector: pixel shuffle + MLP
    3. MoE-extended LLM: causal transformer with modality-specific experts

    Training stages:
    - Stage 1a: Freeze text params, train visual encoder + connector + MoE visual experts
                on 300M web-scale image-text pairs
    - Stage 1b: Unfreeze text attention params, train on 185M high-quality data
    - Stage 2: Unfreeze all params, SFT on 68M high-quality multimodal data
    """

    def __init__(self, config: NaViLConfig):
        super().__init__()
        self.config = config

        # Visual encoder
        ve_config = VisualEncoderConfig(
            depth=config.visual_encoder_depth,
            width=config.visual_encoder_width,
            num_heads=config.visual_encoder_num_heads,
            patch_size=config.visual_encoder_patch_size,
            pixel_shuffle_factor=config.visual_encoder_pixel_shuffle_factor,
            llm_hidden_size=config.llm_hidden_size,
        )
        self.visual_encoder = VisualEncoder(ve_config)

        # Multi-scale packing
        self.multiscale_packing = VisualMultiScalePacking(config)

        # LLM components
        self.embed_tokens = nn.Embedding(config.llm_vocab_size, config.llm_hidden_size)
        self.norm = RMSNorm(config.llm_hidden_size, eps=config.llm_norm_eps)
        self.lm_head = nn.Linear(config.llm_hidden_size, config.llm_vocab_size, bias=False)

        # Determine which layers use MoE
        if config.use_moe:
            if config.moe_layer_indices is None:
                moe_layers = set(range(config.llm_num_layers))
            else:
                moe_layers = set(config.moe_layer_indices)
        else:
            moe_layers = set()

        # LLM layers
        self.layers = nn.ModuleList([
            NaViLLLMLayer(config, use_moe=(i in moe_layers))
            for i in range(config.llm_num_layers)
        ])

    def get_visual_tokens(
        self,
        images: torch.Tensor,
        use_multiscale: bool = True,
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """
        Encode images to visual tokens.

        Args:
            images: (B, C, H, W) or list of images with different sizes
            use_multiscale: whether to use multi-scale packing
        Returns:
            visual_tokens: (B, N_v, llm_hidden_size)
            grid_sizes: list of grid sizes
        """
        if use_multiscale and self.config.use_multiscale:
            # Process each image with multi-scale packing
            all_tokens = []
            all_grids = []
            for i in range(images.shape[0]):
                tokens_i, grids_i = self.multiscale_packing(
                    images[i:i+1], self.visual_encoder
                )
                all_tokens.append(tokens_i)
                all_grids.append(grids_i)
            # Note: different images may have different numbers of tokens
            # In practice, padding is needed for batching
            return all_tokens, all_grids
        else:
            tokens, grid_size = self.visual_encoder(images)
            return tokens, [grid_size] * images.shape[0]

    def build_multimodal_sequence(
        self,
        input_ids: torch.Tensor,
        visual_tokens: torch.Tensor,
        visual_token_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build multimodal token sequence by replacing image placeholder tokens
        with actual visual tokens.

        Args:
            input_ids: (B, seq_len) text token IDs with image placeholders
            visual_tokens: (B, N_v, llm_hidden_size) visual tokens
            visual_token_mask: (B, seq_len) boolean mask for image token positions
        Returns:
            embeddings: (B, seq_len, llm_hidden_size)
            modality_ids: (B, seq_len) 0=visual, 1=linguistic
        """
        B, seq_len = input_ids.shape

        # Get text embeddings
        text_embeds = self.embed_tokens(input_ids)  # (B, seq_len, hidden_size)

        # Replace visual token positions with visual tokens
        embeddings = text_embeds.clone()
        modality_ids = torch.ones(B, seq_len, dtype=torch.long, device=input_ids.device)

        for b in range(B):
            vis_positions = visual_token_mask[b].nonzero(as_tuple=True)[0]
            n_vis = vis_positions.shape[0]
            if n_vis > 0 and visual_tokens is not None:
                n_available = min(n_vis, visual_tokens.shape[1])
                embeddings[b, vis_positions[:n_available]] = visual_tokens[b, :n_available]
                modality_ids[b, vis_positions[:n_available]] = VISUAL_MODALITY

        return embeddings, modality_ids

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        visual_token_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_multiscale: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: (B, seq_len) token IDs
            images: (B, C, H, W) input images
            visual_token_mask: (B, seq_len) mask for image token positions
            attention_mask: (B, seq_len) padding mask
            position_ids: (B, seq_len) position IDs
            labels: (B, seq_len) target token IDs for language modeling loss
            past_key_values: KV cache for inference
            inputs_embeds: pre-computed embeddings (alternative to input_ids)
            use_multiscale: whether to use multi-scale packing
        Returns:
            dict with 'loss', 'logits', 'past_key_values'
        """
        # Encode images if provided
        visual_tokens = None
        if images is not None:
            if use_multiscale and self.config.use_multiscale:
                # For simplicity in training, process without multi-scale
                # Multi-scale is primarily used at inference
                visual_tokens, _ = self.visual_encoder(images)
            else:
                visual_tokens, _ = self.visual_encoder(images)

        # Build input embeddings
        if inputs_embeds is None:
            if visual_tokens is not None and visual_token_mask is not None:
                inputs_embeds, modality_ids = self.build_multimodal_sequence(
                    input_ids, visual_tokens, visual_token_mask
                )
            else:
                inputs_embeds = self.embed_tokens(input_ids)
                modality_ids = torch.ones(
                    input_ids.shape, dtype=torch.long, device=input_ids.device
                )
        else:
            B, seq_len, _ = inputs_embeds.shape
            modality_ids = torch.ones(B, seq_len, dtype=torch.long, device=inputs_embeds.device)

        B, seq_len, _ = inputs_embeds.shape

        # Build causal attention mask
        if attention_mask is None:
            attention_mask = torch.ones(B, seq_len, device=inputs_embeds.device)

        # Create 4D causal mask
        causal_mask = self._make_causal_mask(
            seq_len, inputs_embeds.dtype, inputs_embeds.device,
            past_key_values_length=past_key_values[0][0].shape[2] if past_key_values else 0,
        )

        # Position IDs
        if position_ids is None:
            past_len = past_key_values[0][0].shape[2] if past_key_values else 0
            position_ids = torch.arange(
                past_len, past_len + seq_len,
                device=inputs_embeds.device
            ).unsqueeze(0).expand(B, -1)

        # Forward through LLM layers
        hidden_states = inputs_embeds
        new_past_key_values = []

        for i, layer in enumerate(self.layers):
            pkv = past_key_values[i] if past_key_values is not None else None
            hidden_states, new_pkv = layer(
                hidden_states,
                modality_ids=modality_ids,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=pkv,
            )
            new_past_key_values.append(new_pkv)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        # Compute loss
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.llm_vocab_size),
                shift_labels.view(-1),
                ignore_index=self.config.ignore_index,
            )

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": new_past_key_values,
            "hidden_states": hidden_states,
        }

    def _make_causal_mask(
        self,
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        """Create causal attention mask."""
        total_len = seq_len + past_key_values_length
        mask = torch.full((seq_len, total_len), float("-inf"), dtype=dtype, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).reshape(mask.size(0), 1), 0)
        if past_key_values_length > 0:
            mask = torch.cat([
                torch.zeros(seq_len, past_key_values_length, dtype=dtype, device=device),
                mask
            ], dim=-1)
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, total_len)

    def freeze_text_params(self):
        """Stage 1a: Freeze all text parameters."""
        for name, param in self.named_parameters():
            if "visual_encoder" not in name and "connector" not in name:
                # Check if it's a MoE visual expert
                if "moe" in name and ("visual" in name or "_projs.0" in name):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        # Ensure visual encoder and connector are trainable
        for param in self.visual_encoder.parameters():
            param.requires_grad = True

    def unfreeze_text_attention(self):
        """Stage 1b: Unfreeze text attention parameters."""
        for name, param in self.named_parameters():
            if any(x in name for x in ["q_proj", "k_proj", "v_proj", "o_proj"]):
                param.requires_grad = True

    def unfreeze_all(self):
        """Stage 2: Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_active_parameters(self) -> int:
        """Number of activated parameters (excluding inactive MoE experts)."""
        # In NaViL, each token activates exactly 1 expert per layer
        # So active params = total params - (num_modalities - 1) * expert_params_per_layer
        return self.num_parameters  # Simplified; actual depends on batch composition

    @classmethod
    def from_config(cls, config: NaViLConfig) -> "NaViLModel":
        return cls(config)
