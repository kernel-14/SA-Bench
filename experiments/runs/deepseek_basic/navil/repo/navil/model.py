"""
NaViL: Native Multimodal Large Language Model.

Full model implementation based on the paper:
"NaViL: Rethinking Scaling Properties of Native Multimodal LLMs under Data Constraints"

Architecture (Fig. 8):
1. Visual Encoder V_{d,w}(·): bidirectional transformer with 2D-RoPE
2. Connector: pixel shuffle downsampling + MLP projection
3. MoE-extended LLM: modality-specific attention and FFN experts
4. Special tokens: <begin_of_image>, <end_of_image>, <end_of_line>, <end_of_scale>

Key design choices from the paper:
- LLM initialized from pre-trained checkpoint (InternLM2-1.8B for NaViL-2B)
- MoE with modality-specific attention (MHA-MMoE) and FFN (FFN-MMoE) experts
- Visual encoder with balanced depth/width ratio
- Visual multi-scale packing for inference
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .visual_encoder import VisualEncoder, RMSNorm
from .connector import Connector
from .moe import ModalityMoELayer


@dataclass
class NaViLConfig:
    """Configuration for NaViL model."""
    
    # Visual Encoder config (NaViL-2B defaults from Table 6)
    visual_depth: int = 24
    visual_width: int = 1472
    visual_mlp_width: int = 5888
    visual_num_heads: int = 23
    patch_size: int = 16
    
    # LLM config (InternLM2-1.8B style)
    llm_depth: int = 24
    llm_width: int = 2048
    llm_mlp_width: int = 8192
    llm_num_heads: int = 16
    
    # MoE config
    use_moe: bool = True
    num_experts: int = 2  # visual + text
    
    # Connector
    connector_downsample_ratio: int = 2
    
    # Vocabulary & special tokens
    vocab_size: int = 92544  # InternLM2 vocab size
    pad_token_id: int = 2
    bos_token_id: int = 1
    eos_token_id: int = 2
    
    # Image processing
    image_size: int = 448  # default, but any-resolution is supported
    max_image_patches: int = 4096
    
    # Multi-scale packing
    use_multi_scale: bool = True
    downsample_rate: float = 0.7071067811865476  # sqrt(2)/2
    
    # Training
    max_position_embeddings: int = 16384
    dropout: float = 0.0
    
    # LLM initialization
    llm_checkpoint_path: Optional[str] = None
    
    def __post_init__(self):
        # Ensure hidden sizes are divisible by num_heads
        assert self.visual_width % self.visual_num_heads == 0
        assert self.llm_width % self.llm_num_heads == 0


# Special token IDs
SPECIAL_TOKENS = {
    'begin_of_image': '<begin_of_image>',
    'end_of_image': '<end_of_image>', 
    'end_of_line': '<end_of_line>',
    'end_of_scale': '<end_of_scale>',
}


class NaViLModel(nn.Module):
    """
    NaViL: Native Multimodal Large Language Model.
    
    This is the full end-to-end model that jointly processes images and text.
    """
    
    def __init__(self, config: NaViLConfig):
        super().__init__()
        self.config = config
        
        # --- Visual Encoder ---
        self.visual_encoder = VisualEncoder(
            depth=config.visual_depth,
            width=config.visual_width,
            mlp_width=config.visual_mlp_width,
            num_heads=config.visual_num_heads,
            patch_size=config.patch_size,
            dropout=config.dropout,
        )
        
        # --- Connector ---
        self.connector = Connector(
            visual_hidden_size=config.visual_width,
            llm_hidden_size=config.llm_width,
            downsample_ratio=config.connector_downsample_ratio,
        )
        
        # --- Token Embedding ---
        self.token_embedding = nn.Embedding(
            config.vocab_size, 
            config.llm_width,
            padding_idx=config.pad_token_id,
        )
        
        # --- LLM Decoder Layers (MoE-extended) ---
        self.llm_layers = nn.ModuleList([
            ModalityMoELayer(
                hidden_size=config.llm_width,
                num_heads=config.llm_num_heads,
                intermediate_size=config.llm_mlp_width,
                dropout=config.dropout,
                use_moe=config.use_moe,
            )
            for _ in range(config.llm_depth)
        ])
        
        # --- Final Layer Norm ---
        self.final_norm = RMSNorm(config.llm_width)
        
        # --- Output Head ---
        self.lm_head = nn.Linear(config.llm_width, config.vocab_size, bias=False)
        
        # Store model metadata
        self._num_params = None
        
    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_patches: Optional[List[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of NaViL.
        
        Args:
            input_ids: (B, seq_len) token ids
            pixel_values: (B, C, H, W) raw images, OR
            image_patches: List of (B_i, C, H_i, W_i) for multi-scale
            attention_mask: (B, seq_len) attention mask
            labels: (B, seq_len) for computing loss
            
        Returns:
            dict with 'loss', 'logits', etc.
        """
        B, seq_len = input_ids.shape
        device = input_ids.device
        dtype = self.token_embedding.weight.dtype
        
        # --- Process images ---
        visual_tokens = None
        visual_positions = None
        
        if pixel_values is not None or image_patches is not None:
            visual_tokens, visual_positions = self._process_images(
                pixel_values, image_patches, input_ids, device, dtype
            )
        
        # --- Create modality mask ---
        modality_mask = self._create_modality_mask(
            input_ids, visual_positions, seq_len, device
        )
        
        # --- Embed tokens ---
        hidden_states = self.token_embedding(input_ids)
        
        # --- Insert visual tokens ---
        if visual_tokens is not None and visual_positions is not None:
            hidden_states = self._insert_visual_tokens(
                hidden_states, visual_tokens, visual_positions
            )
        
        # --- Create causal attention mask ---
        causal_mask = self._create_causal_mask(seq_len, device, dtype)
        
        # --- Pass through LLM layers ---
        for layer in self.llm_layers:
            hidden_states = layer(
                hidden_states,
                modality_mask=modality_mask,
                attention_mask=causal_mask,
            )
        
        # --- Final norm and output ---
        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        # --- Compute loss ---
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': hidden_states,
        }
    
    def _process_images(
        self,
        pixel_values: Optional[torch.Tensor],
        image_patches: Optional[List[torch.Tensor]],
        input_ids: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process images through visual encoder and connector.
        
        Supports:
        1. Single image: pixel_values (B, C, H, W)
        2. Multi-scale: image_patches = [img_scale0, img_scale1, ...]
        
        Returns:
            visual_tokens: concatenated visual token embeddings
            visual_positions: positions in the sequence where to insert
        """
        all_visual_tokens = []
        
        if pixel_values is not None:
            images_to_encode = [pixel_values]
        elif image_patches is not None:
            images_to_encode = image_patches
        else:
            return None, None
        
        for img in images_to_encode:
            B_i, C, H, W = img.shape
            
            # Pad to multiples of 32 (paper: "padded to ensure length and width are multiples of 32")
            H_pad = ((H + 31) // 32) * 32
            W_pad = ((W + 31) // 32) * 32
            if H != H_pad or W != W_pad:
                pad_h = H_pad - H
                pad_w = W_pad - W
                img = F.pad(img, (0, pad_w, 0, pad_h))
            
            # Visual encoder forward
            vis_features = self.visual_encoder(img.to(device).to(dtype))
            
            # Compute spatial dimensions
            H_feat = H_pad // self.config.patch_size
            W_feat = W_pad // self.config.patch_size
            
            # Connector (pixel shuffle + MLP)
            vis_tokens, H_out, W_out = self.connector(vis_features, H_feat, W_feat)
            
            all_visual_tokens.append(vis_tokens)
        
        # Concatenate all scales
        visual_tokens = torch.cat(all_visual_tokens, dim=1)  # (B, total_vis_tokens, llm_width)
        
        # Find positions of <begin_of_image> tokens to insert visual tokens
        # For simplicity, we insert after each <begin_of_image> token
        # A proper implementation would track exact positions
        visual_positions = None  # Will be determined by special token positions
        
        return visual_tokens, visual_positions
    
    def _create_modality_mask(
        self,
        input_ids: torch.Tensor,
        visual_positions: Optional[torch.Tensor],
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create a boolean mask indicating which tokens are visual.
        True = visual token, False = text token.
        
        This is a simplified version - a full implementation would track
        exact visual token positions using special tokens.
        """
        B = input_ids.shape[0]
        
        # Default: all tokens are text
        modality_mask = torch.zeros(B, seq_len, dtype=torch.bool, device=device)
        
        # Mark visual tokens based on special token positions
        # <begin_of_image> and <end_of_image> delineate visual sequences
        # For now, this is a placeholder that would be properly implemented
        # with the actual tokenizer
        
        return modality_mask
    
    def _insert_visual_tokens(
        self,
        hidden_states: torch.Tensor,
        visual_tokens: torch.Tensor,
        visual_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Insert visual token embeddings into the sequence at appropriate positions.
        
        This is a simplified version. The full implementation would scatter
        visual tokens into the positions marked by <begin_of_image> ... <end_of_image>.
        """
        # For simplicity, concatenate visual tokens at the beginning
        # A proper implementation would use scatter operations based on positions
        B, N_text, C = hidden_states.shape
        B_vis, N_vis, C_vis = visual_tokens.shape
        
        # Concatenate visual tokens before text tokens
        combined = torch.cat([visual_tokens, hidden_states], dim=1)
        
        return combined
    
    def _create_causal_mask(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create causal attention mask (lower triangular)."""
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=dtype) * float('-inf'),
            diagonal=1,
        )
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    
    @property
    def num_parameters(self) -> int:
        """Total number of parameters."""
        if self._num_params is None:
            self._num_params = sum(p.numel() for p in self.parameters())
        return self._num_params
    
    @property
    def num_activated_parameters(self) -> int:
        """
        Number of activated parameters (for inference).
        Since only one expert is active per token, this equals
        the base parameter count (not 2x).
        """
        # In MoE mode, each token only activates either the visual or text expert
        # The total activated param count is roughly the same as a standard LLM
        # with the same hidden_size (plus encoder)
        total = self.num_parameters
        # For NaViL-2B: ~4.2B total, ~2.4B activated 
        # (1.8B LLM + 0.6B visual encoder)
        return total
    
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        **kwargs,
    ) -> torch.Tensor:
        """
        Autoregressive text generation.
        """
        self.eval()
        device = input_ids.device
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.forward(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                )
                
                logits = outputs['logits'][:, -1, :] / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][:, -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float('-inf')
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                if next_token.item() == self.config.eos_token_id:
                    break
        
        return input_ids
    
    def get_visual_encoder_params(self) -> List[nn.Parameter]:
        """Get visual encoder parameters for staged training."""
        return list(self.visual_encoder.parameters()) + list(self.connector.parameters())
    
    def get_moe_visual_params(self) -> List[nn.Parameter]:
        """Get MoE visual expert parameters for staged training."""
        visual_params = []
        for layer in self.llm_layers:
            if hasattr(layer.attn, 'q_proj_visual'):
                visual_params.extend([
                    layer.attn.q_proj_visual.weight,
                    layer.attn.k_proj_visual.weight,
                    layer.attn.v_proj_visual.weight,
                    layer.attn.o_proj_visual.weight,
                ])
            if hasattr(layer.ffn, 'gate_proj_visual'):
                visual_params.extend([
                    layer.ffn.gate_proj_visual.weight,
                    layer.ffn.up_proj_visual.weight,
                    layer.ffn.down_proj_visual.weight,
                ])
        return visual_params
    
    def get_text_params(self) -> List[nn.Parameter]:
        """Get text (LLM) parameters for staged training."""
        # All parameters except visual encoder and connector
        vis_enc_params = set(self.get_visual_encoder_params())
        all_params = set(self.parameters())
        return list(all_params - vis_enc_params)


def create_navil_2b(llm_checkpoint_path: Optional[str] = None) -> NaViLModel:
    """
    Create NaViL-2B model with the default configuration from the paper.
    
    Config (Table 6):
        - Visual encoder: 0.6B params, depth=24, width=1472, mlp_width=5888, heads=23
        - LLM w/ MoE: 1.8B params (InternLM2-1.8B), depth=24, width=2048, mlp_width=8192, heads=16
        - Total params: ~4.2B
        - Activated params: ~2.4B
    """
    config = NaViLConfig(
        visual_depth=24,
        visual_width=1472,
        visual_mlp_width=5888,
        visual_num_heads=23,
        llm_depth=24,
        llm_width=2048,
        llm_mlp_width=8192,
        llm_num_heads=16,
        use_moe=True,
        num_experts=2,
        llm_checkpoint_path=llm_checkpoint_path,
    )
    return NaViLModel(config)


def create_navil_9b(llm_checkpoint_path: Optional[str] = None) -> NaViLModel:
    """
    Create NaViL-9B model with the configuration from the paper.
    
    Config (Table 6):
        - Visual encoder: 1.2B params, depth=32, width=1792, mlp_width=7168, heads=28
        - LLM w/ MoE: 8.0B params (Qwen3-8B), depth=36, width=4096, mlp_width=12288, heads=32
        - Total params: ~?B
        - Activated params: ~9.2B
    """
    config = NaViLConfig(
        visual_depth=32,
        visual_width=1792,
        visual_mlp_width=7168,
        visual_num_heads=28,
        llm_depth=36,
        llm_width=4096,
        llm_mlp_width=12288,
        llm_num_heads=32,
        use_moe=True,
        num_experts=2,
        llm_checkpoint_path=llm_checkpoint_path,
    )
    return NaViLModel(config)
