```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Tuple, List, Dict, Any, Optional, Union
from transformers import T5EncoderModel, T5Tokenizer, CLIPTextModel, CLIPTokenizer

# Assuming Config class and utility functions are available
try:
    from config import Config
    from utils import get_spatial_pos_embed, get_temporal_rope_embed
except ImportError:
    print("Warning: Could not import Config or utils functions. Using stub classes for model.py.")
    
    # Minimal stub for Config and its nested classes for local testing/linting
    class TextEncoderConfig:
        t5_model_name: str = "google/t5-v1_1-large"
        clip_model_name: str = "openai/clip-vit-large-patch14"
        max_text_length: int = 77

    class VaeConfig:
        latent_channels: int = 4

    class DitParamsConfig:
        num_layers: int = 24
        num_attention_heads: int = 16
        hidden_size: int = 1024
        mlp_ratio: int = 4
        in_channels: int = 4
        out_channels: int = 4
        block_wise_causal_attention: bool = True

    class ModelConfig:
        dit_backbone_name: str = "SD3_Medium"
        dit_params: DitParamsConfig = DitParamsConfig()
        text_encoder: TextEncoderConfig = TextEncoderConfig()
        vae: VaeConfig = VaeConfig()
        pyramid_stages: int = 3

    class ComputeConfig:
        device: str = "cpu"

    class Config:
        model: ModelConfig = ModelConfig()
        compute: ComputeConfig = ComputeConfig()
    
    # Stubs for utils functions
    def get_spatial_pos_embed(D: int, H: int, W: int, embed_dim: int, device: torch.device) -> torch.Tensor:
        # Dummy spatial PE
        return torch.zeros(1, D * H * W, embed_dim, device=device)

    def get_temporal_rope_embed(T: int, head_dim: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        # Dummy RoPE freqs
        return torch.ones(T, head_dim // 2, device=device), torch.zeros(T, head_dim // 2, device=device)


# --- 1. TextEncoder ---

class TextEncoder(nn.Module):
    """
    Wraps Hugging Face's T5EncoderModel and CLIPTextModel for text conditioning.
    """
    def __init__(self, config: Config):
        super().__