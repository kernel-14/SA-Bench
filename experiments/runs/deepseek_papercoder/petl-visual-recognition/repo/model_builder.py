```python
## model_builder.py

"""
PEFTModel: assembles a backbone (ViT-B/16 from ImageNet‑21K or CLIP),
freezes appropriate parameters, injects the selected PEFT method, and
provides a modified forward pass that uses the h1…h9 feature notation.
Also supports weight‑space ensemble (WiSE) interpolation.
"""

import copy
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# Optional open_clip
try:
    import open_clip
except ImportError:
    open_clip = None

from config import Config
from utils import set_seed, compute_trainable_params, merge_dicts
from peft_modules import (
    VPTShallow, VPTDeep,
    AdapterBlock, ConvpassBlock, RepAdapterBlock,
    LoRA, FacT_TT, FacT_TK, LoRALinear, FacTLinear,
    create_peft_module
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------ #
#  Helper sub‑modules for combining QKV with LoRA / FacT
# ------------------------------------------------------------------------ #
class LoRACombinedQKV(nn.Module):
    """
    Replaces the original single qkv linear by three separate projections,
    where Q and V are augmented with LoRA residuals.
    """
    def __init__(self, base_q: nn.Linear, base_k: nn.Linear, base_v: nn.Linear,
                 lora_q: Optional[nn.Module] = None,
                 lora_v: Optional[nn.Module] = None):
        super().__init__()
        self.base_q = base_q
        self.base_k = base_k
        self.base_v = base_v
        self.lora_q = lora_q
        self.lora_v = lora_v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.lora_q is not None:
            q = self.base_q(x) + self.lora_q(x)
        else:
            q = self.base_q(x)
        k = self.base_k(x)
        if self.lora_v is not None:
            v = self.base_v(x) + self.lora_v(x)
        else:
            v = self.base_v(x)
        return torch.cat([q, k, v], dim=-1)


class FacTCombinedLinear(nn.Module):
    """
    Wraps a linear layer and adds a delta from a global FacT module.
    """
