"""PEFT methods for Vision Transformers in visual recognition.

This module implements 14 PEFT methods organized into 4 categories:
- Prompt-based: VPT-Shallow, VPT-Deep
- Adapter-based: Houlsby, Pfeiffer, AdaptFormer, ConvPass, RepAdapter
- Direct Selective Tuning: BitFit, LayerNorm, DiffFit
- Efficient Selective Tuning: LoRA, FacT_TT, FacT_TK, SSF
"""

from .prompt_based import VPTShallow, VPTDeep
from .adapter_based import (HoulsbyAdapter, PfeifferAdapter, AdaptFormer,
                             ConvPass, RepAdapter)
from .direct_selective import BitFit, LayerNorm, DiffFit
from .efficient_selective import LoRA, FacT_TT, FacT_TK
from .ssf import SSF

__all__ = [
    'VPTShallow', 'VPTDeep',
    'HoulsbyAdapter', 'PfeifferAdapter', 'AdaptFormer', 'ConvPass', 'RepAdapter',
    'BitFit', 'LayerNorm', 'DiffFit',
    'LoRA', 'FacT_TT', 'FacT_TK',
    'SSF',
]

# Map method names to classes
METHOD_MAP = {
    'vpt_shallow': VPTShallow,
    'vpt_deep': VPTDeep,
    'houlsby_adapter': HoulsbyAdapter,
    'pfeiffer_adapter': PfeifferAdapter,
    'adaptformer': AdaptFormer,
    'convpass': ConvPass,
    'repadapter': RepAdapter,
    'bitfit': BitFit,
    'layernorm': LayerNorm,
    'difffit': DiffFit,
    'lora': LoRA,
    'fact_tt': FacT_TT,
    'fact_tk': FacT_TK,
    'ssf': SSF,
}

# Method categories
METHOD_CATEGORIES = {
    'prompt_based': ['vpt_shallow', 'vpt_deep'],
    'adapter_based': ['houlsby_adapter', 'pfeiffer_adapter', 'adaptformer', 'convpass', 'repadapter'],
    'direct_selective': ['bitfit', 'layernorm', 'difffit'],
    'efficient_selective': ['lora', 'fact_tt', 'fact_tk', 'ssf'],
}

# VTAB-1K hyperparameter grids from Table 3
HP_GRID = {
    'vpt_shallow': {'prompt_number': [5, 10, 50, 100, 200]},
    'vpt_deep': {'prompt_number': [5, 10, 50, 100]},
    'bitfit': {},
    'difffit': {},
    'layernorm': {},
    'ssf': {},
    'pfeiffer_adapter': {'adapter_scale': [0.01, 0.1, 1, 10], 'adapter_bottleneck': [4, 8, 16, 32]},
    'houlsby_adapter': {'adapter_scale': [0.01, 0.1, 1, 10], 'adapter_bottleneck': [4, 8, 16, 32]},
    'adaptformer': {'adapter_scale': [0.05, 0.1, 0.2], 'adapter_bottleneck': [4, 16, 32]},
    'repadapter': {'adapter_scale': [0.1, 0.5, 1, 5, 10], 'adapter_bottleneck': [8, 16, 32]},
    'convpass': {'adapter_scale': [0.01, 0.1, 1, 10, 100], 'adapter_bottleneck': [8, 16], 'xavier_init': [True, False]},
    'lora': {'lora_rank': [1, 8, 16, 32]},
    'fact_tt': {'fact_scale': [0.01, 0.1, 1, 10, 100], 'fact_bottleneck': [8, 16, 32]},
    'fact_tk': {'fact_bottleneck': [16, 32, 64], 'fact_scale': [0.01, 0.1, 1, 10, 100]},
}
