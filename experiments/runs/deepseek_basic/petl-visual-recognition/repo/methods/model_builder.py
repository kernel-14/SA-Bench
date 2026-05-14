"""Model builder: constructs a ViT model with a specified PEFT method applied.

Supports both ImageNet-21K ViT and CLIP ViT backbones.
"""

import torch
import torch.nn as nn
from .backbone import ViTBackbone, CLIPViTBackbone
from .prompt_based import VPTShallow, VPTDeep
from .adapter_based import (HoulsbyAdapter, PfeifferAdapter, AdaptFormer,
                             ConvPass, RepAdapter)
from .direct_selective import BitFit, LayerNorm, DiffFit
from .efficient_selective import LoRA, FacT_TT, FacT_TK
from .ssf import SSF


class PEFTModel(nn.Module):
    """Unified model that combines a ViT backbone with a PEFT method.
    
    This provides a consistent interface for all PEFT methods, handling
    the different forward pass requirements of each method type.
    """
    
    def __init__(self, backbone, peft_method, num_classes, method_name=''):
        super().__init__()
        self.backbone = backbone
        self.peft_method = peft_method
        self.num_classes = num_classes
        self.method_name = method_name
        
        # Prediction head (randomly initialized)
        self.head = nn.Linear(backbone.embed_dim, num_classes)
        
        # Freeze backbone for PEFT
        self._setup_training()
        
        # Store original backbone forward for methods that need direct access
        self._original_vit_forward = backbone.vit.forward if backbone.vit else None
    
    def _setup_training(self):
        """Configure which parameters are trainable based on the PEFT method."""
        method_name = self.method_name
        
        # Methods that need special backbone setup
        if method_name in ['bitfit', 'layernorm']:
            # These methods modify backbone params directly
            if hasattr(self.peft_method, 'apply'):
                self.peft_method.apply(self.backbone)
        elif method_name == 'difffit':
            if hasattr(self.peft_method, 'apply'):
                self.peft_method.apply(self.backbone)
        else:
            # For other methods, freeze backbone
            for p in self.backbone.parameters():
                p.requires_grad = False
        
        # Head is always trainable
        for p in self.head.parameters():
            p.requires_grad = True
    
    def forward(self, x):
        """Forward pass through the PEFT-augmented model.
        
        Methods that modify the forward pass (adapter-based, prompt-based,
        SSF, DiffFit) handle the forward differently from methods that
        only modify parameters (BitFit, LayerNorm, LoRA).
        """
        method_name = self.method_name
        
        # Methods that need to intercept and modify the forward pass
        if method_name in ['vpt_shallow', 'vpt_deep']:
            # Prompt-based: needs to insert prompts
            logits = self.peft_method(x, self.backbone)
            if logits.shape[-1] == self.backbone.embed_dim:
                logits = self.head(logits)
            return logits
        
        elif method_name in ['houlsby_adapter', 'pfeiffer_adapter', 'adaptformer',
                              'convpass', 'repadapter']:
            # Adapter-based: modifies features inside transformer layers
            logits = self.peft_method(x, self.backbone)
            if logits.shape[-1] == self.backbone.embed_dim:
                logits = self.head(logits)
            return logits
        
        elif method_name == 'difffit':
            # DiffFit: modifies forward with scale factors
            logits = self.peft_method(x, self.backbone)
            if logits.shape[-1] == self.backbone.embed_dim:
                logits = self.head(logits)
            return logits
        
        elif method_name == 'ssf':
            # SSF: modifies forward with scale and shift
            logits = self.peft_method(x, self.backbone)
            if logits.shape[-1] == self.backbone.embed_dim:
                logits = self.head(logits)
            return logits
        
        elif method_name in ['fact_tt', 'fact_tk']:
            # FacT: modifies forward with tensor decomposition
            logits = self.peft_method(x, self.backbone)
            if logits.shape[-1] == self.backbone.embed_dim:
                logits = self.head(logits)
            return logits
        
        elif method_name == 'lora':
            # LoRA modifies attention weights; standard forward
            features = self.backbone(x)
            return self.head(features)
        
        else:
            # BitFit, LayerNorm, linear probing, full FT
            features = self.backbone(x)
            return self.head(features)
    
    def get_trainable_param_count(self):
        """Return count of trainable parameters in millions."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6


def build_model(method_name, num_classes, backbone_type='in21k', 
                drop_path_rate=0.0, **method_kwargs):
    """Build a PEFT model with the specified method.
    
    Args:
        method_name: Name of the PEFT method (or 'linear', 'full')
        num_classes: Number of output classes
        backbone_type: 'in21k' for ImageNet-21K ViT or 'clip' for CLIP ViT
        drop_path_rate: Drop path rate (0.0 or 0.1)
        **method_kwargs: Method-specific hyperparameters
    
    Returns:
        PEFTModel instance
    """
    # Build backbone
    if backbone_type == 'clip':
        backbone = CLIPViTBackbone(drop_path_rate=drop_path_rate)
    else:
        backbone = ViTBackbone(
            model_name='vit_base_patch16_224_in21k',
            pretrained=True,
            drop_path_rate=drop_path_rate,
        )
    
    embed_dim = backbone.embed_dim
    num_layers = backbone.num_layers
    
    # Build PEFT method
    if method_name == 'linear':
        # Linear probing: freeze backbone, train head only
        peft_method = None
        for p in backbone.parameters():
            p.requires_grad = False
    
    elif method_name == 'full':
        # Full fine-tuning: train everything
        peft_method = None
        for p in backbone.parameters():
            p.requires_grad = True
    
    elif method_name == 'vpt_shallow':
        prompt_number = method_kwargs.get('prompt_number', 50)
        peft_method = VPTShallow(embed_dim, num_layers, prompt_number)
    
    elif method_name == 'vpt_deep':
        prompt_number = method_kwargs.get('prompt_number', 10)
        peft_method = VPTDeep(embed_dim, num_layers, prompt_number)
    
    elif method_name == 'bitfit':
        peft_method = BitFit()
    
    elif method_name == 'difffit':
        peft_method = DiffFit(embed_dim, num_layers)
    
    elif method_name == 'layernorm':
        peft_method = LayerNorm()
    
    elif method_name == 'ssf':
        peft_method = SSF(embed_dim, num_layers)
    
    elif method_name == 'pfeiffer_adapter':
        bottleneck = method_kwargs.get('adapter_bottleneck', 16)
        scale = method_kwargs.get('adapter_scale', 1.0)
        peft_method = PfeifferAdapter(embed_dim, num_layers, bottleneck, scale)
    
    elif method_name == 'houlsby_adapter':
        bottleneck = method_kwargs.get('adapter_bottleneck', 16)
        scale = method_kwargs.get('adapter_scale', 1.0)
        peft_method = HoulsbyAdapter(embed_dim, num_layers, bottleneck, scale)
    
    elif method_name == 'adaptformer':
        bottleneck = method_kwargs.get('adapter_bottleneck', 16)
        scale = method_kwargs.get('adapter_scale', 0.1)
        peft_method = AdaptFormer(embed_dim, num_layers, bottleneck, scale)
    
    elif method_name == 'repadapter':
        bottleneck = method_kwargs.get('adapter_bottleneck', 16)
        scale = method_kwargs.get('adapter_scale', 1.0)
        num_groups = method_kwargs.get('num_groups', 4)
        peft_method = RepAdapter(embed_dim, num_layers, bottleneck, scale, num_groups)
    
    elif method_name == 'convpass':
        bottleneck = method_kwargs.get('adapter_bottleneck', 16)
        scale = method_kwargs.get('adapter_scale', 1.0)
        xavier_init = method_kwargs.get('xavier_init', True)
        peft_method = ConvPass(embed_dim, num_layers, bottleneck, scale, xavier_init)
    
    elif method_name == 'lora':
        rank = method_kwargs.get('lora_rank', 8)
        alpha = method_kwargs.get('lora_alpha', 1.0)
        peft_method = LoRA(embed_dim, num_layers, rank, alpha)
    
    elif method_name == 'fact_tt':
        bottleneck = method_kwargs.get('fact_bottleneck', 16)
        scale = method_kwargs.get('fact_scale', 1.0)
        peft_method = FacT_TT(embed_dim, num_layers, bottleneck, scale)
    
    elif method_name == 'fact_tk':
        bottleneck = method_kwargs.get('fact_bottleneck', 16)
        scale = method_kwargs.get('fact_scale', 1.0)
        peft_method = FacT_TK(embed_dim, num_layers, bottleneck, scale)
    
    else:
        raise ValueError(f"Unknown method: {method_name}")
    
    model = PEFTModel(backbone, peft_method, num_classes, method_name)
    return model
