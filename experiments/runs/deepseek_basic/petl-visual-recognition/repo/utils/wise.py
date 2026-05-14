"""Weight-Space Ensembles (WiSE) for PEFT methods.

WiSE linearly interpolates between fine-tuned and pre-trained model weights:
    θ_wise = α * θ_ft + (1-α) * θ_pretrained

This is adapted for different PEFT method types as described in Section 7.
"""

import torch
import torch.nn as nn
import copy
import numpy as np


def apply_wise_to_model(model, pretrained_backbone, alpha=0.5):
    """Apply WiSE interpolation between fine-tuned and pre-trained weights.
    
    For different PEFT method types:
    - Direct selective tuning (BitFit, LayerNorm): merge PEFT-tuned params
    - Adapter-based: scale adapter modules by α
    - Efficient selective (LoRA, FacT): merge additive residuals with α scaling
    - Full FT: interpolate all weights
    
    Args:
        model: The fine-tuned PEFTModel
        pretrained_backbone: Fresh pre-trained backbone (not fine-tuned)
        alpha: Mixing coefficient (0=pretrained, 1=ft)
    
    Returns:
        wise_model: Model with WiSE-applied weights
    """
    method_name = model.method_name
    
    if method_name == 'full':
        # Full FT: interpolate all weights between FT and pretrained
        # Get FT model state
        ft_state = model.backbone.state_dict()
        pt_state = pretrained_backbone.state_dict()
        
        wise_state = {}
        for key in ft_state:
            if key in pt_state:
                wise_state[key] = alpha * ft_state[key] + (1 - alpha) * pt_state[key]
            else:
                wise_state[key] = ft_state[key]
        
        model.backbone.load_state_dict(wise_state)
        
        # Also interpolate head
        # For head we keep it as is (head is task-specific)
    
    elif method_name in ['bitfit', 'layernorm']:
        # Direct selective: interpolate trainable params with pretrained
        ft_state = model.backbone.state_dict()
        pt_state = pretrained_backbone.state_dict()
        
        # Only interpolate the parameters that were trained
        for name, param in model.backbone.named_parameters():
            if param.requires_grad:
                ft_val = ft_state[name]
                pt_val = pt_state[name]
                ft_state[name] = alpha * ft_val + (1 - alpha) * pt_val
        
        model.backbone.load_state_dict(ft_state)
    
    elif method_name == 'difffit':
        # DiffFit: interpolate bias/LN/scale factors
        ft_state = model.backbone.state_dict()
        pt_state = pretrained_backbone.state_dict()
        
        for name, param in model.backbone.named_parameters():
            if param.requires_grad:
                ft_val = ft_state[name]
                if name in pt_state:
                    pt_val = pt_state[name]
                    ft_state[name] = alpha * ft_val + (1 - alpha) * pt_val
        
        model.backbone.load_state_dict(ft_state)
        
        # Scale the DiffFit scale factors
        for p in model.peft_method.msa_scales:
            p.data = alpha * p.data + (1 - alpha) * torch.ones_like(p.data)
        for p in model.peft_method.mlp_scales:
            p.data = alpha * p.data + (1 - alpha) * torch.ones_like(p.data)
    
    elif method_name in ['houlsby_adapter', 'pfeiffer_adapter', 'adaptformer',
                          'convpass', 'repadapter']:
        # Adapter-based: scale adapter modules by α
        # This implements WiSE as feature ensembles:
        # output = backbone(x) + α * adapter(x)
        # By scaling adapter weights, we effectively interpolate features
        
        _scale_adapter_weights(model.peft_method, alpha)
    
    elif method_name in ['lora', 'fact_tt', 'fact_tk']:
        # Efficient selective: scale additive residuals by α
        # W_wise = W_pretrained + α * ΔW
        _scale_additive_weights(model.peft_method, alpha, method_name)
    
    elif method_name == 'ssf':
        # SSF: interpolate scale/shift factors towards identity
        _scale_ssf_weights(model.peft_method, alpha)
    
    # For prompt-based, WiSE isn't directly applicable (prompts are new params)
    # But we can scale prompts towards zero
    elif method_name in ['vpt_shallow', 'vpt_deep']:
        _scale_prompts(model.peft_method, alpha)
    
    return model


def _scale_adapter_weights(peft_method, alpha):
    """Scale adapter weights by alpha."""
    for name, param in peft_method.named_parameters():
        if 'down_proj' in name or 'up_proj' in name or 'conv' in name:
            # Scale adapter parameters toward zero
            param.data = alpha * param.data


def _scale_additive_weights(peft_method, alpha, method_name):
    """Scale additive residuals (LoRA, FacT) by alpha."""
    for name, param in peft_method.named_parameters():
        param.data = alpha * param.data


def _scale_ssf_weights(peft_method, alpha):
    """Scale SSF weights: scale -> identity(1), shift -> identity(0)."""
    for name, param in peft_method.named_parameters():
        if 'scale' in name:
            param.data = alpha * param.data + (1 - alpha)
        elif 'shift' in name:
            param.data = alpha * param.data


def _scale_prompts(peft_method, alpha):
    """Scale prompts towards zero for WiSE."""
    for name, param in peft_method.named_parameters():
        param.data = alpha * param.data


def compute_wise_accuracy_curve(model, pretrained_backbone, eval_fn, 
                                 alphas=None):
    """Compute accuracy for different WiSE mixing coefficients.
    
    Args:
        model: Fine-tuned PEFTModel
        pretrained_backbone: Pre-trained backbone
        eval_fn: Function that takes a model and returns accuracy dict
        alphas: List of α values to try
    
    Returns:
        List of (alpha, target_acc, shift_acc) tuples
    """
    if alphas is None:
        alphas = np.arange(0.0, 1.05, 0.05)
    
    results = []
    
    # Save original state
    original_state = copy.deepcopy(model.state_dict())
    
    for alpha in alphas:
        # Apply WiSE
        model.load_state_dict(original_state)  # Reset
        wise_model = apply_wise_to_model(model, pretrained_backbone, alpha=alpha)
        
        # Evaluate
        acc_dict = eval_fn(wise_model)
        
        results.append({
            'alpha': alpha,
            **acc_dict,
        })
    
    # Restore original
    model.load_state_dict(original_state)
    
    return results
