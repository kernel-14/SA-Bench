"""
WiSE (Weight-Space Ensembles) for PEFT methods.
Wortsman et al., 2022.

WiSE linearly interpolates the weights of a fine-tuned model with those of the original backbone.
For PEFT methods, we adapt WiSE to work with the specific parameter structures.
"""

import torch
import torch.nn as nn
import copy


def apply_wise(ft_model, pretrained_model, alpha, method_type='adapter'):
    """
    Apply WiSE (Weight-Space Ensemble) to a PEFT fine-tuned model.
    
    For different PEFT types:
    - Direct selective tuning (BitFit, LayerNorm, DiffFit): Merge tuned params with original
    - Adapter-based: Scale adapter modules by alpha
    - Efficient selective (LoRA, FacT): Scale additive residuals by alpha
    
    Args:
        ft_model: Fine-tuned model
        pretrained_model: Original pre-trained model
        alpha: Mixing coefficient (0 = pretrained, 1 = fine-tuned)
        method_type: Type of PEFT method ('adapter', 'selective', 'lora', 'vpt')
    
    Returns:
        WiSE model
    """
    wise_model = copy.deepcopy(ft_model)
    
    if method_type in ['bitfit', 'layernorm', 'difffit']:
        # Direct selective tuning: interpolate tuned parameters
        ft_state = ft_model.state_dict()
        pt_state = pretrained_model.state_dict()
        
        wise_state = {}
        for key in ft_state:
            if key in pt_state:
                # Interpolate between pretrained and fine-tuned
                wise_state[key] = alpha * ft_state[key] + (1 - alpha) * pt_state[key]
            else:
                # New parameters (e.g., head) - keep fine-tuned
                wise_state[key] = ft_state[key]
        
        wise_model.load_state_dict(wise_state)
    
    elif method_type in ['adapter', 'adaptformer', 'convpass', 'repadapter']:
        # Adapter-based: Scale adapter modules by alpha
        # The adapter output is: h + adapter(h)
        # WiSE: h + alpha * adapter(h)
        # This is equivalent to scaling the adapter's output by alpha
        for name, module in wise_model.named_modules():
            if hasattr(module, 'scale') and 'adapter' in name.lower():
                module.scale = module.scale * alpha
    
    elif method_type in ['lora', 'fact_tt', 'fact_tk']:
        # Efficient selective: Scale additive residuals by alpha
        # The update is: W + delta_W
        # WiSE: W + alpha * delta_W
        # Scale LoRA/FacT parameters by alpha
        for name, param in wise_model.named_parameters():
            if 'lora_up' in name or 'lora_q_up' in name or 'lora_v_up' in name:
                param.data = param.data * alpha
            elif hasattr(wise_model, 'fact_module'):
                # Scale FacT parameters
                if 'Sigma' in name or 'B' in name or 'A' in name:
                    param.data = param.data * alpha
    
    elif method_type == 'ssf':
        # SSF: Scale the SSF parameters towards identity
        # SSF: h * scale + shift
        # WiSE: h * (alpha * scale + (1-alpha) * 1) + alpha * shift
        for name, module in wise_model.named_modules():
            if hasattr(module, 'scale') and hasattr(module, 'shift') and 'ssf' in name.lower():
                module.scale.data = alpha * module.scale.data + (1 - alpha) * torch.ones_like(module.scale.data)
                module.shift.data = alpha * module.shift.data
    
    # Always interpolate the head
    ft_head_state = {k: v for k, v in ft_model.state_dict().items() if 'head' in k}
    pt_head_state = {k: v for k, v in pretrained_model.state_dict().items() if 'head' in k}
    
    wise_state = wise_model.state_dict()
    for key in ft_head_state:
        if key in pt_head_state:
            wise_state[key] = alpha * ft_head_state[key] + (1 - alpha) * pt_head_state[key]
    wise_model.load_state_dict(wise_state)
    
    return wise_model


def wise_sweep(ft_model, pretrained_model, test_loader, dist_shift_loaders, 
               method_type, device='cuda', alphas=None):
    """
    Sweep over alpha values for WiSE and find the best trade-off.
    
    Args:
        ft_model: Fine-tuned model
        pretrained_model: Original pre-trained model
        test_loader: DataLoader for target distribution
        dist_shift_loaders: Dict of DataLoaders for distribution shifts
        method_type: Type of PEFT method
        device: Device to use
        alphas: List of alpha values to try
    
    Returns:
        results: Dict mapping alpha to (target_acc, avg_shift_acc)
    """
    from .evaluator import evaluate
    
    if alphas is None:
        alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    results = {}
    
    for alpha in alphas:
        wise_model = apply_wise(ft_model, pretrained_model, alpha, method_type)
        wise_model = wise_model.to(device)
        
        # Evaluate on target distribution
        target_acc = evaluate(wise_model, test_loader, device)
        
        # Evaluate on distribution shifts
        shift_accs = []
        for name, loader in dist_shift_loaders.items():
            shift_acc = evaluate(wise_model, loader, device)
            shift_accs.append(shift_acc)
        
        avg_shift_acc = sum(shift_accs) / len(shift_accs) if shift_accs else 0.0
        results[alpha] = (target_acc, avg_shift_acc)
        
        print(f'Alpha={alpha:.1f}: Target={target_acc:.2f}%, Avg Shift={avg_shift_acc:.2f}%')
    
    return results
