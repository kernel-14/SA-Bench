
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Tuple

from lora_sb_layers import LoRASBLayer
from config import Config

def get_trainable_modules(model: nn.Module, target_modules: List[str]):
    """
    Identifies and returns original Linear/Conv2d modules that will be replaced by LoRASBLayer.
    This is necessary to compute initial gradients on W_0.
    """
    trainable_modules = {}
    for name, module in model.named_modules():
        if any(target_module_key in name for target_module_key in target_modules):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                trainable_modules[name] = module
                # Temporarily enable gradients for W_0 to estimate first step gradient
                if hasattr(module, 'weight') and module.weight is not None:
                    module.weight.requires_grad = True
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.requires_grad = True
    return trainable_modules


def estimate_first_step_gradients(
    model: nn.Module,
    dataloader: DataLoader,
    num_samples: int,
    device: torch.device,
    target_modules: List[str]
) -> Dict[str, torch.Tensor]:
    """
    Estimates the averaged first-step gradients (ΔW_avg) for specified modules.
    This involves a temporary forward and backward pass on a subset of data.
    """
    model.eval() # Use eval mode for gradient estimation, no dropout etc.
    original_weights_to_grad = {}
    
    # Store original requires_grad state and set for gradient computation
    original_requires_grad_states = {}
    for name, module in model.named_modules():
        if any(target_module_key in name for target_module_key in target_modules):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if hasattr(module, 'weight') and module.weight is not None:
                    original_requires_grad_states[f"{name}.weight"] = module.weight.requires_grad
                    module.weight.requires_grad = True
                if hasattr(module, 'bias') and module.bias is not None:
                    original_requires_grad_states[f"{name}.bias"] = module.bias.requires_grad
                    if module.bias is not None:
                        module.bias.requires_grad = True

    model.zero_grad()
    
    gradient_sums: Dict[str, torch.Tensor] = {}
    samples_processed = 0

    for batch in tqdm(dataloader, desc="Estimating first-step gradients", total=min(num_samples, len(dataloader))):
        if samples_processed >= num_samples:
            break
        
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss

        # Backward pass
        loss.backward()

        # Aggregate gradients for relevant modules
        for name, module in model.named_modules():
            if any(target_module_key in name for target_module_key in target_modules):
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    if hasattr(module, 'weight') and module.weight is not None and module.weight.grad is not None:
                        if name not in gradient_sums:
                            gradient_sums[name] = torch.zeros_like(module.weight.data)
                        gradient_sums[name] += module.weight.grad.data
        
        model.zero_grad() # Clear gradients after each sample/batch
        samples_processed += batch['input_ids'].shape[0] if 'input_ids' in batch else 1 # Assuming input_ids as batch size indicator

    delta_w_avg: Dict[str, torch.Tensor] = {}
    for name, grad_sum in gradient_sums.items():
        # Apply sign function as per AdamW's first step approximation in paper Appendix C
        # The paper uses -eta * sign(grad_sum), but since eta is just a scalar for direction,
        # we can just use -sign(grad_sum) for the SVD and let the LR handle magnitude.
        delta_w_avg[name] = -torch.sign(grad_sum)

    # Restore original requires_grad states
    for name, module in model.named_modules():
        if any(target_module_key in name for target_module_key in target_modules):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if hasattr(module, 'weight') and module.weight is not None:
                    module.weight.requires_grad = original_requires_grad_states[f"{name}.weight"]
                if hasattr(module, 'bias') and module.bias is not None:
                    if module.bias is not None:
                        module.bias.requires_grad = original_requires_grad_states[f"{name}.bias"]

    model.train() # Set back to train mode
    return delta_w_avg

def perform_truncated_svd(matrix: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Performs truncated SVD on a matrix to obtain U, S, V.
    """
    # torch.linalg.svd returns U, S, Vh (conjugate transpose). We need V.
    U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
    V = Vh.transpose(-2, -1) # Get V from Vh

    # Truncate to rank
    U_trunc = U[:, :rank]
    S_trunc = torch.diag(S[:rank])
    V_trunc = V[:, :rank] # This is V_trunc, not Vh_trunc

    return U_trunc, S_trunc, V_trunc

def lora_sb_initialization(
    lora_sb_layer: LoRASBLayer,
    delta_w_avg_layer: torch.Tensor,
    rank: int,
    scaling_factor: float
):
    """
    Initializes B, R, A matrices of a LoRASBLayer using truncated SVD of delta_w_avg_layer.
    """
    # Ensure delta_w_avg_layer has the correct dimensions (out_features, in_features)
    # The weight matrix from nn.Linear is typically (out_features, in_features)
    # If it's Conv2d, it's (out_channels, in_channels/groups, kH, kW)
    # We need to flatten Conv2d weights if they are to be treated like linear.
    # The paper's SVD formulation implies 2D matrices.
    
    if len(delta_w_avg_layer.shape) > 2: # Handle Conv2d type weights
        original_shape = delta_w_avg_layer.shape
        # For simplicity, reshape to 2D: (out_channels, in_channels * kH * kW)
        delta_w_avg_layer_2d = delta_w_avg_layer.reshape(original_shape[0], -1)
    else:
        delta_w_avg_layer_2d = delta_w_avg_layer

    U, S, V = perform_truncated_svd(delta_w_avg_layer_2d, rank)
    
    # Paper's initialization:
    # B_init = U[1:r]
    # A_init = V[1:r]
    # R_init = (1/s) * S[1:r, 1:r]

    # U is m x r, S is r x r (diagonal), V is n x r (from Vh transpose)
    # LoRASBLayer expects:
    # B: out_features x rank
    # A: rank x in_features
    # R: rank x rank

    # If delta_w_avg_layer is (out_features, in_features):
    # U is (out_features, rank), S is (rank, rank), V is (in_features, rank)
    # So U corresponds to B
    # V.T corresponds to A
    
    # B_init is U[:, :rank]
    # A_init is V[:, :rank].T (from original Vh) or V_trunc.T if V_trunc is (in_features, rank)
    
    # Let's align with typical SVD output and LoRA-SB layer:
    # U is (m, r), S is (r,), Vh is (r, n)
    # A: r x n (matrix Vh)
    # B: m x r (matrix U)
    # R: r x r (diagonal matrix from S)

    # In `perform_truncated_svd`, we return U_trunc, S_trunc (diag), V_trunc (which is V, not Vh)
    # U_trunc: (out_features, rank) - This is B
    # V_trunc: (in_features, rank) - We need V_trunc.T for A
    # S_trunc: (rank, rank) - This is a diagonal matrix, we need to multiply by (1/s) for R

    # B_init = U
    lora_sb_layer.B.data = U
    
    # A_init = V.T
    lora_sb_layer.A.data = V.T # A needs to be rank x in_features

    # R_init = (1/s) * S (diagonal matrix)
    lora_sb_layer.R.data = (1.0 / scaling_factor) * S

    # If original_weight was Conv2d, need to ensure the shapes match expectations for LoRASBLayer
    # This might require more sophisticated handling for Conv2d or restricting LoRA-SB to Linear layers
    # For now, assuming it's flattened for SVD and the output of SVD is compatible with Linear-like.
    # The paper's formulation W = W0 + sBRA assumes 2D matrices.
    # Conv2d weights are 4D. This is a common challenge in PEFT for Conv2d.
    # If the LoRASBLayer was initialized with Conv2d, its B, A, R need to adapt.
    # The current LoRASBLayer assumes 2D weights (out_features, in_features).
    # If delta_w_avg_layer was from a Conv2d, its 'in_features' is actually in_channels * kH * kW.
    # This implies that the 'in_features' and 'out_features' need to be carefully handled for Conv2d.
    # For simplicity, assuming LoRA-SB is primarily applied to Linear layers as in many LLMs.
    
    # If delta_w_avg_layer came from a Conv2d and was flattened, B and A will have corresponding flattened dimensions.
    # This needs to be correctly handled in the LoRASBLayer's forward pass if Conv2d.
    # For now, proceeding with the assumption of 2D weights for SVD and then assigning to B, A, R.
    # The LoRASBLayer's forward for Conv2d needs to re-reshape the updated_weight correctly.
    # This will be addressed when integrating LoRASBLayer with Conv2d.
