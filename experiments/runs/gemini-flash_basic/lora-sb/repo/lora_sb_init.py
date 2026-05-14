import torch
from torch import nn
import torch.nn.functional as F

from lora_sb_model import LoRASBLayer

def compute_delta_w_avg_and_sign(model: nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: nn.Module, num_samples: int = 50, learning_rate: float = 1e-4) -> dict[str, torch.Tensor]:
    """
    Computes the averaged first full FT gradient (approximated with sign for AdamW) for each linear layer.
    This function simulates the first step of full fine-tuning to get the initial update direction.

    Args:
        model: The full pre-trained model.
        data_loader: A DataLoader for a subset of the training data.
        loss_fn: The loss function to compute gradients.
        num_samples: The number of samples to use for averaging gradients.
        learning_rate: The learning rate for the initial update step.

    Returns:
        A dictionary mapping layer names (or identifiers) to their corresponding ΔW_avg tensors.
    """
    delta_w_avg_by_layer = {}

    original_linear_layers = {name: module for name, module in model.named_modules() if isinstance(module, nn.Linear)}

    accumulated_gradients = {name: torch.zeros_like(module.weight) for name, module in original_linear_layers.items()}
    sample_count = 0

    model.eval() # Ensure model is in eval mode during gradient collection if batch norm/dropout are present

    for batch_idx, batch in enumerate(data_loader):
        if sample_count >= num_samples:
            break

        # Prepare inputs and labels from batch
        # This part is highly dependent on your specific data structure
        inputs = batch['input_ids'] if 'input_ids' in batch else batch[0]
        labels = batch['labels'] if 'labels' in batch else batch[1]

        # Temporarily enable gradients for original weights
        for name, module in original_linear_layers.items():
            module.weight.requires_grad_(True)

        model.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()

        for name, module in original_linear_layers.items():
            if module.weight.grad is not None:
                accumulated_gradients[name] += module.weight.grad.data.clone()
            module.weight.requires_grad_(False) # Disable gradients again
        sample_count += inputs.size(0) # Assuming batch size is inputs.size(0)

    # Approximate AdamW first step: -eta * sign(summed_gradients)
    for name, grad_sum in accumulated_gradients.items():
        delta_w_avg_by_layer[name] = -learning_rate * torch.sign(grad_sum)

    return delta_w_avg_by_layer

def lora_sb_initialization(lora_sb_layer: LoRASBLayer, delta_w_avg: torch.Tensor):
    """
    Initializes B, A, and R matrices for a LoRASBLayer based on the SVD of delta_w_avg.
    This implements Equation 7, 8, and 9 from the paper.
    B_init = U[:, :r]
    A_init = Vh[:r, :]
    R_init = (1/s) * S_diag[:r, :r]
    """
    rank = lora_sb_layer.rank
    scaling_factor = lora_sb_layer.scaling_factor

    # Perform Truncated SVD: U, S, Vh = SVD(delta_w_avg)
    # delta_w_avg can be (out_features, in_features)
    # U: (out_features, out_features), S: (min(out_features, in_features)), Vh: (in_features, in_features) (V_h is V^H, conjugate transpose)
    U, S, Vh = torch.linalg.svd(delta_w_avg, full_matrices=False) # full_matrices=False for truncated SVD

    # Select the top 'rank' components
    U_r = U[:, :rank]
    S_r_diag = torch.diag_embed(S[:rank]) # Make S a diagonal matrix (rank, rank)
    Vh_r = Vh[:rank, :] # Vh is already V^H (transpose for real inputs). So we take first 'rank' rows for A.

    # Initialize B_init, A_init, R_init (Equation 7, 8, 9)
    # B_init is U[:, :r]
    lora_sb_layer.B.data = U_r

    # A_init is Vh[:r, :]
    lora_sb_layer.A.data = Vh_r

    # R_init is (1/s) * S_diag[:r, :r] (diagonal matrix)
    lora_sb_layer.R.data = (1.0 / scaling_factor) * S_r_diag

    # Freeze B and A
    lora_sb_layer.B.requires_grad_(False)
    lora_sb_layer.A.requires_grad_(False)
    # R is trainable, so its requires_grad remains True (default for nn.Parameter)


def apply_lora_sb_initialization_to_model(model: nn.Module, delta_w_avg_dict: dict[str, torch.Tensor]):
    """
    Applies the LoRA-SB initialization to all LoRASBLayer instances in the model.

    Args:
        model: The model with LoRASBLayer instances injected.
        delta_w_avg_dict: A dictionary mapping original linear layer names to their delta_w_avg tensors.
    """
    for name, module in model.named_modules():
        if isinstance(module, LoRASBLayer):
            # The name for the LoRASBLayer should correspond to the original linear layer's name
            # to fetch the correct delta_w_avg.
            # In `inject_lora_sb_layers`, we replaced the original. So the `name` here should be correct.
            # Need to convert the name to match how it would be in the original model if it was a linear layer
            # before being replaced by LoRASBLayer. Typically, the name remains the same.
            if name in delta_w_avg_dict:
                print(f"Applying LoRA-SB initialization to {name}")
                lora_sb_initialization(module, delta_w_avg_dict[name])
            else:
                print(f"Warning: No delta_w_avg found for LoRASBLayer {name}. Skipping initialization.")

