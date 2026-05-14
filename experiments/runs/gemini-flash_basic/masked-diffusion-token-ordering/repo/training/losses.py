import torch
import torch.nn.functional as F

# Equation 1: MDM Loss
def mdm_loss(log_probs_pred, x_0_masked_tokens, masked_positions_mask):
    """
    Calculates the MDM loss as defined in Equation 1 of the paper.
    L_theta = integral (alpha_t' / (1 - alpha_t)) * E_x0~p_data [ sum_{i: x_t^i=0} -log p_theta(x_0^i | x_t, t) ] dt

    In practice, for a single training step, we approximate the expectation and integral.
    The denoising network p_theta(x_0^i | x_t) predicts log probabilities for original tokens
    at masked positions.

    Args:
        log_probs_pred (torch.Tensor): Log probabilities predicted by the denoising network for all positions.
                                       Shape: (batch_size, sequence_length, vocab_size).
        x_0_masked_tokens (torch.Tensor): The true original tokens for the masked positions.
                                          Shape: (batch_size, sequence_length).
        masked_positions_mask (torch.BoolTensor): A boolean mask indicating which positions were masked.
                                                Shape: (batch_size, sequence_length).

    Returns:
        torch.Tensor: Scalar loss value.
    """
    # We only care about the loss at masked positions.
    # Gather log probabilities for the true tokens at masked positions.
    # Using NLLLoss for categorical prediction.
    
    # Select predictions for masked positions only
    # log_probs_pred_masked = log_probs_pred[masked_positions_mask]
    # x_0_true_masked = x_0_masked_tokens[masked_positions_mask]

    # More robust way: apply mask to log_probs_pred and then compute NLL
    # For a general categorical cross-entropy, we expect target to be class indices.
    # The log_probs_pred are already log_softmax outputs.
    
    # This assumes that log_probs_pred contains the log-probabilities for all tokens
    # at all positions. We need to select the log-probability of the *true* token
    # at the *masked* position.
    
    # Reshape for F.nll_loss: (N, C) for input, (N) for target
    # N = number of masked tokens across the batch and sequence
    # C = vocab_size
    
    # Flatten log_probs_pred and x_0_masked_tokens
    log_probs_flat = log_probs_pred.view(-1, log_probs_pred.size(-1)) # (batch_size * seq_len, vocab_size)
    targets_flat = x_0_masked_tokens.view(-1) # (batch_size * seq_len)
    mask_flat = masked_positions_mask.view(-1) # (batch_size * seq_len)
    
    # Select only the masked entries
    log_probs_masked_only = log_probs_flat[mask_flat]
    targets_masked_only = targets_flat[mask_flat]
    
    if targets_masked_only.numel() == 0:
        return torch.tensor(0.0, device=log_probs_pred.device, requires_grad=True)

    # Compute negative log likelihood loss for masked tokens
    # F.nll_loss expects log-probabilities, which log_softmax returns.
    loss = F.nll_loss(log_probs_masked_only, targets_masked_only, reduction='mean')
    
    # The paper's loss also includes a factor (alpha_t' / (1 - alpha_t)).
    # This factor depends on the noise schedule and would typically be applied
    # during the sampling of t. For a single step, we might integrate it
    # into the weighting of the loss or the sampling strategy.
    # For this conceptual implementation, we'll assume a 'mean' reduction handles
    # the expectation over x_0 and sum over i. The integral over t would be handled
    # by sampling t and weighting the loss.
    # We return the mean NLL over the masked tokens.
    return loss

# Equation 2: ARM Loss (Standard Left-to-Right)
def arm_loss_left_to_right(log_probs_pred, x_0):
    """
    Calculates the standard left-to-right ARM loss as defined in Equation 2 of the paper.
    log p_theta(x_0) = sum_{i=0}^{L-1} log p_theta(x_0^i | x_0[{i, ..., L-1}])

    This is essentially causal language modeling loss.

    Args:
        log_probs_pred (torch.Tensor): Log probabilities predicted by the ARM model for the *next* token.
                                       Shape: (batch_size, sequence_length, vocab_size).
        x_0 (torch.Tensor): The true original sequence.
                             Shape: (batch_size, sequence_length).

    Returns:
        torch.Tensor: Scalar loss value.
    """
    # For causal language modeling, we predict x_0[i] based on x_0[0...i-1].
    # The log_probs_pred[..., i, :] corresponds to the prediction for x_0[i].
    # The target for log_probs_pred[..., i, :] is x_0[..., i+1].
    # We usually shift the targets by one for next token prediction.

    # Targets are x_0 tokens shifted by one position.
    targets = x_0[..., 1:].contiguous()
    
    # Predictions are for positions 0 to L-2, predicting tokens 1 to L-1.
    log_probs_for_loss = log_probs_pred[..., :-1, :].contiguous()

    # Reshape for F.nll_loss
    log_probs_flat = log_probs_for_loss.view(-1, log_probs_for_loss.size(-1))
    targets_flat = targets.view(-1)

    loss = F.nll_loss(log_probs_flat, targets_flat, reduction='mean')
    return loss

# Equation 3: ARM Loss (Order-Aware / Permuted)
def arm_loss_order_aware(log_probs_pred, x_0_permuted_tokens, permutation_indices):
    """
    Calculates the order-aware ARM loss based on a given permutation.
    This is a generalization of the left-to-right ARM loss.
    
    L_theta = - E_{pi~Unif(S_L)} [ sum_{i=0}^{L-1} log p_theta(x_0^pi(i) | x_0[pi{i, ..., L-1}]) ]
    
    For a specific permutation pi, the loss is sum_{i=0}^{L-1} -log p_theta(x_0^pi(i) | x_0_masked_by_pi).

    Args:
        log_probs_pred (torch.Tensor): Log probabilities predicted by the ARM model for the *next* token
                                       in the permuted sequence. Shape: (batch_size, sequence_length, vocab_size).
        x_0_permuted_tokens (torch.Tensor): The true original sequence, permuted according to .
                                           Shape: (batch_size, sequence_length).
        permutation_indices (torch.Tensor): The permutation indices applied to x_0.
                                            Shape: (sequence_length).

    Returns:
        torch.Tensor: Scalar loss value.
    """
    # Similar to left-to-right, but the input x_0_permuted_tokens is already permuted.
    # The model predicts the next token in this permuted sequence.
    
    # Targets are x_0_permuted_tokens shifted by one position.
    targets = x_0_permuted_tokens[..., 1:].contiguous()
    
    # Predictions are for positions 0 to L-2, predicting tokens 1 to L-1 of the permuted sequence.
    log_probs_for_loss = log_probs_pred[..., :-1, :].contiguous()

    # Reshape for F.nll_loss
    log_probs_flat = log_probs_for_loss.view(-1, log_probs_for_loss.size(-1))
    targets_flat = targets.view(-1)

    loss = F.nll_loss(log_probs_flat, targets_flat, reduction='mean')
    return loss

