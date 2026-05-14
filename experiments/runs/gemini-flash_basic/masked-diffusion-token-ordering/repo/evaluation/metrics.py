import torch
import math

def calculate_accuracy(predictions, targets, masked_positions_mask=None):
    """
    Calculates accuracy, optionally only for masked positions.
    Used for tasks like Sudoku where exact match is required.

    Args:
        predictions (torch.Tensor): Predicted token IDs. Shape: (batch_size, sequence_length).
        targets (torch.Tensor): True token IDs. Shape: (batch_size, sequence_length).
        masked_positions_mask (torch.BoolTensor, optional): If provided, accuracy is calculated only for these positions.

    Returns:
        float: Accuracy (0.0 to 1.0).
    """
    if masked_positions_mask is not None:
        # Calculate accuracy only for masked positions
        correct_predictions = (predictions == targets) & masked_positions_mask
        total_predictions = torch.sum(masked_positions_mask).item()
    else:
        # Calculate accuracy for all positions
        correct_predictions = (predictions == targets)
        total_predictions = targets.numel()

    if total_predictions == 0:
        return 0.0

    accuracy = torch.sum(correct_predictions).item() / total_predictions
    return accuracy

def calculate_generative_perplexity(log_probs_pred, targets):
    """
    Calculates generative perplexity for language models.
    PPL = exp(- (1/N) * sum(log P(token_i)))
    
    Args:
        log_probs_pred (torch.Tensor): Log probabilities for the *next* token at each position.
                                       Shape: (batch_size, sequence_length, vocab_size).
        targets (torch.Tensor): True token IDs (shifted by one for next token prediction).
                                Shape: (batch_size, sequence_length).

    Returns:
        float: Generative perplexity.
    """
    # Assuming log_probs_pred are from an ARM-like model where
    # log_probs_pred[b, i] is the log-prob distribution for targets[b, i].
    # We need to gather the log-probability of the actual target token.

    # Remove the last token from predictions as it predicts nothing
    # And the first token from targets as it has no preceding prediction
    log_probs_for_ppl = log_probs_pred[..., :-1, :]
    targets_for_ppl = targets[..., 1:]

    # Gather the log probabilities of the true tokens
    # log_probs_for_ppl: (B, L-1, V)
    # targets_for_ppl: (B, L-1)
    log_likelihoods = torch.gather(log_probs_for_ppl, -1, targets_for_ppl.unsqueeze(-1)).squeeze(-1)

    # Sum over all log likelihoods and divide by total number of tokens
    total_log_likelihood = log_likelihoods.sum()
    total_tokens = targets_for_ppl.numel()

    if total_tokens == 0:
        return float('inf') # Or a reasonable default for empty sequences

    mean_log_likelihood = total_log_likelihood / total_tokens
    perplexity = math.exp(-mean_log_likelihood.item())
    
    return perplexity

def calculate_entropy(probabilities):
    """
    Calculates the entropy of the predicted token distributions.
    H = - sum(p * log(p))

    Args:
        probabilities (torch.Tensor): Predicted token probabilities. Shape: (batch_size, sequence_length, vocab_size).

    Returns:
        float: Average entropy across all positions and batch items.
    """
    # Filter out probabilities that are zero to avoid log(0)
    probabilities = probabilities + 1e-9 # Add small epsilon to avoid log(0)
    
    # H = - sum(p * log(p)) over the vocab dimension, then average over batch and sequence
    entropy_per_position = -torch.sum(probabilities * torch.log(probabilities), dim=-1)

    return torch.mean(entropy_per_position).item()


