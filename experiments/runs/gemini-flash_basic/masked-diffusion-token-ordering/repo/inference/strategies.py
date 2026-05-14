import torch
import random

def vanilla_mdm_inference_step(mdm_model, x_t, s, t):
    """
    Implements a single step of vanilla MDM inference as described in the paper (lines 121-123).
    (a) Sample a set of masked tokens S randomly.
    (b) For each i in S, sample x_s^i ~ p_theta(x^i | x_t).

    Args:
        mdm_model (MDM): The conceptual MDM model.
        x_t (torch.Tensor): Current sequence with masked tokens (0 indicates masked).
                            Shape: (batch_size, sequence_length).
        s (float): The next noise level (s < t).
        t (float): The current noise level.

    Returns:
        torch.Tensor: The sequence x_s after unmasking some tokens. Shape: (batch_size, sequence_length).
    """
    batch_size, seq_len = x_t.shape
    x_s = x_t.clone() # Start with current state

    # Identify masked positions (where token is 0)
    # Assuming 0 is the mask token as stated in line 45
    masked_positions = (x_t == 0)

    # Calculate probability of a masked token being sampled for unmasking
    # P(i in S) = (alpha_s - alpha_t) / (1 - alpha_t) as per line 121
    alpha_s = mdm_model.get_alpha_t(s)
    alpha_t = mdm_model.get_alpha_t(t)

    if alpha_t >= 1.0:
        # Handle cases where alpha_t might be 1 (fully clean, no masks to sample)
        # or very close to 1, leading to division by zero or negative probabilities.
        # This happens early in the reverse process or if alpha_t is not strictly decreasing.
        unmask_prob_per_token = 0.0
    else:
        unmask_prob_per_token = (alpha_s - alpha_t) / (1.0 - alpha_t)

    # (a) Sample a set of masked tokens S
    # For each masked position, decide if it should be unmasked in this step
    # This is a conceptual sampling. In a real implementation, this would involve
    # sampling a certain number of tokens or applying the probability per position.
    # Here, we'll simulate sampling based on unmask_prob_per_token.
    
    # Create a tensor of probabilities for each masked position
    # This is a simplification; the paper implies a coordinate-independent process
    # leading to sampling a set S.
    # For simplicity, we randomly select a subset of currently masked positions to unmask.
    
    # Get indices of currently masked positions for each item in batch
    for b in range(batch_size):
        current_masked_indices = torch.nonzero(masked_positions[b], as_tuple=False).squeeze(-1)
        if current_masked_indices.numel() == 0:
            continue

        # Number of tokens to unmask in this step for this sequence
        # This is a heuristic. The exact K is often decided by a schedule.
        # Let's say we unmask a proportion of the currently masked tokens.
        num_to_unmask = max(1, int(unmask_prob_per_token * current_masked_indices.numel()))
        
        # Randomly select positions to unmask
        # Ensure num_to_unmask does not exceed available masked tokens
        num_to_unmask = min(num_to_unmask, current_masked_indices.numel())
        
        if num_to_unmask == 0:
            continue

        sampled_unmask_indices_global = random.sample(current_masked_indices.tolist(), num_to_unmask)
        sampled_unmask_indices_global = torch.tensor(sampled_unmask_indices_global, device=x_t.device)

        if sampled_unmask_indices_global.numel() > 0:
            # (b) Assign token values to the selected positions
            # Get predicted probabilities for x_0 given x_t for the current batch item
            # We need to pass the current x_t (with batch dim) to the denoising network.
            log_probs_all_positions = mdm_model.p_theta(x_t[b].unsqueeze(0)).squeeze(0) # (seq_len, vocab_size)

            # Get predicted probabilities only for the positions to be unmasked
            predicted_log_probs_for_unmasking = log_probs_all_positions[sampled_unmask_indices_global, :]

            # Sample tokens from these predicted probabilities
            # Use categorical sampling
            sampled_tokens = torch.distributions.Categorical(logits=predicted_log_probs_for_unmasking).sample()

            # Update x_s with the sampled tokens
            x_s[b, sampled_unmask_indices_global] = sampled_tokens

    return x_s


def _select_k_positions(probabilities, k, masked_positions):
    """
    Helper function to select K positions based on some metric.
    Only considers currently masked positions.
    """
    # Filter probabilities to only include masked positions
    relevant_probabilities = probabilities.clone()
    relevant_probabilities[~masked_positions] = -torch.inf # Mask out non-masked positions

    # Get top K indices. If fewer than K masked positions, take all of them.
    num_masked = torch.sum(masked_positions).item()
    actual_k = min(k, num_masked)
    
    if actual_k == 0:
        return torch.tensor([], dtype=torch.long, device=probabilities.device)

    # Get the indices of the top 'actual_k' probabilities among masked positions
    _, top_indices = torch.topk(relevant_probabilities, actual_k, dim=-1)
    return top_indices

def adaptive_mdm_inference_step(mdm_model, x_t, k, ordering_oracle):
    """
    Implements a single step of adaptive MDM inference (lines 211-213).
    (a) Sample a set of masked tokens S = F(theta, x_t) strategically.
    (b) For each i in S, sample x_s^i ~ p_theta(x^i | x_t).

    Args:
        mdm_model (MDM): The conceptual MDM model.
        x_t (torch.Tensor): Current sequence with masked tokens (0 indicates masked).
                            Shape: (batch_size, sequence_length).
        k (int): Number of tokens to unmask in this step.
        ordering_oracle (str): 'top_probability' or 'top_probability_margin'.

    Returns:
        torch.Tensor: The sequence x_s after unmasking some tokens. Shape: (batch_size, sequence_length).
    """
    batch_size, seq_len = x_t.shape
    x_s = x_t.clone()

    # Identify masked positions (where token is 0)
    masked_positions = (x_t == 0)

    # Get predicted log probabilities for all positions from the denoising network
    # Shape: (batch_size, seq_len, vocab_size)
    log_probs_all_positions = mdm_model.p_theta(x_t)
    probs_all_positions = torch.exp(log_probs_all_positions)

    for b in range(batch_size):
        current_masked_positions = masked_positions[b]
        if torch.sum(current_masked_positions) == 0:
            continue # No tokens to unmask for this sequence

        selected_indices = None
        if ordering_oracle == 'top_probability':
            # Top probability: max probability assigned to any value (lines 225-226)
            # For each position, find the maximum probability over its vocabulary.
            max_probs_per_position = torch.max(probs_all_positions[b], dim=-1).values # (seq_len,)
            selected_indices = _select_k_positions(max_probs_per_position, k, current_masked_positions)

        elif ordering_oracle == 'top_probability_margin':
            # Top probability margin: difference between the two most probable values (lines 229-230)
            # For each position, sort probabilities and take the difference of the top two.
            sorted_probs_per_position, _ = torch.sort(probs_all_positions[b], dim=-1, descending=True)
            # Ensure there are at least two values to compare (vocab_size >= 2)
            if mdm_model.vocab_size < 2:
                # Fallback to top_probability if not enough vocab for margin
                max_probs_per_position = torch.max(probs_all_positions[b], dim=-1).values
                margins_per_position = max_probs_per_position # Simplification
            else:
                margins_per_position = sorted_probs_per_position[:, 0] - sorted_probs_per_position[:, 1]
            selected_indices = _select_k_positions(margins_per_position, k, current_masked_positions)
        else:
            raise ValueError(f"Unknown ordering oracle: {ordering_oracle}")

        if selected_indices.numel() > 0:
            # (b) Assign token values to the selected positions
            predicted_log_probs_for_unmasking = log_probs_all_positions[b, selected_indices, :]
            sampled_tokens = torch.distributions.Categorical(logits=predicted_log_probs_for_unmasking).sample()
            x_s[b, selected_indices] = sampled_tokens

    return x_s

# Helper function to generate an initial fully masked sequence
def generate_fully_masked_sequence(sequence_length, batch_size, device='cpu'):
    return torch.full((batch_size, sequence_length), 0, dtype=torch.long, device=device)

# Conceptual alpha_t schedule function (linear schedule for simplicity)
def conceptual_alpha_schedule(t):
    """
    A conceptual linear alpha_t schedule where alpha_0=1 and alpha_1=0.
    The paper does not specify the exact schedule, so this is a simple approximation.
    """
    return 1.0 - t

# Full sampling loop (conceptual)
def mdm_sampling(mdm_model, num_steps, k_unmask_per_step, strategy, initial_x_t=None, verbose=False):
    """
    Conceptual full MDM sampling process.

    Args:
        mdm_model (MDM): The conceptual MDM model.
        num_steps (int): Number of denoising steps.
        k_unmask_per_step (int): Number of tokens to unmask in each adaptive step.
        strategy (str): 'vanilla', 'top_probability', or 'top_probability_margin'.
        initial_x_t (torch.Tensor, optional): Starting state. If None, starts with a fully masked sequence.
                                              Shape: (batch_size, sequence_length).
        verbose (bool): If True, print progress.

    Returns:
        torch.Tensor: The final generated sequence.
    """
    batch_size, seq_len = initial_x_t.shape if initial_x_t is not None else (1, mdm_model.sequence_length)
    
    if initial_x_t is None:
        x_t = generate_fully_masked_sequence(mdm_model.sequence_length, batch_size, device='cpu')
    else:
        x_t = initial_x_t.clone()

    # Discretize time steps from 1 down to 0
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1)

    for i in range(num_steps):
        t_current = timesteps[i].item()
        t_next = timesteps[i+1].item()

        if verbose:
            print(f"Sampling step {i+1}/{num_steps}, t={t_current:.4f} -> s={t_next:.4f}")

        if strategy == 'vanilla':
            x_t = vanilla_mdm_inference_step(mdm_model, x_t, t_next, t_current)
        elif strategy in ['top_probability', 'top_probability_margin']:
            x_t = adaptive_mdm_inference_step(mdm_model, x_t, k_unmask_per_step, strategy)
        else:
            raise ValueError(f"Unknown inference strategy: {strategy}")
            
        # Check if all tokens are unmasked (no more 0s)
        if torch.sum(x_t == 0) == 0:
            if verbose:
                print("All tokens unmasked. Early stopping.")
            break

    return x_t

