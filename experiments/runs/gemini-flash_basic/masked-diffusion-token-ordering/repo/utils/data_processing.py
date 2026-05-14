import torch
import random

def mask_sequence(x_0, mask_ratio, mask_token_id=0):
    """
    Applies masking to a sequence x_0 to create x_t.
    For MDMs, this is a coordinate-independent masking process (line 47).

    Args:
        x_0 (torch.Tensor): The original (clean) sequence. Shape: (batch_size, sequence_length).
        mask_ratio (float): The proportion of tokens to mask (corresponds to 1 - alpha_t).
        mask_token_id (int): The ID for the mask token.

    Returns:
        tuple: x_t (masked sequence) and masked_positions_mask (boolean mask).
    """
    batch_size, seq_len = x_0.shape
    masked_positions_mask = torch.rand(x_0.shape, device=x_0.device) < mask_ratio
    
    x_t = x_0.clone()
    x_t[masked_positions_mask] = mask_token_id
    
    return x_t, masked_positions_mask

def generate_loe_nae_sat_data(num_sequences, L, N, m, device='cpu'):
    """
    Generates conceptual L&O-NAE-SAT data as described in Definition 3.1 and Example 3.2.
    This is a simplified conceptual generation for demonstration.
    
    Args:
        num_sequences (int): Number of sequences to generate.
        L (int): Total sequence length (N + P).
        N (int): Number of latent tokens.
        m (int): Vocabulary size for tokens.
        device (str): Device to place the tensors.
        
    Returns:
        torch.Tensor: Generated sequences (x_0). Shape: (num_sequences, L).
    """
    assert L > N, "L must be greater than N for observation tokens to exist."
    P = L - N # Number of observation tokens
    
    # Simplified: assume identity permutation for now for simplicity of conceptual generation
    # i.e., first N tokens are latents, rest are observations.
    
    x_0 = torch.zeros((num_sequences, L), dtype=torch.long, device=device)
    
    # 1. Latent tokens: sample independently from prior distribution p_prior
    # (Assumed uniform over {1, ..., m} for simplicity)
    x_0[:, :N] = torch.randint(1, m + 1, (num_sequences, N), device=device) # Tokens from 1 to m
    
    # 2. Observation tokens: depend on latent tokens via observation functions O_j.
    # Example 3.2: Sparse predicate observations (NAE(x_i1, x_i2, x_i3))
    # This is highly complex to implement generally without specific predicate functions.
    # For conceptual purposes, we'll simplify: observation tokens are a deterministic,
    # but arbitrary, function of some latent tokens. This is not a faithful implementation
    # of NAE-SAT but captures the *idea* of dependence.
    
    # For simplicity, let's say each observation token is just a sum modulo m + 1 of 3 latent tokens
    # This is a very rough placeholder for the complexity of NAE-SAT.
    for seq_idx in range(num_sequences):
        for p_idx in range(P):
            # Select 3 random latent indices for this observation token
            # Ensure these indices are within the latent token range [0, N-1]
            latent_indices = random.sample(range(N), min(3, N)) # Pick up to 3 latents
            
            if len(latent_indices) > 0:
                # Sum of latents (conceptual operation for observation)
                # Add 1 to ensure tokens are within vocab {1, ..., m}
                obs_value = (x_0[seq_idx, latent_indices].sum() % m) + 1 
            else:
                # If no latents (e.g., N<3), just pick a random value
                obs_value = torch.randint(1, m + 1, (1,), device=device).item()
                
            x_0[seq_idx, N + p_idx] = obs_value
            
    return x_0


def apply_permutation(sequence, permutation_indices):
    """
    Applies a given permutation to a sequence.

    Args:
        sequence (torch.Tensor): The original sequence. Shape: (batch_size, sequence_length).
        permutation_indices (torch.Tensor): The permutation to apply. Shape: (sequence_length,).

    Returns:
        torch.Tensor: The permuted sequence.
    """
    return sequence[:, permutation_indices]
