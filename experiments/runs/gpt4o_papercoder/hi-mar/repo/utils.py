# utils.py

import torch
import numpy as np
from typing import Dict, Union

def generate_sinusoidal_embedding(resolution: int, d_model: int) -> torch.Tensor:
    """
    Generate sinusoidal embeddings for positional or resolution-specific encoding.

    Args:
        resolution (int): The resolution (e.g., 128 for low scale, 256 for high scale).
        d_model (int): The dimensionality of the embedding vector.

    Returns:
        torch.Tensor: Sinusoidal embeddings of shape (resolution, d_model).
    """
    position = torch.arange(resolution, dtype=torch.float32).unsqueeze(1)  # Shape: (resolution, 1)
    div_term = torch.exp(
        -np.log(10000.0) * torch.arange(0, d_model, 2, dtype=torch.float32) / d_model
    )  # Shape: (d_model / 2)

    # Sinusoidal encoding with alternating sine and cosine functions
    embedding = torch.zeros((resolution, d_model), dtype=torch.float32)  # Initialize
    embedding[:, 0::2] = torch.sin(position * div_term)  # Even indices -> sine
    embedding[:, 1::2] = torch.cos(position * div_term)  # Odd indices -> cosine

    return embedding


def build_scale_vector(resolution: int, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Generate scale-specific vectors for hierarchical prediction tasks.

    Args:
        resolution (int): The resolution for which the scale vector is being generated.
        params (Dict[str, torch.Tensor]): Dictionary containing MLP weights for scale transformation.
                                          Expected keys: `mlp_weight`, `mlp_bias`.

    Returns:
        torch.Tensor: Learned scale vector of shape (d_model,).
    """
    # Generate base sinusoidal embedding first
    sinusoidal_embedding = generate_sinusoidal_embedding(resolution, params['d_model'])  # Shape: (resolution, d_model)

    # Aggregate embeddings along resolution axis (average to condense context)
    aggregated_context = torch.mean(sinusoidal_embedding, dim=0)  # Shape: (d_model,)

    # Pass embedding through the MLP layer for transformation
    mlp_output = torch.matmul(aggregated_context, params['mlp_weight']) + params['mlp_bias']

    # Return scale-aware vector split for residual and normalization purposes
    return mlp_output


def mask_tokens(sequence: torch.Tensor, ratio: float, strategy: str = "uniform_random") -> torch.Tensor:
    """
    Apply phase-specific masking strategies to token sequences.

    Args:
        sequence (torch.Tensor): The full sequence of tokens (e.g., visual tokens from VAE).
                                 Shape: (batch_size, sequence_length, hidden_dim)
        ratio (float): The masking ratio (fraction of tokens masked).
        strategy (str): Masking strategy to be applied. Options are:
                        - "uniform_random": Randomly mask tokens uniformly.
                        - "cosine": Apply masking weights using a cosine schedule.
                        - "beta": Sample masking ratio from Beta distribution.

    Returns:
        torch.Tensor: Masked token sequence with some tokens replaced by placeholders.
    """
    batch_size, seq_length, hidden_dim = sequence.shape
    mask = torch.zeros((batch_size, seq_length), dtype=torch.bool)  # Initialize mask

    if strategy == "uniform_random":
        # Uniform Random Masking: Mask tokens based on a fixed ratio uniformly
        mask = torch.rand(batch_size, seq_length) < ratio

    elif strategy == "cosine":
        # Cosine masking across token positions
        token_pos = torch.arange(seq_length, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)  # (batch_size, seq_length)
        cosine_weights = 0.5 * (1 + torch.cos(np.pi * token_pos / seq_length))  # Weight each token position
        random_prob = torch.rand(batch_size, seq_length)
        mask = random_prob < (cosine_weights * ratio)  # Mask tokens with probabilistic thresholds

    elif strategy == "beta":
        # Beta distribution-based masking (use Beta distribution to sample ratios)
        alpha, beta = 4, 1  # Configurable parameters
        sampled_ratios = torch.distributions.Beta(alpha, beta).sample((batch_size,)).unsqueeze(1)  # One sampled ratio per batch
        masking_thresholds = (torch.arange(seq_length).unsqueeze(0) + 1) / seq_length  # Uniform threshold for masking
        mask = masking_thresholds < sampled_ratios  # Apply sample-based masking

    else:
        raise ValueError("Unknown masking strategy. Supported strategies: 'uniform_random', 'cosine', 'beta'.")

    # Apply mask to the sequence by replacing masked tokens with placeholders
    masked_sequence = sequence.clone()
    masked_placeholder = torch.zeros_like(sequence)  # Placeholder tokens (encoded zeros)
    masked_sequence[mask] = masked_placeholder[mask]

    return masked_sequence


# Helper methods for testing the utilities in isolation
if __name__ == "__main__":
    d_model = 512
    resolution = 128

    # Test sinusoidal embedding generation
    sinusoidal_embeds = generate_sinusoidal_embedding(resolution, d_model)
    print(f"Sinusoidal Embedding Shape: {sinusoidal_embeds.shape}")

    # Test scale-specific vector generation (mock parameters)
    mlp_weight = torch.randn(d_model, d_model)  # Mock MLP weight
    mlp_bias = torch.randn(d_model)  # Mock MLP bias
    params = {"mlp_weight": mlp_weight, "mlp_bias": mlp_bias, "d_model": d_model}
    scale_vector = build_scale_vector(resolution, params)
    print(f"Scale Vector Shape: {scale_vector.shape}")

    # Test uniform random masking
    batch_size, seq_length, hidden_dim = 4, 128, 512
    token_sequence = torch.randn(batch_size, seq_length, hidden_dim)
    masked_sequence = mask_tokens(token_sequence, ratio=0.75, strategy="uniform_random")
    print(f"Masked Sequence Shape: {masked_sequence.shape}")
