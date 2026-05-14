"""
Tokenizer for the Simformer.

Each variable is represented as a token consisting of:
  1. Identifier embedding (learnable vector)
  2. Value embedding (scalar value repeated to match token dimension)
  3. Metadata embedding (optional, e.g., random Fourier features for time index)
  4. Condition state embedding (learnable: True -> learnable vector, False -> zeros)

From the addendum:
  - Value embedding: scalar value repeated to match desired dimensionality
  - Condition mask embedding: True -> learnable vector, False -> zeros
  - Token = concat(identifier, value, metadata, condition_state)
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional


class RandomFourierEmbedding(nn.Module):
    """
    Random Gaussian Fourier embedding for scalar inputs (e.g., time indices).
    Used for embedding metadata (time/space coordinates) and diffusion time.
    """
    embed_dim: int
    scale: float = 1.0

    @nn.compact
    def __call__(self, x):
        # x: scalar or array of shape (...)
        # Random frequencies (fixed at init, not learned)
        B = self.param('B', nn.initializers.normal(stddev=self.scale),
                       (self.embed_dim // 2,))
        # x shape: (...,) -> (..., embed_dim)
        x_proj = x[..., None] * B[None, :] * 2 * jnp.pi
        return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)


class Tokenizer(nn.Module):
    """
    Tokenizer for the Simformer.

    Converts a sequence of (value, identifier, condition_state) tuples into tokens.

    Token = concat(id_embed, val_embed, [meta_embed], cond_embed)

    Args:
        num_variables: total number of variables (parameters + data)
        token_dim: dimension of each token component (default 50 from paper)
        use_metadata: whether to include metadata (e.g., time index) in tokens
        metadata_dim: dimension of metadata embedding
    """
    num_variables: int
    token_dim: int = 50
    use_metadata: bool = False
    metadata_dim: int = 50

    @nn.compact
    def __call__(self, values, condition_mask, metadata=None):
        """
        Args:
            values: (batch, num_variables) - the variable values
            condition_mask: (batch, num_variables) bool - True if conditioned on
            metadata: (batch, num_variables, metadata_raw_dim) optional metadata

        Returns:
            tokens: (batch, num_variables, token_dim * num_components)
        """
        batch_size, n_vars = values.shape

        # 1. Identifier embeddings: learnable vector per variable
        id_embeds = self.param(
            'id_embeds',
            nn.initializers.normal(stddev=0.02),
            (self.num_variables, self.token_dim)
        )
        # Broadcast to batch: (batch, n_vars, token_dim)
        id_tokens = jnp.broadcast_to(id_embeds[None, :n_vars, :],
                                      (batch_size, n_vars, self.token_dim))

        # 2. Value embeddings: repeat scalar to match token_dim
        # values: (batch, n_vars) -> (batch, n_vars, token_dim)
        val_tokens = jnp.repeat(values[:, :, None], self.token_dim, axis=-1)

        # 3. Condition state embeddings:
        #    True (conditioned) -> learnable vector
        #    False (latent) -> zeros
        cond_embeds = self.param(
            'cond_embeds',
            nn.initializers.normal(stddev=0.02),
            (self.num_variables, self.token_dim)
        )
        # condition_mask: (batch, n_vars) bool
        # True -> use learnable embed, False -> zeros
        cond_mask_float = condition_mask.astype(jnp.float32)
        cond_tokens = cond_mask_float[:, :, None] * cond_embeds[None, :n_vars, :]

        # 4. Concatenate components
        if metadata is not None and self.use_metadata:
            # metadata: (batch, n_vars, metadata_dim)
            tokens = jnp.concatenate([id_tokens, val_tokens, metadata, cond_tokens], axis=-1)
        else:
            tokens = jnp.concatenate([id_tokens, val_tokens, cond_tokens], axis=-1)

        return tokens


class FunctionTokenizer(nn.Module):
    """
    Tokenizer for function-valued parameters (infinite-dimensional).

    For function-valued variables, the identifier includes:
    - A shared learnable embedding vector
    - A random Fourier embedding of the index (time/space coordinate)

    This allows the model to handle arbitrary query points.
    """
    token_dim: int = 50
    fourier_dim: int = 50
    fourier_scale: float = 1.0

    @nn.compact
    def __call__(self, values, indices, condition_mask):
        """
        Args:
            values: (batch, n_points) - function values at query points
            indices: (batch, n_points) - query point coordinates (e.g., time)
            condition_mask: (batch, n_points) bool

        Returns:
            tokens: (batch, n_points, token_dim * 3 + fourier_dim)
        """
        batch_size, n_points = values.shape

        # Shared identifier embedding
        shared_id = self.param(
            'shared_id',
            nn.initializers.normal(stddev=0.02),
            (self.token_dim,)
        )
        id_tokens = jnp.broadcast_to(
            shared_id[None, None, :],
            (batch_size, n_points, self.token_dim)
        )

        # Random Fourier embedding of indices
        B = self.param(
            'fourier_B',
            nn.initializers.normal(stddev=self.fourier_scale),
            (self.fourier_dim // 2,)
        )
        idx_proj = indices[:, :, None] * B[None, None, :] * 2 * jnp.pi
        meta_tokens = jnp.concatenate([jnp.sin(idx_proj), jnp.cos(idx_proj)], axis=-1)

        # Value embedding
        val_tokens = jnp.repeat(values[:, :, None], self.token_dim, axis=-1)

        # Condition state embedding
        cond_embed = self.param(
            'cond_embed',
            nn.initializers.normal(stddev=0.02),
            (self.token_dim,)
        )
        cond_mask_float = condition_mask.astype(jnp.float32)
        cond_tokens = cond_mask_float[:, :, None] * cond_embed[None, None, :]

        tokens = jnp.concatenate([id_tokens, val_tokens, meta_tokens, cond_tokens], axis=-1)
        return tokens
