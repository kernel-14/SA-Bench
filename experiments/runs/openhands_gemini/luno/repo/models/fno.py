
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Optional, Any

class FourierBlock(nn.Module):
    """
    A single Fourier block for the FNO.
    Applies linear transformation, Fourier transform, spectral convolution,
    inverse Fourier transform, and another linear transformation.
    """
    modes: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        # x shape: (batch, spatial_res, hidden_dim)

        # 1. Lift to higher dimension if necessary (handled by FNO main class)
        # 2. Linear transformation before Fourier operations (implicitly done by FNO main class or part of it)

        # Fourier transform
        # Move hidden_dim to the last axis if it's not already, assuming FNO applies across spatial_res
        x_ft = jnp.fft.rfft(x, axis=1) # (batch, spatial_res // 2 + 1, hidden_dim)

        # Apply spectral convolution
        # We need to filter out higher frequencies. `modes` defines the number of modes to keep.
        # This effectively slices the Fourier transformed data.
        # For 1D, we keep modes up to `modes` along the spatial frequency dimension.
        # Initialize spectral weights (R in the paper)
        # R_k_ij in the paper -> R_k for each mode k, and for each pair of input/output hidden_dim
        # The paper says R is complex: R^(l) in C^(k_max x d_v' x d_v')
        # Here, we have (modes, hidden_dim, hidden_dim) complex weights
        spectral_weights = self.param(
            'spectral_weights',
            nn.initializers.variance_scaling(1.0, 'fan_in', 'normal', dtype=jnp.complex64),
            (self.modes, x_ft.shape[-1], x_ft.shape[-1]),
            jnp.complex64
        )

        # Truncate and multiply
        out_ft = jnp.zeros_like(x_ft)
        # Only apply to the first 'modes' frequencies
        # (batch, modes, hidden_dim) @ (modes, hidden_dim, hidden_dim) is not quite right
        # It's (batch, modes, hidden_dim) * (modes, 1, hidden_dim) for filtering
        # Or, more generally, sum_j R_kij * F(v_j)_k
        # This means, for each mode k, we apply a (hidden_dim x hidden_dim) matrix.
        
        # This implementation follows the typical FNO spectral convolution:
        # For each mode k, output_ft[..., k, :] = E_k @ input_ft[..., k, :]
        # where E_k is a (hidden_dim x hidden_dim) complex matrix.
        # Or, simpler, out_ft[..., :self.modes, :] = x_ft[..., :self.modes, :] @ spectral_weights_at_k
        
        # Let's assume spectral_weights is (self.modes, hidden_dim, hidden_dim)
        # Then for each mode k in 0..self.modes-1:
        # out_ft[:, k, :] = jnp.einsum('bi,io->bo', x_ft[:, k, :], spectral_weights[k]) # No, this is for one mode.
        
        # The equation shows: (R_kij * F(v_j)_k). This implies that for a given output channel i,
        # and a given Fourier mode k, the contribution comes from a sum over input channels j.
        # So, the spectral weights are effectively applied as a linear layer in the channel dimension for each mode.
        # This is (batch, modes, hidden_dim) @ (hidden_dim, hidden_dim) for each mode k.
        # A common implementation for spectral convolution is to have learnable complex parameters for each mode,
        # which acts as a filter.
        
        # Re-interpret: R_k (d_v' x d_v') is multiplied with F(v)_k (d_v'). This means F(v)_k is a vector.
        # F(v)_k is the k-th Fourier coefficient of the vector v.
        # (batch, k_max, d_v') * (d_v', d_v') -> (batch, k_max, d_v')
        # This sounds like a set of linear layers, one for each mode.
        
        # My `spectral_weights` are (self.modes, hidden_dim, hidden_dim)
        # Slice input: x_ft_slice = x_ft[:, :self.modes, :] # (batch, modes, hidden_dim)
        # Apply transformation for each mode:
        # out_ft_slice = jax.vmap(lambda ft_slice, weights: ft_slice @ weights, in_axes=(1, 0), out_axes=1)(x_ft_slice, spectral_weights)
        # This would be (batch, modes, hidden_dim)
        
        x_ft_truncated = x_ft[:, :self.modes, :]
        out_ft_truncated = jnp.einsum('bki,kio->bko', x_ft_truncated, spectral_weights)
        out_ft = out_ft.at[:, :self.modes, :].set(out_ft_truncated)

        # Inverse Fourier transform
        x_ifft = jnp.fft.irfft(out_ft, n=x.shape[1], axis=1) # (batch, spatial_res, hidden_dim)

        # Linear transformation and skip connection (W_ij * v_j^(l)(x) in paper)
        # This is a pointwise linear layer.
        w_weights = self.param(
            'w_weights',
            nn.initializers.lecun_normal(), # Common for linear layers
            (x.shape[-1], x.shape[-1]) # (hidden_dim, hidden_dim)
        )
        x_linear = jnp.einsum('bji,io->bjo', x, w_weights)
        
        # Sum with input (skip connection)
        x_out = x_ifft + x_linear

        # Activation function (sigma in the paper)
        x_out = nn.gelu(x_out) # Common activation in FNOs, paper says sigma^(l)

        return x_out

class FNO(nn.Module):
    """
    Fourier Neural Operator (FNO) model.
    Maps input functions to output functions.
    """
    modes: int
    hidden_dim: int
    num_fourier_blocks: int
    output_dim: int
    add_pos_encoding: bool = True

    @nn.compact
    def __call__(self, x):
        # x shape: (batch, spatial_res, num_initial_steps * input_channels)
        # If add_pos_encoding is True, the input channels will be augmented.

        # Positional Encoding (if enabled)
        if self.add_pos_encoding:
            # Create a grid for positional encoding
            spatial_res = x.shape[1]
            grid = jnp.linspace(0, 1, spatial_res)[jnp.newaxis, :, jnp.newaxis] # (1, spatial_res, 1)
            grid = jnp.tile(grid, (x.shape[0], 1, 1)) # (batch, spatial_res, 1)
            x = jnp.concatenate([x, grid], axis=-1) # (batch, spatial_res, num_initial_steps*input_channels + 1)

        # Lifting layer (p in the paper)
        # Maps input features to hidden_dim
        # (batch, spatial_res, input_features) -> (batch, spatial_res, hidden_dim)
        x = nn.Dense(self.hidden_dim, name='lifting_layer')(x)

        # Fourier Blocks
        for i in range(self.num_fourier_blocks):
            x = FourierBlock(modes=self.modes, hidden_dim=self.hidden_dim, name=f'fourier_block_{i}')(x)

        # Projection layer (q in the paper)
        # Maps hidden_dim to output_dim
        x = nn.Dense(self.hidden_dim, name='projection_layer_1')(x)
        x = nn.gelu(x)
        x = nn.Dense(self.output_dim, name='projection_layer_2')(x)

        return x # (batch, spatial_res, output_dim)
