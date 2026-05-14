import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from typing import Any, Callable, Sequence, Tuple, Optional, List

# Assuming utils.py is in the same directory or accessible via PYTHONPATH
from utils import rfft_transform, irfft_transform, pad_input


class FourierBlock(nn.Module):
    """
    A single Fourier layer (block) in a Fourier Neural Operator.
    Implements spectral convolutions and pointwise operations as described in the paper.
    """
    modes: int  # Number of Fourier modes to keep per spatial dimension
    hidden_dims: int  # Number of channels (d_v')
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu # Sigma function (sigma^(l) in paper)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Forward pass for a Fourier Block.

        Args:
            x: Input tensor, typically shape (batch, spatial_x, [spatial_y], hidden_dims).
               `spatial_x` and `spatial_y` can vary due to the resolution-agnostic nature.

        Returns:
            A tuple (activated_output, pre_activation_output):
            - activated_output: Output after activation (v^(l+1) in paper).
            - pre_activation_output: Output before activation (z^(l+1) in paper),
                                     crucial for LUNO's last-layer linearization.
        """
        # Ensure x is float32 for numerical stability and consistency
        x = x.astype(jnp.float32)

        # Determine spatial axes for FFT based on input dimensions
        # x.ndim will be 3 for 1D (batch, X, channels) or 4 for 2D (batch, X, Y, channels)
        spatial_axes = tuple(range(1, x.ndim - 1))

        # 1. Spectral Convolution part
        # Perform Real Fast Fourier Transform (RFFT)
        x_ft = rfft_transform(x, spatial_axes) # complex-valued

        # Initialize output Fourier tensor (will be filled with transformed modes)
        out_ft = jnp.zeros_like(x_ft, dtype=jnp.complex64)

        if len(spatial_axes) == 1: # 1D case: (batch, modes_x_half_plus_1, hidden_dims)
            # Truncate to specified modes
            x_ft_truncated = x_ft[:, :self.modes, :]

            # Define complex spectral weights (R_k) of shape (modes, hidden_dims, hidden_dims)
            # Implemented as real and imaginary components for Flax compatibility
            r_w_real = self.param('spectral_weights_1d_real',
                                  nn.initializers.normal(stddev=0.02),
                                  (self.modes, self.hidden_dims, self.hidden_dims),
                                  jnp.float32)
            r_w_imag = self.param('spectral_weights_1d_imag',
                                  nn.initializers.normal(stddev=0.02),
                                  (self.modes, self.hidden_dims, self.hidden_dims),
                                  jnp.float32)

            # Apply complex multiplication: (a+bi)*(c+di) = (ac-bd) + (ad+bc)i
            x_ft_real = x_ft_truncated.real
            x_ft_imag = x_ft_truncated.imag
            
            out_ft_real = jnp.einsum('bkc,kcd->bkd', x_ft_real, r_w_real) - \
                          jnp.einsum('bkc,kcd->bkd', x_ft_imag, r_w_imag)
            out_ft_imag = jnp.einsum('bkc,kcd->bkd', x_ft_real, r_w_imag) + \
                          jnp.einsum('bkc,kcd->bkd', x_ft_imag, r_w_real)
            
            out_ft_truncated = out_ft_real + 1j * out_ft_imag

            # Pad the transformed modes back to the original FFT shape
            # using .at[...].set for efficient immutable array updates
            out_ft = out_ft.at[:, :self.modes, :].set(out_ft_truncated)

        else: # 2D case: (batch, modes_x, modes_y_half_plus_1, hidden_dims)
            modes_x = self.modes
            # The last spatial dimension of rfftn output is reduced to N//2 + 1.
            # We must truncate the modes for this dimension correctly.
            modes_y_fft_output = x_ft.shape[-2] # W_half_plus_1
            modes_y_to_use = min(self.modes, modes_y_fft_output)
            
            # Truncate to specified modes
            x_ft_truncated = x_ft[:, :modes_x, :modes_y_to_use, :]

            # Define complex spectral weights (R_k) of shape (modes_x, modes_y, hidden_dims, hidden_dims)
            r_w_real = self.param('spectral_weights_2d_real',
                                  nn.initializers.normal(stddev=0.02),
                                  (modes_x, modes_y_to_use, self.hidden_dims, self.hidden_dims),
                                  jnp.float32)
            r_w_imag = self.param('spectral_weights_2d_imag',
                                  nn.initializers.normal(stddev=0.02),
                                  (modes_x, modes_y_to_use, self.hidden_dims, self.hidden_dims),
                                  jnp.float32)
            
            # Apply complex multiplication
            x_ft_real = x_ft_truncated.real
            x_ft_imag = x_ft_truncated.imag
            
            out_ft_real = jnp.einsum('bxyc,xycd->bxyd', x_ft_real, r_w_real) - \
                          jnp.einsum('bxyc,xycd->bxyd', x_ft_imag, r_w_imag)
            out_ft_imag = jnp.einsum('bxyc,xycd->bxyd', x_ft_real, r_w_imag) + \
                          jnp.einsum('bxyc,xycd->bxyd', x_ft_imag, r_w_real)
            
            out_ft_truncated = out_ft_real + 1j * out_ft_imag

            # Pad the transformed modes back to the original FFT shape
            out_ft = out_ft.at[:, :modes_x, :modes_y_to_use, :].set(out_ft_truncated)

        # Perform Inverse Real Fast Fourier Transform (IRFFT)
        original_spatial_dims = x.shape[1:-1] # E.g., (spatial_x,) for 1D, (spatial_x, spatial_y) for 2D
        x_spectral_out = irfft_transform(out_ft, spatial_axes, original_spatial_dims)
        x_spectral_out = x_spectral_out.astype(jnp.float32) # Convert back to float

        # 2. Pointwise Convolution part (W_ij^(l) * v_j^(l))
        # Applies a linear transformation to the channel dimension at each spatial point.
        x_pointwise_out = nn.Dense(features=self.hidden_dims,
                                    kernel_init=nn.initializers.normal(stddev=0.02),
                                    name="pointwise_conv")(x)

        # 3. Sum spectral and pointwise outputs (before activation)
        pre_activation_output = x_spectral_out + x_pointwise_out

        # 4. Apply activation function (sigma^(l))
        activated_output = self.activation(pre_activation_output)

        return activated_output, pre_activation_output


class FNO(nn.Module):
    """
    Fourier Neural Operator (FNO) model. This Flax Linen Module implements the FNO
    architecture for learning maps between function spaces, primarily for PDE solutions.
    """
    modes: int = 12
    hidden_dims: int = 18
    num_fourier_blocks: int = 4
    output_channels: int = 1
    initial_time_steps: int = 10  # Number of initial time steps provided as input
    input_padding: int = 2  # Number of zero grid points to pad on each side of spatial dims
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu # Default activation for Fourier blocks

    @nn.compact
    def _apply_fno_layers(
        self, x_in: jnp.ndarray, conditions: jnp.ndarray, return_last_z: bool
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """
        Helper method to apply FNO layers (lifting, Fourier blocks, projection)
        and handle input/output padding.

        Args:
            x_in: Input state tensor. Shape: (batch, initial_time_steps, spatial_x, [spatial_y], 1).
                  This represents the history of the scalar field.
            conditions: Conditions tensor. Shape: (batch, spatial_x, [spatial_y], channels_cond).
                        This represents additional fields (e.g., velocity, reaction term)
                        that are constant over the input `initial_time_steps`.
            return_last_z: If True, the pre-activation output (z^(L-1)) of the last
                           Fourier block is returned along with the final output.

        Returns:
            A tuple (final_output_unpadded, z_last_block_output_unpadded).
            - final_output_unpadded: The FNO's final prediction, with padding removed.
                                     Shape: (batch, spatial_x, [spatial_y], output_channels).
            - z_last_block_output_unpadded: The pre-activation output of the last
                                            Fourier block, with padding removed,
                                            or None if `return_last_z` is False.
                                            Shape: (batch, spatial_x, [spatial_y], hidden_dims).
        """
        # Ensure x_in is float32
        x_in = x_in.astype(jnp.float32)

        # 1. Input Preprocessing: Flatten initial_time_steps into the channel dimension
        # x_in shape: (batch, initial_time_steps, spatial_x, [spatial_y], 1)
        # Target shape for concatenation: (batch, spatial_x, [spatial_y], initial_time_steps)
        # Determine spatial dimensions from x_in's shape
        spatial_dims_start_idx = 2
        original_spatial_shape = x_in.shape[spatial_dims_start_idx:-1]

        # Reshape x_in to merge the initial_time_steps with the last channel dimension
        x_flat_channels = x_in.reshape(x_in.shape[0], *original_spatial_shape, -1)
        # x_flat_channels shape: (batch, spatial_x, [spatial_y], initial_time_steps)

        # Pad the flattened state and the conditions spatially
        # The `conditions` input is assumed to be unpadded, and its spatial dimensions
        # match `x_in`'s spatial dimensions.
        spatial_axes_to_pad = tuple(range(1, x_flat_channels.ndim - 1)) # (1,) for 1D, (1,2) for 2D
        padded_x_flat = pad_input(x_flat_channels, self.input_padding, spatial_axes_to_pad)
        padded_conditions = pad_input(conditions, self.input_padding, spatial_axes_to_pad)

        # Concatenate padded state and padded conditions along the last (channel) dimension
        # This forms the input to the lifting layer (v0 in paper's notation for first layer input).
        # v0 shape: (batch, spatial_x_padded, [spatial_y_padded], initial_time_steps + channels_cond)
        v0 = jnp.concatenate([padded_x_flat, padded_conditions], axis=-1)

        # 2. Lifting Layer (p in paper)
        # Maps the combined input channels to the FNO's internal hidden_dims.
        v = nn.Dense(features=self.hidden_dims, name="lifting")(v0)

        z_last_block_output: Optional[jnp.ndarray] = None # To store pre-activation output for LUNO

        # 3. Fourier Blocks (L layers)
        for i in range(self.num_fourier_blocks):
            # Each FourierBlock returns (activated_output, pre_activation_output)
            v, z_block = FourierBlock(
                modes=self.modes,
                hidden_dims=self.hidden_dims,
                activation=self.activation,
                name=f"fourier_block_{i}"
            )(v)
            # Store the pre-activation output if it's the last block and requested
            if return_last_z and i == self.num_fourier_blocks - 1:
                z_last_block_output = z_block

        # 4. Projection Layer (q in paper)
        # Maps from hidden_dims back to the desired output_channels.
        # Common FNO projection uses two Dense layers with an activation in between.
        # The input `v` here is the activated output of the last FourierBlock.
        v_out = nn.Dense(features=self.hidden_dims, name="projection_dense1")(v)
        v_out = self.activation(v_out) # Apply activation (e.g., GELU)
        final_output_padded = nn.Dense(features=self.output_channels, name="projection_dense2")(v_out)

        # 5. Post-processing: Remove padding from the final output and z_last_block_output
        # The original unpadded spatial shape is required for slicing.
        # Padded dimension = Original dimension + 2 * input_padding.
        # Original dimension = Padded dimension - 2 * input_padding.
        
        padded_spatial_shape = final_output_padded.shape[1:-1]
        unpadded_spatial_shape = tuple(dim - 2 * self.input_padding for dim in padded_spatial_shape)
        
        # Calculate slicing indices
        slice_start = self.input_padding
        slice_end_x = self.input_padding + unpadded_spatial_shape[0]

        if len(unpadded_spatial_shape) == 1: # 1D spatial data
            final_output_unpadded = final_output_padded[:, slice_start:slice_end_x, :]
            if z_last_block_output is not None:
                z_last_block_output_unpadded = z_last_block_output[:, slice_start:slice_end_x, :]
            else:
                z_last_block_output_unpadded = None
        else: # 2D spatial data
            slice_end_y = self.input_padding + unpadded_spatial_shape[1]
            final_output_unpadded = final_output_padded[:, slice_start:slice_end_x, slice_start:slice_end_y, :]
            if z_last_block_output is not None:
                z_last_block_output_unpadded = z_last_block_output[:, slice_start:slice_end_x, slice_start:slice_end_y, :]
            else:
                z_last_block_output_unpadded = None
            
        return final_output_unpadded, z_last_block_output_unpadded

    @nn.compact
    def __call__(self, x_in: jnp.ndarray, conditions: jnp.ndarray) -> jnp.ndarray:
        """
        Performs the full forward pass of the FNO model for next-step prediction.

        Args:
            x_in: Input state tensor. Shape: (batch, initial_time_steps, spatial_x, [spatial_y], 1).
            conditions: Conditions tensor. Shape: (batch, spatial_x, [spatial_y], channels_cond).

        Returns:
            The predicted next time step (output function).
            Shape: (batch, spatial_x, [spatial_y], output_channels).
        """
        # Call the internal helper method, indicating not to return the last block's 'z'
        final_output, _ = self._apply_fno_layers(x_in, conditions, return_last_z=False)
        return final_output

    @nn.compact
    def get_last_block_pre_activation_output(
        self, x_in: jnp.ndarray, conditions: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Returns the pre-activation output (z^(L-1)) of the last Fourier block.
        This is a specific interface required for LUNO's last-layer linearization,
        where linearization occurs around this intermediate output.

        Args:
            x_in: Input state tensor. Shape: (batch, initial_time_steps, spatial_x, [spatial_y], 1).
            conditions: Conditions tensor. Shape: (batch, spatial_x, [spatial_y], channels_cond).

        Returns:
            The unpadded pre-activation output (z^(L-1)).
            Shape: (batch, spatial_x, [spatial_y], hidden_dims).
        """
        # Call the internal helper method, indicating to return the last block's 'z'
        _, z_last_block_output = self._apply_fno_layers(x_in, conditions, return_last_z=True)
        # Assert that the output was indeed captured (should always be true if return_last_z is True)
        assert z_last_block_output is not None, "Last block pre-activation output was not captured."
        return z_last_block_output

