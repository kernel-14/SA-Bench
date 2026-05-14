## model.py

from typing import Any, Dict, Tuple
import jax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict


class FourierLayer(nn.Module):
    """
    Represents a single Fourier layer which applies FFT transformations,
    spectral convolution, and inverse FFT back to spatial domain.
    """
    modes: int  # Number of Fourier modes used for spectral convolution.
    hidden_dimensions: int  # Number of hidden dimensions in input/output.

    def setup(self):
        """
        Initialize trainable weights for Fourier layer.
        Two sets of weights are maintained for real and imaginary parts.
        """
        self.weights_real = self.param(
            "weights_real",
            nn.initializers.xavier_uniform(),
            (self.modes, self.hidden_dimensions, self.hidden_dimensions),
        )
        self.weights_imag = self.param(
            "weights_imag",
            nn.initializers.xavier_uniform(),
            (self.modes, self.hidden_dimensions, self.hidden_dimensions),
        )

    def spectral_convolution(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """
        Perform spectral convolution in the Fourier domain.

        Args:
            inputs (jnp.ndarray): Input features in the Fourier domain.

        Returns:
            jnp.ndarray: Convolved features in the Fourier domain.
        """
        real_part = jnp.einsum(
            "bkm,mhi->bki", inputs.real[..., : self.modes], self.weights_real
        )
        imag_part = jnp.einsum(
            "bkm,mhi->bki", inputs.imag[..., : self.modes], self.weights_imag
        )
        return real_part + 1j * imag_part

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through the Fourier layer.

        Args:
            inputs (jnp.ndarray): Input features in the spatial domain.

        Returns:
            jnp.ndarray: Output features in the spatial domain.
        """
        # Forward FFT
        inputs_fft = jax.numpy.fft.rfft(inputs, axis=1)  # Shape: (Batch, Modes, Features)
        
        # Apply spectral convolution
        convolved_fft = self.spectral_convolution(inputs_fft)
        
        # Inverse FFT
        outputs = jax.numpy.fft.irfft(convolved_fft, n=inputs.shape[1], axis=1)
        
        return outputs


class FourierNeuralOperator(nn.Module):
    """
    Implements a Fourier Neural Operator (FNO) architecture.
    Handles lifting, Fourier layers, and projection.
    """
    modes: int  # Number of Fourier modes for spectral convolution.
    hidden_dimensions: int  # Number of hidden dimensions in the network.
    num_blocks: int  # Number of Fourier blocks in the architecture.

    def setup(self):
        """
        Setup architecture including lifting layer, Fourier blocks, and projection layer.
        """
        # Lifting layer to project input features to high-dimensional space
        self.lifting_layer = nn.Dense(features=self.hidden_dimensions)

        # Fourier blocks
        self.fourier_blocks = [
            FourierLayer(modes=self.modes, hidden_dimensions=self.hidden_dimensions)
            for _ in range(self.num_blocks)
        ]

        # Projection layer to reduce back to input/output feature space
        self.projection_layer = nn.Dense(features=1)  # Output features assumed to be 1 (e.g., scalar PDE output)

    def forward(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """
        Forward method for a single pass through the FNO.

        Args:
            inputs (jnp.ndarray): Input tensor (Batch, Spatial Resolution, Features).

        Returns:
            jnp.ndarray: Predicted tensor of same shape as the input.
        """
        # Project input to higher dimensional feature space
        x = self.lifting_layer(inputs)  # Shape: (Batch, Spatial Resolution, Hidden Dimensions)

        # Apply Fourier blocks sequentially
        for block in self.fourier_blocks:
            x = x + block(x)  # Residual connection included for stability

        # Project back to low-dimensional output space
        outputs = self.projection_layer(x)  # Shape: (Batch, Spatial Resolution, 1)

        return outputs

    def trainable_weights(self) -> FrozenDict:
        """
        Retrieve all trainable weights (parameters) of the model.

        Returns:
            FrozenDict: Trainable weight parameters of the model.
        """
        return self.variables.get("params", {})


# Example Usage:
if __name__ == "__main__":
    # Define default configuration values for the FourierNeuralOperator
    config = {
        "modes": 12,
        "hidden_dimensions": 18,
        "num_blocks": 4,
    }

    # Initialize the model
    model = FourierNeuralOperator(
        modes=config["modes"],
        hidden_dimensions=config["hidden_dimensions"],
        num_blocks=config["num_blocks"],
    )

    # Define dummy inputs (e.g., single trajectory batch)
    key = jax.random.PRNGKey(0)
    dummy_inputs = jax.random.normal(key, (1, 256, 1))  # Single trajectory: (Batch, Grid Resolution, Features)

    # Perform forward pass
    outputs = model.init(key, dummy_inputs).apply({"params": model.trainable_weights()}, dummy_inputs)
    print("Output shape:", outputs.shape)

    # Access trainable weights
    weights = model.trainable_weights()
    print("Number of trainable parameters:", sum([np.prod(w.shape) for w in weights.values()]))
