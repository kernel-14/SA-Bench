"""
utils.py
Contains utility functions crucial for JAX-based operations and numerical methods,
logging, and evaluation metrics.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core import FrozenDict
from typing import Any, Callable, Sequence, Tuple, Optional, List
import numpy as np
import os
from pathlib import Path
from absl import logging


# JAX FFT transforms for FNOs
def rfft_transform(x: jnp.ndarray, axes: Sequence[int]) -> jnp.ndarray:
    """
    Performs the N-dimensional Real Fast Fourier Transform (RFFT) on a given JAX array.

    Args:
        x: A jax.numpy.ndarray representing the input tensor (e.g., spatial domain data).
           Expected to be real-valued.
        axes: A sequence of integers specifying the dimensions along which to compute the FFT.

    Returns:
        A jax.numpy.ndarray of complex numbers representing the RFFT of x.
    """
    return jnp.fft.rfftn(x, axes=axes)


def irfft_transform(x_fft: jnp.ndarray, axes: Sequence[int], s: Sequence[int]) -> jnp.ndarray:
    """
    Performs the N-dimensional Inverse Real Fast Fourier Transform (IRFFT) on a given
    complex JAX array.

    Args:
        x_fft: A jax.numpy.ndarray of complex numbers, typically the output of rfft_transform.
        axes: A sequence of integers specifying the dimensions along which to compute the IRFFT.
              Must match the axes used for rfft_transform.
        s: A sequence of integers representing the size of the real output along the transformed axes.
           This is crucial for irfftn to correctly infer the original real input size.

    Returns:
        A jax.numpy.ndarray of real numbers representing the IRFFT of x_fft.
    """
    return jnp.fft.irfftn(x_fft, axes=axes, s=s)


# JVP function for LUNO
def get_jvp_fn(
    model_apply_fn: Callable[..., jnp.ndarray],
    params: FrozenDict,
    rng_key: jnp.ndarray,
    dummy_input: jnp.ndarray,
    dummy_conditions: jnp.ndarray,
) -> Callable[[FrozenDict], Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Creates a specialized function for computing Jacobian-Vector Products (JVP) for a given
    Flax module's method (e.g., 'apply' or 'get_last_block_output') with respect to its parameters.

    Args:
        model_apply_fn: The Flax method (e.g., fno_module.apply) for which to compute the JVP.
                        It should expect arguments `(params, input, conditions, rngs={'params': rng_key})`.
        params: The flax.core.FrozenDict of model parameters around which the JVP is linearized.
        rng_key: A JAX PRNGKey for any stochastic layers within model_apply_fn.
        dummy_input: A jax.numpy.ndarray with the shape and dtype of a typical FNO input state.
        dummy_conditions: A jax.numpy.ndarray with the shape and dtype of the combined
                          conditions (velocity, reaction terms).

    Returns:
        A callable `jvp_fn` that takes a `FrozenDict` of tangent vectors (structured like `params`)
        and returns a tuple `(primal_output: jnp.ndarray, jvp_result: jnp.ndarray)`.
    """
    # Define a target function that takes only params as input for jax.jvp
    def target_fn(p: FrozenDict) -> jnp.ndarray:
        return model_apply_fn(p, dummy_input, dummy_conditions, rngs={"params": rng_key})

    def jvp_fn(dw: FrozenDict) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Computes the JVP for the `model_apply_fn` around `params` in the direction `dw`.
        """
        return jax.jvp(target_fn, (params,), (dw,))

    return jvp_fn


# FNO Parameter Initialization
def initialize_fno_params(
    rng_key: jnp.ndarray,
    fno_module: nn.Module,
    dummy_input: jnp.ndarray,
    dummy_conditions: jnp.ndarray,
) -> FrozenDict:
    """
    Initializes the parameters of a Flax FNO module using a PRNGKey and dummy inputs
    for shape inference.

    Args:
        rng_key: A JAX PRNGKey for parameter initialization.
        fno_module: An instance of the FNO class (a flax.linen.Module).
        dummy_input: A jax.numpy.ndarray matching the expected shape and dtype of the FNO's `x` input.
        dummy_conditions: A jax.numpy.ndarray matching the expected shape and dtype of the FNO's `conditions` input.

    Returns:
        A flax.core.FrozenDict containing the initialized model parameters.
    """
    params = fno_module.init({"params": rng_key}, dummy_input, dummy_conditions)["params"]
    return params


# Evaluation Metrics
def compute_rmse(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """
    Calculates the Root Mean Squared Error (RMSE) between true values and predictions.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted mean values.
        Both are jax.numpy.ndarray and must have compatible shapes.

    Returns:
        A scalar jax.numpy.ndarray representing the RMSE.
    """
    return jnp.sqrt(jnp.mean(jnp.square(y_true - y_pred)))


def compute_nll(y_true: jnp.ndarray, y_pred_mean: jnp.ndarray, y_pred_std: jnp.ndarray) -> jnp.ndarray:
    """
    Calculates the Marginal Negative Log-Likelihood (NLL) assuming a Gaussian predictive distribution.
    The NLL is computed as the sum of negative log-probabilities across all elements,
    as implied by the paper's formula.

    Args:
        y_true: Ground truth values.
        y_pred_mean: Predicted mean values.
        y_pred_std: Predicted standard deviation values.
        All are jax.numpy.ndarray and must have compatible shapes.

    Returns:
        A scalar jax.numpy.ndarray representing the total NLL.
    """
    EPS = 1e-6  # Small constant for numerical stability
    # Ensure std is positive and avoid log(0)
    y_pred_std_stable = y_pred_std + EPS
    
    # Calculate log probability density for a Gaussian distribution
    # log(1 / sqrt(2*pi*sigma^2) * exp(- (y - mu)^2 / (2*sigma^2)))
    # = -0.5 * log(2*pi) - log(sigma) - (y - mu)^2 / (2*sigma^2)
    # The paper's formula is NLL = - sum_i log(PDF_i)
    log_pdf_term1 = -0.5 * jnp.log(2 * jnp.pi)
    log_pdf_term2 = -jnp.log(y_pred_std_stable)
    log_pdf_term3 = -jnp.square(y_true - y_pred_mean) / (2 * jnp.square(y_pred_std_stable))
    
    log_pdf = log_pdf_term1 + log_pdf_term2 + log_pdf_term3
    
    # Sum over all elements to get the total NLL as per paper's summation notation
    nll = -jnp.sum(log_pdf)
    return nll


def compute_chi_squared(y_true: jnp.ndarray, y_pred_mean: jnp.ndarray, y_pred_std: jnp.ndarray) -> jnp.ndarray:
    """
    Calculates the chi-squared statistic, which assesses the calibration of uncertainty predictions.
    A value close to 1 indicates well-calibrated uncertainty.

    Args:
        y_true: Ground truth values.
        y_pred_mean: Predicted mean values.
        y_pred_std: Predicted standard deviation values.
        All are jax.numpy.ndarray and must have compatible shapes.

    Returns:
        A scalar jax.numpy.ndarray representing the chi-squared statistic.
    """
    EPS = 1e-6  # Small constant for numerical stability
    # Ensure std is positive to prevent division by zero
    y_pred_std_stable = y_pred_std + EPS
    
    # Calculate (y - mu)^2 / sigma^2 for each point
    squared_normalized_error = jnp.square(y_true - y_pred_mean) / jnp.square(y_pred_std_stable)
    
    # Average over all elements as per paper's formula Q = (1/n) * sum_i (...)
    chi_squared_statistic = jnp.mean(squared_normalized_error)
    return chi_squared_statistic


# Logging Setup
def setup_logging(log_dir: str, experiment_name: str) -> None:
    """
    Configures the Python logging system using `absl.logging` to output messages
    to console and a file.

    Args:
        log_dir: Base directory for logs.
        experiment_name: Name of the current experiment, used for log file naming.
    """
    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Get the root logger
    root_logger = logging.get_absl_logger()
    root_logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicate logs if called multiple times
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_path / f"{experiment_name}.log")
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    logging.info(f"Logging configured. Logs will be saved to: {log_path / f'{experiment_name}.log'}")


# Array Manipulation Utilities
def pad_input(
    array: jnp.ndarray, padding_size: int, spatial_axes: Sequence[int]
) -> jnp.ndarray:
    """
    Pads the spatial dimensions of an input array with zeros. Used for FNO input preprocessing.

    Args:
        array: The input jax.numpy.ndarray to be padded.
        padding_size: The number of zero grid points to add on each side of the spatial dimensions.
        spatial_axes: A sequence of integers indicating the indices of the spatial dimensions
                      (e.g., (1, 2) for (batch, H, W, channels)).

    Returns:
        The padded jax.numpy.ndarray.
    """
    num_dims = array.ndim
    pad_width: List[Tuple[int, int]] = [(0, 0)] * num_dims

    for axis in spatial_axes:
        if 0 <= axis < num_dims:
            pad_width[axis] = (padding_size, padding_size)
        else:
            logging.warning(f"Spatial axis {axis} is out of bounds for array with {num_dims} dimensions. Skipping padding for this axis.")

    return jnp.pad(array, pad_width, mode="constant", constant_values=0)


def stack_conditions(
    state_field: jnp.ndarray,
    velocity_field: Optional[jnp.ndarray] = None,
    reaction_term_field: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Concatenates the initial state field with optional velocity and reaction terms
    along the feature (channel) dimension, forming the FNO's combined input.

    Args:
        state_field: The jax.numpy.ndarray representing the initial PDE state.
                     Expected shape: `(batch, ..., num_time_steps_in_input)`.
        velocity_field: The optional jax.numpy.ndarray for the velocity field.
                        Expected shape: `(batch, ..., num_velocity_components)`.
                        If None, it's omitted.
        reaction_term_field: The optional jax.numpy.ndarray for the reaction term.
                             Expected shape: `(batch, ..., num_reaction_components)`.
                             If None, it's omitted.

    Returns:
        A single jax.numpy.ndarray with combined features/channels.
        The last dimension will be the concatenated feature dimension.
    """
    tensors_to_stack: List[jnp.ndarray] = [state_field]

    if velocity_field is not None:
        tensors_to_stack.append(velocity_field)
    if reaction_term_field is not None:
        tensors_to_stack.append(reaction_term_field)

    # The last axis is assumed to be the channel/feature dimension
    combined_input = jnp.concatenate(tensors_to_stack, axis=-1)
    return combined_input

