"""
LUNO: Core implementation of Linearization Turns Neural Operators into 
Function-Valued Gaussian Processes.

This module implements the main LUNO framework as described in Section 3.2 of the paper.

Key steps:
  Step 0: Start with a trained neural operator F: A x W -> U
  Step 1: Uncurry F to get f: (A x D_U) x W -> R^{d_U'}
  Step 2: Obtain Gaussian weight-space belief w ~ N(mu, Sigma)
  Step 3: Linearize f around mu to get a GP belief f ~ GP(m, K)
  Step 4: Probabilistic currying gives F ~ GP(M, K) as a function-valued GP

For FNOs, we implement the efficient last-layer LUNO from Appendix C.1.
"""

import jax
import jax.numpy as jnp
import flax.nnx as nnx
from typing import Callable, Optional, Tuple, Union, NamedTuple
from dataclasses import dataclass
import functools

from .weight_space import IsotropicGaussian, LaplaceApproximation


WeightBelief = Union[IsotropicGaussian, LaplaceApproximation]


class LUNOPrediction(NamedTuple):
    """Result of LUNO prediction.
    
    Represents the function-valued GP F(a) ~ GP(m_a, K_a) evaluated at
    a set of output points x.
    
    Attributes:
        mean: Predictive mean, shape (n_x, out_channels) or (batch, n_x, out_channels)
        variance: Marginal predictive variance, shape (n_x, out_channels) or (batch, n_x, out_channels)
        std: Predictive standard deviation (sqrt of variance)
    """
    mean: jnp.ndarray
    variance: jnp.ndarray
    std: jnp.ndarray


class LUNO:
    """LUNO: Linearized Uncertainty in Neural Operators.
    
    Converts a trained neural operator into a function-valued Gaussian process
    by linearizing around the MAP weights and propagating weight-space uncertainty.
    
    This implements the framework from Section 3.2 of the paper, with the
    efficient last-layer variant from Appendix C.1 for FNOs.
    
    Usage:
        # After training an FNO:
        luno = LUNO(model, weight_belief)
        prediction = luno.predict_marginal_variance(input_fn)
        # prediction.mean, prediction.std give the GP mean and std
    """

    def __init__(
        self,
        model: nnx.Module,
        weight_belief: WeightBelief,
        last_layer_only: bool = True,
        is_2d: bool = False,
    ):
        """
        Args:
            model: Trained neural operator (FNO1d or FNO2d)
            weight_belief: Gaussian weight-space belief (IsotropicGaussian or LaplaceApproximation)
            last_layer_only: If True, use last-layer LUNO (Appendix C.1) for efficiency.
                             Only considers uncertainty in the last Fourier block parameters.
            is_2d: Whether the model is a 2D FNO (affects batching detection)
        """
        self.model = model
        self.weight_belief = weight_belief
        self.last_layer_only = last_layer_only
        self.is_2d = is_2d

    def predict_marginal_variance(
        self,
        x: jnp.ndarray,
        output_points: Optional[jnp.ndarray] = None,
    ) -> LUNOPrediction:
        """Compute the LUNO predictive mean and marginal variance.
        
        Implements the GP prediction from Section 3.2:
        - Mean: m_a(x) = F(a, w*)(x)  [MAP prediction]
        - Variance: K_a(x, x) = J_w F(a, w)(x)|_{w*} Sigma J_w F(a, w)(x)|_{w*}^T
        
        For last-layer LUNO (Appendix C.1):
        K_a(x, x) = D_tilde_q(m_{z^{L-1}}(x)) K_{z^{L-1}}(x, x) D_tilde_q(m_{z^{L-1}}(x))^T
        
        Args:
            x: Input function discretized at grid points.
               For 1D FNO: shape (n_x, in_ch) or (batch, n_x, in_ch)
               For 2D FNO: shape (n_x, n_y, in_ch) or (batch, n_x, n_y, in_ch)
            output_points: Output evaluation points (not needed for grid-based FNOs)
        
        Returns:
            LUNOPrediction with mean and variance
        """
        # Detect if input is batched
        # 1D: unbatched = 2D array (n_x, in_ch), batched = 3D (batch, n_x, in_ch)
        # 2D: unbatched = 3D array (n_x, n_y, in_ch), batched = 4D (batch, n_x, n_y, in_ch)
        if self.is_2d:
            batched = x.ndim == 4
        else:
            batched = x.ndim == 3

        if not batched:
            x = x[None]  # Add batch dim

        if self.last_layer_only:
            return self._predict_last_layer(x, batched)
        else:
            return self._predict_full(x, batched)

    def _predict_last_layer(self, x: jnp.ndarray, batched: bool) -> LUNOPrediction:
        """Last-layer LUNO prediction (Appendix C.1).
        
        Efficient implementation exploiting the structure of the last Fourier block.
        
        The key insight is that z^{L-1}(x, w_{L-1}) is linear in w_{L-1},
        so the GP over z^{L-1} has a simple parametric form.
        
        F(a)(x) = tilde_q(m_{z^{L-1}}(x)) + D_tilde_q(m_{z^{L-1}}(x)) * (z^{L-1}(x) - m_{z^{L-1}}(x))
        
        K_a(x1, x2) = D_tilde_q(m_{z^{L-1}}(x1)) K_{z^{L-1}}(x1, x2) D_tilde_q(m_{z^{L-1}}(x2))^T
        """
        # Get last Fourier block parameters
        last_block = self.model.fourier_blocks[-1]
        last_block_graphdef, last_block_params = nnx.split(last_block)
        last_params_flat, last_unravel_fn = jax.flatten_util.ravel_pytree(last_block_params)

        # Compute v^{L-1}: input to last Fourier block
        v_prev = jax.vmap(self.model.get_last_layer_input)(x)  # (batch, ..., hidden_ch)

        # Compute mean prediction (MAP)
        mean = jax.vmap(self.model)(x)  # (batch, ..., out_ch)

        # For last-layer LUNO, we need the Jacobian of the full output
        # w.r.t. the last Fourier block parameters
        # f(a, x, w_{L-1}) = proj(last_block(v_prev, w_{L-1}))
        # where proj = proj2 o activation o proj1

        def output_from_last_params(last_params_flat, v_prev_single):
            """Compute output from last block params and v_prev for a single input."""
            last_block_params = last_unravel_fn(last_params_flat)
            last_block = nnx.merge(last_block_graphdef, last_block_params)
            # v_prev_single: (..., hidden_ch) - spatial dims + hidden channels
            z = last_block(v_prev_single[None])[0]  # (..., hidden_ch)
            # Apply projection
            h = self.model.activation(self.model.proj1(z))
            out = self.model.proj2(h)
            return out  # (..., out_ch)

        # Compute Jacobian of output w.r.t. last block parameters
        def compute_variance_single(v_prev_single):
            """Compute marginal variance for a single input."""
            # J shape: (..., out_ch, n_last_params)
            J = jax.jacobian(output_from_last_params)(last_params_flat, v_prev_single)
            # Flatten spatial dims: (n_spatial * out_ch, n_last_params)
            spatial_shape = J.shape[:-2]
            out_ch = J.shape[-2]
            n_last_params = J.shape[-1]
            n_spatial = 1
            for s in spatial_shape:
                n_spatial *= s
            J_flat = J.reshape(n_spatial * out_ch, n_last_params)

            # Compute marginal variance: diag(J @ Sigma @ J^T)
            def compute_point_variance(j_row):
                """Compute j_row @ Sigma @ j_row^T."""
                sigma_j = self.weight_belief.covariance_matvec(j_row)
                return jnp.dot(j_row, sigma_j)

            variances_flat = jax.vmap(compute_point_variance)(J_flat)  # (n_spatial * out_ch,)
            variances = variances_flat.reshape(*spatial_shape, out_ch)
            return variances

        # Compute variances for each batch element
        variances = jax.vmap(compute_variance_single)(v_prev)  # (batch, ..., out_ch)

        if not batched:
            mean = mean[0]
            variances = variances[0]

        std = jnp.sqrt(jnp.maximum(variances, 0.0))
        return LUNOPrediction(mean=mean, variance=variances, std=std)

    def _predict_full(self, x: jnp.ndarray, batched: bool) -> LUNOPrediction:
        """Full LUNO prediction using all parameters.
        
        Computes the GP prediction using the full Jacobian w.r.t. all parameters.
        More expensive than last-layer LUNO but considers all parameter uncertainty.
        """
        graphdef, params = nnx.split(self.model)
        params_flat, unravel_fn = jax.flatten_util.ravel_pytree(params)

        def model_fn(params_flat, x_single):
            params = unravel_fn(params_flat)
            model = nnx.merge(graphdef, params)
            return model(x_single[None])[0]  # Remove/add batch dim

        # Compute mean prediction
        mean = jax.vmap(lambda xi: model_fn(params_flat, xi))(x)  # (batch, ..., out_ch)

        def compute_variance_single(x_single):
            """Compute marginal variance for a single input."""
            # J: (..., out_ch, n_params)
            J = jax.jacobian(model_fn)(params_flat, x_single)
            spatial_shape = J.shape[:-2]
            out_ch = J.shape[-2]
            n_params = J.shape[-1]
            n_spatial = 1
            for s in spatial_shape:
                n_spatial *= s
            J_flat = J.reshape(n_spatial * out_ch, n_params)

            def compute_point_variance(j_row):
                sigma_j = self.weight_belief.covariance_matvec(j_row)
                return jnp.dot(j_row, sigma_j)

            variances_flat = jax.vmap(compute_point_variance)(J_flat)
            return variances_flat.reshape(*spatial_shape, out_ch)

        variances = jax.vmap(compute_variance_single)(x)  # (batch, ..., out_ch)

        if not batched:
            mean = mean[0]
            variances = variances[0]

        std = jnp.sqrt(jnp.maximum(variances, 0.0))
        return LUNOPrediction(mean=mean, variance=variances, std=std)

    def predict_covariance(
        self,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute the full predictive covariance matrix K_a(x1, x2).
        
        K_a(x1, x2) = J(x1) @ Sigma @ J(x2)^T
        
        where J(x) = D_w f(a, x, w)|_{w*} is the Jacobian of the output
        w.r.t. the parameters.
        
        Args:
            x: Input function, shape (..., in_ch) (unbatched)
        
        Returns:
            Covariance matrix, shape (n_spatial * out_ch, n_spatial * out_ch)
        """
        if self.is_2d:
            if x.ndim == 3:
                x = x[None]
        else:
            if x.ndim == 2:
                x = x[None]

        last_block = self.model.fourier_blocks[-1]
        last_block_graphdef, last_block_params = nnx.split(last_block)
        last_params_flat, last_unravel_fn = jax.flatten_util.ravel_pytree(last_block_params)

        v_prev = self.model.get_last_layer_input(x)[0]  # (..., hidden_ch)

        def output_from_last_params(last_params_flat, v_prev):
            last_block_params = last_unravel_fn(last_params_flat)
            last_block = nnx.merge(last_block_graphdef, last_block_params)
            z = last_block(v_prev[None])[0]
            h = self.model.activation(self.model.proj1(z))
            out = self.model.proj2(h)
            return out  # (..., out_ch)

        # Compute full Jacobian
        J = jax.jacobian(output_from_last_params)(last_params_flat, v_prev)
        # J shape: (..., out_ch, n_last_params)
        spatial_shape = J.shape[:-2]
        out_ch = J.shape[-2]
        n_last_params = J.shape[-1]
        n_spatial = 1
        for s in spatial_shape:
            n_spatial *= s
        J_flat = J.reshape(n_spatial * out_ch, n_last_params)  # (n_spatial * out_ch, n_last_params)

        # Compute K = J @ Sigma @ J^T
        def sigma_jt_col(j_row):
            return self.weight_belief.covariance_matvec(j_row)

        Sigma_Jt = jax.vmap(sigma_jt_col)(J_flat)  # (n_spatial * out_ch, n_last_params)
        K = J_flat @ Sigma_Jt.T  # (n_spatial * out_ch, n_spatial * out_ch)

        return K

    def sample(
        self,
        x: jnp.ndarray,
        key: jax.Array,
        n_samples: int = 1,
    ) -> jnp.ndarray:
        """Draw samples from the function-valued GP F(a).
        
        Samples are drawn by:
        1. Sampling weight perturbations delta_w ~ N(0, Sigma)
        2. Computing f_lin(a, x, w* + delta_w) = f(a, x, w*) + J(a, x) @ delta_w
        
        This gives lazy functional samples that can be evaluated at arbitrary points.
        
        Args:
            x: Input function, shape (..., in_ch) or (batch, ..., in_ch)
            key: JAX random key
            n_samples: Number of samples to draw
        
        Returns:
            Samples, shape (n_samples, ..., out_ch) or (n_samples, batch, ..., out_ch)
        """
        if self.is_2d:
            batched = x.ndim == 4
        else:
            batched = x.ndim == 3

        if not batched:
            x = x[None]

        last_block = self.model.fourier_blocks[-1]
        last_block_graphdef, last_block_params = nnx.split(last_block)
        last_params_flat, last_unravel_fn = jax.flatten_util.ravel_pytree(last_block_params)
        n_last_params = last_params_flat.shape[0]

        # Sample weight perturbations
        delta_w_samples = self.weight_belief.sample_weight_perturbation(
            key, n_last_params, n_samples
        )  # (n_samples, n_last_params)

        # Compute mean prediction
        mean = jax.vmap(self.model)(x)  # (batch, ..., out_ch)

        # Compute v_prev for each batch element
        v_prev = jax.vmap(self.model.get_last_layer_input)(x)  # (batch, ..., hidden_ch)

        def output_from_last_params(last_params_flat, v_prev_single):
            last_block_params = last_unravel_fn(last_params_flat)
            last_block = nnx.merge(last_block_graphdef, last_block_params)
            z = last_block(v_prev_single[None])[0]
            h = self.model.activation(self.model.proj1(z))
            out = self.model.proj2(h)
            return out.ravel()  # Flatten for JVP

        def compute_samples_for_input(v_prev_single, mean_single):
            """Compute samples for a single input."""
            # J: (n_out, n_last_params)
            J = jax.jacobian(output_from_last_params)(last_params_flat, v_prev_single)
            n_out = J.shape[0]
            output_shape = mean_single.shape

            # Compute J @ delta_w for each sample
            # delta_w_samples: (n_samples, n_last_params)
            perturbations = delta_w_samples @ J.T  # (n_samples, n_out)
            perturbations = perturbations.reshape(n_samples, *output_shape)

            # Add mean
            samples = mean_single[None] + perturbations  # (n_samples, ..., out_ch)
            return samples

        all_samples = jax.vmap(compute_samples_for_input)(v_prev, mean)
        # all_samples: (batch, n_samples, ..., out_ch)
        # Transpose to (n_samples, batch, ..., out_ch)
        all_samples = jnp.moveaxis(all_samples, 1, 0)

        if not batched:
            all_samples = all_samples[:, 0]  # (n_samples, ..., out_ch)

        return all_samples
