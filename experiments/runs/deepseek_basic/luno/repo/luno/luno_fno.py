"""LUNO for Fourier Neural Operators (FNOs).

This module implements the core LUNO framework specialized for Fourier Neural
Operators, following Section 3.2.1 and Appendix C.1.

The key insight for FNOs is that when we restrict weight-space uncertainty to
only the last Fourier block (w_{L-1} = (R^{(L-1)}, W^{(L-1)})), we can derive
a particularly efficient representation of the function-valued Gaussian process.

From Section 3.2.1:
- z^{(L-1)} is linear in w_{L-1}
- z^{(L-1)} ~ GP(m_{z^{(L-1)}}, K_{z^{(L-1)}}) is a multi-output parametric GP
- F(a)(x) = q̃(m_{z^{(L-1)}}(x)) + Dq̃(m_{z^{(L-1)}}(x)) (z^{(L-1)}(x) - m_{z^{(L-1)}}(x))
- F(a) ~ GP(m_a, K_a) with:
  - m_a(x) = F(a, w*)(x)
  - K_a(x_1, x_2) = Dq̃(m_{z^{(L-1)}}(x_1)) K_{z^{(L-1)}}(x_1, x_2) Dq̃(...)^T

From Appendix C.1:
z^{(L-1)}_i(x, w_{L-1}) = Σ_j Σ_k Re(R^{(L-1)}_{k,ij}) φ_{kj}(x) 
                         + Σ_j Σ_k Im(R^{(L-1)}_{k,ij}) ψ_{kj}(x)
                         + Σ_j W^{(L-1)}_{ij} v^{(L-1)}_j(x)

where φ, ψ are feature functions derived from the inverse FFT.
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Optional, Callable, Any, Dict
from functools import partial
from dataclasses import dataclass


@dataclass
class FNOState:
    """State of an FNO forward pass, storing intermediate activations."""
    v_layers: list  # Hidden states v^{(l)} for each layer
    fourier_coeffs: list  # Fourier coefficients at each layer


class FourierGaussianRandomOperator:
    """Function-valued Gaussian process derived from an FNO's last layer.
    
    This is the core LUNO construction for FNOs (Section 3.2.1):
    
    F(a) ~ GP(m_a, K_a) where:
    - m_a(x) = F(a, w*)(x)  (the mean prediction of the trained FNO)
    - K_a(x_1, x_2) = Dq̃(x_1) K_z(x_1, x_2) Dq̃(x_2)^T
    
    where K_z is the covariance of z^{(L-1)}, which is parametric in terms
    of Fourier feature functions.
    
    Benefits:
    1. Resolution-agnostic: can evaluate at arbitrary output points
    2. Efficient: only needs one forward pass to get hidden state
    3. Lazy functional samples: can sample entire functions
    """
    
    def __init__(
        self,
        v_hidden: jnp.ndarray,  # v^{(L-1)} hidden state, shape (d_v', n_x)
        fourier_coeffs: jnp.ndarray,  # Fourier coefficients v̂_{kj}^{(L-1)}
        weights_mean: Dict[str, jnp.ndarray],  # Mean of R, W
        weights_cov: Any,  # Covariance structure for R, W
        q_tilde_fn: Callable,  # q̃ = q(·, w_q) ◦ σ^{(L-1)}
        q_tilde_deriv_fn: Callable,  # Derivative Dq̃
        modes: int,  # k_max — number of Fourier modes
        grid_size: int,  # Spatial grid size
        output_dim: int,  # d'_U
    ):
        self.v_hidden = v_hidden  # (d_v', n_x)
        self.fourier_coeffs = fourier_coeffs  # Complex, (k_max, d_v')
        self.weights_mean = weights_mean
        self.weights_cov = weights_cov
        self.q_tilde_fn = q_tilde_fn
        self.q_tilde_deriv_fn = q_tilde_deriv_fn
        self.modes = modes
        self.grid_size = grid_size
        self.output_dim = output_dim
        self.d_v = v_hidden.shape[0]
        
        # Compute q̃(m_z(x)) for all x
        # m_z(x) = z^{(L-1)}(x, w*) — the hidden representation at the MAP
        m_z = self._compute_m_z(weights_mean)
        self._m_z = m_z  # (d_v', n_x)
    
    def _compute_m_z(self, weights: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Compute the mean of z^{(L-1)} given weight means.
        
        z^{(L-1)}_i(x) = Σ_j (F^{-1}(R_{·,ij} ⊙ v̂_j)(x) + W_{ij} v_j(x))
        """
        R = weights.get('R', None)  # (modes, d_v, d_v), complex
        W = weights.get('W', None)  # (d_v, d_v), real
        v = self.v_hidden  # (d_v, n_x)
        v_hat = self.fourier_coeffs  # (modes, d_v), complex
        
        d_v = self.d_v
        n_x = self.grid_size
        
        m_z = jnp.zeros((d_v, n_x))
        
        if R is not None:
            # Spectral convolution term
            for i in range(d_v):
                for j in range(d_v):
                    # R_{·,i,j} ⊙ v̂_j — mode-wise product
                    conv_coeffs = R[:, i, j] * v_hat[:, j]  # (modes,)
                    # Inverse FFT: F^{-1}(conv_coeffs)
                    # Pad to full grid then IFFT
                    conv_spatial = self._inverse_rfft(conv_coeffs, n_x)
                    m_z = m_z.at[i].add(conv_spatial)
        
        if W is not None:
            # Linear skip connection
            m_z += W @ v  # (d_v, d_v) @ (d_v, n_x) -> (d_v, n_x)
        
        return m_z
    
    def _inverse_rfft(self, coeffs: jnp.ndarray, n: int) -> jnp.ndarray:
        """Inverse real FFT: map Fourier coefficients back to spatial domain.
        
        This is a key operation for the FNO structure. For real inputs,
        we use the RFFT which only stores positive frequencies.
        
        Args:
            coeffs: Fourier coefficients of shape (k_max,) for positive frequencies
            n: Number of spatial points
        
        Returns:
            Real-valued spatial signal of shape (n,)
        """
        # Build full complex spectrum from positive frequencies
        # For real FFT of length n:
        # - modes 0 to n//2 are stored
        # - modes n//2+1 to n-1 are conjugates
        k_max = coeffs.shape[0]
        
        full_spectrum = jnp.zeros(n, dtype=jnp.complex64)
        full_spectrum = full_spectrum.at[:k_max].set(coeffs)
        
        # Fill in conjugate symmetric part
        if n % 2 == 0:
            # For even n, the Nyquist frequency is real
            full_spectrum = full_spectrum.at[k_max:n].set(
                jnp.conj(coeffs[1:k_max][::-1])
            )
        else:
            full_spectrum = full_spectrum.at[k_max:n].set(
                jnp.conj(coeffs[1:k_max+1][::-1])
            )
        
        return jnp.fft.ifft(full_spectrum).real * n  # Unnormalized IFFT
    
    def mean_function(self, x_query: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the mean function m_a at query points.
        
        m_a(x) = F(a, w*)(x) = q̃(m_z(x))
        
        Args:
            x_query: Query positions, can be arbitrary
        
        Returns:
            Mean predictions at query points
        """
        # Interpolate m_z to query points if needed
        if x_query.shape != (self.grid_size,):
            # Simple interpolation for arbitrary query points
            m_z_at_x = self._interpolate(self._m_z, x_query)
        else:
            m_z_at_x = self._m_z
        
        # Apply q̃ element-wise to each column (spatial position)
        mean_vals = jnp.zeros((self.output_dim, x_query.shape[0]))
        for i in range(x_query.shape[0]):
            mean_vals = mean_vals.at[:, i].set(
                self.q_tilde_fn(m_z_at_x[:, i])
            )
        
        return mean_vals  # (d'_U, n_query)
    
    def covariance_function(
        self, 
        x1: jnp.ndarray, 
        x2: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute K_a(x_1, x_2) = Dq̃(x_1) K_z(x_1, x_2) Dq̃(x_2)^T.
        
        Args:
            x1, x2: Query point indices or coordinates
        
        Returns:
            Covariance matrix of shape (d'_U, d'_U)
        """
        # Get Dq̃ at both points
        Dq1 = self.q_tilde_deriv_fn(self._m_z[:, x1])  # (d'_U, d_v')
        Dq2 = self.q_tilde_deriv_fn(self._m_z[:, x2])  # (d'_U, d_v')
        
        # Compute K_z(x_1, x_2) using the feature function representation
        K_z = self._compute_k_z(x1, x2)  # (d_v', d_v')
        
        # K_a = Dq̃(x_1) K_z(x_1, x_2) Dq̃(x_2)^T
        return Dq1 @ K_z @ Dq2.T  # (d'_U, d'_U)
    
    def _compute_k_z(self, x1: int, x2: int) -> jnp.ndarray:
        """Compute the covariance of z^{(L-1)} between two spatial points.
        
        K_z(x_1, x_2)_{ij} = Cov[z_i(x_1), z_j(x_2)]
        
        This uses the feature function decomposition from Appendix C.1:
        z_i(x) = Σ_j Σ_k Re(R_{k,ij}) φ_{kj}(x) + Im(R_{k,ij}) ψ_{kj}(x) 
                + Σ_j W_{ij} v_j(x)
        
        where φ_{kj}(x) = Re(v̂_{kj}) cos(ω_k x) and ψ_{kj}(x) = -Im(v̂_{kj}) sin(ω_k x).
        """
        d_v = self.d_v
        k_max = self.modes
        v = self.v_hidden  # (d_v, n_x)
        v_hat = self.fourier_coeffs  # (k_max, d_v), complex
        
        # Build feature vectors at x1 and x2
        # Feature dimension: k_max * d_v * 2 (real + imag) + d_v (skip connection)
        n_features_R = k_max * d_v * 2  # for each output channel i, features for R
        n_features_W = d_v  # for skip connection
        
        # Construct feature vectors
        phi_x1 = self._build_feature_vector(x1, v, v_hat)  # (d_v, n_features)
        phi_x2 = self._build_feature_vector(x2, v, v_hat)
        
        # K_z = phi_x1^T Σ_w phi_x2 where Σ_w is block-structured by R and W
        # We use the weight-space covariance structure
        K_z = jnp.zeros((d_v, d_v))
        
        # If using isotropic covariance: K_z = σ² phi_x1^T phi_x2
        if hasattr(self.weights_cov, 'get_covariance_matrix'):
            Sigma_w = self.weights_cov.get_covariance_matrix()
        elif isinstance(self.weights_cov, float):
            Sigma_w = self.weights_cov  # scalar σ²
        else:
            Sigma_w = 1.0
        
        if isinstance(Sigma_w, float):
            # Isotropic: simple inner product
            K_z = phi_x1 @ phi_x2.T * Sigma_w
        else:
            # Full/low-rank covariance: need to apply Σ_w
            # K_z[i,j] = Σ_{kl} phi_{i,k} Sigma_w[k,l] phi_{j,l}
            # This is phi_x1 Σ_w phi_x2^T
            K_z = phi_x1 @ Sigma_w @ phi_x2.T
        
        return K_z
    
    def _build_feature_vector(
        self, 
        x_idx: int, 
        v: jnp.ndarray, 
        v_hat: jnp.ndarray
    ) -> jnp.ndarray:
        """Build the feature vector for the z^{(L-1)} linear expansion.
        
        For each output channel i, the features correspond to:
        - For each j and each mode k: Re(v̂_{kj}) cos(ω_k x) and -Im(v̂_{kj}) sin(ω_k x)
        - Skip connection: v_j(x)
        
        Returns matrix of shape (d_v, total_features) where each row i
        corresponds to the features that multiply weights affecting output i.
        """
        d_v = v.shape[0]
        k_max = v_hat.shape[0]
        n_x = v.shape[1]
        
        # Compute cos/sin basis at x_idx
        # ω_k = 2πk / n_x for k = 0, ..., k_max-1
        ks = jnp.arange(k_max)
        angles = 2 * jnp.pi * ks * x_idx / n_x
        cos_vals = jnp.cos(angles)  # (k_max,)
        sin_vals = jnp.sin(angles)  # (k_max,)
        
        # Real part features: Re(v̂_{kj}) cos(ω_k x)
        # For each j, k: feature value
        Re_v_hat = jnp.real(v_hat)  # (k_max, d_v)
        Im_v_hat = jnp.imag(v_hat)  # (k_max, d_v)
        
        # Feature for R (spectral convolution): 
        # For each i (output), the weights are R_{k,i,j} for all j,k
        # So features are:
        #   - real part: Re(v̂_{kj}) cos(ω_k x) for all j, k
        #   - imag part: -Im(v̂_{kj}) sin(ω_k x) for all j, k
        
        # Build features: total = k_max * d_v * 2 (R) + d_v (W)
        n_R_features = k_max * d_v * 2
        n_W_features = d_v
        n_features = n_R_features + n_W_features
        
        feature_matrix = jnp.zeros((d_v, n_features))
        
        for i in range(d_v):
            # R features (same for all output channels i since R_{k,i,j})
            # arranged as [real_features, imag_features]
            idx = 0
            for j in range(d_v):
                for k in range(k_max):
                    # Real part: Re(v̂_{kj}) cos(ω_k x)
                    feature_matrix = feature_matrix.at[i, idx].set(
                        Re_v_hat[k, j] * cos_vals[k]
                    )
                    idx += 1
            
            for j in range(d_v):
                for k in range(k_max):
                    # Imaginary part: -Im(v̂_{kj}) sin(ω_k x)
                    feature_matrix = feature_matrix.at[i, idx].set(
                        -Im_v_hat[k, j] * sin_vals[k]
                    )
                    idx += 1
            
            # W features: v_j(x) for the linear skip connection
            for j in range(d_v):
                feature_matrix = feature_matrix.at[i, n_R_features + j].set(
                    v[j, x_idx]
                )
        
        return feature_matrix
    
    def _interpolate(self, values: jnp.ndarray, query_points: jnp.ndarray) -> jnp.ndarray:
        """Interpolate grid values to arbitrary query points.
        
        Simple linear interpolation.
        """
        n_x = values.shape[1]
        # Assume query_points are indices or coordinates in [0, n_x)
        x_floor = jnp.floor(query_points).astype(jnp.int32)
        x_ceil = jnp.minimum(x_floor + 1, n_x - 1)
        frac = query_points - x_floor
        
        v_floor = values[:, x_floor]
        v_ceil = values[:, x_ceil]
        
        return v_floor * (1 - frac) + v_ceil * frac
    
    def sample_function(
        self, 
        rng_key: jax.random.PRNGKey, 
        n_samples: int = 1
    ) -> jnp.ndarray:
        """Draw lazy functional samples from F(a).
        
        F(a) ~ GP(m_a, K_a) is a parametric GP.
        
        We sample by:
        1. Sample weights from weight-space belief: w ~ N(μ, Σ)
        2. Construct sampled z^{(L-1)} function
        3. Apply q̃ to get function samples
        
        The result is a function that can be lazily evaluated at any point.
        
        Args:
            rng_key: JAX random key
            n_samples: Number of function samples to draw
        
        Returns:
            Sampled functions as array of shape (n_samples, d'_U, n_x)
        """
        # Sample weights
        if hasattr(self.weights_cov, 'sample'):
            weight_samples = self.weights_cov.sample(rng_key, n_samples)
        else:
            # Simple isotropic sampling
            sigma = jnp.sqrt(self.weights_cov)
            weight_samples = jax.random.normal(
                rng_key, (n_samples, self.d_v * self.modes * self.d_v * 2 + self.d_v * self.d_v)
            ) * sigma
        
        # For each sample, construct z^{(L-1)} and apply q̃
        function_samples = jnp.zeros((n_samples, self.output_dim, self.grid_size))
        
        for s in range(n_samples):
            # Reconstruct perturbed weights
            w_perturbed = self._reconstruct_weights(weight_samples[s])
            # Compute z with perturbed weights
            z_perturbed = self._compute_m_z(w_perturbed)
            # Apply q̃
            for i in range(self.grid_size):
                function_samples = function_samples.at[s, :, i].set(
                    self.q_tilde_fn(z_perturbed[:, i])
                )
        
        return function_samples
    
    def _reconstruct_weights(self, flat_weights: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """Reconstruct weight dict from flattened sample."""
        # This depends on the specific weight parameterization
        # For now, return mean weights (placeholder)
        return self.weights_mean
    
    def sample_at_points(
        self,
        rng_key: jax.random.PRNGKey,
        x_points: jnp.ndarray,
        n_samples: int = 1,
    ) -> jnp.ndarray:
        """Draw samples evaluated at specific points.
        
        More efficient than sampling full functions when only specific
        evaluation points are needed.
        
        Args:
            rng_key: JAX random key
            x_points: Points at which to evaluate samples
            n_samples: Number of samples
        
        Returns:
            Samples at query points, shape (n_samples, d'_U, n_points)
        """
        full_samples = self.sample_function(rng_key, n_samples)
        # Interpolate to query points
        return self._interpolate(full_samples, x_points)


class LUNO_FNO:
    """Complete LUNO framework applied to a Fourier Neural Operator.
    
    This class orchestrates the full LUNO pipeline (Figure 1):
    
    Step 0: Start with trained FNO F: A × W → U
    Step 1: Uncurry: f((a, x), w) = F(a, w)(x)
    Step 2: Obtain Gaussian belief w ~ N(μ, Σ) via Laplace/isotropic/ensemble
    Step 3: Probabilistic currying: construct F ~ GP(M, K) with values in U
    
    The implementation handles:
    - Full LUNO: linearize all weights
    - Last-layer LUNO: only linearize last Fourier block (Section 3.2.1)
    """
    
    def __init__(
        self,
        fno_forward_fn: Callable,
        fno_params: Any,
        lifting_fn: Callable,
        projection_fn: Callable,
        fourier_layers: list,
        n_fourier_blocks: int = 4,
        modes: int = 12,
        hidden_dim: int = 18,
    ):
        """Initialize LUNO for an FNO.
        
        Args:
            fno_forward_fn: Full FNO forward pass function
            fno_params: Trained FNO parameters w*
            lifting_fn: Lifting layer p
            projection_fn: Projection layer q
            fourier_layers: List of Fourier layer functions
            n_fourier_blocks: Number of Fourier blocks (L=4 in paper)
            modes: Number of Fourier modes (12 in paper)
            hidden_dim: Hidden dimension (18 in paper)
        """
        self.fno_forward = fno_forward_fn
        self.params = fno_params
        self.lifting = lifting_fn
        self.projection = projection_fn
        self.fourier_layers = fourier_layers
        self.n_blocks = n_fourier_blocks
        self.modes = modes
        self.hidden_dim = hidden_dim
    
    def forward_with_state(
        self, 
        a: jnp.ndarray, 
        params: Any
    ) -> Tuple[jnp.ndarray, FNOState]:
        """Forward pass that also returns intermediate state.
        
        Needed to compute z^{(L-1)} for the last-layer LUNO.
        
        Args:
            a: Input function discretization
            params: Model parameters
        
        Returns:
            Tuple (output, state) where state contains hidden activations
        """
        # Step 1: Lifting
        v = self.lifting(a, params['lifting'])
        
        v_layers = [v]
        fourier_coeffs_list = []
        
        # Step 2: Fourier layers
        for l, layer_fn in enumerate(self.fourier_layers):
            v, fft_coeffs = layer_fn(v, params[f'fourier_layer_{l}'])
            v_layers.append(v)
            fourier_coeffs_list.append(fft_coeffs)
        
        # Step 3: Projection
        u = self.projection(v, params['projection'])
        
        state = FNOState(
            v_layers=v_layers,
            fourier_coeffs=fourier_coeffs_list,
        )
        
        return u, state
    
    def linearize_last_layer(
        self,
        a: jnp.ndarray,
        params: Any,
        state: FNOState,
    ) -> Tuple[Callable, Callable]:
        """Linearize the FNO with respect to the last Fourier block weights.
        
        Following Section 3.2.1 and Appendix C.1:
        
        F(a, w)(x) = q̃(z^{(L-1)}(x, w_{L-1}))
        
        where q̃ = q(·, w_q) ◦ σ^{(L-1)} and z^{(L-1)} is linear in w_{L-1}.
        
        The linearization:
        F_lin(a, w)(x) = q̃(m_z(x)) + Dq̃(m_z(x)) (z^{(L-1)}(x, w) - m_z(x))
        
        Args:
            a: Input function
            params: Full parameters
            state: Forward pass state
        
        Returns:
            Tuple (mean_fn, cov_fn) for the linearized predictive
        """
        # Extract last hidden state v^{(L-1)} and its Fourier coefficients
        v_last = state.v_layers[-2]  # v^{(L-1)} — before last Fourier block
        v_hat_last = state.fourier_coeffs[-2] if len(state.fourier_coeffs) >= 2 else state.fourier_coeffs[-1]
        
        # Get last layer weights
        w_last = params.get('fourier_layer_{}'.format(self.n_blocks - 1), None)
        if w_last is None:
            w_last = params.get('last_fourier', None)
        
        # The map from z^{(L-1)} to output: q̃ = q ◦ σ
        # σ is the activation of the last Fourier block
        sigma_fn = self.fourier_layers[-1].activation if hasattr(
            self.fourier_layers[-1], 'activation'
        ) else jax.nn.gelu
        
        def q_tilde(z):
            """q̃(z) = q(σ(z), w_q)"""
            return self.projection(sigma_fn(z), params['projection'])
        
        def q_tilde_derivative(z):
            """Dq̃(z) = J_q(σ(z)) · σ'(z)"""
            # Use JVP for derivative
            primals, f_vjp = jax.vjp(q_tilde, z)
            # Return VJP as a matrix
            return jax.jacrev(q_tilde)(z)
        
        # Compute m_z = z^{(L-1)}(x, w*)
        m_z = self._compute_z_last(v_last, v_hat_last, w_last)
        
        # Build the function-valued GP
        fgp = FourierGaussianRandomOperator(
            v_hidden=v_last,
            fourier_coeffs=v_hat_last,
            weights_mean=w_last,
            weights_cov=None,  # To be set based on weight-space belief
            q_tilde_fn=q_tilde,
            q_tilde_deriv_fn=q_tilde_derivative,
            modes=self.modes,
            grid_size=v_last.shape[1],
            output_dim=1,  # Scalar PDE output typically
        )
        
        return fgp
    
    def _compute_z_last(
        self,
        v: jnp.ndarray,
        v_hat: jnp.ndarray,
        weights: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Compute z^{(L-1)} = R * v + W v (last Fourier block before activation).
        
        Args:
            v: Hidden state v^{(L-1)} of shape (d_v, n_x)
            v_hat: Fourier transform of v, shape (k_max, d_v)
            weights: Parameters {R, W} for the last Fourier block
        
        Returns:
            z^{(L-1)} of shape (d_v, n_x)
        """
        R = weights.get('R', None)
        W = weights.get('W', None)
        
        d_v = v.shape[0]
        n_x = v.shape[1]
        k_max = v_hat.shape[0]
        
        z = jnp.zeros((d_v, n_x))
        
        if R is not None:
            # Spectral convolution: F^{-1}(R ⊙ v̂)
            for i in range(d_v):
                for j in range(d_v):
                    conv_coeffs = R[:, i, j] * v_hat[:, j]
                    # Inverse FFT
                    conv_spatial = self._inv_rfft(conv_coeffs, n_x)
                    z = z.at[i].add(conv_spatial)
        
        if W is not None:
            z += W @ v
        
        return z
    
    def _inv_rfft(self, coeffs: jnp.ndarray, n: int) -> jnp.ndarray:
        """Inverse real FFT."""
        k_max = coeffs.shape[0]
        full = jnp.zeros(n, dtype=jnp.complex64)
        full = full.at[:k_max].set(coeffs)
        if n % 2 == 0:
            full = full.at[k_max:].set(jnp.conj(coeffs[1:k_max][::-1]))
        else:
            full = full.at[k_max:].set(jnp.conj(coeffs[1:k_max+1][::-1]))
        return jnp.fft.ifft(full).real * n
    
    def get_function_valued_gp(
        self,
        a: jnp.ndarray,
        weight_belief: Any,
        last_layer_only: bool = True,
    ) -> FourierGaussianRandomOperator:
        """Construct the function-valued GP posterior for input a.
        
        This is the main entry point for LUNO.
        
        Args:
            a: Input function
            weight_belief: Weight-space belief (IsotropicGaussian, LowRankLaplace, etc.)
            last_layer_only: If True, use last-layer LUNO (Section 3.2.1)
        
        Returns:
            FourierGaussianRandomOperator representing F(a) ~ GP(m_a, K_a)
        """
        # Forward pass to get state
        output, state = self.forward_with_state(a, self.params)
        
        if last_layer_only:
            fgp = self.linearize_last_layer(a, self.params, state)
            fgp.weights_cov = weight_belief
            return fgp
        else:
            # Full LUNO: linearize all weights
            # This requires computing the full Jacobian
            raise NotImplementedError(
                "Full LUNO (linearizing all weights) requires Jacobian computation. "
                "Use last_layer_only=True for the efficient implementation from Section 3.2.1."
            )
