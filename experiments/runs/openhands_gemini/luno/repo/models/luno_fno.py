
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core import FrozenDict
from typing import Callable, Any, Tuple
from functools import partial
import ml_collections

from luno.models.fno import FNO

# Helper to flatten and unflatten parameters for linearization
def _tree_flatten_dict(pytree):
    flat_params, treedef = jax.tree_util.tree_flatten(pytree)
    return flat_params, treedef

def _tree_unflatten_dict(treedef, flat_params):
    return jax.tree_util.tree_unflatten(treedef, flat_params)

class LUNOFNO:
    """
    LUNO framework for Fourier Neural Operators.
    Implements linearization and probabilistic currying to obtain
    function-valued Gaussian process beliefs over FNO predictions.
    """
    def __init__(self, fno_model: FNO, params: FrozenDict, uq_config: ml_collections.ConfigDict):
        self.fno_model = fno_model
        self.params = params
        self.uq_config = uq_config

        # Extract last layer parameters for LLA/LUNO
        # In FNO, the "last layer" refers to the final projection layer.
        # However, the paper discusses last-layer LUNO for the *last Fourier block* weights.
        # This means the parameters (R and W) of the last FourierBlock and the subsequent projection layers.
        # We need to adapt the FNO structure to allow access to z^(L-1) and then q(z^(L-1)).

        # A common way to implement last-layer LA is to linearize the function after the penultimate layer.
        # In our FNO, this means linearizing q around z^(L-1).
        # Let's assume `q` is the sequence of `projection_layer_1` -> gelu -> `projection_layer_2`.
        # And `z^(L-1)` is the output of the `num_fourier_blocks-1` FourierBlock.

        # We need to re-initialize the FNO model but with a way to access intermediate activations
        # or separate the final projection layers.
        # For simplicity and adherence to the paper's "last-layer Laplace approximation",
        # we'll consider the full FNO as f(w) and linearize around *all* its weights,
        # but the covariance will be defined only for the "last layer" weights.
        # The paper says: "Gaussian belief is restricted to the parameters of the final Fourier block w_L-1".
        # This implies we linearize the output with respect to w_L-1.

        # Identify parameters of the last Fourier block and projection layers
        self.last_layer_param_names = []
        if uq_config.last_layer_la:
            # Parameters of the last FourierBlock
            last_fourier_block_name = f'fourier_block_{self.fno_model.num_fourier_blocks - 1}'
            # Iterate through the dictionary to get full paths
            def get_param_paths(prefix, params_dict):
                paths = []
                for k, v in params_dict.items():
                    if isinstance(v, FrozenDict):
                        paths.extend(get_param_paths(f"{prefix}/{k}", v))
                    else:
                        paths.append(f"{prefix}/{k}")
                return paths

            # Get paths for the last fourier block
            if last_fourier_block_name in self.params:
                self.last_layer_param_names.extend(get_param_paths(last_fourier_block_name, self.params[last_fourier_block_name]))
            
            # Get paths for projection layers
            if 'projection_layer_1' in self.params:
                self.last_layer_param_names.extend(get_param_paths('projection_layer_1', self.params['projection_layer_1']))
            if 'projection_layer_2' in self.params:
                self.last_layer_param_names.extend(get_param_paths('projection_layer_2', self.params['projection_layer_2']))

            print(f"LUNO: Identified {len(self.last_layer_param_names)} last layer parameters for UQ.")

        else:
            # If not last-layer LA, consider all parameters.
            # For simplicity, LUNO context assumes some selection of parameters.
            # For a full FNO LUNO, we would linearize over all parameters.
            # For now, let's assume `last_layer_la` is always True in this context.
            raise NotImplementedError("Full FNO LUNO (not last-layer) is not implemented yet.")
        
        # Flatten all parameters and get a mask for the last-layer parameters
        self.flat_all_params_tree, self.param_treedef = jax.tree_util.tree_flatten(self.params)
        self.last_layer_mask_flat = self._create_last_layer_mask_flat()

        # Extract only the "last layer" parameters
        self.flat_last_layer_params = [p for p, mask_val in zip(self.flat_all_params_tree, self.last_layer_mask_flat) if mask_val]
        
        # The mean parameter `mu` for the weight-space Gaussian
        self.mu_w = jnp.array(self.flat_last_layer_params)

        # Initialize weight-space covariance Sigma
        self.Sigma_w = self._initialize_covariance()

    def _create_last_layer_mask_flat(self):
        # Create a tree of boolean masks with the same structure as self.params
        mask_tree = jax.tree_util.tree_map_with_path(
            lambda path, x: '/'.join([p.key for p in path]) in self.last_layer_param_names,
            self.params,
            is_leaf=lambda x: not isinstance(x, FrozenDict)
        )
        # Flatten this mask tree
        flat_mask, _ = jax.tree_util.tree_flatten(mask_tree)
        return flat_mask
    
    def _initialize_covariance(self):
        num_last_layer_params = len(self.mu_w)
        if num_last_layer_params == 0:
            raise ValueError("No last layer parameters found for UQ. Check FNO model and config.")

        if self.uq_config.method == 'LUNO-Iso':
            # Isotropic Gaussian: Sigma = sigma^2 * I
            if self.uq_config.sigma_iso is None:
                # Default sigma_iso if not provided, for demo purposes.
                # In a real scenario, this would be calibrated.
                print("WARNING: sigma_iso not specified for LUNO-Iso, using default 1e-3.")
                self.uq_config.sigma_iso = 1e-3
            return self.uq_config.sigma_iso * jnp.eye(num_last_layer_params)
        elif self.uq_config.method == 'LUNO-LA':
            # Laplace Approximation: Sigma = P_inv where P is the GGN matrix.
            # The paper states Sigma = (n V V^T + sigma I)^-1 where V V^T is low-rank GGN approx
            # This requires actual computation based on data and loss.
            # For this reproduction, we will use a placeholder.
            print("WARNING: GGN computation for LUNO-LA is complex and depends on data and loss.")
            print("Using a placeholder identity matrix for Sigma_w. This needs to be replaced by actual GGN computation.")
            # Placeholder: small variance for placeholder, to be replaced by actual GGN inverse
            return jnp.eye(num_last_layer_params) * 1e-4
        else:
            raise ValueError(f"Unsupported LUNO method: {self.uq_config.method}")

    def _fno_forward_with_last_layer_params(self, last_layer_flat_params_vector: jnp.ndarray, x_input: jnp.ndarray) -> jnp.ndarray:
        """
        Performs a forward pass of the FNO, but dynamically constructs the `params` dict
        using `last_layer_flat_params_vector` for relevant parameters and `self.params` for others.
        """
        # Reconstruct full params from flattened vector and mask
        current_ll_param_idx = 0
        reconstructed_flat_params = []
        for is_ll_param in self.last_layer_mask_flat:
            if is_ll_param:
                if current_ll_param_idx >= len(last_layer_flat_params_vector):
                    raise IndexError("Not enough last_layer_flat_params_vector elements to fill mask.")
                reconstructed_flat_params.append(last_layer_flat_params_vector[current_ll_param_idx])
                current_ll_param_idx += 1
            else:
                # Use the original parameter for non-last-layer parts
                reconstructed_flat_params.append(self.flat_all_params_tree[len(reconstructed_flat_params)])
        
        # Verify all elements from `last_layer_flat_params_vector` were used
        if current_ll_param_idx != len(last_layer_flat_params_vector):
             raise ValueError(f"Mismatch in last layer parameters: {current_ll_param_idx} used, but {len(last_layer_flat_params_vector)} provided.")

        reconstructed_params = jax.tree_util.tree_unflatten(self.param_treedef, reconstructed_flat_params)
        
        return self.fno_model.apply({'params': reconstructed_params}, x_input)

    def predict_mean_and_cov(self, x_input: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Computes the mean and covariance of the function-valued Gaussian process
        for a given input `x_input`.
        
        Args:
            x_input: Input to the FNO (batch, spatial_res, num_initial_steps * input_channels).
                     For LUNO, this is `a` in F(a).

        Returns:
            mean_f: Mean of the predicted function (batch, spatial_res, output_dim).
            cov_f: Covariance function evaluated.
                   Shape: (batch, spatial_res, output_dim, spatial_res, output_dim)
                   For scalar output, this would be (batch, spatial_res, spatial_res).
        """
        # Mean prediction: F(a, mu)
        # This is simply the standard FNO prediction with the trained weights.
        # The paper says: E[F(a)(x)] = F(a, mu)(x)
        # Assuming `self.params` already represents mu (the trained weights w*)
        mean_f = self.fno_model.apply({'params': self.params}, x_input) # (batch, spatial_res, output_dim)

        # Compute Jacobian of fno_output wrt last_layer_params (J)
        # J_w F(a, w)(x) = D_w f((a,x), w) |_mu
        
        # We need the Jacobian of the FNO output with respect to `self.mu_w` (flattened last-layer params).
        # This Jacobian will have shape (output_elements, num_last_layer_params).
        # Output elements = spatial_res * output_dim (for a single input in the batch)

        # Define the function to get output for a single input, given last_layer_params_vector
        def single_fno_output_fn(ll_params_vector, single_x_input):
            output = self._fno_forward_with_last_layer_params(ll_params_vector, single_x_input)
            return output # shape (spatial_res, output_dim)

        # Compute Jacobian for each input in the batch.
        # jax.vmap is used to vectorize across the batch dimension of x_input.
        # jax.jacrev computes the reverse-mode Jacobian.
        # J_F shape: (batch, spatial_res, output_dim, num_last_layer_params)
        J_F = jax.vmap(jax.jacrev(single_fno_output_fn, argnums=0), in_axes=(None, 0))(self.mu_w, x_input)
        
        # Reshape J_F for covariance calculation.
        # J_F_flat: (batch, spatial_res * output_dim, num_last_layer_params)
        spatial_res = x_input.shape[1]
        output_dim = self.fno_model.output_dim
        num_output_elements_per_batch = spatial_res * output_dim
        J_F_flat = J_F.reshape(x_input.shape[0], num_output_elements_per_batch, len(self.mu_w))

        # Covariance calculation: K(x1, x2) = J(x1) @ Sigma_w @ J(x2).T
        # The formula given is Cov[F(a1)(x1), F(a2)(x2)] = D_w F(a1,w)(x1)|_mu Sigma D_w F(a2,w)(x2)|_mu.T
        # Where D_w F(a,w)(x) is the Jacobian of the specific pointwise output F(a,w)(x) wrt w.
        # This is a matrix-vector product.
        
        # Here we calculate the full covariance matrix of the concatenated output vector (spatial_res * output_dim)
        # cov_f_flat shape: (batch, num_output_elements_per_batch, num_output_elements_per_batch)
        cov_f_flat = jnp.einsum('bpd,dq,bqk->bpk', J_F_flat, self.Sigma_w, J_F_flat.transpose(0, 2, 1))

        # Reshape cov_f_flat back to (batch, spatial_res, output_dim, spatial_res, output_dim)
        cov_f = cov_f_flat.reshape(
            x_input.shape[0], spatial_res, output_dim, spatial_res, output_dim
        )
        
        return mean_f, cov_f

    def sample_functions(self, x_input: jnp.ndarray, num_func_samples: int, key: jax.random.PRNGKey) -> jnp.ndarray:
        """
        Samples entire functions from the function-valued Gaussian process.
        
        Args:
            x_input: Input to the FNO (batch, spatial_res, num_initial_steps * input_channels).
            num_func_samples: Number of function samples to draw.
            key: JAX PRNG key.

        Returns:
            function_samples: (num_func_samples, batch, spatial_res, output_dim)
        """
        mean_f, cov_f = self.predict_mean_and_cov(x_input) # (batch, spatial_res, output_dim), (batch, S, OD, S, OD)

        spatial_res = x_input.shape[1]
        output_dim = self.fno_model.output_dim
        num_output_elements_per_batch = spatial_res * output_dim

        # Reshape mean_f to (batch, num_output_elements_per_batch)
        mean_f_flat = mean_f.reshape(x_input.shape[0], num_output_elements_per_batch)
        
        # Reshape cov_f to (batch, num_output_elements_per_batch, num_output_elements_per_batch)
        cov_f_flat = cov_f.reshape(x_input.shape[0], num_output_elements_per_batch, num_output_elements_per_batch)
        
        # Ensure covariance matrix is symmetric and positive semi-definite for Cholesky decomposition
        # Add a small diagonal term for numerical stability
        eps = 1e-6
        cov_f_flat_stable = cov_f_flat + eps * jnp.eye(num_output_elements_per_batch)

        # Draw samples for each item in the batch
        all_func_samples_flat = []
        for i in range(x_input.shape[0]):
            key, subkey = jax.random.split(key)
            mvn_sample = jax.random.multivariate_normal(
                subkey,
                mean_f_flat[i],
                cov_f_flat_stable[i],
                (num_func_samples,)
            ) # (num_func_samples, num_output_elements_per_batch)
            all_func_samples_flat.append(mvn_sample)
        
        # Stack and reshape
        # (batch, num_func_samples, num_output_elements_per_batch)
        all_func_samples_flat_stacked = jnp.stack(all_func_samples_flat, axis=0)

        # Reshape to (num_func_samples, batch, spatial_res, output_dim)
        function_samples = all_func_samples_flat_stacked.transpose(1, 0, 2).reshape(
            num_func_samples, x_input.shape[0], spatial_res, output_dim
        )
        
        return function_samples
