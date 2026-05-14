import abc
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as tree_util
from flax.core import FrozenDict, unfreeze
import flax.linen as nn
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from tqdm import tqdm
from absl import logging
import scipy.sparse.linalg


# Local application imports
from config import Config
from fno_model import FNO
from utils import get_jvp_fn, initialize_fno_params, rfft_transform, irfft_transform, stack_conditions


# Type aliases for clarity
Params = FrozenDict
PRNGKey = jax.random.PRNGKey


class UQMethod(abc.ABC):
    """
    Abstract Base Class for Uncertainty Quantification (UQ) methods.
    Defines a common interface for all UQ strategies.
    """

    def __init__(self, fno_module: FNO, trained_params: Union[Params, List[Params]], config: Config):
        """
        Initializes the base UQ method.

        Args:
            fno_module: An instance of the FNO model.
            trained_params: Parameters of the trained FNO model. Can be a single FrozenDict
                            for most methods, or a List[FrozenDict] for DeepEnsemble.
            config: An instance of the Config class.
        """
        self.fno_module = fno_module
        self.trained_params = trained_params
        self.config = config
        self.rng_key = jr.PRNGKey(self.config.seed)
        logging.info(f"Initialized UQMethod: {self.__class__.__name__}")

    @abc.abstractmethod
    def fit(self, train_data: Optional[Tuple] = None) -> None:
        """
        Performs any method-specific computations (e.g., GGN matrix calculation for LLA)
        that need to happen after FNO training but before prediction.

        Args:
            train_data: Optional. A tuple (inputs, targets, conditions) used for fitting,
                        e.g., to compute the GGN matrix.
        """
        pass

    @abc.abstractmethod
    def predict_uncertainty(
        self, input_func: jnp.ndarray, conditions: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Computes the mean and standard deviation of the predictive distribution for a given input.

        Args:
            input_func: The input function (state history) for which to make a prediction.
                        Shape: (batch, initial_time_steps, *spatial_dims, 1).
            conditions: Additional conditioning variables (e.g., velocity, reaction terms).
                        Shape: (batch, *spatial_dims, condition_channels).

        Returns:
            A tuple (mean_predictions, std_dev_predictions):
            - mean_predictions: Predicted mean of the output function.
                                Shape: (batch, *spatial_dims, output_channels).
            - std_dev_predictions: Predicted standard deviation of the output function.
                                   Shape: (batch, *spatial_dims, output_channels).
        """
        pass

    @abc.abstractmethod
    def get_samples(
        self, input_func: jnp.ndarray, conditions: jnp.ndarray, num_samples: int
    ) -> jnp.ndarray:
        """
        Returns an array of `num_samples` predictions, representing individual samples
        from the predictive distribution.

        Args:
            input_func: The input function (state history).
            conditions: Additional conditioning variables.
            num_samples: The number of samples to draw.

        Returns:
            A jax.numpy.ndarray of samples.
            Shape: (num_samples, batch, *spatial_dims, output_channels).
        """
        pass


class _LunoBase(UQMethod):
    """
    Base class for LUNO methods, handling common logic for last-layer linearization.
    """

    def __init__(self, fno_module: FNO, trained_params: Params, config: Config):
        super().__init__(fno_module, trained_params, config)
        self.trained_params: Params = trained_params # Ensure type hint is specific for LUNO

        # Identify and extract relevant parameters for w_L-1 and w_q
        # w_L-1 are params of the last FourierBlock: fourier_block_{self.config.fno_blocks - 1}
        # w_q are params of the projection layers: projection_dense1, projection_dense2
        self.w_L_minus_1_path_prefix = f'fourier_block_{self.config.fno_blocks - 1}'
        self.w_q_path_prefixes = ('projection_dense1', 'projection_dense2')
        self.activation_fn = self.fno_module.activation # Activation used in FNO's projection layers

        # Flatten specific subtrees of parameters
        self.w_L_minus_1_params_flat, self.w_L_minus_1_unflatten_fn = self._extract_and_flatten_subtree_params(
            self.trained_params, self.w_L_minus_1_path_prefix
        )
        self.w_q_params_flat, self.w_q_unflatten_fn = self._extract_and_flatten_subtree_params(
            self.trained_params, self.w_q_path_prefixes
        )
        
        # JIT-compile common Jacobian computation functions for efficiency
        # These will be vmapped over batch dimension in predict_uncertainty/get_samples
        self._compute_last_block_output_jacobian_wrt_w_L_minus_1_jitted = jax.jit(self._compute_last_block_output_jacobian_wrt_w_L_minus_1)
        self._compute_final_output_jacobian_wrt_z_L_minus_1_jitted = jax.jit(self._compute_final_output_jacobian_wrt_z_L_minus_1)

        # Dummy inputs for shape inference when dealing with JAX transforms
        # These are usually created in training.py. Since fno_model.py doesn't have them
        # as class members, we create them here if needed for `_compute_full_jacobian_J_f_w_L_minus_1`.
        # However, the functions are jitted and take concrete values, so dummy inputs are mostly for init.
        # Let's ensure `self.fno_module` has dummy_x_in and dummy_conditions set during its init in `main.py` if not present.
        # For now, let's assume actual `input_func` and `conditions` will be passed correctly.

    def _extract_and_flatten_subtree_params(self, full_params: Params, target_path_prefixes: Union[str, Tuple[str, ...]]) -> Tuple[jnp.ndarray, Callable]:
        """
        Extracts parameters at specified path prefixes from the full params tree and flattens them.
        Returns the flattened array and a function to unflatten it back into the
        subtree structure defined by target_path_prefixes.
        """
        if isinstance(target_path_prefixes, str):
            target_path_prefixes = (target_path_prefixes,)

        # Build a dictionary of the subtrees to be extracted
        extracted_subtrees = {}
        for path_prefix in target_path_prefixes:
            # Navigate the Flax FrozenDict to find the desired subtree
            current_level = full_params
            try:
                for key in path_prefix.split('/'): # e.g., 'fourier_block_0/spectral_weights_1d_real'
                    current_level = current_level[key]
            except KeyError as e:
                logging.error(f"Parameter path prefix '{path_prefix}' not found in params tree: {e}")
                raise
            extracted_subtrees[path_prefix] = current_level

        flat_target_params, target_unflatten_fn = tree_util.tree_flatten(extracted_subtrees)
        return jnp.concatenate([jnp.array(x) for x in flat_target_params]), target_unflatten_fn

    def _reconstruct_params_tree_with_subtree(self, base_params: Params, flat_subtree_params: jnp.ndarray, unflatten_fn: Callable, target_path_prefixes: Union[str, Tuple[str, ...]]) -> Params:
        """
        Reconstructs a full params tree by inserting the unflattened subtree parameters
        back into a copy of the base params tree.
        """
        if isinstance(target_path_prefixes, str):
            target_path_prefixes = (target_path_prefixes,)

        # Unflatten the target parameters back to their original subtree structure
        unflattened_target_subtrees_dict = unflatten_fn(list(flat_subtree_params))

        # Create a mutable copy of the base parameters
        temp_params_tree = unfreeze(base_params)

        for path_prefix in target_path_prefixes:
            keys = path_prefix.split('/')
            
            # Navigate to the parent of the target key
            current_level = temp_params_tree
            for key_idx in range(len(keys) - 1):
                if keys[key_idx] not in current_level:
                    logging.error(f"Key '{keys[key_idx]}' not found in params path during reconstruction.")
                    raise KeyError(f"Key '{keys[key_idx]}' not found.")
                current_level = current_level[keys[key_idx]]
            
            # Replace the target subtree
            if keys[-1] not in current_level:
                logging.error(f"Final key '{keys[-1]}' not found in params path during reconstruction.")
                raise KeyError(f"Final key '{keys[-1]}' not found.")
            current_level[keys[-1]] = unflattened_target_subtrees_dict[path_prefix]
        
        return FrozenDict(temp_params_tree)
    
    def _compute_last_block_output_jacobian_wrt_w_L_minus_1(
        self,
        x_in_item: jnp.ndarray,
        conditions_item: jnp.ndarray,
        params_w_L_minus_1_flat: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Computes the Jacobian of z^(L-1) (pre-activation output of last Fourier block)
        with respect to w_L-1 (parameters of the last Fourier block).
        This is J_z,w_L-1 in the paper.
        """
        # Define a wrapper function for jacrev that takes only the w_L_minus_1_params as input
        def z_L_minus_1_output_fn_flat_w(w_L_minus_1_candidate_flat: jnp.ndarray) -> jnp.ndarray:
            # Reconstruct the full params tree with candidate w_L_minus_1 and fixed other params
            # The 'base_params' here are the full trained_params, from which all parts but w_L-1 are fixed
            full_params_with_candidate_w_L_minus_1 = self._reconstruct_params_tree_with_subtree(
                self.trained_params, w_L_minus_1_candidate_flat, self.w_L_minus_1_unflatten_fn, self.w_L_minus_1_path_prefix
            )
            z_output_tree = self.fno_module.apply(
                {'params': full_params_with_candidate_w_L_minus_1},
                jnp.expand_dims(x_in_item, 0), # Add batch dim for single item
                jnp.expand_dims(conditions_item, 0),
                method=self.fno_module.get_last_block_pre_activation_output
            )
            z_output_flat, _ = tree_util.tree_flatten(z_output_tree)
            return jnp.concatenate(z_output_flat)

        jacobian_matrix = jax.jacrev(z_L_minus_1_output_fn_flat_w)(params_w_L_minus_1_flat)
        return jacobian_matrix # This is J_z,w_L-1, shape: (output_flat_dim, w_L_minus_1_flat_dim)

    def _compute_final_output_jacobian_wrt_z_L_minus_1(
        self, z_L_minus_1_output_star: jnp.ndarray, params_q_flat: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Computes the Jacobian of the final FNO output with respect to z^(L-1)
        (pre-activation output of last Fourier block).
        This is D_tilde_q in the paper (Jacobian of `q(sigma^(L-1)(z))` w.r.t. `z`).
        """
        def final_output_fn_flat_z(z_candidate_item: jnp.ndarray) -> jnp.ndarray:
            # Reconstruct w_q_params from flat_params_q_only
            w_q_params_tree = self.w_q_unflatten_fn(list(params_q_flat))
            
            # Apply `tilde_q` operations manually for differentiation
            # `tilde_q = q(., w_q) o sigma^(L-1)` as per paper
            # The `z_candidate_item` here is the pre-activation output.
            
            # Apply sigma^(L-1) (the activation of the last Fourier block)
            activated_output_item = self.activation_fn(z_candidate_item)
            
            # Apply projection_dense1
            proj_dense1_params = w_q_params_tree[self.w_q_path_prefixes[0]]
            v_out_item = self.fno_module.projection_dense1.apply({'params': proj_dense1_params}, activated_output_item)
            
            # Apply second activation in projection head
            v_out_item = self.activation_fn(v_out_item)
            
            # Apply projection_dense2
            proj_dense2_params = w_q_params_tree[self.w_q_path_prefixes[1]]
            final_output_item = self.fno_module.projection_dense2.apply({'params': proj_dense2_params}, v_out_item)
            
            # Flatten final output for jacrev
            final_output_flat, _ = tree_util.tree_flatten(final_output_item)
            return jnp.concatenate(final_output_flat)
        
        # `z_L_minus_1_output_star` is expected to be a single item (not batched for this calculation)
        # and has shape (*spatial_dims, hidden_dims) - pre-activation output
        z_L_minus_1_output_star_flat, _ = tree_util.tree_flatten(z_L_minus_1_output_star)
        z_L_minus_1_output_star_flat = jnp.concatenate(z_L_minus_1_output_star_flat)

        jacobian_matrix = jax.jacrev(final_output_fn_flat_z)(z_L_minus_1_output_star_flat, params_q_flat)
        return jacobian_matrix # This is D_tilde_q, shape: (output_flat_dim, z_L_minus_1_flat_dim)

    def _compute_full_jacobian_J_f_w_L_minus_1(self, x_in_item: jnp.ndarray, conditions_item: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the full Jacobian J_f,w_L-1 = D_tilde_q @ J_z,w_L-1.
        This represents the Jacobian of the FNO's final output w.r.t. the parameters
        of the last Fourier block for a single input item.
        """
        # First, compute z^(L-1) evaluated at w_star
        # z_L_minus_1_star will be (1, *spatial_dims, hidden_dims) if FNO.apply adds batch dim
        z_L_minus_1_star_batched = self.fno_module.apply(
            {'params': self.trained_params},
            jnp.expand_dims(x_in_item, 0), # Add batch dim for single item
            jnp.expand_dims(conditions_item, 0),
            method=self.fno_module.get_last_block_pre_activation_output
        )
        z_L_minus_1_star = jnp.squeeze(z_L_minus_1_star_batched, axis=0) # Remove the added batch dim

        # Compute J_z,w_L-1 (Jacobian of z^(L-1) w.r.t. w_L-1)
        # This will be (z_output_flat_dim, w_L_minus_1_flat_dim)
        J_z_w_L_minus_1 = self._compute_last_block_output_jacobian_wrt_w_L_minus_1_jitted(
            x_in_item, conditions_item, self.w_L_minus_1_params_flat
        )

        # Compute D_tilde_q (Jacobian of tilde_q w.r.t. z^(L-1))
        # This will be (final_output_flat_dim, z_output_flat_dim)
        D_tilde_q = self._compute_final_output_jacobian_wrt_z_L_minus_1_jitted(
            z_L_minus_1_star, self.w_q_params_flat
        )
        
        # The chain rule: J_f,w_L-1 = D_tilde_q @ J_z,w_L-1
        J_f_w_L_minus_1 = D_tilde_q @ J_z_w_L_minus_1
        return J_f_w_L_minus_1


class LunoIso(_LunoBase):
    """
    Implements LUNO with an isotropic Gaussian weight belief, restricting uncertainty
    to the parameters of the last Fourier block.
    """

    def __init__(self, fno_module: FNO, trained_params: Params, config: Config):
        super().__init__(fno_module, trained_params, config)
        # Initialize sigma_squared from config (will be calibrated). It's `sigma^2` not `sigma`.
        self.sigma_squared = config.uq_methods_config["luno_iso"]["sigma_squared_init"]
        logging.info(f"LunoIso initialized with initial sigma_squared: {self.sigma_squared}")
        
    def fit(self, train_data: Optional[Tuple] = None) -> None:
        """
        For LunoIso, `fit` is mainly used to update `sigma_squared` after calibration.
        No complex pre-computation needed here.
        """
        logging.info("LunoIso fit method called. No complex computations, expects calibration to set sigma_squared.")
        pass

    def predict_uncertainty(
        self, input_func: jnp.ndarray, conditions: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Computes the mean and standard deviation of the predictive distribution using LUNO-Iso.
        """
        # Ensure input_func and conditions are batched for consistent JAX operations
        added_batch_dim = False
        if input_func.ndim < self.fno_module.dummy_x_in.ndim: # If unbatched, add batch dim
            input_func = jnp.expand_dims(input_func, axis=0)
            conditions = jnp.expand_dims(conditions, axis=0)
            added_batch_dim = True
            
        batch_size = input_func.shape[0]

        # Mean prediction: Simply the FNO's output with the trained weights
        # FNO apply returns (batch, *spatial_dims, output_channels)
        mean_predictions = self.fno_module.apply(
            {'params': self.trained_params}, input_func, conditions
        )
        
        # For standard deviation, we need the Jacobian J_f,w_L-1
        # This Jacobian depends on input, so we vmap it over the batch dimension.
        J_f_w_L_minus_1_batch = jax.vmap(self._compute_full_jacobian_J_f_w_L_minus_1, in_axes=(0, 0))(input_func, conditions)
        
        # J_f_w_L_minus_1_batch shape: (batch_size, output_flat_dim, w_L_minus_1_flat_dim)
        
        # Predictive variance for each batch item: sigma_squared * diag(J @ J.T) (for marginal variance)
        # Covariance: K_a(x, x) = sigma_squared * J_f,w_L-1 @ J_f,w_L-1.T
        # We need std_dev for marginals, so sqrt of diagonal of this matrix.
        std_dev_predictions_flat = jax.vmap(
            lambda J_item: jnp.sqrt(self.sigma_squared * jnp.sum(J_item**2, axis=-1))
        )(J_f_w_L_minus_1_batch)
        # The sum J_item**2 along last axis gives (J_item @ J_item.T)_diag assuming isotropic covariance (sigma^2 * I)
        
        # Reshape std_dev_predictions_flat back to original output shape
        std_dev_predictions = std_dev_predictions_flat.reshape(mean_predictions.shape)
        
        if added_batch_dim: # Remove batch dim if it was added
            mean_predictions = jnp.squeeze(mean_predictions, axis=0)
            std_dev_predictions = jnp.squeeze(std_dev_predictions, axis=0)
        
        return mean_predictions, std_dev_predictions

    def get_samples(
        self, input_func: jnp.ndarray, conditions: jnp.ndarray, num_samples: int
    ) -> jnp.ndarray:
        """
        Generates samples from the LUNO-Iso predictive distribution.
        """
        # Ensure input_func and conditions are batched
        added_batch_dim = False
        if input_func.ndim < self.fno_module.dummy_x_in.ndim:
            input_func = jnp.expand_dims(input_func, axis=0)
            conditions = jnp.expand_dims(conditions, axis=0)
            added_batch_dim = True

        batch_size = input_func.shape[0]

        mean_predictions = self.fno_module.apply(
            {'params': self.trained_params}, input_func, conditions
        ) # (batch, *spatial_dims, output_channels)

        # Compute J_f,w_L-1 for each item in the batch
        J_f_w_L_minus_1_batch = jax.vmap(self._compute_full_jacobian_J_f_w_L_minus_1, in_axes=(0, 0))(input_func, conditions)
        # J_f_w_L_minus_1_batch shape: (batch_size, output_flat_dim, w_L_minus_1_flat_dim)
        
        w_L_minus_1_flat_dim = self.w_L_minus_1_params_flat.shape[0]
        
        # Generate random weights for delta_w_L_minus_1
        self.rng_key, sample_key = jr.split(self.rng_key)
        delta_w_L_minus_1_samples = jr.normal(
            sample_key, (num_samples, w_L_minus_1_flat_dim)
        ) * jnp.sqrt(self.sigma_squared)
        # delta_w_L_minus_1_samples shape: (num_samples, w_L_minus_1_flat_dim)

        # Compute samples for each batch item and each delta_w sample
        # Sampled output = mean_predictions_flat + J_f,w_L-1 @ delta_w_L-1_sample
        
        # Need to reshape mean_predictions to be flat per batch item
        mean_predictions_flat = mean_predictions.reshape(batch_size, -1) # (batch, output_flat_dim)

        def compute_single_sample_for_batch(J_batch_item: jnp.ndarray, mean_flat_batch_item: jnp.ndarray, delta_w_sample: jnp.ndarray) -> jnp.ndarray:
            jacobian_vec_product = J_batch_item @ delta_w_sample
            return mean_flat_batch_item + jacobian_vec_product

        # vmap over batch items first, then over samples
        samples_flat_batch = jax.vmap(
            lambda J_item, mean_flat_item: jax.vmap(
                lambda delta_w: compute_single_sample_for_batch(J_item, mean_flat_item, delta_w)
            )(delta_w_L_minus_1_samples)
        )(J_f_w_L_minus_1_batch, mean_predictions_flat)
        # samples_flat_batch shape: (batch_size, num_samples, output_flat_dim)
        
        # Reshape back to (num_samples, batch, *spatial_dims, output_channels)
        output_spatial_shape = mean_predictions.shape[1:] # (*spatial_dims, output_channels)
        all_samples = samples_flat_batch.transpose((1, 0, 2)).reshape(num_samples, batch_size, *output_spatial_shape)
        
        if added_batch_dim:
            all_samples = jnp.squeeze(all_samples, axis=1) # Remove batch dim (which was 1)
        
        return all_samples


class LunoLA(_LunoBase):
    """
    Implements LUNO with Linearized Laplace Approximation for the weight belief,
    restricting uncertainty to the parameters of the last Fourier block.
    """

    def __init__(self, fno_module: FNO, trained_params: Params, config: Config):
        super().__init__(fno_module, trained_params, config)
        self.prior_sigma_squared = config.uq_methods_config["luno_la"]["prior_sigma_init"] ** 2 # Square std_dev for variance
        self.ggn_low_rank = config.uq_methods_config["luno_la"]["ggn_low_rank"]
        self.ggn_data_minibatch_size = config.uq_methods_config["luno_la"]["ggn_data_minibatch_size"]
        
        self.weight_covariance_factors: Optional[jnp.ndarray] = None # Will store P_dagger
        logging.info(f"LunoLA initialized with initial prior_sigma_squared: {self.prior_sigma_squared}")

    def fit(self, train_data: Optional[Tuple] = None) -> None:
        """
        Computes the low-rank GGN approximation for the linearized Laplace approximation
        of the weight posterior covariance for w_L-1.
        """
        if train_data is None:
            raise ValueError("train_data must be provided for LunoLA.fit to compute GGN.")
        
        logging.info("Starting LunoLA GGN computation for last Fourier block parameters...")

        train_x_in, train_conditions, train_targets = train_data
        num_train_data = train_x_in.shape[0]
        
        # Determine number of data points to use for GGN (full for low-data, minibatch for OOD)
        current_dataset_name = self.config.datasets.get("pde_name", "")
        is_low_data_regime = (current_dataset_name in ["Burgers", "Hyper Diffusion", "Kuramoto-Sivashinsky (cons.)"])
        
        if is_low_data_regime:
            data_for_ggn_x_in = train_x_in
            data_for_ggn_conditions = train_conditions
            # data_for_ggn_targets = train_targets # Not directly used by GGN calculation
            n_data_ggn = num_train_data
        else: # OOD regime
            self.rng_key, ggn_data_key = jr.split(self.rng_key)
            indices = jr.choice(ggn_data_key, num_train_data, (self.ggn_data_minibatch_size,), replace=False)
            data_for_ggn_x_in = train_x_in[indices]
            data_for_ggn_conditions = train_conditions[indices]
            # data_for_ggn_targets = train_targets[indices]
            n_data_ggn = self.ggn_data_minibatch_size
        
        logging.info(f"Using {n_data_ggn} data points for GGN approximation.")
        
        w_L_minus_1_flat_dim = self.w_L_minus_1_params_flat.shape[0]

        def _ggn_matvec(v: jnp.ndarray) -> jnp.ndarray:
            """
            Computes GGN @ v for the parameters of the last Fourier block.
            G = sum_i (J_i.T @ J_i) for MSE.
            """
            # Define inner function for vmap over batch items
            def batch_ggn_matvec_product(x_in_item: jnp.ndarray, conditions_item: jnp.ndarray, v_ggn: jnp.ndarray) -> jnp.ndarray:
                # The function whose Jacobian we need for GGN
                def f_output_wrt_w_L_minus_1_for_item(w_L_minus_1_candidate_flat: jnp.ndarray) -> jnp.ndarray:
                    full_params_with_candidate_w_L_minus_1 = self._reconstruct_params_tree_with_subtree(
                        self.trained_params, w_L_minus_1_candidate_flat, self.w_L_minus_1_unflatten_fn, self.w_L_minus_1_path_prefix
                    )
                    f_out_tree = self.fno_module.apply(
                        {'params': full_params_with_candidate_w_L_minus_1},
                        jnp.expand_dims(x_in_item, 0), # Add batch dim for single item inference
                        jnp.expand_dims(conditions_item, 0)
                    )
                    f_out_flat, _ = tree_util.tree_flatten(f_out_tree)
                    return jnp.concatenate(f_out_flat)

                # Compute J @ v_ggn
                _, jvp_result = jax.jvp(f_output_wrt_w_L_minus_1_for_item, (self.w_L_minus_1_params_flat,), (v_ggn,))
                
                # Compute J.T @ (J @ v_ggn) using vjp
                vjp_fn = jax.vjp(f_output_wrt_w_L_minus_1_for_item, self.w_L_minus_1_params_flat)[1]
                vjp_result_tree = vjp_fn(jvp_result) # vjp_result_tree is (G @ v, )
                return vjp_result_tree[0]

            # Sum over batch items
            total_gv = jax.vmap(batch_ggn_matvec_product, in_axes=(0, 0, None))(
                data_for_ggn_x_in, data_for_ggn_conditions, v
            )
            return jnp.sum(total_gv, axis=0)

        logging.info(f"Solving for top {self.ggn_low_rank} eigenvectors of GGN...")
        
        try:
            # We need k < N for eigsh. If dim is too small, use full matrix.
            k_val = min(self.ggn_low_rank, w_L_minus_1_flat_dim - 1)
            if k_val < 1: # Handle very small parameter spaces
                logging.warning(f"Parameter dimension ({w_L_minus_1_flat_dim}) too small for low-rank GGN. Using isotropic prior.")
                self.weight_covariance_factors = self.prior_sigma_squared * jnp.eye(w_L_minus_1_flat_dim)
                return

            eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
                A=_ggn_matvec,
                k=k_val,
                which='LM', # Largest magnitude eigenvalues
                ncv=min(2 * k_val + 1, w_L_minus_1_flat_dim - 1), # Number of Lanczos vectors
                maxiter=5000,
                tol=1e-3 # Tolerance for convergence
            )
            eigenvalues = jnp.maximum(eigenvalues, 0.0) # Ensure non-negative eigenvalues
            
            # Construct P_dagger using Woodbury-like identity as derived in related work (Immer et al., 2021)
            # P = GGN + (1/prior_sigma^2) * I
            # If GGN_approx = V @ diag(eigenvalues) @ V.T (where V are eigenvectors)
            # P_inv = (V @ diag(eigenvalues) @ V.T + (1/prior_sigma^2) * I)^-1
            # P_inv = prior_sigma^2 * I - (prior_sigma^2)^2 * V @ (I + prior_sigma^2 * diag(eigenvalues))^-1 @ diag(eigenvalues) @ V.T
            
            prior_precision = 1.0 / self.prior_sigma_squared
            
            # Inverse of (I + prior_sigma^2 * diag(eigenvalues))
            diag_inverse_term = jnp.diag(1.0 / (1.0 + prior_precision * eigenvalues))
            
            # The term (I + prior_sigma^2 * Lambda)^-1 @ Lambda
            intermediate_matrix = diag_inverse_term @ jnp.diag(eigenvalues)

            # P_dagger = prior_sigma^2 * I - (prior_sigma^2)^2 * V @ intermediate_matrix @ V.T
            P_dagger = self.prior_sigma_squared * jnp.eye(w_L_minus_1_flat_dim) - \
                       (self.prior_sigma_squared ** 2) * eigenvectors @ intermediate_matrix @ eigenvectors.T

            self.weight_covariance_factors = P_dagger
            
            logging.info("LunoLA GGN computation complete. P_dagger stored.")

        except Exception as e:
            logging.error(f"Error computing GGN or its inverse for LunoLA: {e}")
            logging.warning(f"Falling back to isotropic weight covariance for LunoLA due to GGN computation error. Using prior_sigma_squared={self.prior_sigma_squared}")
            self.weight_covariance_factors = self.prior_sigma_squared * jnp.eye(w_L_minus_1_flat_dim)


    def predict_uncertainty(
        self, input_func: jnp.ndarray, conditions: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Computes the mean and standard deviation of the predictive distribution using LUNO-LA.
        """
        if self.weight_covariance_factors is None:
            # Fallback to isotropic prior if fit was not successful
            logging.warning("LunoLA.fit() was not called or failed. Falling back to isotropic prior for prediction.")
            w_L_minus_1_flat_dim = self.w_L_minus_1_params_flat.shape[0]
            self.weight_covariance_factors = self.prior_sigma_squared * jnp.eye(w_L_minus_1_flat_dim)

        # Ensure input_func and conditions are batched
        added_batch_dim = False
        if input_func.ndim < self.fno_module.dummy_x_in.ndim: # If unbatched, add batch dim
            input_func = jnp.expand_dims(input_func, axis=0)
            conditions = jnp.expand_dims(conditions, axis=0)
            added_batch_dim = True
        
        batch_size = input_func.shape[0]

        # Mean prediction: Simply the FNO's output with the trained weights
        mean_predictions = self.fno_module.apply(
            {'params': self.trained_params}, input_func, conditions
        )
        
        # For standard deviation, we need the Jacobian J_f,w_L-1
        J_f_w_L_minus_1_batch = jax.vmap(self._compute_full_jacobian_J_f_w_L_minus_1, in_axes=(0, 0))(input_func, conditions)
        # J_f_w_L_minus_1_batch shape: (batch_size, output_flat_dim, w_L_minus_1_flat_dim)
        
        # Predictive covariance for each batch item: J_f,w_L-1 @ P_dagger @ J_f,w_L-1.T
        predictive_covariance_batch = jax.vmap(
            lambda J_item: J_item @ self.weight_covariance_factors @ J_item.T
        )(J_f_w_L_minus_1_batch)
        
        # Standard deviation is sqrt of the diagonal elements
        std_dev_predictions_flat = jnp.sqrt(jnp.diagonal(predictive_covariance_batch, axis1=1, axis2=2))
        
        # Reshape std_dev_predictions_flat back to original output shape
        std_dev_predictions = std_dev_predictions_flat.reshape(mean_predictions.shape)
        
        if added_batch_dim:
            mean_predictions = jnp.squeeze(mean_predictions, axis=0)
            std_dev_predictions = jnp.squeeze(std_dev_predictions, axis=0)

        return mean_predictions, std_dev_predictions

    def get_samples(
        self, input_func: jnp.ndarray, conditions: jnp.ndarray, num_samples: int
    ) -> jnp.ndarray:
        """
        Generates samples from the LUNO-LA predictive distribution.
        """
        if self.weight_covariance_factors is None:
            raise RuntimeError("LunoLA.fit() must be called before generating samples.")

        # Ensure input_func and conditions are batched
        added_batch_dim = False
        if input_func.ndim < self.fno_module.dummy_x_in.ndim:
            input_func = jnp.expand_dims(input_func, axis=0)
            conditions = jnp.expand_dims(conditions, axis=0)
            added_batch_dim = True

        batch_size = input_func.shape[0]

        mean_predictions = self.fno_module.apply(
            {'params': self.trained_params}, input_func, conditions
        ) # (batch, *spatial_dims, output_channels)

        # Compute J_f,w_L-1 for each item in the batch
        J_f_w_L_minus_1_batch = jax.vmap(self._compute_full_jacobian_J_f_w_L_minus_1, in_axes=(0, 0))(input_func, conditions)
        # J_f_w_L_minus_1_batch shape: (batch_size, output_flat_dim, w_L_minus_1_flat_dim)
        
        w_L_minus_1_flat_dim = self.w_L_minus_1_params_flat.shape[0]
        
        # Sample delta_w_L_minus_1 from N(0, P_dagger)
        self.rng_key, sample_key = jr.split(self.rng_key)
        # jax.random.multivariate_normal requires covariance matrix
        delta_w_L_minus_1_samples = jr.multivariate_normal(
            sample_key,
            mean=jnp.zeros(w_L_minus_1_flat_dim),
            cov=self.weight_covariance_factors,
            shape=(num_samples,)
        )
        # delta_w_L_minus_1_samples shape: (num_samples, w_L_minus_1_flat_dim)

        mean_predictions_flat = mean_predictions.reshape(batch_size, -1)

        def compute_single_sample_for_batch(J_batch_item: jnp.ndarray, mean_flat_batch_item: jnp.ndarray, delta_w_sample: jnp.ndarray) -> jnp.ndarray:
            jacobian_vec_product = J_batch_item @ delta_w_sample
            return mean_flat_batch_item + jacobian_vec_product

        samples_flat_batch = jax.vmap(
            lambda J_item, mean_flat_item: jax.vmap(
                lambda delta_w: compute_single_sample_for_batch(J_item, mean_flat_item, delta_w)
            )(delta_w_L_minus_1_samples)
        )(J_f_w_L_minus_1_batch, mean_predictions_flat)
        
        output_spatial_shape = mean_predictions.shape[1:]
        all_samples = samples_flat_batch.transpose((1, 0, 2)).reshape(num_samples, batch_size, *output_spatial_shape)
        
        if added_batch_dim:
            all_samples = jnp.squeeze(all_samples, axis=1)

        return all_samples


class _SamplingBase(UQMethod):
    """
    Base class for sampling-based UQ methods, handling common logic for weight flattening.
    """

    def __init__(self, fno_module: FNO, trained_params: Params, config: Config):
        super().__init__(fno_module, trained_params, config)
        self.trained_params: Params = trained_params # Ensure specific type for sampling methods

        # Flatten all trained parameters for sampling in weight space
        self.w_star_flat, self.unflatten_fn_full_model = tree_util.tree_flatten(self.trained_params)
        self.w_dim = self.w_star_flat.shape[0]

        # num_samples can be specified per method in config, using a default for Safety
        self.num_samples = config.uq_methods_config.get(self.__class__.__name__.lower(), {}).get(
            "num_samples", 200
        )
        logging.info(f"{self.__class__.__name__} initialized. Total parameters: {self.w_dim}")

    def _sample_weights(self, rng_key: PRNGKey, cov_matrix: jnp.ndarray) -> List[Params]:
        """
        Generates `self.num_samples` weight vectors by sampling from a Gaussian
        distribution with mean `self.w_star_flat` and `cov_matrix`.
        """
        sampled_w_flat = jr.multivariate_normal(
            rng_key, self.w_star_flat, cov_matrix, shape=(self.num_samples,)
        )
        
        # Unflatten each sampled vector back to FrozenDict structure
        sampled_params_list = [
            self.unflatten_fn_full_model(list(w_flat)) for w_flat in sampled_w_flat
        ]
        return sampled_params_list

    def predict_uncertainty(
        self, input_func: jnp.ndarray, conditions: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Computes empirical mean and standard deviation from sampled predictions.
        """
        all_predictions = self.get_samples(input_func, conditions, self.num_samples)
        
        # all_predictions shape: (num_samples, batch, *spatial_dims, output_channels)
        # Compute mean and std dev across the sample dimension (axis=0)
        empirical_mean = jnp.mean(all_predictions, axis=0)
        empirical_std_dev = jnp.std(all_predictions, axis=0)
        
        return empirical_mean, empirical_std_dev

    @abc.abstractmethod
    def get_samples(
        self, input_func: jnp.ndarray, conditions: jnp.ndarray, num_samples: int
    ) -> jnp.ndarray:
        """
        Returns an array of `num_samples` predictions using sampled weights.
        This abstract method is implemented in subclasses.
        """
        pass


class SampleIso(_SamplingBase):
    """
    Implements sampling-based UQ with an isotropic Gaussian weight belief for full model.
    """

    def __init__(self, fno_module: FNO, trained_params: Params, config: Config):
        super().__init__(fno_module, trained_params, config)
        self.sigma_squared = config.uq_methods_config["sample_iso"]["sigma_squared_init"]
        logging.info(f"SampleIso initialized with initial sigma_squared: {self.sigma_squared}")

    def fit(self, train_data: Optional[Tuple] = None) -> None:
        """
        For SampleIso, `fit` is mainly used to update `sigma_squared` after calibration.
        """
        logging.info("SampleIso fit method called. No complex computations, expects calibration to set sigma_squared.")
        pass

    def get_samples(
        self, input_func: jnp.ndarray, conditions: jnp.ndarray, num_samples: int
    ) -> jnp.ndarray:
        """
        Generates samples from the Sample-Iso predictive distribution.
        """
        self.rng_key, sample_w_key = jr.split(self.rng_key, 2)

        # Create isotropic covariance matrix
        isotropic_cov = self.sigma_squared * jnp.eye(self.w_dim)
        
        # Sample weights
        sampled_params_list = self._sample_weights(sample_w_key, isotropic_cov)
        
        # Use vmap to apply FNO for each sampled parameter set
        # This will return (num_samples, batch, *spatial_dims, output_channels)
        all_predictions = jax.vmap(
            lambda params: self.fno_module.apply({'params': params}, input_func, conditions)
        )(sampled_params_list)

        return all_predictions


class SampleLA(_SamplingBase):
    """
    Implements sampling-based UQ with Linearized Laplace Approximation weight belief for full model.
    """

    def __init__(self, fno_module: FNO, trained_params: Params, config: Config):
        super().__init__(fno_module, trained_params, config)
        self.prior_sigma_squared = config.uq_methods_config["sample_la"]["prior_sigma_init"] ** 2 # Square for variance
        self.ggn_low_rank = config.uq_methods_config["sample_la"]["ggn_low_rank"]
        self.ggn_data_minibatch_size = config.uq_methods_config["sample_la"]["ggn_data_minibatch_size"]
        
        self.weight_covariance_factors: Optional[jnp.ndarray] = None # Will store P_dagger
        logging.info(f"SampleLA initialized with initial prior_sigma_squared: {self.prior_sigma_squared}")

    def fit(self, train_data: Optional[Tuple] = None) -> None:
        """
        Computes the low-rank GGN approximation for the linearized Laplace approximation
        of the weight posterior covariance for the full model parameters.
        """
        if train_data is None:
            raise ValueError("train_data must be provided for SampleLA.fit to compute GGN.")
        
        logging.info("Starting SampleLA GGN computation for full model parameters...")

        train_x_in, train_conditions, train_targets = train_data
        num_train_data = train_x_in.shape[0]
        
        current_dataset_name = self.config.datasets.get("pde_name", "")
        is_low_data_regime = (current_dataset_name in ["Burgers", "Hyper Diffusion", "Kuramoto-Sivashinsky (cons.)"])

        # Determine number of data points to use for GGN
        if is_low_data_regime:
            data_for_ggn_x_in = train_x_in
            data_for_ggn_conditions = train_conditions
            # data_for_ggn_targets = train_targets
            n_data_ggn = num_train_data
        else: # OOD regime
            self.rng_key, ggn_data_key = jr.split(self.rng_key)
            indices = jr.choice(ggn_data_key, num_train_data, (self.ggn_data_minibatch_size,), replace=False)
            data_for_ggn_x_in = train_x_in[indices]
            data_for_ggn_conditions = train_conditions[indices]
            # data_for_ggn_targets = train_targets[indices]
            n_data_ggn = self.ggn_data_minibatch_size
        
        logging.info(f"Using {n_data_ggn} data points for GGN approximation.")

        def _ggn_matvec_full_model(v: jnp.ndarray) -> jnp.ndarray:
            """
            Computes GGN @ v for the full model parameters.
            """
            def batch_ggn_matvec_product_full_model(x_in_item: jnp.ndarray, conditions_item: jnp.ndarray, v_ggn: jnp.ndarray) -> jnp.ndarray:
                def f_output_wrt_w_full(w_full_candidate_flat: jnp.ndarray) -> jnp.ndarray:
                    w_full_candidate_tree = self.unflatten_fn_full_model(list(w_full_candidate_flat))
                    f_out_tree = self.fno_module.apply(
                        {'params': w_full_candidate_tree},
                        jnp.expand_dims(x_in_item, 0),
                        jnp.expand_dims(conditions_item, 0)
                    )
                    f_out_flat, _ = tree_util.tree_flatten(f_out_tree)
                    return jnp.concatenate(f_out_flat)

                _, jvp_result = jax.jvp(f_output_wrt_w_full, (self.w_star_flat,), (v_ggn,))
                vjp_fn = jax.vjp(f_output_wrt_w_full, self.w_star_flat)[1]
                return vjp_fn(jvp_result)[0]

            total_gv = jax.vmap(batch_ggn_matvec_product_full_model, in_axes=(0, 0, None))(
                data_for_ggn_x_in, data_for_ggn_conditions, v
            )
            return jnp.sum(total_gv, axis=0)

        logging.info(f"Solving for top {self.ggn_low_rank} eigenvectors of GGN (full model)...")
        try:
            k_val = min(self.ggn_low_rank, self.w_dim - 1)
            if k_val < 1:
                logging.warning(f"Parameter dimension ({self.w_dim}) too small for low-rank GGN. Using isotropic prior.")
                self.weight_covariance_factors = self.prior_sigma_squared * jnp.eye(self.w_dim)
                return

            eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
                A=_ggn_matvec_full_model,
                k=k_val,
                which='LM',
                ncv=min(2 * k_val + 1, self.w_dim - 1),
                maxiter=5000,
                tol=1e-3
            )
            eigenvalues = jnp.maximum(eigenvalues, 0.0) # Ensure non-negative eigenvalues

            prior_precision = 1.0 / self.prior_sigma_squared
            
            diag_inverse_term = jnp.diag(1.0 / (1.0 + prior_precision * eigenvalues))
            intermediate_matrix = diag_inverse_term @ jnp.diag(eigenvalues)
            
            P_dagger = self.prior_sigma_squared * jnp.eye(self.w_dim) - \
                       (self.prior_sigma_squared ** 2) * eigenvectors @ intermediate_matrix @ eigenvectors.T
            
            self.weight_covariance_factors = P_dagger
            logging.info("SampleLA GGN computation complete. P_dagger stored.")

        except Exception as e:
            logging.error(f"Error computing GGN or its inverse for SampleLA: {e}")
            logging.warning(f"Falling back to isotropic weight covariance for SampleLA due to GGN computation error. Using prior_sigma_squared={self.prior_sigma_squared}")
            self.weight_covariance_factors = self.prior_sigma_squared * jnp.eye(self.w_dim)

    def get_samples(
        self, input_func: jnp.ndarray, conditions: jnp.ndarray, num_samples: int
    ) -> jnp.ndarray:
        """
        Generates samples from the Sample-LA predictive distribution.
        """
        if self.weight_covariance_factors is None:
            raise RuntimeError("SampleLA.fit() must be called before generating samples.")

        self.rng_key, sample_w_key = jr.split(self.rng_key, 2)

        sampled_params_list = self._sample_weights(sample_w_key, self.weight_covariance_factors)

        all_predictions = jax.vmap(
            lambda params: self.fno_module.apply({'params': params}, input_func, conditions)
        )(sampled_params_list)

        return all_predictions


class InputPerturbations(UQMethod):
    """
    Implements UQ by perturbing input data with Gaussian noise.
    """

    def __init__(self, fno_module: FNO, trained_params: Params, config: Config):
        super().__init__(fno_module, trained_params, config)
        self.trained_params: Params = trained_params # Ensure specific type
        self.noise_sigma = config.uq_methods_config["input_perturbations"]["noise_sigma_init"]
        self.num_perturbations = config.uq_methods_config["input_perturbations"]["num_perturbations"]
        logging.info(f"InputPerturbations initialized with initial noise_sigma: {self.noise_sigma}")

    def fit(self, train_data: Optional[Tuple] = None) -> None:
        """
        For InputPerturbations, `fit` is mainly used to update `noise_sigma` after calibration.
        """
        logging.info("InputPerturbations fit method called. No complex computations, expects calibration to set noise_sigma.")
        pass

    def _perturb_input(self, rng_key: PRNGKey, input_func_item: jnp.ndarray) -> jnp.ndarray:
        """
        Adds Gaussian noise to a single input function.
        """
        noise = jr.normal(rng_key, input_func_item.shape) * self.noise_sigma
        return input_func_item + noise

    def predict_uncertainty(
        self, input_func: jnp.ndarray, conditions: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Computes mean and standard deviation from predictions on perturbed inputs.
        """
        # Ensure input_func and conditions are batched
        added_batch_dim = False
        if input_func.ndim < self.fno_module.dummy_x_in.ndim:
            input_func = jnp.expand_dims(input_func, axis=0)
            conditions = jnp.expand_dims(conditions, axis=0)
            added_batch_dim = True
            
        all_predictions = self.get_samples(input_func, conditions, self.num_perturbations)
        
        empirical_mean = jnp.mean(all_predictions, axis=0)
        empirical_std_dev = jnp.std(all_predictions, axis=0)

        if added_batch_dim:
            empirical_mean = jnp.squeeze(empirical_mean, axis=0)
            empirical_std_dev = jnp.squeeze(empirical_std_dev, axis=0)
        
        return empirical_mean, empirical_std_dev

    def get_samples(
        self, input_func: jnp.ndarray, conditions: jnp.ndarray, num_samples: int
    ) -> jnp.ndarray:
        """
        Generates samples by perturbing input functions and running FNO.
        """
        self.rng_key, *perturb_keys = jr.split(self.rng_key, num_samples + 1)

        # Perturb input_func `num_samples` times
        # vmap over perturbation keys, broadcast input_func
        perturbed_inputs = jax.vmap(self._perturb_input, in_axes=(0, None))(
            jnp.asarray(perturb_keys), input_func
        )
        
        # vmap over perturbed inputs, conditions are broadcasted (same for all perturbed inputs)
        # All predictions will be (num_samples, batch, *spatial_dims, output_channels)
        all_predictions = jax.vmap(
            lambda perturbed_x_item: self.fno_module.apply(
                {'params': self.trained_params}, perturbed_x_item, conditions
            )
        )(perturbed_inputs)
        
        return all_predictions


class DeepEnsemble(UQMethod):
    """
    Implements UQ using a deep ensemble of independently trained FNOs.
    """

    def __init__(self, fno_module: FNO, trained_params_list: List[Params], config: Config):
        # Override __init__ type hint to explicitly take List[Params]
        super().__init__(fno_module, trained_params_list, config)
        self.trained_params_list: List[Params] = trained_params_list
        self.num_members = len(self.trained_params_list)
        # Assert config value matches actual list length
        if config.uq_methods_config["deep_ensemble"]["num_members"] != self.num_members:
            logging.warning(f"Configured num_members ({config.uq_methods_config['deep_ensemble']['num_members']}) "
                            f"does not match actual number of ensemble members ({self.num_members}). Using actual.")
            # config.uq_methods_config["deep_ensemble"]["num_members"] = self.num_members # Not good to modify config here

        logging.info(f"DeepEnsemble initialized with {self.num_members} members.")

    def fit(self, train_data: Optional[Tuple] = None) -> None:
        """
        For DeepEnsemble, `fit` is a no-op as training happens externally.
        """
        logging.info("DeepEnsemble fit method called. No internal computations, assumes members are pre-trained.")
        pass

    def predict_uncertainty(
        self, input_func: jnp.ndarray, conditions: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Computes empirical mean and standard deviation from ensemble member predictions.
        """
        all_predictions = self.get_samples(input_func, conditions, self.num_members) # Get all member predictions
        
        # all_predictions shape: (num_members, batch, *spatial_dims, output_channels)
        empirical_mean = jnp.mean(all_predictions, axis=0)
        empirical_std_dev = jnp.std(all_predictions, axis=0)
        
        return empirical_mean, empirical_std_dev

    def get_samples(
        self, input_func: jnp.ndarray, conditions: jnp.ndarray, num_samples: int
    ) -> jnp.ndarray:
        """
        Returns predictions from ensemble members. If num_samples is less than
        the total number of members, a subset is returned.
        """
        if num_samples > self.num_members:
            logging.warning(f"Requested {num_samples} samples but only {self.num_members} ensemble members available. "
                            "Returning all available members.")
            num_samples = self.num_members
            
        # Select a subset of members if num_samples < self.num_members
        selected_members_params = self.trained_params_list[:num_samples]
        
        # Use vmap to apply FNO for each ensemble member's parameters
        # This returns (num_selected_members, batch, *spatial_dims, output_channels)
        all_predictions = jax.vmap(
            lambda params: self.fno_module.apply({'params': params}, input_func, conditions)
        )(selected_members_params)

        return all_predictions

