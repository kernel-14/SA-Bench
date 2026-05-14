
import jax
import jax.numpy as jnp
import flax
from flax.core import FrozenDict
import ml_collections
import os
from typing import Tuple, Dict, Any
from tqdm import tqdm
import numpy as np

from luno.models.fno import FNO
from luno.models.luno_fno import LUNOFNO
from luno.data.pde_datasets import PDEDataset
from luno.utils.metrics import compute_metrics, nll

def load_fno_model(config: ml_collections.ConfigDict, params_path: str) -> Tuple[FNO, FrozenDict]:
    """Loads a trained FNO model and its parameters."""
    # Determine input features
    # This requires a dummy dataset to get the input shape dynamically
    dummy_dataset = PDEDataset(config, 'train') # Using train split for dummy input
    dummy_input_seq, _ = dummy_dataset[0]
    input_features = dummy_input_seq.shape[-1] * dummy_input_seq.shape[0]
    if config.model.add_pos_encoding:
        input_features += 1

    fno_model = FNO(
        modes=config.model.modes,
        hidden_dim=config.model.hidden_dim,
        num_fourier_blocks=config.model.num_fourier_blocks,
        output_dim=config.model.output_dim,
        add_pos_encoding=config.model.add_pos_encoding
    )
    
    # Initialize params with a dummy input for flax to build the structure
    dummy_fno_input = jnp.zeros((1, config.data.spatial_resolution, input_features))
    initial_params = fno_model.init(jax.random.PRNGKey(0), dummy_fno_input)['params']

    # Load trained parameters
    with open(params_path, 'rb') as f:
        restored_params = flax.serialization.from_bytes(initial_params, f.read())
    
    return fno_model, restored_params

def evaluate_uq_method(
    fno_model: FNO, 
    fno_params: FrozenDict, 
    config: ml_collections.ConfigDict, 
    test_dataset: PDEDataset,
    rng_key: jax.random.PRNGKey
) -> Dict[str, float]:
    """
    Evaluates a specified Uncertainty Quantification method.
    """
    uq_method = config.uq.method
    print(f"Evaluating UQ method: {uq_method}")

    all_metrics = {metric: [] for metric in config.evaluation.metrics}

    test_dataloader = test_dataset.get_dataloader(config.training.batch_size)
    
    for batch_idx, (batch_inputs_seq, batch_targets) in enumerate(tqdm(test_dataloader, desc=f"Evaluating {uq_method}")):
        rng_key, subkey = jax.random.split(rng_key)
        
        # Reshape inputs for FNO: (batch, spatial_res, num_initial_steps * input_channels)
        batch_inputs_reshaped = batch_inputs_seq.transpose(0, 2, 1, 3).reshape(
            batch_inputs_seq.shape[0], batch_inputs_seq.shape[2], -1
        )

        mean_predictions = None
        cov_predictions = None

        if uq_method.startswith('LUNO'):
            luno_fno = LUNOFNO(fno_model, fno_params, config.uq)
            mean_predictions, cov_predictions = luno_fno.predict_mean_and_cov(batch_inputs_reshaped)
            
        elif uq_method.startswith('Sample'):
            num_samples = config.uq.num_samples
            
            # This is a simplified placeholder.
            # Proper implementation involves:
            # 1. For 'Sample-LA', computing GGN (not done here).
            # 2. For 'Sample-Iso', sampling from N(fno_params, sigma_iso^2 * I).
            # For demonstration, we'll sample from a perturbed version of fno_params.
            
            sampled_predictions = []
            
            # Create a base flattened parameter array for perturbation
            flat_fno_params, treedef = jax.tree_util.tree_flatten(fno_params)
            flat_fno_params_arr = jnp.array(flat_fno_params)

            # Determine variance for sampling (placeholder)
            sample_variance = 1e-4 # Default, replace with calibrated sigma_iso for Sample-Iso
            if uq_method == 'Sample-Iso' and config.uq.sigma_iso is not None:
                sample_variance = config.uq.sigma_iso
            # For Sample-LA, this would involve drawing from (GGN)^-1 which is complex.

            for s_idx in range(num_samples):
                subkey, sample_key = jax.random.split(subkey)
                # Sample noise with determined variance
                noise = jax.random.normal(sample_key, flat_fno_params_arr.shape) * sample_variance
                # Perturb flattened parameters
                perturbed_flat_params_arr = flat_fno_params_arr + noise
                # Reconstruct perturbed params tree
                perturbed_params = jax.tree_util.tree_unflatten(treedef, perturbed_flat_params_arr.tolist()) # Convert back to list for unflatten

                pred = fno_model.apply({'params': perturbed_params}, batch_inputs_reshaped)
                sampled_predictions.append(pred)
            
            sampled_predictions = jnp.stack(sampled_predictions) # (num_samples, batch, spatial_res, output_dim)
            mean_predictions = jnp.mean(sampled_predictions, axis=0) # (batch, spatial_res, output_dim)
            
            # Compute empirical covariance. For compute_metrics, we only need marginal variances.
            marginal_variances = jnp.var(sampled_predictions, axis=0) # (batch, spatial_res, output_dim)
            
            # Construct a diagonal covariance matrix for compute_metrics for simplicity
            spatial_res = batch_inputs_reshaped.shape[1]
            output_dim = config.model.output_dim
            cov_predictions = jnp.zeros(
                (batch_inputs_reshaped.shape[0], spatial_res, output_dim, spatial_res, output_dim)
            )
            for b in range(batch_inputs_reshaped.shape[0]):
                for s in range(spatial_res):
                    for o in range(output_dim):
                        cov_predictions = cov_predictions.at[b, s, o, s, o].set(marginal_variances[b, s, o])
                        
        elif uq_method == 'Ensemble':
            # This would require loading multiple FNO models trained with different seeds.
            # Placeholder: Simulate ensemble predictions by perturbing the single model's params.
            num_ensemble_members = 10 # As per paper
            ensemble_predictions = []
            
            flat_fno_params, treedef = jax.tree_util.tree_flatten(fno_params)
            flat_fno_params_arr = jnp.array(flat_fno_params)

            for e_idx in range(num_ensemble_members):
                subkey, ensemble_key = jax.random.split(subkey)
                # Simulate a different ensemble member by applying a perturbation
                # In a real setup, this would load pre-trained `e_idx` model params.
                perturbed_flat_params_arr = flat_fno_params_arr + jax.random.normal(ensemble_key, flat_fno_params_arr.shape) * 1e-3
                perturbed_params = jax.tree_util.tree_unflatten(treedef, perturbed_flat_params_arr.tolist())
                
                pred = fno_model.apply({'params': perturbed_params}, batch_inputs_reshaped)
                ensemble_predictions.append(pred)
            
            ensemble_predictions = jnp.stack(ensemble_predictions) # (num_ensemble, batch, spatial_res, output_dim)
            mean_predictions = jnp.mean(ensemble_predictions, axis=0)
            
            marginal_variances = jnp.var(ensemble_predictions, axis=0)
            spatial_res = batch_inputs_reshaped.shape[1]
            output_dim = config.model.output_dim
            cov_predictions = jnp.zeros(
                (batch_inputs_reshaped.shape[0], spatial_res, output_dim, spatial_res, output_dim)
            )
            for b in range(batch_inputs_reshaped.shape[0]):
                for s in range(spatial_res):
                    for o in range(output_dim):
                        cov_predictions = cov_predictions.at[b, s, o, s, o].set(marginal_variances[b, s, o])
            
        elif uq_method == 'InputPerturbations':
            num_perturbations = 50 # Example number, actual should be calibrated
            perturbation_std = 0.01 # Calibrated parameter
            
            perturbed_predictions = []
            for p_idx in range(num_perturbations):
                subkey, perturb_key = jax.random.split(subkey)
                # Add noise to input
                perturbed_input = batch_inputs_reshaped + jax.random.normal(perturb_key, batch_inputs_reshaped.shape) * perturbation_std
                pred = fno_model.apply({'params': fno_params}, perturbed_input)
                perturbed_predictions.append(pred)
            
            perturbed_predictions = jnp.stack(perturbed_predictions)
            mean_predictions = jnp.mean(perturbed_predictions, axis=0)
            
            marginal_variances = jnp.var(perturbed_predictions, axis=0)
            spatial_res = batch_inputs_reshaped.shape[1]
            output_dim = config.model.output_dim
            cov_predictions = jnp.zeros(
                (batch_inputs_reshaped.shape[0], spatial_res, output_dim, spatial_res, output_dim)
            )
            for b in range(batch_inputs_reshaped.shape[0]):
                for s in range(spatial_res):
                    for o in range(output_dim):
                        cov_predictions = cov_predictions.at[b, s, o, s, o].set(marginal_variances[b, s, o])

        else:
            raise ValueError(f"Unknown UQ method: {uq_method}")

        # Compute and aggregate metrics
        if mean_predictions is not None and cov_predictions is not None:
            batch_metrics = compute_metrics(batch_targets, mean_predictions, cov_predictions)
            for metric_name, value in batch_metrics.items():
                all_metrics[metric_name].append(value)
        else:
            raise ValueError("Predictions not computed for the selected UQ method.")

    # Average metrics over all batches
    avg_metrics = {name: jnp.mean(jnp.stack(values)) for name, values in all_metrics.items()}
    return avg_metrics

def calibrate_hyperparameters(
    fno_model: FNO,
    fno_params: FrozenDict,
    config: ml_collections.ConfigDict,
    val_dataset: PDEDataset,
    rng_key: jax.random.PRNGKey
) -> ml_collections.ConfigDict:
    """
    Calibrates UQ hyperparameters (e.g., sigma for LUNO-Iso) by minimizing NLL on validation set.
    """
    uq_method = config.uq.method
    print(f"Calibrating hyperparameters for {uq_method}...")

    if uq_method in ['LUNO-Iso', 'Sample-Iso']:
        best_nll = float('inf')
        best_hp_value = None

        # Calibrate sigma_iso. The paper mentions a grid search.
        # Logarithmically spaced grid with 500 points centered around the relevant value.
        # Let's assume relevant range for sigma_iso for initial guess.
        # The scale of the parameters (and thus noise) is usually small.
        # A range like 1e-8 to 1e-1 might be appropriate.
        log_hp_values = jnp.linspace(jnp.log10(1e-8), jnp.log10(1e-1), config.evaluation.calibration_grid_points)
        hp_values = 10**log_hp_values

        for hp_value in tqdm(hp_values, desc=f"Calibrating {uq_method} sigma_iso"):
            temp_uq_config = config.uq.copy_and_resolve_references()
            temp_uq_config.sigma_iso = hp_value
            
            current_nlls = []
            val_dataloader = val_dataset.get_dataloader(config.training.batch_size)
            # Re-initialize LUNO_FNO for each hp_value if it's LUNO-Iso
            # or pass the hp_value to sample_functions for Sample-Iso
            
            for batch_inputs_seq, batch_targets in val_dataloader:
                subkey, _ = jax.random.split(rng_key) 
                
                batch_inputs_reshaped = batch_inputs_seq.transpose(0, 2, 1, 3).reshape(
                    batch_inputs_seq.shape[0], batch_inputs_seq.shape[2], -1
                )

                mean_predictions = None
                cov_predictions = None

                if uq_method == 'LUNO-Iso':
                    luno_fno = LUNOFNO(fno_model, fno_params, temp_uq_config)
                    mean_predictions, cov_predictions = luno_fno.predict_mean_and_cov(batch_inputs_reshaped)
                elif uq_method == 'Sample-Iso':
                    num_samples = config.uq.num_samples
                    sampled_predictions = []
                    
                    flat_fno_params, treedef = jax.tree_util.tree_flatten(fno_params)
                    flat_fno_params_arr = jnp.array(flat_fno_params)

                    for s_idx in range(num_samples):
                        subkey, sample_key = jax.random.split(subkey)
                        noise = jax.random.normal(sample_key, flat_fno_params_arr.shape) * hp_value # Use hp_value as sigma
                        perturbed_flat_params_arr = flat_fno_params_arr + noise
                        perturbed_params = jax.tree_util.tree_unflatten(treedef, perturbed_flat_params_arr.tolist())
                        pred = fno_model.apply({'params': perturbed_params}, batch_inputs_reshaped)
                        sampled_predictions.append(pred)
                    
                    sampled_predictions = jnp.stack(sampled_predictions)
                    mean_predictions = jnp.mean(sampled_predictions, axis=0)
                    marginal_variances = jnp.var(sampled_predictions, axis=0)
                    
                    spatial_res = batch_inputs_reshaped.shape[1]
                    output_dim = config.model.output_dim
                    cov_predictions = jnp.zeros(
                        (batch_inputs_reshaped.shape[0], spatial_res, output_dim, spatial_res, output_dim)
                    )
                    for b in range(batch_inputs_reshaped.shape[0]):
                        for s in range(spatial_res):
                            for o in range(output_dim):
                                cov_predictions = cov_predictions.at[b, s, o, s, o].set(marginal_variances[b, s, o])
                
                if mean_predictions is not None and cov_predictions is not None:
                    metrics = compute_metrics(batch_targets, mean_predictions, cov_predictions)
                    current_nlls.append(metrics['NLL'])
                else:
                    print(f"Warning: No predictions for UQ method {uq_method} during calibration with hp_value {hp_value}.")
                    continue # Skip this hp_value if no predictions were generated

            if current_nlls: # Only if NLLs were actually collected
                avg_nll = jnp.mean(jnp.stack(current_nlls))
                if avg_nll < best_nll:
                    best_nll = avg_nll
                    best_hp_value = hp_value
            
        config.uq.sigma_iso = float(best_hp_value) # Convert to float for config storage
        print(f"Calibrated sigma_iso for {uq_method}: {config.uq.sigma_iso:.6f} with best NLL: {best_nll:.4f}")
    
    # For LUNO-LA, Sample-LA, Ensemble, InputPerturbations, calibration might involve other parameters
    # or no explicit calibration if using default settings from paper.
    # The paper mentions calibrating 'sigma^2' for isotropic, and 'low-rank' for LA.
    # For 'Sample-LA', 'LUNO-LA', the GGN computation itself implicitly defines the covariance,
    # so additional scalar factors might need calibration if not fully defined by the GGN itself.
    
    return config

