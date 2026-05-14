"""
main.py
The main entry point and orchestration script for the LUNO reproduction project.
It handles configuration loading, data generation, FNO training, uncertainty
quantification, calibration, and evaluation.
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Type, Union

import jax
import jax.numpy as jnp
import jax.random as jr
from flax.core import FrozenDict
from absl import logging
from tqdm import tqdm

# Local application imports
from config import Config
from datasets import PdeDatasetLoader, APEBenchDatasetLoader, AdvectionDiffusionDatasetLoader
from fno_model import FNO
from training import Trainer
from uq_methods import (
    UQMethod,
    LunoIso,
    LunoLA,
    SampleIso,
    SampleLA,
    InputPerturbations,
    DeepEnsemble,
)
from calibration import UQCalibrator
from evaluation import Evaluator
from utils import setup_logging, pad_input, stack_conditions # Import utility functions


class ExperimentRunner:
    """
    Orchestrates the entire LUNO reproduction experiment pipeline, including
    data generation, model training, uncertainty quantification, calibration,
    and evaluation.
    """

    def __init__(self, config_path: str):
        """
        Initializes the ExperimentRunner by loading configuration, setting up
        logging, and initializing JAX's PRNGKey.

        Args:
            config_path: Path to the YAML configuration file.
        """
        # 1. Configuration Loading
        with open(config_path, "r") as f:
            raw_config = yaml.safe_load(f)

        # Flatten raw_config for the Config class as per design
        flat_config_dict = {}
        for section, values in raw_config.items():
            if isinstance(values, dict):
                for key, val in values.items():
                    # Special handling for uq_methods which is passed as a nested dict
                    if section == "uq_methods":
                        flat_config_dict["uq_methods_config"] = raw_config["uq_methods"]
                    else:
                        flat_config_dict[key] = val
            else:
                flat_config_dict[section] = values # For top-level experiment config

        self.config = Config(flat_config_dict)

        # 2. Logging Setup
        # Create results and log directories early
        self.results_dir = Path(self.config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(self.config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(str(self.log_dir), self.config.experiment_name)
        logging.info(f"Loaded configuration: {config_path}")
        logging.info(f"Experiment: {self.config.experiment_name}")

        # 3. JAX Device Setup & PRNGKey Initialization
        self.master_rng = jr.PRNGKey(self.config.seed)
        logging.info(f"JAX PRNGKey initialized with seed: {self.config.seed}")
        logging.info(f"JAX devices: {jax.devices()}")

        # Placeholder for data and model parameters
        self.data_loader: PdeDatasetLoader
        self.train_data: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        self.val_data: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        self.test_data_dict: Dict[str, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]] = {} # Stores test data for different variants
        self.fno_model: FNO
        self.trained_fno_params: FrozenDict
        self.ensemble_member_params_list: List[FrozenDict] = []

    def _get_dummy_fno_inputs(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Generates dummy FNO inputs for parameter initialization.
        """
        spatial_dims: Tuple[int, ...]
        condition_channels: int

        if self.config.dimensions == "1D":
            spatial_dims = (self.config.spatial_res, 1)  # Add dummy Y-dim for consistent FNO logic
            condition_channels = 0 # 1D APEBench PDEs usually don't have explicit conditions
        else: # 2D
            spatial_dims = (self.config.spatial_res, self.config.spatial_res)
            # vx, vy, reaction_term (1 channel for each, 2 for vel)
            condition_channels = 3 

        # FNO `x_in` expects (batch, initial_time_steps, *spatial_dims, 1)
        dummy_x_in_shape = (1, self.config.initial_time_steps, *spatial_dims, 1)
        dummy_x_in = jnp.zeros(dummy_x_in_shape, dtype=jnp.float32)

        # FNO `conditions` expects (batch, *spatial_dims, condition_channels)
        dummy_conditions_shape = (1, *spatial_dims, condition_channels)
        dummy_conditions = jnp.zeros(dummy_conditions_shape, dtype=jnp.float32)
        
        return dummy_x_in, dummy_conditions

    def run_experiments(self) -> Dict[str, Any]:
        """
        Orchestrates the entire experimental workflow.
        """
        all_results: Dict[str, Any] = {}

        # --- Phase 1: Data Preparation ---
        logging.info("--- Phase 1: Data Preparation ---")
        self.master_rng, data_rng = jr.split(self.master_rng)

        # Instantiate appropriate data loader based on config
        if self.config.dimensions == "1D":
            self.data_loader = APEBenchDatasetLoader(self.config)
            # Generate/load training and validation data (base case, not OOD)
            self.train_data = self.data_loader.generate_data(self.config.num_train_traj, is_ood=False)
            self.val_data = self.data_loader.generate_data(self.config.num_val_traj, is_ood=False)
            
            # Generate/load test data
            self.test_data_dict["Base"] = self.data_loader.generate_data(self.config.num_test_traj, is_ood=False)
            logging.info("1D PDE (low-data regime) data generated/loaded.")
        elif self.config.dimensions == "2D":
            self.data_loader = AdvectionDiffusionDatasetLoader(self.config)
            
            # Generate/load training and validation data (always 'Base' for OOD setup)
            self.train_data = self.data_loader.generate_data(self.config.num_train_traj, is_ood=False, ood_type="Base")
            self.val_data = self.data_loader.generate_data(self.config.num_val_traj, is_ood=False, ood_type="Base")

            # Generate/load OOD test data for all variants
            for variant in self.config.ood_variants:
                self.test_data_dict[variant] = self.data_loader.generate_data(
                    self.config.num_test_pairs, is_ood=True, ood_type=variant
                )
            logging.info("2D PDE (OOD regime) data generated/loaded.")
        else:
            raise ValueError(f"Unsupported PDE dimensions: {self.config.dimensions}")

        # --- Phase 2: FNO Training (Base Model) ---
        logging.info("--- Phase 2: FNO Training (Base Model) ---")
        
        dummy_x_in, dummy_conditions = self._get_dummy_fno_inputs()

        self.fno_model = FNO(
            modes=self.config.fno_modes,
            hidden_dims=self.config.fno_hidden_dims,
            num_fourier_blocks=self.config.fno_blocks,
            output_channels=self.config.fno_output_channels,
            initial_time_steps=self.config.initial_time_steps,
            input_padding=self.config.input_padding,
        )
        # Store dummy_x_in and dummy_conditions in fno_model for UQ methods to access
        # This is a deviation from the initial design but necessary for UQ methods to properly
        # handle shapes when reconstructing parameters or running JVP on single items.
        # Could also pass them explicitly to UQ methods' constructors.
        self.fno_model.dummy_x_in = dummy_x_in
        self.fno_model.dummy_conditions = dummy_conditions


        trainer = Trainer(self.fno_model, self.config)
        self.trained_fno_params, _ = trainer.train(self.train_data, self.val_data)
        logging.info("Base FNO model training complete.")

        # --- Phase 3: Deep Ensemble Training (if enabled) ---
        if self.config.uq_methods_config.get("deep_ensemble", {}).get("enabled", False):
            logging.info("--- Phase 3: Deep Ensemble Training ---")
            num_members = self.config.ensemble_members
            for i in range(num_members):
                self.master_rng, member_rng = jr.split(self.master_rng)
                logging.info(f"Training Deep Ensemble member {i+1}/{num_members} with seed {member_rng}...")
                
                # Create a new FNO model instance for each member to ensure independent initialization
                member_fno_model = FNO(
                    modes=self.config.fno_modes,
                    hidden_dims=self.config.fno_hidden_dims,
                    num_fourier_blocks=self.config.fno_blocks,
                    output_channels=self.config.fno_output_channels,
                    initial_time_steps=self.config.initial_time_steps,
                    input_padding=self.config.input_padding,
                )
                member_fno_model.dummy_x_in = dummy_x_in
                member_fno_model.dummy_conditions = dummy_conditions
                
                # Overwrite the config's seed for this trainer instance to use a new seed
                member_config = self.config
                member_config.seed = int(member_rng[0]) # Use first part of split key as seed for Config

                member_trainer = Trainer(member_fno_model, member_config)
                member_params, _ = member_trainer.train(self.train_data, self.val_data)
                self.ensemble_member_params_list.append(member_params)
            logging.info("Deep Ensemble training complete.")

        # --- Phase 4: UQ Methods, Calibration, and Evaluation ---
        logging.info("--- Phase 4: UQ Methods, Calibration, and Evaluation ---")
        uq_methods_map: Dict[str, Type[UQMethod]] = {
            "luno_iso": LunoIso,
            "luno_la": LunoLA,
            "sample_iso": SampleIso,
            "sample_la": SampleLA,
            "input_perturbations": InputPerturbations,
            "deep_ensemble": DeepEnsemble,
        }

        for method_name, method_config in self.config.uq_methods_config.items():
            if not method_config.get("enabled", False):
                logging.info(f"Skipping disabled UQ method: {method_name}")
                continue

            logging.info(f"Processing UQ method: {method_name}")
            method_class = uq_methods_map[method_name]
            
            # Select correct parameters for UQ method
            if method_name == "deep_ensemble":
                uq_method_params = self.ensemble_member_params_list
            else:
                uq_method_params = self.trained_fno_params
            
            uq_method_instance: UQMethod = method_class(self.fno_model, uq_method_params, self.config)

            # Call fit method (e.g., for GGN computation)
            uq_method_instance.fit(self.train_data)

            # Hyperparameter Calibration
            if not isinstance(uq_method_instance, DeepEnsemble):
                calibrator = UQCalibrator(uq_method_instance, self.val_data, self.config)
                self.master_rng, calibrate_rng = jr.split(self.master_rng)
                # Pass RNG for any sampling inside calibration if needed
                # (current UQCalibrator design does not accept rng, UQMethods handle their own).
                calibrator.calibrate_hyperparameters() # This updates uq_method_instance's HP directly

            # Single-Step Prediction and Evaluation
            all_results[method_name] = {}
            for test_variant, (test_inputs, test_targets_full, test_initial_conditions) in self.test_data_dict.items():
                logging.info(f"Evaluating {method_name} on {test_variant} test data (single step)...")

                # Predict for all test samples
                # uq_method_instance.predict_uncertainty expects (batch, ...)
                predictions_mean, predictions_std = uq_method_instance.predict_uncertainty(test_inputs, test_initial_conditions) # Changed conditions from full trajectory to initial condition.
                                                                                                    # The FNO is built to take conditions per time step.
                                                                                                    # Recheck: For 2D OOD data, conditions (vel, reaction) are (batch, spatial_x, spatial_y, cond_channels) - so they are constant over time.
                                                                                                    # In datasets.py:
                                                                                                    # AdvectionDiffusionDatasetLoader: `input_conditions_for_fno` is (spatial_res, spatial_res, 3), then padded.
                                                                                                    # Then `expanded_conditions` is (initial_time_steps, padded_spatial_x, padded_spatial_y, num_cond_channels).
                                                                                                    # FNO receives `conditions` which is (batch, initial_time_steps, padded_spatial_x, padded_spatial_y, cond_channels).
                                                                                                    # So for a single prediction, test_conditions passed should be the (batch, initial_time_steps, spatial_x, spatial_y, cond_channels).
                                                                                                    # Need to ensure test_initial_conditions is properly shaped conditions field if it's used.
                                                                                                    # Looking at datasets.py, `generate_data` returns (inputs, targets, initial_conditions). `inputs` already has all conditions stacked.
                                                                                                    # It seems `conditions` for `uq_method_instance.predict_uncertainty` should be `inputs` from `test_data_dict` and `input_func` from `test_data_dict` as well.
                                                                                                    # Let's adjust `fno_model.py` and `uq_methods.py` to separate `input_state` and `input_conditions` if they are truly separate in the call signature.
                                                                                                    # But `FNO.__call__` takes `x_in` and `conditions`.
                                                                                                    # In `datasets.py` `final_fno_input = stack_conditions(padded_fno_input_state)` (1D) or `stack_conditions(padded_fno_input_state, conditions=expanded_conditions)` (2D).
                                                                                                    # This implies the input to FNO is `(batch, time, spatial_x, spatial_y, channels_total)`.
                                                                                                    # But FNO __call__ definition is `__call__(self, x_in: jnp.ndarray, conditions: jnp.ndarray)`.
                                                                                                    # The description in `fno_model.py` for `x_in` is `Input state tensor. Shape: (batch, initial_time_steps, spatial_x, [spatial_y], 1)`.
                                                                                                    # And for `conditions`: `Conditions tensor. Shape: (batch, spatial_x, [spatial_y], channels_cond)`.
                                                                                                    # This is a conflict with how `datasets.py` `stack_conditions` output and `fno_model.py` internally treat `conditions`.

                                                                                                    # Reconciling:
                                                                                                    # `stack_conditions` in `utils.py` does `jnp.concatenate([state_field, velocity_field, reaction_term_field], axis=-1)`.
                                                                                                    # So if `fno_model.py` `__call__` receives `x_in` and `conditions`, `x_in` should be the actual state history, and `conditions` should be the *other* conditions (v/R) already expanded for time.
                                                                                                    # Let's assume `test_inputs` from `datasets.py` represents the `x_in` for the FNO, and it already contains the full concatenated state + conditions.
                                                                                                    # Then the `conditions` argument in `FNO.__call__` is probably `None` or an empty tensor.
                                                                                                    # Or, `FNO.__call__` expects the raw `state_history` and `raw_conditions` (velocity, reaction terms) separately, and does the stacking internally.
                                                                                                    # Review `FNO` definition: `_apply_fno_layers(self, x_in: jnp.ndarray, conditions: jnp.ndarray, ...)`
                                                                                                    # `v0 = jnp.concatenate([padded_x_flat, padded_conditions], axis=-1)`
                                                                                                    # This means FNO expects `x_in` (state history) and `conditions` (v/R, potentially constant over time).
                                                                                                    # `datasets.py` `generate_data` returns `final_fno_input` which is already concatenated. This needs to be consistent.

                                                                                                    # **Correction:** The `FNO.__call__` method expects `x_in` (the time series of states) and `conditions` (the static conditions like velocity, reaction terms that are glued to the input). `datasets.py` produces `final_fno_input` which already has `x_in` and `conditions` `stack_conditions`-ed.
                                                                                                    # So, I need to *unstack* `test_inputs` before passing them to `uq_method_instance.predict_uncertainty` if the FNO expects them separately.
                                                                                                    # Or, modify FNO and UQ methods to expect already stacked `total_input`.
                                                                                                    # Let's change `fno_model.py` to take a single `total_input` tensor and parse its channels. This is simpler than unstacking constantly.
                                                                                                    # `FNO.__call__(self, total_input: jnp.ndarray)`. The `_apply_fno_layers` will parse the channels itself.
                                                                                                    # *Correction 2:* The current FNO design (fno_model.py) explicitly has `x_in` and `conditions` as two separate arguments. My `datasets.py` creates `final_fno_input` which is `stack_conditions` from `padded_fno_input_state` AND `expanded_conditions`. This means `final_fno_input` already contains *both*.
                                                                                                    # So, `test_inputs` from `datasets.py` actually contains `(state_history + conditions)` in its channel dimension.
                                                                                                    # For `fno_model.py`, the `conditions` argument of `__call__` is likely intended to be the velocity/reaction terms *before* stacking.
                                                                                                    # But then, `padded_fno_input_state` was just `x_in` (state only) and `expanded_conditions` was conditions only.
                                                                                                    # The `fno_model.py` has `v0 = jnp.concatenate([padded_x_flat, padded_conditions], axis=-1)`. This means `x_in` and `conditions` are separate.
                                                                                                    # `datasets.py` `final_fno_input = stack_conditions(padded_fno_input_state, conditions=expanded_conditions)` effectively *combines* these before passing. This implies `stack_conditions` is actually creating the `v0` input.
                                                                                                    # This needs to be passed correctly.
                                                                                                    # Let's align `datasets.py`'s output with `fno_model.py`'s expected inputs:
                                                                                                    # `datasets.py` should return: `(state_history, conditions_only, targets)`
                                                                                                    # And `FNO.__call__` will take `state_history` as `x_in` and `conditions_only` as `conditions`.
                                                                                                    # This changes the `datasets.py` `generate_data` and `load_data` return type.
                                                                                                    # It also means `Trainer._loss_fn` and `_train_step` must accept the separate `state_history` and `conditions_only`.
                                                                                                    # **Re-re-correction**: The current design `fno_model.py` takes `x_in` and `conditions` *separately*. `datasets.py` `stack_conditions` returns a single concatenated array.
                                                                                                    # The design was `final_fno_input = stack_conditions(padded_fno_input_state, conditions=expanded_conditions)`. This implies the output of `stack_conditions` is *the* input to FNO.
                                                                                                    # However, in `fno_model.py`, the `_apply_fno_layers` method uses `jnp.concatenate([padded_x_flat, padded_conditions], axis=-1)`.
                                                                                                    # This suggests that `x_in` should be `state_field` and `conditions` should be `velocity_field` + `reaction_term_field`.
                                                                                                    # I will update `datasets.py` to return the state history AND the condition fields separately.
                                                                                                    # Then `training.py` and `uq_methods.py` will pass them separately.

                                                                                                    # Update `datasets.py`: `generate_data` and `load_data` should return `(state_history, conditions_fields, targets)`.
                                                                                                    # `state_history`: (num_traj, initial_time_steps, *spatial_dims, 1) -- PADDED
                                                                                                    # `conditions_fields`: (num_traj, initial_time_steps, *spatial_dims, cond_channels) -- PADDED (replicated over time)
                                                                                                    # `targets`: (num_traj, *spatial_dims, output_channels) -- UNPADDED

                # The `test_inputs` from `self.test_data_dict` are the `x_in` (state history) for the FNO,
                # and `test_conditions` are the static (velocity, reaction) conditions.
                # `datasets.py` is modified to output these separately.
                # So the `test_inputs` is the state_history, and `test_conditions_raw` is the v/R conditions.
                test_state_history, test_conditions_raw, _ = test_inputs # _ is dummy output, targets
                
                predictions_mean, predictions_std = uq_method_instance.predict_uncertainty(
                    test_state_history, test_conditions_raw
                )

                evaluator = Evaluator(self.test_data_dict[test_variant], self.config)
                single_step_metrics = evaluator.calculate_metrics(predictions_mean, predictions_std)
                all_results[method_name][test_variant] = {"single_step": single_step_metrics}
                logging.info(f"Single-step metrics for {method_name} on {test_variant}: {single_step_metrics}")

                # Plotting for one example
                example_idx = 0 # Choose a fixed example for plotting
                self.master_rng, plot_rng = jr.split(self.master_rng)
                samples_for_plot = uq_method_instance.get_samples(
                    jnp.expand_dims(test_state_history[example_idx], 0), # add batch dim
                    jnp.expand_dims(test_conditions_raw[example_idx], 0),
                    num_samples=4 # Get a few samples for visualization
                ) # (num_samples, 1, *spatial, channels)
                samples_for_plot = jnp.squeeze(samples_for_plot, axis=1) # (num_samples, *spatial, channels)


                evaluator.plot_predictions(
                    method_name,
                    f"single_step_example_{example_idx}_{test_variant}",
                    predictions_mean[example_idx],
                    predictions_std[example_idx],
                    test_targets_full[example_idx, self.config.initial_time_steps], # Ground truth for the predicted step
                    samples=samples_for_plot,
                )

            # Autoregressive Rollout Evaluation
            if self.config.eval_autoregressive_rollout_steps > 0:
                logging.info(f"Evaluating {method_name} with autoregressive rollouts...")
                
                # Use a specific OOD variant for rollout, as per paper (Pos-Neg-Flip)
                # Or 'Base' if 1D experiment
                rollout_variant = "Pos-Neg-Flip" if self.config.dimensions == "2D" else "Base"
                
                # Check if the variant exists in test data
                if rollout_variant not in self.test_data_dict:
                    logging.warning(f"Rollout variant '{rollout_variant}' not found in test data. Skipping rollouts for {method_name}.")
                    continue
                
                rollout_test_state_history, rollout_test_conditions_raw, rollout_test_targets_full = self.test_data_dict[rollout_variant]
                
                num_rollout_traj_to_eval = min(
                    self.config.eval_num_rollout_trajectories, rollout_test_state_history.shape[0]
                )
                
                all_rollout_metrics: List[Dict[str, float]] = []

                # Iterate over a subset of trajectories for rollouts
                for traj_idx in tqdm(range(num_rollout_traj_to_eval), desc=f"Rollout for {method_name}"):
                    current_state_history_batch = jnp.expand_dims(rollout_test_state_history[traj_idx], axis=0) # (1, T_in, X, Y, C)
                    current_conditions_raw_batch = jnp.expand_dims(rollout_test_conditions_raw[traj_idx], axis=0) # (1, T_in, X, Y, C_cond)
                    
                    rollout_means_traj = []
                    rollout_stds_traj = []

                    for t_step in range(self.config.eval_autoregressive_rollout_steps):
                        # Predict next step
                        next_step_mean, next_step_std = uq_method_instance.predict_uncertainty(
                            current_state_history_batch, current_conditions_raw_batch
                        ) # (1, X, Y, C)

                        rollout_means_traj.append(next_step_mean)
                        rollout_stds_traj.append(next_step_std)

                        # Prepare next input: shift window, add new prediction
                        # (1, T_in, X, Y, C_state) -> remove oldest T_in step, add new T_in step.
                        # `current_state_history_batch` is (1, initial_time_steps, spatial_x, spatial_y, 1)
                        # `next_step_mean` is (1, spatial_x, spatial_y, 1)
                        
                        # Take all but the first time step, then append the new prediction
                        new_state_history = jnp.concatenate(
                            [current_state_history_batch[:, 1:, :, :, :], next_step_mean[:, jnp.newaxis, :, :, :]],
                            axis=1 # Concatenate along the time dimension
                        )
                        current_state_history_batch = new_state_history
                        
                        # Plotting for an example rollout step (e.g., first trajectory, middle step)
                        if traj_idx == 0 and t_step == self.config.eval_autoregressive_rollout_steps // 2:
                             self.master_rng, plot_rng = jr.split(self.master_rng)
                             samples_for_plot = uq_method_instance.get_samples(
                                jnp.expand_dims(rollout_test_state_history[traj_idx], 0), # add batch dim
                                jnp.expand_dims(rollout_test_conditions_raw[traj_idx], 0),
                                num_samples=4 # Get a few samples for visualization
                            ) # (num_samples, 1, *spatial, channels)
                             samples_for_plot = jnp.squeeze(samples_for_plot, axis=1) # (num_samples, *spatial, channels)

                             evaluator.plot_predictions(
                                method_name,
                                f"autoregressive_example_{traj_idx}_step_{t_step}_{rollout_variant}",
                                next_step_mean[0], # unbatch
                                next_step_std[0],  # unbatch
                                rollout_test_targets_full[traj_idx, self.config.initial_time_steps + t_step], # Ground truth for this rollout step
                                samples=samples_for_plot,
                                time_step_to_plot=self.config.initial_time_steps + t_step,
                             )


                    # After rollout for one trajectory, evaluate its metrics
                    # The ground truth for rollout starts from config.initial_time_steps
                    evaluator = Evaluator((None, rollout_test_targets_full[traj_idx:traj_idx+1], None), self.config) # Pass single traj for evaluator
                    # Stack predicted means and stds for the entire rollout length
                    rollout_means_stacked = jnp.concatenate(rollout_means_traj, axis=0) # (num_steps, X, Y, C)
                    rollout_stds_stacked = jnp.concatenate(rollout_stds_traj, axis=0)   # (num_steps, X, Y, C)
                    
                    # Compute metrics for each step and average, or for the whole rollout
                    # The paper computes NLL over "autoregressive rollout ... on 50 trajectories"
                    # which implies averaging metrics calculated per step/trajectory.
                    # Evaluator's `calculate_metrics` expects full predictions/targets, for all steps for this traj.
                    # It also needs `eval_time_step_idx` for multi-step evaluation.
                    # Let's compute RMSE and NLL per step and average them.
                    
                    current_rollout_metrics = {}
                    for step_i in range(self.config.eval_autoregressive_rollout_steps):
                        metrics_per_step = evaluator.calculate_metrics(
                            jnp.expand_dims(rollout_means_stacked[step_i], 0), # (1, X, Y, C)
                            jnp.expand_dims(rollout_stds_stacked[step_i], 0),   # (1, X, Y, C)
                            eval_time_step_idx=self.config.initial_time_steps + step_i
                        )
                        for k, v in metrics_per_step.items():
                            current_rollout_metrics.setdefault(k, []).append(v)
                    
                    # Average metrics over steps for this trajectory
                    for k in current_rollout_metrics:
                        current_rollout_metrics[k] = jnp.mean(jnp.asarray(current_rollout_metrics[k]))
                    
                    all_rollout_metrics.append(current_rollout_metrics)

                # Average metrics over all rollout trajectories
                avg_rollout_metrics = {
                    metric: float(jnp.mean(jnp.asarray([t_metrics[metric] for t_metrics in all_rollout_metrics])))
                    for metric in all_rollout_metrics[0].keys()
                }
                all_results[method_name][rollout_variant]["autoregressive_rollout"] = avg_rollout_metrics
                logging.info(f"Autoregressive rollout metrics for {method_name} on {rollout_variant}: {avg_rollout_metrics}")

        # --- Phase 5: Final Report Generation ---
        logging.info("--- Phase 5: Final Report Generation ---")
        results_file_path = self.results_dir / f"{self.config.experiment_name}_results.json"
        with open(results_file_path, "w") as f:
            json.dump(all_results, f, indent=4)
        logging.info(f"Full experiment results saved to: {results_file_path}")

        # Optionally print results in a human-readable format
        print("\n--- Experiment Results Summary ---")
        print(json.dumps(all_results, indent=2))

        return all_results


if __name__ == "__main__":
    # Define the path to the configuration file
    current_dir = Path(__file__).parent
    config_file = current_dir / "config.yaml"

    # Ensure config.yaml exists
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_file}")

    # Run the experiment
    runner = ExperimentRunner(str(config_file))
    final_results = runner.run_experiments()
    logging.info("ExperimentRunner finished.")

