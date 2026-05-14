"""
calibration.py
Contains the UQCalibrator class for hyperparameter tuning of Uncertainty Quantification methods.
"""

import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple, Optional, Type, Union
from tqdm import tqdm
from absl import logging

# Local application imports
from config import Config
from uq_methods import (
    UQMethod,
    LunoIso,
    LunoLA,
    SampleIso,
    SampleLA,
    InputPerturbations,
    DeepEnsemble,
)
from utils import compute_nll


class UQCalibrator:
    """
    Manages the calibration of hyperparameters for various Uncertainty Quantification methods.
    It performs a grid search to find the optimal hyperparameter value that minimizes
    the Negative Log-Likelihood (NLL) on a validation dataset.
    """

    def __init__(self, uq_method: UQMethod, val_data: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], config: Config):
        """
        Initializes the UQCalibrator.

        Args:
            uq_method: An instance of a concrete UQMethod subclass. This instance's
                       hyperparameter will be calibrated.
            val_data: A tuple (val_inputs, val_conditions, val_targets) representing
                      the validation dataset.
            config: The global Config object, providing access to calibration settings
                    and initial hyperparameter values.
        """
        self.uq_method = uq_method
        self.val_inputs, self.val_conditions, self.val_targets = val_data
        self.config = config
        self.calibration_grid_size = self.config.calibration_grid_size
        logging.info(f"UQCalibrator initialized for method: {self.uq_method.__class__.__name__}")

    def calibrate_hyperparameters(self) -> Optional[float]:
        """
        Executes the grid search to find the optimal hyperparameter for the
        UQ method instance.

        The calibration process involves:
        1. Identifying the specific hyperparameter to tune for the given UQ method.
        2. Defining a logarithmically spaced grid of values around the initial guess.
        3. Iterating through the grid, setting the hyperparameter, predicting
           uncertainty on the validation set, and computing NLL.
        4. Selecting the hyperparameter value that yields the minimum NLL.
        5. Applying the best hyperparameter value to the UQ method instance.

        Returns:
            The best hyperparameter value found, or None if calibration is skipped
            (e.g., for Deep Ensembles).
        """
        if isinstance(self.uq_method, DeepEnsemble):
            logging.info("DeepEnsemble does not require calibration. Skipping.")
            return None

        hyperparam_name: str
        initial_value_cfg_key: str
        param_section: Dict
        
        # Determine the hyperparameter name and its initial value based on UQ method type
        if isinstance(self.uq_method, LunoIso):
            hyperparam_name = "sigma_squared"
            param_section = self.config.uq_methods_config["luno_iso"]
            initial_value_cfg_key = "sigma_squared_init"
        elif isinstance(self.uq_method, LunoLA):
            hyperparam_name = "prior_sigma" # Calibrating sigma, not sigma_squared
            param_section = self.config.uq_methods_config["luno_la"]
            initial_value_cfg_key = "prior_sigma_init"
        elif isinstance(self.uq_method, SampleIso):
            hyperparam_name = "sigma_squared"
            param_section = self.config.uq_methods_config["sample_iso"]
            initial_value_cfg_key = "sigma_squared_init"
        elif isinstance(self.uq_method, SampleLA):
            hyperparam_name = "prior_sigma" # Calibrating sigma, not sigma_squared
            param_section = self.config.uq_methods_config["sample_la"]
            initial_value_cfg_key = "prior_sigma_init"
        elif isinstance(self.uq_method, InputPerturbations):
            hyperparam_name = "noise_sigma"
            param_section = self.config.uq_methods_config["input_perturbations"]
            initial_value_cfg_key = "noise_sigma_init"
        else:
            logging.warning(f"UQ method {self.uq_method.__class__.__name__} is not recognized for calibration. Skipping.")
            return None

        initial_value = param_section.get(initial_value_cfg_key)
        if initial_value is None:
            logging.error(f"Initial value for '{initial_value_cfg_key}' not found in config for {self.uq_method.__class__.__name__}. Skipping calibration.")
            return None

        logging.info(f"Calibrating '{hyperparam_name}' for {self.uq_method.__class__.__name__} (initial: {initial_value:.2e})...")

        # Define grid search range (logarithmically spaced)
        # Using 4 orders of magnitude (2 below, 2 above) centered around initial_value
        log10_initial = np.log10(initial_value)
        min_log_val = log10_initial - 2.0
        max_log_val = log10_initial + 2.0
        hyperparam_grid = np.logspace(min_log_val, max_log_val, self.calibration_grid_size)

        min_nll = float('inf')
        best_hp_value: Optional[float] = None
        
        nlls: List[float] = []

        # Iterate and evaluate NLL
        for hp_value in tqdm(hyperparam_grid, desc=f"Calibrating {self.uq_method.__class__.__name__}"):
            # Set hyperparameter in the UQ method instance
            self.uq_method.set_hyperparameter(hyperparam_name, hp_value)

            # Predict uncertainty on the validation set
            mean_preds, std_preds = self.uq_method.predict_uncertainty(self.val_inputs, self.val_conditions)

            # Compute NLL
            current_nll = float(compute_nll(self.val_targets, mean_preds, std_preds))
            nlls.append(current_nll)

            # Update best hyperparameter if current NLL is lower
            if current_nll < min_nll:
                min_nll = current_nll
                best_hp_value = hp_value
        
        if best_hp_value is None:
            logging.error("Calibration failed: No best hyperparameter value found.")
            return None

        # Apply the best hyperparameter value to the UQ method instance for subsequent use
        self.uq_method.set_hyperparameter(hyperparam_name, best_hp_value)
        logging.info(f"Calibration complete for {self.uq_method.__class__.__name__}:")
        logging.info(f"Best '{hyperparam_name}': {best_hp_value:.2e} (Min NLL: {min_nll:.4f})")

        return best_hp_value

