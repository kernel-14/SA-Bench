"""
training.py
Contains the Trainer class responsible for FNO model training.
This includes model initialization, optimizer setup, data iteration,
loss calculation, gradient computation, and parameter updates.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.core import FrozenDict
from typing import Any, Tuple, Optional, Callable, Dict, List
from tqdm import tqdm
from absl import logging

# Local application imports
from config import Config
from fno_model import FNO
from utils import compute_rmse, setup_logging, initialize_fno_params


# Type aliases for clarity
OptState = Any
Params = FrozenDict
PRNGKey = jax.random.PRNGKey


class Trainer:
    """
    Manages the training process for an FNO model.

    This class handles:
    - Initializing model parameters and optimizer state.
    - Running the training loop over multiple epochs.
    - Computing loss and gradients.
    - Updating model parameters using an optimizer and learning rate schedule.
    - Evaluating the model on a validation set.
    """

    def __init__(
        self,
        fno_module: FNO,
        config: Config,
        optimizer_state: Optional[OptState] = None,
    ):
        """
        Initializes the Trainer with an FNO model, configuration, and optional optimizer state.

        Args:
            fno_module: An instance of the FNO model (flax.linen.Module).
            config: An instance of the Config class containing all experiment parameters.
            optimizer_state: Optional. The initial state of the Optax optimizer for resuming training.
                             If None, a new optimizer state will be initialized.
        """
        self.config = config
        self.fno_module = fno_module
        self.rng_key = jax.random.PRNGKey(self.config.seed)

        # 1. Prepare dummy inputs for FNO parameter initialization
        self.rng_key, init_key = jax.random.split(self.rng_key)

        spatial_dims: Tuple[int, ...]
        if self.config.dimensions == "1D":
            spatial_dims = (self.config.spatial_res, 1)  # Add dummy Y-dim for consistent FNO logic
            condition_channels = 0 # 1D APEBench PDEs usually don't have explicit conditions
        else: # 2D
            spatial_dims = (self.config.spatial_res, self.config.spatial_res)
            condition_channels = 3 # vx, vy, reaction_term for 2D Advection-Diffusion-Reaction

        # FNO `x_in` expects (batch, initial_time_steps, *spatial_dims, 1)
        dummy_x_in_shape = (1, self.config.initial_time_steps, *spatial_dims, 1)
        dummy_x_in = jnp.zeros(dummy_x_in_shape, dtype=jnp.float32)

        # FNO `conditions` expects (batch, *spatial_dims, condition_channels)
        dummy_conditions_shape = (1, *spatial_dims, condition_channels)
        dummy_conditions = jnp.zeros(dummy_conditions_shape, dtype=jnp.float32)

        # 2. Initialize FNO Parameters
        self.params: Params = initialize_fno_params(
            init_key, self.fno_module, dummy_x_in, dummy_conditions
        )
        logging.info("FNO model parameters initialized.")

        # 3. Optimizer Initialization (Learning rate schedule will be finalized in `train` for precise decay_steps)
        # Use a placeholder learning rate scheduler for init, or re-initialize in `train` for precision.
        # As per the design, self.optimizer is initialized here.
        # The weight_decay value is not specified in paper for training, using a common default.
        weight_decay = 1.0e-4 if self.config.training_optimizer == "AdamW" else 0.0
        self.optimizer = optax.adamw(
            learning_rate=self.config.lr_schedule["peak_value"], # Placeholder LR
            weight_decay=weight_decay
        )

        # 4. Initialize optimizer state
        if optimizer_state:
            self.opt_state: OptState = optimizer_state
            logging.info("Optimizer state loaded for resuming training.")
        else:
            self.opt_state = self.optimizer.init(self.params)
            logging.info("New optimizer state initialized.")

        # JIT compile the training step for performance
        self._jit_train_step = jax.jit(self._train_step)
        logging.info("Training step JIT-compiled.")

    def _loss_fn(
        self,
        params: Params,
        batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        rng_key: PRNGKey,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Computes the mean squared error loss for a given batch.

        Args:
            params: The current FNO model parameters.
            batch: A tuple containing (state_history, conditions_field, targets).
                   - state_history: (batch_size, initial_time_steps, *spatial_dims, 1)
                   - conditions_field: (batch_size, *spatial_dims, condition_channels)
                   - targets: (batch_size, *spatial_dims, output_channels)
            rng_key: A JAX PRNGKey for any stochastic layers within the model.

        Returns:
            A tuple (loss, predictions):
            - loss: The scalar mean squared error loss.
            - predictions: The model's predictions for the given batch.
        """
        state_history, conditions_field, targets = batch

        # FNO forward pass
        predictions: jnp.ndarray = self.fno_module.apply(
            {"params": params}, state_history, conditions_field, rngs={"params": rng_key}
        )

        # Mean Squared Error (MSE) loss
        loss = jnp.mean(jnp.square(predictions - targets))

        return loss, predictions

    def _apply_gradient(
        self, params: Params, opt_state: OptState, grads: Params
    ) -> Tuple[Params, OptState]:
        """
        Applies the computed gradients to update model parameters and optimizer state.

        Args:
            params: The current FNO model parameters.
            opt_state: The current Optax optimizer state.
            grads: The computed gradients.

        Returns:
            A tuple (new_params, new_opt_state) with updated model parameters and optimizer state.
        """
        # Compute parameter updates from gradients and current optimizer state
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)

        # Apply the updates to the parameters
        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_state

    def _train_step(
        self,
        params: Params,
        opt_state: OptState,
        batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        rng_key: PRNGKey,
    ) -> Tuple[Params, OptState, jnp.ndarray, jnp.ndarray]:
        """
        Performs a single training step: calculates loss, gradients, and updates parameters.
        This function is designed to be JIT-compiled.

        Args:
            params: Current model parameters.
            opt_state: Current optimizer state.
            batch: A tuple (state_history, conditions_field, targets) for the current batch.
            rng_key: PRNGKey for stochastic operations in _loss_fn.

        Returns:
            A tuple (new_params, new_opt_state, loss_value, predictions)
            - new_params: Updated model parameters.
            - new_opt_state: Updated optimizer state.
            - loss_value: Scalar loss for the current batch.
            - predictions: Model predictions for the current batch.
        """
        # Calculate loss and gradients
        (loss_value, predictions), grads = jax.value_and_grad(self._loss_fn, argnums=0, has_aux=True)(
            params, batch, rng_key
        )

        # Apply gradients to update parameters and optimizer state
        new_params, new_opt_state = self._apply_gradient(params, opt_state, grads)

        return new_params, new_opt_state, loss_value, predictions

    def train(
        self,
        train_data: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        val_data: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> Tuple[Params, OptState]:
        """
        Executes the main training loop for the FNO model.

        Args:
            train_data: A tuple (state_history, conditions_field, targets) for the training set.
                        Each element is a jnp.ndarray where the first dimension is num_trajectories.
            val_data: A tuple (state_history, conditions_field, targets) for the validation set.

        Returns:
            A tuple (trained_params, final_opt_state) containing the final model parameters
            and optimizer state after training.
        """
        # Unpack training and validation data
        train_state_history, train_conditions_field, train_targets = train_data
        val_state_history, val_conditions_field, val_targets = val_data

        num_train_trajectories = train_state_history.shape[0]
        epochs = self.config.fno_epochs
        batch_size = self.config.training_batch_size # Expected to be 1 for iterating trajectories

        logging.info(f"Starting FNO training for {epochs} epochs...")
        logging.info(f"Training data size: {num_train_trajectories} trajectories")
        logging.info(f"Validation data size: {val_state_history.shape[0]} trajectories")

        # Re-initialize LR schedule and optimizer with correct decay_steps
        # The paper specifies 'batch_size' is effectively 1 trajectory per step,
        # so num_steps_per_epoch is num_train_trajectories.
        num_steps_per_epoch = num_train_trajectories // batch_size # If batch_size > 1
        if num_train_trajectories % batch_size != 0:
             num_steps_per_epoch += 1 # Handle remainder
        
        total_decay_steps = epochs * num_steps_per_epoch

        lr_schedule_cfg = self.config.lr_schedule
        self.lr_scheduler = optax.warmup_cosine_decay_schedule(
            init_value=lr_schedule_cfg["init_value"],
            peak_value=lr_schedule_cfg["peak_value"],
            warmup_steps=lr_schedule_cfg["warmup_steps"],
            decay_steps=total_decay_steps,
            end_value=lr_schedule_cfg.get("end_value", lr_schedule_cfg["init_value"]), # Optional, for robustness
        )
        
        weight_decay = 1.0e-4 if self.config.training_optimizer == "AdamW" else 0.0
        self.optimizer = optax.adamw(learning_rate=self.lr_scheduler, weight_decay=weight_decay)
        self.opt_state = self.optimizer.init(self.params) # Re-init opt_state with new scheduler

        logging.info(f"Learning rate schedule configured with total decay steps: {total_decay_steps}")


        # Training loop
        for epoch in tqdm(range(epochs), desc="Training FNO"):
            self.rng_key, shuffle_key = jax.random.split(self.rng_key)
            shuffled_indices = jax.random.permutation(shuffle_key, num_train_trajectories)

            epoch_train_loss = 0.0
            for i in range(0, num_train_trajectories, batch_size):
                batch_indices = shuffled_indices[i : i + batch_size]

                # Extract batch data using advanced indexing
                batch_state_history = train_state_history[batch_indices]
                batch_conditions_field = train_conditions_field[batch_indices]
                batch_targets = train_targets[batch_indices]

                # Split PRNGKey for the current step
                self.rng_key, step_key = jax.random.split(self.rng_key)

                # Perform a single training step (JIT-compiled)
                self.params, self.opt_state, loss_value, _ = self._jit_train_step(
                    self.params,
                    self.opt_state,
                    (batch_state_history, batch_conditions_field, batch_targets),
                    step_key,
                )
                epoch_train_loss += loss_value

            avg_train_loss = epoch_train_loss / num_steps_per_epoch

            # Validation step
            self.rng_key, val_key = jax.random.split(self.rng_key)
            val_predictions = self.fno_module.apply(
                {"params": self.params},
                val_state_history,
                val_conditions_field,
                rngs={"params": val_key},
            )
            val_rmse = compute_rmse(val_targets, val_predictions)

            logging.info(
                f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val RMSE: {val_rmse:.6f}"
            )

        logging.info("FNO training complete.")
        return self.params, self.opt_state

