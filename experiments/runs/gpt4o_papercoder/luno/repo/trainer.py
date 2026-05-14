## trainer.py

import time
from typing import Dict, Any
import jax
import jax.numpy as jnp
import optax

class Trainer:
    """
    Trainer class for optimizing the Fourier Neural Operator (FNO) model.
    Handles training with the AdamW optimizer, cosine decay learning rate,
    and mean squared error loss computation.
    """
    
    def __init__(self, model: Any, optimizer: Any, config: Dict[str, Any]):
        """
        Initialize Trainer with the model, optimizer, and configuration.

        Args:
            model (Any): An instance of the FourierNeuralOperator model.
            optimizer (Any): Configured optimizer instance using optax.
            config (dict): Configuration dictionary loaded from config.yaml.
        """
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.batch_size = config.get("training", {}).get("batch_size", 1)
        self.epochs = config.get("training", {}).get("epochs_low_data", 100)
        self.learning_rate = config.get("training", {}).get("learning_rate", 1e-3)

        # Setup learning rate scheduler
        warmup_steps = config.get("scheduler", {}).get("warmup_steps", 10)
        self.lr_scheduler = optax.cosine_decay_schedule(
            init_value=self.learning_rate,
            decay_steps=self.epochs
        )

        # Initialize optimizer state
        self.tx = optax.chain(
            optax.additive_weight_decay(config.get("optimizer", {}).get("weight_decay", 1e-4)),
            optax.adamw(self.learning_rate)
        )
        self.opt_state = self.tx.init(self.model.trainable_weights())

    @staticmethod
    def mean_squared_error(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
        """
        Compute Mean Squared Error (MSE) loss between predictions and targets.

        Args:
            predictions (jnp.ndarray): Predicted values from the model.
            targets (jnp.ndarray): Ground truth values.

        Returns:
            jnp.ndarray: Computed MSE loss.
        """
        return jnp.mean((predictions - targets) ** 2)

    def train(self, data: Dict[str, Dict[str, jnp.ndarray]], epochs: int = None) -> Dict[str, Any]:
        """
        Train the FNO model on the provided data.

        Args:
            data (dict): Dictionary containing training data with 'inputs' and 'outputs'.
            epochs (int, optional): Number of epochs to train. Defaults to config value.

        Returns:
            dict: Dictionary containing training summary:
                  - losses: List of epoch-wise losses.
                  - weights: Final trained weights.
                  - runtime: Total time taken for training.
        """
        if epochs is None:
            epochs = self.epochs

        train_inputs, train_outputs = data["train"]["inputs"], data["train"]["outputs"]

        # Training metrics
        losses = []
        start_time = time.time()

        # Define gradient and update function
        def loss_fn(weights, inputs, targets):
            predictions = self.model.apply({"params": weights}, inputs)
            return self.mean_squared_error(predictions, targets)

        grad_fn = jax.jit(jax.value_and_grad(loss_fn))

        # Training loop
        for epoch in range(epochs):
            epoch_start_time = time.time()

            batch_losses = []

            for batch_start in range(0, len(train_inputs), self.batch_size):
                # Extract batch
                batch_inputs = train_inputs[batch_start:batch_start + self.batch_size]
                batch_targets = train_outputs[batch_start:batch_start + self.batch_size]

                # Compute gradients and loss
                loss, grads = grad_fn(self.model.trainable_weights(), batch_inputs, batch_targets)

                # Update weights
                updates, self.opt_state = self.tx.update(grads, self.opt_state, self.model.trainable_weights())
                self.model.variables["params"] = optax.apply_updates(self.model.trainable_weights(), updates)

                batch_losses.append(jax.device_get(loss))

            # Compute epoch-wise loss
            epoch_loss = jnp.mean(jnp.array(batch_losses))
            losses.append(epoch_loss)

            # Print progress
            epoch_runtime = time.time() - epoch_start_time
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.6f}, Time: {epoch_runtime:.2f}s")

        # Final training time
        total_runtime = time.time() - start_time

        return {
            "losses": losses,
            "weights": self.model.trainable_weights(),
            "runtime": total_runtime
        }
