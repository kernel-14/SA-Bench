"""Training script for Fourier Neural Operators.

Implements the training pipeline from the LUNO paper:
- AdamW optimizer with cosine decay learning rate schedule
- MSE loss
- Supports training for low-data and OOD experiments
- Saves trained model weights for downstream UQ
"""

import os
from typing import Optional, Callable, Tuple, Dict
import jax
import jax.numpy as jnp
import optax
import numpy as np
from flax import nnx
from tqdm import tqdm

from config import ExperimentConfig, TrainingConfig
from model import create_fno


def create_optimizer(config: TrainingConfig, total_steps: int) -> optax.GradientTransformation:
    """Create AdamW optimizer with cosine decay schedule and warmup."""
    schedule_fn = optax.cosine_decay_schedule(
        init_value=config.learning_rate,
        decay_steps=total_steps - config.warmup_steps,
        alpha=0.0,
    )

    # Linear warmup
    if config.warmup_steps > 0:
        warmup_fn = optax.linear_schedule(
            init_value=0.0,
            end_value=config.learning_rate,
            transition_steps=config.warmup_steps,
        )
        # Combine: warmup then cosine decay
        schedule_fn = optax.join_schedules(
            schedules=[warmup_fn, schedule_fn],
            boundaries=[config.warmup_steps],
        )

    optimizer = optax.adamw(
        learning_rate=schedule_fn,
        weight_decay=config.weight_decay,
    )
    return optimizer


def mse_loss(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    """Mean squared error loss."""
    return jnp.mean((y_pred - y_true) ** 2)


def train_single_epoch(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    X_train: jnp.ndarray,
    y_train: jnp.ndarray,
    batch_size: int,
    rngs: nnx.Rngs,
) -> Tuple[float, int]:
    """Train for one epoch over the data.

    The paper iterates through a single input-output pair per trajectory in the training set.
    This corresponds to batch_size=1 with the pair constructed from each trajectory.

    Args:
        model: FNO model with nnx training state
        optimizer: nnx optimizer wrapper
        X_train: Training inputs (n_samples, ...)
        y_train: Training targets (n_samples, ...)
        batch_size: Batch size (typically 1 as per paper)
        rngs: Random key streams

    Returns:
        avg_loss: Average loss over epoch
        n_batches: Number of batches
    """
    n_samples = X_train.shape[0]
    indices = jax.random.permutation(rngs.dropout(), n_samples)
    total_loss = 0.0
    n_batches = 0

    for start in range(0, n_samples, batch_size):
        batch_idx = indices[start:start + batch_size]
        x_batch = X_train[batch_idx]
        y_batch = y_train[batch_idx]

        def loss_fn(model):
            y_pred = model(x_batch)
            return mse_loss(y_pred, y_batch)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)

        total_loss += loss
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    return float(avg_loss), n_batches


def evaluate_model(
    model: nnx.Module,
    X: jnp.ndarray,
    y: jnp.ndarray,
    batch_size: int = 64,
) -> Dict[str, float]:
    """Evaluate model on a dataset.

    Returns dict with 'loss', 'rmse'.
    """
    n_samples = X.shape[0]
    total_mse = 0.0

    for start in range(0, n_samples, batch_size):
        x_batch = X[start:start + batch_size]
        y_batch = y[start:start + batch_size]
        y_pred = model(x_batch)
        total_mse += jnp.sum((y_pred - y_batch) ** 2)

    mse = float(total_mse / n_samples)
    rmse = float(jnp.sqrt(mse))
    return {"loss": mse, "rmse": rmse}


def train_ensemble(
    config: ExperimentConfig,
    X_train: jnp.ndarray,
    y_train: jnp.ndarray,
    X_val: jnp.ndarray,
    y_val: jnp.ndarray,
    n_members: int = 10,
) -> list:
    """Train an ensemble of FNOs with different random seeds.

    The paper uses 10 ensemble members trained from different random initializations.
    """
    models = []
    for i in range(n_members):
        seed = config.seed + i
        rngs = nnx.Rngs(seed)

        model = create_fno(
            spatial_dim=config.data.spatial_dim,
            input_dim=config.fno.input_dim,
            output_dim=config.fno.output_dim,
            hidden_dim=config.fno.hidden_dim,
            n_modes=config.fno.n_modes,
            n_blocks=config.fno.n_blocks,
            rngs=rngs,
        )

        optimizer_config = create_optimizer(config.training, config.training.epochs)
        optimizer = nnx.Optimizer(model, optimizer_config)

        pbar = tqdm(range(config.training.epochs), desc=f"Ensemble {i+1}/{n_members}")
        for epoch in pbar:
            avg_loss, _ = train_single_epoch(
                model, optimizer, X_train, y_train, config.training.batch_size, rngs
            )
            if epoch % 25 == 0:
                val_metrics = evaluate_model(model, X_val, y_val)
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "val_rmse": f"{val_metrics['rmse']:.4f}"})

        models.append(model)

    return models


def train(
    config: ExperimentConfig,
    X_train: jnp.ndarray,
    y_train: jnp.ndarray,
    X_val: jnp.ndarray,
    y_val: jnp.ndarray,
) -> nnx.Module:
    """Train a single FNO model.

    Args:
        config: Experiment configuration
        X_train, y_train: Training data
        X_val, y_val: Validation data

    Returns:
        Trained model
    """
    rngs = nnx.Rngs(config.seed)

    model = create_fno(
        spatial_dim=config.data.spatial_dim,
        input_dim=config.fno.input_dim,
        output_dim=config.fno.output_dim,
        hidden_dim=config.fno.hidden_dim,
        n_modes=config.fno.n_modes,
        n_blocks=config.fno.n_blocks,
        rngs=rngs,
    )

    optimizer_config = create_optimizer(config.training, config.training.epochs)
    optimizer = nnx.Optimizer(model, optimizer_config)

    best_val_rmse = float("inf")
    best_params = None

    pbar = tqdm(range(config.training.epochs), desc="Training FNO")
    for epoch in pbar:
        avg_loss, _ = train_single_epoch(
            model, optimizer, X_train, y_train, config.training.batch_size, rngs
        )

        if epoch % max(1, config.training.epochs // 20) == 0:
            val_metrics = evaluate_model(model, X_val, y_val)
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "val_rmse": f"{val_metrics['rmse']:.4f}"})

            if val_metrics["rmse"] < best_val_rmse:
                best_val_rmse = val_metrics["rmse"]
                best_params = nnx.state(model)

    # Restore best params if tracked
    if best_params is not None:
        nnx.update(model, best_params)

    print(f"Training complete. Best val RMSE: {best_val_rmse:.6f}")
    return model


def save_model(model: nnx.Module, path: str):
    """Save model parameters."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    params = nnx.state(model)
    np.savez(path, **jax.tree_util.tree_map(np.array, params))


def load_model(
    model_class,
    path: str,
    spatial_dim: int,
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    n_modes,
    n_blocks: int,
    rngs: nnx.Rngs,
) -> nnx.Module:
    """Load model from saved parameters."""
    model = create_fno(spatial_dim, input_dim, output_dim, hidden_dim, n_modes, n_blocks, rngs)
    loaded = np.load(path, allow_pickle=True)
    params = dict(loaded)
    nnx.update(model, params)
    return model
