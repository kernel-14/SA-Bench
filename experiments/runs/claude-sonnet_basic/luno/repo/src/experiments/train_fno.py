"""
Training script for Fourier Neural Operators.

From Appendix D.2:
- Architecture: 12 modes, 18 hidden dimensions, 4 Fourier blocks
- Training: AdamW optimizer with cosine decay learning rate scheduler with warmup
- Loss: Mean Squared Error
- Low data regime: 100 epochs
- OOD experiment: 1000 epochs
- One epoch = iterating through a single input-output pair per trajectory
"""

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import numpy as np
from typing import Optional, Tuple, Dict
import time
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fno import FNO1d, FNO2d
from data.dataset import PDEDataset, batch_iterator


def create_fno_1d(
    in_channels: int,
    out_channels: int = 1,
    hidden_channels: int = 18,
    n_modes: int = 12,
    n_layers: int = 4,
    seed: int = 42,
) -> FNO1d:
    """Create a 1D FNO with the paper's default hyperparameters."""
    rngs = nnx.Rngs(params=seed)
    return FNO1d(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        n_modes=n_modes,
        n_layers=n_layers,
        rngs=rngs,
    )


def create_fno_2d(
    in_channels: int,
    out_channels: int = 1,
    hidden_channels: int = 18,
    n_modes_x: int = 12,
    n_modes_y: int = 12,
    n_layers: int = 4,
    seed: int = 42,
) -> FNO2d:
    """Create a 2D FNO with the paper's default hyperparameters."""
    rngs = nnx.Rngs(params=seed)
    return FNO2d(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        n_modes_x=n_modes_x,
        n_modes_y=n_modes_y,
        n_layers=n_layers,
        rngs=rngs,
    )


def create_optimizer(
    n_epochs: int,
    n_steps_per_epoch: int,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    warmup_fraction: float = 0.05,
) -> optax.GradientTransformation:
    """Create AdamW optimizer with cosine decay and warmup.
    
    From Appendix D.2: AdamW with cosine decay learning rate scheduler with warmup.
    """
    total_steps = n_epochs * n_steps_per_epoch
    warmup_steps = int(warmup_fraction * total_steps)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=1e-6,
    )

    optimizer = optax.adamw(
        learning_rate=schedule,
        weight_decay=weight_decay,
    )

    return optimizer


@nnx.jit
def train_step(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    x: jnp.ndarray,
    y: jnp.ndarray,
) -> jnp.ndarray:
    """Single training step.
    
    Args:
        model: FNO model
        optimizer: Optax optimizer wrapped in nnx.Optimizer
        x: Input batch, shape (batch, n_x, in_ch) or (batch, n_x, n_y, in_ch)
        y: Target batch, shape (batch, n_x, out_ch) or (batch, n_x, n_y, out_ch)
    
    Returns:
        Loss value
    """
    def loss_fn(model):
        y_pred = model(x)
        loss = jnp.mean((y_pred - y) ** 2)
        return loss

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)
    return loss


def train_fno(
    model: nnx.Module,
    train_dataset: PDEDataset,
    val_dataset: PDEDataset,
    n_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """Train an FNO model.
    
    From Appendix D.2:
    - One epoch = iterating through a single input-output pair per trajectory
    - Low data regime: 100 epochs
    - OOD experiment: 1000 epochs
    
    Args:
        model: FNO model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        n_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Peak learning rate
        weight_decay: Weight decay for AdamW
        seed: Random seed
        verbose: Whether to print progress
    
    Returns:
        Dictionary with training history
    """
    n_train = len(train_dataset)
    n_steps_per_epoch = max(1, n_train // batch_size)

    optimizer_tx = create_optimizer(
        n_epochs=n_epochs,
        n_steps_per_epoch=n_steps_per_epoch,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    optimizer = nnx.Optimizer(model, optimizer_tx)

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')

    for epoch in range(n_epochs):
        # Training
        train_losses = []
        for x_batch, y_batch in batch_iterator(train_dataset, batch_size, shuffle=True, seed=seed + epoch):
            x_batch = jnp.array(x_batch, dtype=jnp.float32)
            y_batch = jnp.array(y_batch, dtype=jnp.float32)
            loss = train_step(model, optimizer, x_batch, y_batch)
            train_losses.append(float(loss))

        train_loss = np.mean(train_losses)
        history['train_loss'].append(train_loss)

        # Validation
        val_losses = []
        for x_batch, y_batch in batch_iterator(val_dataset, batch_size, shuffle=False):
            x_batch = jnp.array(x_batch, dtype=jnp.float32)
            y_batch = jnp.array(y_batch, dtype=jnp.float32)
            y_pred = model(x_batch)
            val_loss = float(jnp.mean((y_pred - y_batch) ** 2))
            val_losses.append(val_loss)

        val_loss = np.mean(val_losses)
        history['val_loss'].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

    return history


def train_ensemble(
    in_channels: int,
    out_channels: int,
    train_dataset: PDEDataset,
    val_dataset: PDEDataset,
    n_members: int = 10,
    n_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    is_2d: bool = False,
    verbose: bool = True,
) -> list:
    """Train an ensemble of FNO models.
    
    From Section 5: Deep ensembles trained 10 times with different random seeds.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        train_dataset: Training dataset
        val_dataset: Validation dataset
        n_members: Number of ensemble members (10 in paper)
        n_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        is_2d: Whether to use 2D FNO
        verbose: Whether to print progress
    
    Returns:
        List of trained FNO models
    """
    ensemble = []

    for i in range(n_members):
        print(f"\nTraining ensemble member {i+1}/{n_members}...")

        if is_2d:
            model = create_fno_2d(in_channels, out_channels, seed=i)
        else:
            model = create_fno_1d(in_channels, out_channels, seed=i)

        train_fno(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=i,
            verbose=verbose,
        )

        ensemble.append(model)

    return ensemble
