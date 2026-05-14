"""
Training loop for FNO models.

Training details from the paper (Appendix D.2):
  - Loss: MSE
  - Optimizer: AdamW (Loshchilov & Hutter, 2019)
  - LR scheduler: cosine decay with warmup
  - Low-data regime: 100 epochs
  - OOD experiments: 1000 epochs
  - One epoch = iterating through one input-output pair per trajectory
  - Input: 10 history time steps + velocity field + reaction term
  - Padding: 2 constant zero grid points at borders (handled in FNO)
"""

from typing import Callable, Dict, Iterator, List, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from tqdm import tqdm


def mse_loss(pred: jax.Array, target: jax.Array) -> jax.Array:
    """Mean squared error loss."""
    return jnp.mean((pred - target) ** 2)


def create_optimizer(
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    warmup_steps: int = 100,
    total_steps: int = 10000,
) -> optax.GradientTransformation:
    """
    Create AdamW optimizer with cosine decay learning rate schedule and warmup.

    Args:
        learning_rate: peak learning rate
        weight_decay: L2 regularization coefficient
        warmup_steps: number of linear warmup steps
        total_steps: total training steps for cosine decay
    Returns:
        optax optimizer
    """
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=learning_rate * 1e-2,
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
    a: jax.Array,
    u: jax.Array,
) -> jax.Array:
    """
    Single training step.

    Args:
        model: FNO model
        optimizer: Flax NNX optimizer
        a: input function batch, shape (batch, n_x, d_in) or (batch, n_x, n_y, d_in)
        u: target output batch, shape (batch, n_x, d_out) or (batch, n_x, n_y, d_out)
    Returns:
        loss: scalar MSE loss
    """
    def loss_fn(model):
        pred = model(a)
        return mse_loss(pred, u)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)
    return loss


def train_epoch(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    data_loader,
) -> float:
    """
    Train for one epoch.

    Args:
        model: FNO model
        optimizer: Flax NNX optimizer
        data_loader: iterable of (a, u) batches
    Returns:
        mean epoch loss
    """
    total_loss = 0.0
    n_batches = 0

    for a_batch, u_batch in data_loader:
        loss = train_step(model, optimizer, a_batch, u_batch)
        total_loss += float(loss)
        n_batches += 1

    return total_loss / max(n_batches, 1)


def train(
    model: nnx.Module,
    train_loader,
    val_loader=None,
    n_epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    warmup_steps: int = 100,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """
    Full training loop.

    Args:
        model: FNO model
        train_loader: training data loader
        val_loader: optional validation data loader
        n_epochs: number of training epochs
        learning_rate: peak learning rate
        weight_decay: AdamW weight decay
        warmup_steps: warmup steps for LR schedule
        verbose: whether to print progress
    Returns:
        history: dict with "train_loss" and optionally "val_loss"
    """
    n_steps_per_epoch = len(train_loader) if hasattr(train_loader, "__len__") else 100
    total_steps = n_epochs * n_steps_per_epoch

    optimizer_tx = create_optimizer(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )
    optimizer = nnx.Optimizer(model, optimizer_tx)

    history = {"train_loss": [], "val_loss": []}

    epoch_iter = range(n_epochs)
    if verbose:
        epoch_iter = tqdm(epoch_iter, desc="Training")

    for epoch in epoch_iter:
        train_loss = train_epoch(model, optimizer, train_loader)
        history["train_loss"].append(train_loss)

        if val_loader is not None:
            val_loss = evaluate_loss(model, val_loader)
            history["val_loss"].append(val_loss)

            if verbose:
                epoch_iter.set_postfix(
                    train_loss=f"{train_loss:.4e}",
                    val_loss=f"{val_loss:.4e}",
                )
        elif verbose:
            epoch_iter.set_postfix(train_loss=f"{train_loss:.4e}")

    return history


@nnx.jit
def eval_step(model: nnx.Module, a: jax.Array, u: jax.Array) -> jax.Array:
    """Single evaluation step."""
    pred = model(a)
    return mse_loss(pred, u)


def evaluate_loss(model: nnx.Module, data_loader) -> float:
    """Evaluate MSE loss on a dataset."""
    total_loss = 0.0
    n_batches = 0

    for a_batch, u_batch in data_loader:
        loss = eval_step(model, a_batch, u_batch)
        total_loss += float(loss)
        n_batches += 1

    return total_loss / max(n_batches, 1)


def train_ensemble(
    model_factory: Callable,
    train_loader,
    val_loader=None,
    n_ensemble: int = 10,
    n_epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    warmup_steps: int = 100,
    base_seed: int = 0,
    verbose: bool = True,
) -> List[nnx.Module]:
    """
    Train an ensemble of FNO models with different random seeds.

    Args:
        model_factory: callable that creates a new FNO model given a seed
        train_loader: training data loader
        val_loader: optional validation data loader
        n_ensemble: number of ensemble members (paper uses 10)
        n_epochs: training epochs per member
        learning_rate: peak learning rate
        weight_decay: AdamW weight decay
        warmup_steps: warmup steps
        base_seed: base random seed (each member uses base_seed + i)
        verbose: whether to print progress
    Returns:
        list of trained FNO models
    """
    ensemble = []

    for i in range(n_ensemble):
        if verbose:
            print(f"\nTraining ensemble member {i + 1}/{n_ensemble} (seed={base_seed + i})")

        model = model_factory(seed=base_seed + i)
        train(
            model,
            train_loader,
            val_loader=val_loader,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            verbose=verbose,
        )
        ensemble.append(model)

    return ensemble
