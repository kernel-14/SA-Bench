"""
trainer.py
===========
Training loop for the Fourier Neural Operator (FNO).

The ``Trainer`` class provides a simple interface:

    trainer = Trainer(model, train_ds, val_ds, train_config, batch_size)
    best_params = trainer.train(rng)

It uses AdamW with a cosine decay learning‑rate schedule including a linear
warm‑up, exactly as described in the LUNO paper.  The best parameters (in terms
of validation loss) are saved to disk and returned.
"""

from __future__ import annotations

import logging
import os
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Tuple,
)

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn  # for type hints
from tqdm.auto import tqdm

from config import TrainConfig  # safe import: uses immutable dataclass
from utils import save_pytree

logger = logging.getLogger(__name__)


# =============================================================================
# Helper: jitted training / validation steps
# =============================================================================


def _make_train_step(
    apply_fn: Callable[[Any, jnp.ndarray], jnp.ndarray],
    optimizer_update_fn: Callable,
) -> Callable:
    """
    Create a jitted training step function.

    Parameters
    ----------
    apply_fn : callable
        A stateless function ``apply_fn(params, x_batch) -> predictions``.
        Typically ``FourierNeuralOperator.apply``.
    optimizer_update_fn : callable
        The ``update`` function of an Optax gradient transformation:

            ``updates, new_opt_state = update_fn(grads, opt_state, params)``

    Returns
    -------
    train_step : callable
        Jitted function ``(params, opt_state, x, y) ->
        (new_params, new_opt_state, loss)``.
    """

    def loss_fn(params: Any, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        pred = apply_fn(params, x)
        return jnp.mean((pred - y) ** 2)

    @jax.jit
    def _train_step(
        params: Any,
        opt_state: Any,
        x: jnp.ndarray,
        y: jnp.ndarray,
    ) -> Tuple[Any, Any, jnp.ndarray]:
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        updates, new_opt_state = optimizer_update_fn(
            grads, opt_state, params
        )
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    return _train_step


def _make_val_step(
    apply_fn: Callable[[Any, jnp.ndarray], jnp.ndarray],
) -> Callable:
    """
    Create a jitted validation step that returns the MSE loss.

    Parameters
    ----------
    apply_fn : callable
        As in :func:`_make_train_step`.

    Returns
    -------
    val_step : callable
        Jitted ``(params, x, y) -> scalar MSE loss``.
    """

    @jax.jit
    def _val_step(
        params: Any, x: jnp.ndarray, y: jnp.ndarray
    ) -> jnp.ndarray:
        pred = apply_fn(params, x)
        return jnp.mean((pred - y) ** 2)

    return _val_step


# =============================================================================
# Trainer class
# =============================================================================


class Trainer:
    """
    Orchestrates the training of an FNO on given datasets.

    Parameters
    ----------
    model : FourierNeuralOperator
        The FNO model (a Flax ``Module``).  Its ``apply`` method is used
        for forward passes.
    train_ds : tuple of ndarray
        ``(train_x, train_y)``. Shapes:
        - ``train_x`` : ``(N, T, *spatial, C)``
        - ``train_y`` : ``(N, *spatial, 1)``
    val_ds : tuple of ndarray
        Same format as ``train_ds``; may be empty (``None`` is accepted).
    config : TrainConfig
        Training hyperparameters (epochs, learning rate, weight decay, …).
    batch_size : int
        Mini‑batch size (from the data configuration).
    """

    def __init__(
        self,
        model: nn.Module,
        train_ds: Tuple[np.ndarray, np.ndarray],
        val_ds: Optional[Tuple[np.ndarray, np.ndarray]],
        config: TrainConfig,
        batch_size: int,
    ) -> None:
        self.model = model
        self.train_x, self.train_y = train_ds
        self.val_x, self.val_y = val_ds if val_ds is not None else (None, None)
        self.config = config
        self.batch_size = batch_size

        # Will be initialised in train()
        self.params: Optional[Any] = None
        self.best_params: Optional[Any] = None
        self.best_val_loss: float = float("inf")
        self.opt_state: Optional[Any] = None

        # Create the optimizer once (schedule is built inside train() because it
        # depends on total steps, but the transformation object itself is
        # independent of the schedule – the schedule is a function).  So we
        # construct the optimizer without the schedule; the learning-rate
        # schedule will be injected in train() via `optax.inject_hyperparams`.
        # However, simple approach: build the optimizer with the schedule at
        # train time.  We'll do that in `train()` and store it.
        # We'll just store a placeholder; the actual optimizer will be created
        # in `train()` to capture `steps_per_epoch`.
        self.optimizer = None
        self.optimizer_update_fn = None

        # jitted step functions – set in train() as well
        self.train_step_jit = None
        self.val_step_jit = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train(self, rng: jax.random.PRNGKey) -> Dict[str, Any]:
        """
        Run the full training loop.

        Parameters
        ----------
        rng : jax.random.PRNGKey
            Base random key.  It is split internally for reproducibility.

        Returns
        -------
        best_params : dict
            The Flax parameter dictionary that achieved the lowest validation
            loss.  (Also saved to disk in ``config.checkpoint_dir``).
        """
        rng, init_rng, shuffle_root = jax.random.split(rng, 3)

        # ---- 1. Initialise model parameters -----------------------------------
        # Use a single training sample to infer shapes (dummy input).
        dummy = self.train_x[:1]
        # Flax Modules' .init returns a FrozenDict containing 'params' and
        # possibly other collections; we only need 'params'.
        initialised = self.model.init(init_rng, dummy, train=False)
        params = initialised.get("params", initialised)
        self.params = params

        # ---- 2. Build learning rate schedule & optimizer -----------------------
        num_train_samples = self.train_x.shape[0]
        steps_per_epoch = max(
            1, int(np.ceil(num_train_samples / self.batch_size))
        )
        total_steps = self.config.epochs * steps_per_epoch
        warmup_steps = self.config.warmup_epochs * steps_per_epoch

        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.config.learning_rate,
            warmup_steps=max(0, warmup_steps),
            decay_steps=max(1, total_steps - warmup_steps),
            end_value=0.0,
        )
        self.optimizer = optax.adamw(
            learning_rate=lr_schedule,
            weight_decay=self.config.weight_decay,
        )
        self.optimizer_update_fn = self.optimizer.update
        self.opt_state = self.optimizer.init(params)

        # ---- 3. Prepare jitted step functions (capture apply_fn) ------------
        apply_fn = self.model.apply
        train_step = _make_train_step(apply_fn, self.optimizer_update_fn)
        val_step_fn = _make_val_step(apply_fn)

        # ---- 4. Training loop ------------------------------------------------
        best_params = None
        best_val_loss = float("inf")
        shuffle_key = shuffle_root

        epoch_pbar = tqdm(
            range(1, self.config.epochs + 1),
            desc="Training epochs",
            leave=True,
        )

        for epoch in epoch_pbar:
            # Shuffle training data
            shuffle_key, epoch_key = jax.random.split(shuffle_key)
            perm = jax.random.permutation(epoch_key, num_train_samples)

            # --- Train one epoch ---
            train_loss_sum = 0.0
            n_batches = 0
            for start in range(0, num_train_samples, self.batch_size):
                batch_idx = perm[start : start + self.batch_size]
                xb = jnp.asarray(self.train_x[batch_idx])
                yb = jnp.asarray(self.train_y[batch_idx])

                self.params, self.opt_state, loss = train_step(
                    self.params, self.opt_state, xb, yb
                )
                train_loss_sum += float(loss) * len(batch_idx)
                n_batches += 1
            avg_train_loss = train_loss_sum / num_train_samples

            # --- Validation ---
            val_loss = None
            if self.val_x is not None and self.val_y is not None:
                val_loss_sum = 0.0
                n_val = self.val_x.shape[0]
                for start in range(0, n_val, self.batch_size):
                    idx = slice(start, min(start + self.batch_size, n_val))
                    xb = jnp.asarray(self.val_x[idx])
                    yb = jnp.asarray(self.val_y[idx])
                    batch_val_loss = val_step_fn(self.params, xb, yb)
                    val_loss_sum += float(batch_val_loss) * (idx.stop - idx.start)
                val_loss = val_loss_sum / n_val

                # Track best
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = jax.tree_map(
                        lambda x: x.copy(), self.params
                    )

            # Logging
            if epoch % self.config.log_interval == 0 or epoch == 1:
                log_str = (
                    f"Epoch {epoch:4d}/{self.config.epochs} | "
                    f"train loss: {avg_train_loss:.6f}"
                )
                if val_loss is not None:
                    log_str += f" | val loss: {val_loss:.6f}"
                lr = float(lr_schedule((epoch - 1) * steps_per_epoch))
                log_str += f" | lr: {lr:.2e}"
                epoch_pbar.write(log_str)

            # Save checkpoint if validation improved (and validation exists)
            if val_loss is not None and val_loss == best_val_loss:
                save_pytree(
                    best_params,
                    os.path.join(
                        self.config.checkpoint_dir, "best_params.pkl"
                    ),
                )

        # ---- 5. Finalise --------------------------------------------------
        self.best_params = best_params if best_params is not None else self.params
        self.best_val_loss = best_val_loss

        # Save final best parameters
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        final_path = os.path.join(
            self.config.checkpoint_dir, "best_params.pkl"
        )
        save_pytree(self.best_params, final_path)
        logger.info("Best parameters saved to %s", final_path)

        return self.best_params

