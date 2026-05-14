## training/trainer.py
"""Training module for the LUNO reproduction.

Implements the Trainer class that manages the full FNO training lifecycle:
  - AdamW optimizer with cosine decay + linear warmup schedule
  - Epoch definition from paper: one input-output pair per trajectory per epoch
  - MSE loss
  - Orbax checkpointing

Paper references:
  - Appendix D.2: "mean squared error loss was minimized using AdamW combined
    with a cosine decay learning rate scheduler with warmup"
  - Appendix D.2: "one epoch corresponds to iterating through a single
    input-output pair per trajectory in the training set"
  - Appendix D.2: "Networks for the low data experiment are trained for 100
    epochs, all remaining networks are trained for 1000 epochs"

Design notes:
  - Uses Flax NNX's nnx.split/nnx.merge for functional parameter access,
    enabling jax.jit and jax.value_and_grad over model parameters.
  - The optimizer is constructed inside train() once the dataset size is
    known (total_steps = n_epochs * n_traj requires n_traj from dataset).
  - _train_step is a module-level JIT-compiled function (not a method) to
    avoid recompilation when self changes.
  - Pairs are stored in dataset in trajectory-major order:
    pair_idx = traj_idx * pairs_per_traj + local_t_idx
    Each epoch samples one local_t_idx per trajectory uniformly at random.
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from config import Config
from data.dataset import PDEDataset
from models.fno import FNO

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level JIT-compiled training step
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=("optimizer",))
def _jit_train_step(
    graphdef: Any,
    state: Any,
    batch: Tuple[jnp.ndarray, jnp.ndarray],
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
) -> Tuple[Any, optax.OptState, jnp.ndarray]:
    """JIT-compiled single gradient update step.

    Computes the MSE loss and its gradient w.r.t. the model state, then
    applies the optimizer update. Uses Flax NNX's split/merge pattern to
    enable functional JAX transforms over the stateful NNX module.

    Args:
        graphdef: NNX graph definition (static structure of the FNO).
            Captured as a static argument for JIT compilation.
        state: NNX state (parameter values as a pytree). This is the
            differentiable argument.
        batch: Tuple ``(inputs, targets)`` where:
            - ``inputs``: shape ``[batch_size, spatial, in_channels]`` (1D)
              or ``[batch_size, H, W, in_channels]`` (2D).
            - ``targets``: shape ``[batch_size, spatial, out_channels]`` (1D)
              or ``[batch_size, H, W, out_channels]`` (2D).
        opt_state: Current optimizer state.
        optimizer: Optax gradient transformation (static, not traced).

    Returns:
        Tuple ``(new_state, new_opt_state, loss)`` where:
        - ``new_state``: Updated NNX state after the gradient step.
        - ``new_opt_state``: Updated optimizer state.
        - ``loss``: Scalar MSE loss value before the update.

    Notes:
        - ``optimizer`` is marked as a static argument so JIT treats it as
          a compile-time constant. This avoids recompilation when the
          optimizer object is the same Python object.
        - The loss is computed as ``mean((predictions - targets)^2)`` over
          all spatial points and channels.
    """
    inputs: jnp.ndarray = batch[0]
    targets: jnp.ndarray = batch[1]

    def loss_fn(state_inner: Any) -> jnp.ndarray:
        """Compute MSE loss for a given model state.

        Args:
            state_inner: NNX state to evaluate.

        Returns:
            Scalar MSE loss.
        """
        model_copy: FNO = nnx.merge(graphdef, state_inner)
        predictions: jnp.ndarray = model_copy(inputs)
        loss: jnp.ndarray = jnp.mean((predictions - targets) ** 2)
        return loss

    # Compute loss and gradients w.r.t. state
    loss_val: jnp.ndarray
    grads: Any
    loss_val, grads = jax.value_and_grad(loss_fn)(state)

    # Apply optimizer update
    updates: Any
    new_opt_state: optax.OptState
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_state: Any = optax.apply_updates(state, updates)

    return new_state, new_opt_state, loss_val


# ---------------------------------------------------------------------------
# Trainer class
# ---------------------------------------------------------------------------

class Trainer:
    """Manages FNO training with AdamW + cosine decay schedule.

    Implements the training procedure from Appendix D.2 of the LUNO paper:
    - MSE loss
    - AdamW optimizer with cosine decay + linear warmup
    - Epoch definition: one input-output pair per trajectory per epoch
    - Orbax checkpointing

    The optimizer is constructed lazily inside ``train()`` once the dataset
    size is known (required for computing ``total_steps``).

    Attributes:
        model: The FNO instance to train.
        config: Configuration dataclass with all hyperparameters.
        optimizer: Optax gradient transformation. Set to ``None`` until
            ``train()`` is called.
        opt_state: Optax optimizer state. Set to ``None`` until ``train()``
            is called.

    Example::

        model = FNO(modes=12, channels=18, n_blocks=4, in_channels=12,
                    out_channels=1, spatial_dims=1, rngs=nnx.Rngs(params=42))
        trainer = Trainer(model=model, config=config)
        key = jax.random.PRNGKey(0)
        graphdef, final_state = trainer.train(train_dataset, key)
    """

    def __init__(
        self,
        model: FNO,
        config: Config,
    ) -> None:
        """Initialise the Trainer.

        Stores the model and config. The optimizer is not constructed here
        because ``total_steps`` requires the dataset size, which is only
        known at ``train()`` time.

        Args:
            model: The FNO instance to train. Must be a Flax NNX module
                with parameters already initialised (via ``FNO.__init__``).
            config: Configuration dataclass. Relevant fields:
                - ``config.lr``: Peak learning rate (default 1e-3).
                - ``config.weight_decay``: AdamW weight decay (default 1e-4).
                - ``config.warmup_fraction``: Warmup fraction (default 0.05).
                - ``config.epochs``: Total training epochs (100 or 1000).
                - ``config.log_every_n_epochs``: Logging frequency (default 10).
                - ``config.save_checkpoints``: Whether to save checkpoints.
                - ``config.checkpoint_dir``: Checkpoint directory.
        """
        self.model: FNO = model
        self.config: Config = config

        # Optimizer and state are set lazily in train()
        self.optimizer: Optional[optax.GradientTransformation] = None
        self.opt_state: Optional[optax.OptState] = None

        logger.info(
            "Trainer initialised: lr=%.2e, weight_decay=%.2e, "
            "warmup_fraction=%.2f, epochs=%d",
            config.lr,
            config.weight_decay,
            config.warmup_fraction,
            config.epochs,
        )

    # -----------------------------------------------------------------------
    # Optimizer Construction
    # -----------------------------------------------------------------------

    def _make_optimizer(
        self,
        total_steps: int,
        warmup_steps: int,
    ) -> optax.GradientTransformation:
        """Construct the AdamW optimizer with cosine decay + warmup schedule.

        Implements the schedule from Appendix D.2:
        "AdamW combined with a cosine decay learning rate scheduler with warmup"

        The schedule is:
        - Linear warmup from 0 to ``peak_value`` over ``warmup_steps`` steps.
        - Cosine decay from ``peak_value`` to 0 over the remaining steps.

        Args:
            total_steps: Total number of gradient steps across all epochs.
                Computed as ``n_epochs * n_traj`` (one step per trajectory
                per epoch).
            warmup_steps: Number of warmup steps. Computed as
                ``int(warmup_fraction * total_steps)``.

        Returns:
            An ``optax.GradientTransformation`` combining the cosine decay
            schedule with AdamW weight decay.

        Notes:
            - ``end_value=0.0`` means the learning rate decays to zero at
              the end of training.
            - ``optax.adamw`` applies weight decay directly to parameters
              (not to the gradient), which is the correct AdamW behavior.
        """
        # Cosine decay schedule with linear warmup
        schedule: optax.Schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.config.lr,
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
            end_value=0.0,
        )

        # AdamW optimizer with the schedule
        optimizer: optax.GradientTransformation = optax.adamw(
            learning_rate=schedule,
            weight_decay=self.config.weight_decay,
        )

        logger.info(
            "_make_optimizer: total_steps=%d, warmup_steps=%d, "
            "peak_lr=%.2e, weight_decay=%.2e",
            total_steps,
            warmup_steps,
            self.config.lr,
            self.config.weight_decay,
        )

        return optimizer

    # -----------------------------------------------------------------------
    # Loss Function
    # -----------------------------------------------------------------------

    def _loss_fn(
        self,
        state: Any,
        graphdef: Any,
        batch: Tuple[jnp.ndarray, jnp.ndarray],
    ) -> jnp.ndarray:
        """Compute MSE loss for a given model state and batch.

        This method is provided for external use (e.g., validation loss
        computation). The JIT-compiled training step uses the module-level
        ``_jit_train_step`` function instead.

        Args:
            state: NNX state (parameter values).
            graphdef: NNX graph definition.
            batch: Tuple ``(inputs, targets)``.

        Returns:
            Scalar MSE loss: ``mean((predictions - targets)^2)``.
        """
        inputs: jnp.ndarray = batch[0]
        targets: jnp.ndarray = batch[1]
        model_copy: FNO = nnx.merge(graphdef, state)
        predictions: jnp.ndarray = model_copy(inputs)
        return jnp.mean((predictions - targets) ** 2)

    # -----------------------------------------------------------------------
    # Training Step (non-JIT wrapper for external use)
    # -----------------------------------------------------------------------

    def _train_step(
        self,
        graphdef: Any,
        state: Any,
        batch: Tuple[jnp.ndarray, jnp.ndarray],
        opt_state: optax.OptState,
    ) -> Tuple[Any, optax.OptState, jnp.ndarray]:
        """Apply one gradient update step (wraps the JIT-compiled function).

        Args:
            graphdef: NNX graph definition (static).
            state: Current NNX state.
            batch: Tuple ``(inputs, targets)``.
            opt_state: Current optimizer state.

        Returns:
            Tuple ``(new_state, new_opt_state, loss)``.

        Raises:
            RuntimeError: If ``self.optimizer`` has not been initialised
                (i.e., ``train()`` has not been called yet).
        """
        if self.optimizer is None:
            raise RuntimeError(
                "_train_step called before optimizer was initialised. "
                "Call train() first."
            )
        return _jit_train_step(
            graphdef=graphdef,
            state=state,
            batch=batch,
            opt_state=opt_state,
            optimizer=self.optimizer,
        )

    # -----------------------------------------------------------------------
    # Main Training Loop
    # -----------------------------------------------------------------------

    def train(
        self,
        dataset: PDEDataset,
        key: jax.Array,
    ) -> Tuple[Any, Any]:
        """Run the full training loop and return the trained model state.

        Implements the paper's training procedure (Appendix D.2):
        - One epoch = one input-output pair per trajectory (n_traj steps).
        - Trajectory order is shuffled each epoch.
        - Within each trajectory, one pair is selected uniformly at random.
        - Loss is logged every ``config.log_every_n_epochs`` epochs.
        - Checkpoints are saved at the end of training if
          ``config.save_checkpoints`` is True.

        The dataset is assumed to store pairs in trajectory-major order:
        ``pair_idx = traj_idx * pairs_per_traj + local_t_idx``
        where ``pairs_per_traj = dataset.n_pairs // dataset.n_traj``.

        Args:
            dataset: Training dataset. Must have ``n_traj`` set (not None).
                Pairs are assumed to be in trajectory-major order.
            key: JAX PRNG key for reproducible shuffling and pair selection.

        Returns:
            Tuple ``(graphdef, final_state)`` where:
            - ``graphdef``: NNX graph definition (static structure).
            - ``final_state``: NNX state after training (parameter values).

        Raises:
            ValueError: If ``dataset.n_traj`` is None (trajectory count
                required for the paper's epoch definition).
            ValueError: If ``dataset.n_pairs`` is not divisible by
                ``dataset.n_traj`` (pairs must be in trajectory-major order).

        Example::

            key = jax.random.PRNGKey(42)
            graphdef, state = trainer.train(train_dataset, key)
            # Use state for inference:
            model_copy = nnx.merge(graphdef, state)
            predictions = model_copy(test_inputs)
        """
        # ------------------------------------------------------------------
        # Validate dataset
        # ------------------------------------------------------------------
        if dataset.n_traj is None:
            raise ValueError(
                "dataset.n_traj must be set for the paper's epoch definition "
                "(one pair per trajectory per epoch). "
                "Ensure the dataset was created with n_traj specified."
            )

        n_traj: int = dataset.n_traj
        n_pairs: int = dataset.n_pairs

        if n_pairs % n_traj != 0:
            raise ValueError(
                f"dataset.n_pairs ({n_pairs}) must be divisible by "
                f"dataset.n_traj ({n_traj}) for trajectory-major pair ordering. "
                f"Got {n_pairs} % {n_traj} = {n_pairs % n_traj}."
            )

        pairs_per_traj: int = n_pairs // n_traj
        n_epochs: int = self.config.epochs

        logger.info(
            "Starting training: n_epochs=%d, n_traj=%d, pairs_per_traj=%d, "
            "steps_per_epoch=%d, total_steps=%d",
            n_epochs,
            n_traj,
            pairs_per_traj,
            n_traj,
            n_epochs * n_traj,
        )

        # ------------------------------------------------------------------
        # Construct optimizer with correct total_steps
        # ------------------------------------------------------------------
        total_steps: int = n_epochs * n_traj
        warmup_steps: int = max(1, int(self.config.warmup_fraction * total_steps))
        self.optimizer = self._make_optimizer(total_steps, warmup_steps)

        # ------------------------------------------------------------------
        # Extract NNX graphdef and initial state
        # ------------------------------------------------------------------
        graphdef: Any
        state: Any
        graphdef, state = nnx.split(self.model)

        # ------------------------------------------------------------------
        # Initialise optimizer state
        # ------------------------------------------------------------------
        self.opt_state = self.optimizer.init(state)

        opt_state: optax.OptState = self.opt_state

        # ------------------------------------------------------------------
        # Training loop
        # ------------------------------------------------------------------
        for epoch in range(n_epochs):
            key, key_shuffle, key_time = jax.random.split(key, 3)

            # Shuffle trajectory order for this epoch
            traj_order: jnp.ndarray = jax.random.permutation(
                key_shuffle, n_traj
            )  # [n_traj]

            # For each trajectory, sample one local time index uniformly
            # local_t_idx ∈ [0, pairs_per_traj)
            local_t_indices: jnp.ndarray = jax.random.randint(
                key_time,
                shape=(n_traj,),
                minval=0,
                maxval=pairs_per_traj,
            )  # [n_traj]

            # Compute global pair indices for this epoch
            # pair_idx = traj_idx * pairs_per_traj + local_t_idx
            epoch_pair_indices: jnp.ndarray = (
                traj_order * pairs_per_traj + local_t_indices
            )  # [n_traj]

            # Fetch the batch for this epoch (one pair per trajectory)
            batch_inputs: jnp.ndarray
            batch_targets: jnp.ndarray
            batch_inputs, batch_targets = dataset.get_batch(epoch_pair_indices)
            # batch_inputs: [n_traj, spatial, in_channels]
            # batch_targets: [n_traj, spatial, out_channels]

            batch: Tuple[jnp.ndarray, jnp.ndarray] = (batch_inputs, batch_targets)

            # Apply one gradient update step
            state, opt_state, loss_val = _jit_train_step(
                graphdef=graphdef,
                state=state,
                batch=batch,
                opt_state=opt_state,
                optimizer=self.optimizer,
            )

            # ------------------------------------------------------------------
            # Logging
            # ------------------------------------------------------------------
            if (epoch + 1) % self.config.log_every_n_epochs == 0 or epoch == 0:
                logger.info(
                    "Epoch [%d/%d] | Loss: %.6f | LR: %.2e",
                    epoch + 1,
                    n_epochs,
                    float(loss_val),
                    float(self._get_current_lr(opt_state)),
                )

        # ------------------------------------------------------------------
        # Update stored optimizer state
        # ------------------------------------------------------------------
        self.opt_state = opt_state

        # ------------------------------------------------------------------
        # Save checkpoint if requested
        # ------------------------------------------------------------------
        if self.config.save_checkpoints:
            checkpoint_path: str = os.path.join(
                self.config.checkpoint_dir,
                f"fno_{self.config.experiment}_{self.config.pde_name}",
            )
            try:
                self.save_params(state, checkpoint_path)
                logger.info("Checkpoint saved to: %s", checkpoint_path)
            except Exception as e:  # pylint: disable=broad-except
                logger.warning(
                    "Failed to save checkpoint to %s: %s", checkpoint_path, e
                )

        logger.info(
            "Training complete: %d epochs, final loss: %.6f",
            n_epochs,
            float(loss_val),
        )

        return graphdef, state

    # -----------------------------------------------------------------------
    # Checkpointing
    # -----------------------------------------------------------------------

    def save_params(
        self,
        state: Any,
        path: str,
    ) -> None:
        """Save model parameters to disk using Orbax.

        Creates the checkpoint directory if it does not exist. The state
        is saved as an Orbax checkpoint, which handles JAX pytrees natively.

        Args:
            state: NNX state (parameter values as a pytree) to save.
                Typically the ``state`` returned by ``train()``.
            path: Directory path where the checkpoint will be saved.
                Created automatically if it does not exist.

        Raises:
            ImportError: If ``orbax-checkpoint`` is not installed.
            Exception: If the checkpoint save fails (e.g., disk full,
                permission denied).

        Example::

            trainer.save_params(state, "checkpoints/fno_burgers")
        """
        try:
            import orbax.checkpoint as ocp  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "orbax-checkpoint is required for saving checkpoints. "
                "Install it with: pip install orbax-checkpoint"
            ) from exc

        # Create directory if it does not exist
        os.makedirs(path, exist_ok=True)

        # Save using StandardCheckpointer
        checkpointer = ocp.StandardCheckpointer()
        checkpointer.save(path, state)

        logger.debug("Checkpoint saved: %s", path)

    def load_params(
        self,
        path: str,
        target_state: Optional[Any] = None,
    ) -> Any:
        """Load model parameters from an Orbax checkpoint.

        Restores the NNX state from a previously saved checkpoint. If
        ``target_state`` is provided, it is used as the target structure
        for shape inference during restoration.

        Args:
            path: Directory path of the checkpoint to restore.
            target_state: Optional target NNX state with the correct
                pytree structure. If ``None``, attempts to restore without
                a target (may fail for some Orbax versions).

        Returns:
            Restored NNX state (parameter values as a pytree).

        Raises:
            ImportError: If ``orbax-checkpoint`` is not installed.
            FileNotFoundError: If the checkpoint directory does not exist.
            Exception: If restoration fails.

        Example::

            # Restore with target structure from current model
            _, init_state = nnx.split(model)
            state = trainer.load_params("checkpoints/fno_burgers", init_state)
            model_copy = nnx.merge(graphdef, state)
        """
        try:
            import orbax.checkpoint as ocp  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "orbax-checkpoint is required for loading checkpoints. "
                "Install it with: pip install orbax-checkpoint"
            ) from exc

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Checkpoint directory not found: {path}"
            )

        checkpointer = ocp.StandardCheckpointer()

        if target_state is not None:
            # Restore with target structure for correct shape inference
            restored_state: Any = checkpointer.restore(path, target=target_state)
        else:
            # Attempt restoration without target (may work for simple pytrees)
            restored_state = checkpointer.restore(path)

        logger.debug("Checkpoint loaded: %s", path)

        return restored_state

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def _get_current_lr(
        self,
        opt_state: optax.OptState,
    ) -> float:
        """Extract the current learning rate from the optimizer state.

        Traverses the optimizer state tree to find the scale-by-learning-rate
        state, which contains the current learning rate value.

        Args:
            opt_state: Current optimizer state from optax.

        Returns:
            Current learning rate as a Python float. Returns ``0.0`` if
            the learning rate cannot be extracted (e.g., unexpected state
            structure).

        Notes:
            This is a best-effort extraction for logging purposes only.
            The exact state structure depends on the optax version and
            optimizer configuration.
        """
        try:
            # optax.adamw state is a tuple; the schedule state is typically
            # in the ScaleByScheduleState which has a 'count' field.
            # The actual LR is computed from the schedule function.
            # We extract it by looking for the scale state.
            for state_item in jax.tree_util.tree_leaves(opt_state):
                if hasattr(state_item, "count"):
                    # This is likely the schedule state; compute LR from count
                    if self.optimizer is not None:
                        # Use the schedule directly if accessible
                        # For warmup_cosine_decay_schedule, we can recompute
                        count: int = int(state_item.count)
                        # Reconstruct schedule to get current LR
                        schedule: optax.Schedule = optax.warmup_cosine_decay_schedule(
                            init_value=0.0,
                            peak_value=self.config.lr,
                            warmup_steps=max(
                                1,
                                int(
                                    self.config.warmup_fraction
                                    * (self.config.epochs * 25)  # approximate
                                ),
                            ),
                            decay_steps=self.config.epochs * 25,
                            end_value=0.0,
                        )
                        return float(schedule(count))
        except Exception:  # pylint: disable=broad-except
            pass
        return 0.0

    def evaluate_loss(
        self,
        graphdef: Any,
        state: Any,
        dataset: PDEDataset,
        n_pairs: int = 250,
        key: Optional[jax.Array] = None,
    ) -> float:
        """Compute the average MSE loss on a subset of the dataset.

        Useful for monitoring validation loss during or after training.

        Args:
            graphdef: NNX graph definition.
            state: NNX state (parameter values).
            dataset: Dataset to evaluate on (typically validation set).
            n_pairs: Number of pairs to evaluate. If larger than
                ``dataset.n_pairs``, all pairs are used.
            key: Optional JAX PRNG key for random pair selection. If
                ``None``, uses the first ``n_pairs`` pairs sequentially.

        Returns:
            Average MSE loss over the selected pairs.

        Example::

            val_loss = trainer.evaluate_loss(graphdef, state, val_dataset)
            print(f"Validation loss: {val_loss:.6f}")
        """
        n_eval: int = min(n_pairs, dataset.n_pairs)

        if key is not None:
            indices: jnp.ndarray = jax.random.choice(
                key, dataset.n_pairs, shape=(n_eval,), replace=False
            )
        else:
            indices = jnp.arange(n_eval)

        inputs: jnp.ndarray
        targets: jnp.ndarray
        inputs, targets = dataset.get_batch(indices)

        loss: jnp.ndarray = self._loss_fn(
            state=state,
            graphdef=graphdef,
            batch=(inputs, targets),
        )

        return float(loss)
