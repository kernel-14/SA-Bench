
import jax
import jax.numpy as jnp
import optax
import flax
from flax.training import train_state
from flax.core import FrozenDict
import ml_collections
from tqdm import tqdm
import os

from luno.models.fno import FNO
from luno.data.pde_datasets import PDEDataset
from luno.utils.metrics import rmse

class TrainState(train_state.TrainState):
    # A simple TrainState that includes the optimizer
    pass

def create_train_state(rng, config: ml_collections.ConfigDict, model: FNO, dummy_input: jnp.ndarray) -> TrainState:
    """Creates initial `TrainState`."""
    params = model.init(rng, dummy_input)['params']
    
    # Optimizer setup (AdamW with cosine decay and warmup)
    learning_rate_scheduler = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
        decay_steps=config.training.cosine_decay_steps,
        end_value=0.0,
    )
    optimizer = optax.adamw(
        learning_rate=learning_rate_scheduler,
        weight_decay=config.training.weight_decay
    )
    return TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

def train_step(state: TrainState, batch: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple[TrainState, jnp.ndarray]:
    """Performs a single training step."""
    inputs, targets = batch

    def loss_fn(params):
        predictions = state.apply_fn({'params': params}, inputs)
        loss = jnp.mean(jnp.square(predictions - targets))
        return loss, predictions

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, predictions), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

def evaluate_step(state: TrainState, batch: Tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
    """Evaluates the model on a batch."""
    inputs, targets = batch
    predictions = state.apply_fn({'params': state.params}, inputs)
    loss = jnp.mean(jnp.square(predictions - targets))
    return loss

def train_model(config: ml_collections.ConfigDict):
    """Trains the FNO model."""
    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng = jax.random.split(rng)

    # Load datasets
    train_dataset = PDEDataset(config, 'train')
    val_dataset = PDEDataset(config, 'validation')

    # Determine input channels for FNO based on data and positional encoding
    # Input shape to FNO: (batch, spatial_res, num_initial_steps * input_channels (+ 1 for pos_encoding))
    # Assuming the first element of train_dataset is representative
    dummy_input, _ = train_dataset[0]
    # dummy_input shape: (num_initial_steps, spatial_res, input_channels)
    # We need to flatten num_initial_steps and input_channels for FNO input
    input_features = dummy_input.shape[-1] * dummy_input.shape[0]
    if config.model.add_pos_encoding:
        input_features += 1 # For positional encoding

    # FNO model initialization
    fno_model = FNO(
        modes=config.model.modes,
        hidden_dim=config.model.hidden_dim,
        num_fourier_blocks=config.model.num_fourier_blocks,
        output_dim=config.model.output_dim,
        add_pos_encoding=config.model.add_pos_encoding
    )
    
    # Dummy input for model initialization
    # (batch_size, spatial_res, input_features)
    dummy_fno_input = jnp.zeros(
        (config.training.batch_size, config.data.spatial_resolution, input_features)
    )

    # Calculate total steps for learning rate schedule
    total_train_steps = (len(train_dataset) // config.training.batch_size) * \
                        (config.training.epochs_ood if not config.data.low_data_regime else config.training.epochs)
    config.training.cosine_decay_steps = total_train_steps
    config.training.warmup_steps = int(0.1 * total_train_steps) # 10% warmup

    state = create_train_state(init_rng, config, fno_model, dummy_fno_input)

    best_val_loss = float('inf')
    
    epochs = config.training.epochs_ood if not config.data.low_data_regime else config.training.epochs

    for epoch in range(epochs):
        # Training loop
        train_dataloader = train_dataset.get_dataloader(config.training.batch_size)
        train_loss_sum = 0.0
        num_train_batches = 0
        for batch_inputs, batch_targets in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} Training"):
            # Reshape inputs: (batch, num_initial_steps, spatial_res, input_channels)
            # -> (batch, spatial_res, num_initial_steps * input_channels)
            batch_inputs_reshaped = batch_inputs.transpose(0, 2, 1, 3).reshape(
                batch_inputs.shape[0], batch_inputs.shape[2], -1
            )
            state, loss = train_step(state, (batch_inputs_reshaped, batch_targets))
            train_loss_sum += loss
            num_train_batches += 1
        avg_train_loss = train_loss_sum / num_train_batches

        # Validation loop
        val_dataloader = val_dataset.get_dataloader(config.training.batch_size)
        val_loss_sum = 0.0
        num_val_batches = 0
        for batch_inputs, batch_targets in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs} Validation"):
            batch_inputs_reshaped = batch_inputs.transpose(0, 2, 1, 3).reshape(
                batch_inputs.shape[0], batch_inputs.shape[2], -1
            )
            loss = evaluate_step(state, (batch_inputs_reshaped, batch_targets))
            val_loss_sum += loss
            num_val_batches += 1
        avg_val_loss = val_loss_sum / num_val_batches

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_dir = os.path.join(config.save_dir, config.pde_name)
            os.makedirs(checkpoint_dir, exist_ok=True)
            with open(os.path.join(checkpoint_dir, 'best_fno_params.msgpack'), 'wb') as f:
                f.write(flax.serialization.to_bytes(state.params))
            print(f"Saved best model with Val Loss: {best_val_loss:.4f}")

    return state.params

if __name__ == '__main__':
    from luno.config import get_config
    config = get_config()

    # Adjust config for a quick test run
    config.data.pde_name = 'Burgers'
    config.data.low_data_regime = True
    config.data.n_train_low_data = 2
    config.data.n_val = 1
    config.data.spatial_resolution = 32
    config.data.temporal_resolution = 15
    config.data.num_initial_steps = 2
    config.training.epochs = 2
    config.training.batch_size = 1

    trained_params = train_model(config)
    print("Training complete.")

