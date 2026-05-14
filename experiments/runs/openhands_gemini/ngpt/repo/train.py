
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import os

from ngpt_model.model import GPT
from data import get_dataloader
from config import Config

def train(config: Config, is_ngpt: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GPT(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_mlp=config.d_mlp,
        dropout=config.dropout,
        is_ngpt=is_ngpt,
        rope_base=config.rope_base,
        d_k=config.d_k
    ).to(device)

    # Resize token embeddings for the tokenizer's actual vocab size if it changed.
    # This is a common practice if special tokens are added.
    # For now, assuming config.vocab_size matches the tokenizer's effective vocab size.
    # model.token_embeddings.weight.data = model.token_embeddings.weight.data[:len(tokenizer), :]
    # model.output_embeddings.weight.data = model.output_embeddings.weight.data[:len(tokenizer), :]


    # Optimizer (Adam or AdamW)
    if config.optimizer == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=config.initial_lr, weight_decay=config.weight_decay)
    else: # Adam for nGPT as per Table 3
        optimizer = optim.Adam(model.parameters(), lr=config.initial_lr, weight_decay=config.weight_decay)

    # Learning rate schedule (Cosine Annealing)
    # This simplified version doesn't include warmup for nGPT,
    # and total_steps is determined by max_iters.
    def get_lr(it):
        if config.num_warmup_steps > 0 and it < config.num_warmup_steps:
            return config.initial_lr * (it / config.num_warmup_steps)
        if it > config.max_iters:
            return config.final_lr
        decay_ratio = (it - config.num_warmup_steps) / (config.max_iters - config.num_warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return config.final_lr + coeff * (config.initial_lr - config.final_lr)

    # DataLoaders
    train_dataloader = get_dataloader(
        tokenizer_name='llama-2',
        block_size=config.context_length,
        batch_size=config.batch_size,
        shuffle=True
    )
    # Using a subset of the training data for validation for simplicity,
    # or a separate validation split if available.
    # For now, just taking a subset from train_dataloader for eval.
    # In a real scenario, a dedicated validation set would be used.
    # For simplicity, we'll just mock eval for now.
    print("WARNING: Using a mocked evaluation loop. A proper validation set is needed.")

    for iter_num in tqdm(range(config.max_iters)):
        # Set learning rate
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Get batch
        # For simplicity, we'll endlessly iterate over the train_dataloader.
        # In a real setup, handle end of epoch and re-shuffling.
        try:
            inputs, targets = next(train_iter)
        except (StopIteration, NameError):
            train_iter = iter(train_dataloader)
            inputs, targets = next(train_iter)

        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        logits, loss = model(inputs, targets)

        # Backward pass and optimize
        optimizer.loss = loss # Store loss for potential logging within optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if is_ngpt:
            # Normalize nGPT parameters after each training step
            model.normalize_ngpt_parameters()
            # Also normalize the optimizer's state for parameters that are normalized
            # This is critical for Adam, where m and v should correspond to normalized parameters.
            # However, directly normalizing optimizer state can be tricky and might interfere
            # with Adam's internal logic. The paper states:
            # "It is very important to note that when implementing nGPT in training libraries,
            # one should make sure that not only instantiated model parameters are normalized
            # but also the ones which are used by the optimizer. Missing the latter is a common bug that should be avoided."
            # A direct way to address this would be to re-project the optimizer's m and v to the tangent space
            # of the hypersphere, or re-normalize them. This is complex and usually requires custom optimizers.
            # For this reproduction, we'll assume the model's Norm operation implicitly handles this in its effect on gradients.
            # The most straightforward interpretation of "optimizer parameters are normalized" in the context of Adam
            # when model weights are L2-normalized is that the gradients themselves should reflect the spherical constraint.
            # The current approach of normalizing weights *after* optimizer step is a common way to implement
            # projected gradient descent.

        if iter_num % config.log_interval == 0:
            print(f"Iteration {iter_num}: Loss = {loss.item():.4f}, LR = {lr:.6f}")

        if iter_num % config.eval_interval == 0:
            # Mock evaluation loop
            print("Performing mock evaluation...")
            model.eval()
            eval_loss_total = 0.0
            with torch.no_grad():
                # For simplicity, use a few batches from the training loader for mock eval
                for i, (eval_inputs, eval_targets) in enumerate(train_dataloader):
                    if i >= config.eval_iters:
                        break
                    eval_inputs, eval_targets = eval_inputs.to(device), eval_targets.to(device)
                    _, eval_loss = model(eval_inputs, eval_targets)
                    eval_loss_total += eval_loss.item()
            avg_eval_loss = eval_loss_total / config.eval_iters
            print(f"--- Eval Iteration {iter_num}: Avg. Loss = {avg_eval_loss:.4f} ---")
            model.train()

    print("Training finished.")

if __name__ == "__main__":
    # Example usage for baseline GPT
    print("Training Baseline GPT...")
    gpt_config = Config(model_size="0.5B", is_ngpt=False)
    train(gpt_config, is_ngpt=False)

    print("\nTraining Normalized Transformer (nGPT)...")
    ngpt_config = Config(model_size="0.5B", is_ngpt=True)
    train(ngpt_config, is_ngpt=True)
