
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_scheduler
from accelerate import Accelerator
import os
import math
import tqdm

from config import current_model_config as ModelConfig
from config import current_training_config as TrainingConfig
from data import get_dataloader, load_tokenizer
from model import get_model

def calculate_perplexity(loss: float) -> float:
    """Calculates perplexity from the given loss."""
    return math.exp(loss)

def train():
    # 1. Initialize Accelerator
    accelerator = Accelerator(mixed_precision="bf16" if TrainingConfig.use_bf16 else "no")

    accelerator.print("Loading configurations...")
    model_config = ModelConfig()
    training_config = TrainingConfig()

    accelerator.print(f"Using device: {accelerator.device}")

    # 2. Load tokenizer
    tokenizer = load_tokenizer(training_config.model_name_or_path)
    # Add a pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': training_config.pad_token})
        # Note: model.resize_token_embeddings is called after model initialization if needed

    # 3. Get data loaders
    # For a real scenario, this would load data from a large corpus.
    # Here we use dummy data.
    accelerator.print("Loading data...")
    train_dataloader = get_dataloader(
        tokenizer=tokenizer,
        seq_len=training_config.context_length,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_samples=10000 # Placeholder for a small dummy dataset
    )
    eval_dataloader = get_dataloader(
        tokenizer=tokenizer,
        seq_len=training_config.context_length,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_samples=1000 # Placeholder for a small dummy dataset
    )

    # 4. Get model
    accelerator.print("Initializing model...")
    model = get_model(model_config, vocab_size=len(tokenizer))

    if model.vocab_size != len(tokenizer):
        accelerator.print(f"Resizing model embeddings to {len(tokenizer)} from {model.vocab_size}")
        model.resize_token_embeddings(len(tokenizer))
        model.vocab_size = len(tokenizer) # Update vocab_size in model config


    # 5. Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay
    )

    lr_scheduler = get_scheduler(
        name=training_config.lr_decay_strategy,
        optimizer=optimizer,
        num_warmup_steps=training_config.lr_warmup_steps,
        num_training_steps=training_config.total_train_steps,
    )

    # 6. Prepare everything for accelerator
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # Loss function for language modeling
    # The output of the model is logits for the next token.
    # We want to predict the next token, so we shift labels by one.
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    accelerator.print("Starting training...")
    model.train()
    completed_steps = 0
    for epoch in range(1): # Paper typically trains for a fixed number of steps, not epochs
        for step, batch in enumerate(train_dataloader):
            if completed_steps >= training_config.total_train_steps:
                break

            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']

            # Shift tokens to create labels for language modeling
            labels = input_ids.clone()
            # For causal language modeling, the model predicts the next token.
            # So, input is [t0, t1, ..., tn-1] and labels are [t1, t2, ..., tn].
            # We can achieve this by shifting the labels by one position.
            input_ids = input_ids[:, :-1]
            labels = labels[:, 1:]

            # Ensure attention mask is adjusted for the shifted input_ids
            # For simplicity, we'll recreate a causal mask within the model
            # or handle it if attention_mask is for padding only.
            # The model's forward pass handles causal masking if attention_mask is None.
            # If attention_mask is only for padding, we might need to adjust it for shifted labels.
            # For simplicity, let's assume the model handles causal masking based on input_ids length.
            # If attention_mask needs to be passed, it should be adapted for the input_ids length.

            # Forward pass
            if model_config.model_type == "moe":
                outputs, total_router_z_loss = model(input_ids, attention_mask=None)
            else:
                outputs = model(input_ids, attention_mask=None)
                total_router_z_loss = torch.tensor(0.0, device=outputs.device) # No Z-loss for dense models

            # Calculate loss
            # outputs are (batch_size, seq_len-1, vocab_size)
            # labels are (batch_size, seq_len-1)
            lm_loss = loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            
            # Combine language modeling loss and MoE Z-loss
            loss = lm_loss + total_router_z_loss

            # Backward pass
            accelerator.backward(loss)

            # Gradient clipping (not explicitly mentioned but often used)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0) # Placeholder value

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            completed_steps += 1

            if completed_steps % training_config.eval_interval_steps == 0:
                accelerator.print(f"Step {completed_steps}/{training_config.total_train_steps} - Loss: {loss.item():.4f}")
                
                # Evaluation
                model.eval()
                eval_losses = []
                for eval_step, eval_batch in enumerate(eval_dataloader):
                    with torch.no_grad():
                        eval_input_ids = eval_batch['input_ids']
                        eval_labels = eval_input_ids.clone()
                        eval_input_ids = eval_input_ids[:, :-1]
                        eval_labels = eval_labels[:, 1:]

                        if model_config.model_type == "moe":
                            eval_outputs, _ = model(eval_input_ids, attention_mask=None) # Discard Z-loss during eval
                        else:
                            eval_outputs = model(eval_input_ids, attention_mask=None)
                        
                        eval_loss = loss_fn(eval_outputs.view(-1, eval_outputs.size(-1)), eval_labels.view(-1))
                        eval_losses.append(eval_loss.item())
                
                avg_eval_loss = sum(eval_losses) / len(eval_losses)
                eval_ppl = calculate_perplexity(avg_eval_loss)
                accelerator.print(f"Evaluation PPL: {eval_ppl:.2f}")
                model.train()
            
            if completed_steps % training_config.save_interval_steps == 0:
                output_dir = f"checkpoint-{completed_steps}"
                if accelerator.is_main_process:
                    os.makedirs(output_dir, exist_ok=True)
                    accelerator.save_state(output_dir)
                    accelerator.print(f"Saved checkpoint to {output_dir}")

            if completed_steps >= training_config.total_train_steps:
                break

    accelerator.print("Training complete!")
    if accelerator.is_main_process:
        accelerator.save_state("final_model")
        accelerator.print("Saved final model.")

if __name__ == "__main__":
    train()
