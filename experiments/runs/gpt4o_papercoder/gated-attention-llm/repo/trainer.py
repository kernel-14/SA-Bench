# trainer.py

import os
from typing import Any, Dict
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler
from tqdm import tqdm
from model import TransformerModel


class Trainer:
    """
    Trainer class to manage the training of the Transformer model with
    gated attention mechanisms. Handles forward/backward passes, optimization,
    learning rate schedules, model checkpoints, and logging.
    """

    def __init__(self, model: TransformerModel, data: DataLoader, config: Dict[str, Any]):
        """
        Initialize the Trainer instance with model, dataset, and training config.

        Args:
            model (TransformerModel): Transformer model to be trained.
            data (DataLoader): Training dataset loader.
            config (dict): Configuration dictionary from `config.yaml`.
        """
        self.model = model
        self.data = data
        self.config = config

        # Training configurations from YAML
        self.epochs = self.config["training"].get("epochs", 10)
        self.batch_size = self.config["training"].get("batch_size", 1024)
        self.learning_rate = self.config["training"].get("learning_rate", 0.002)
        self.min_learning_rate = self.config["training"].get("min_learning_rate", 0.00003)
        self.warmup_steps = self.config["training"].get("warmup_steps", 1000)
        self.total_steps = self.config["training"].get("total_steps", 100000)
        self.gradient_clipping = self.config["optimization"].get("gradient_clipping", 1.0)

        # Logging directory and checkpoint settings
        self.output_dir = "./outputs"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Optimizer and scheduler
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        self.scheduler = get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps,
        )

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()  # Compatible with language modeling tasks

    def train(self) -> None:
        """
        Execute the training loop, iterating over epochs and batches.
        Logs metrics, applies gradient updates, and saves model checkpoints.
        """
        print(f"Starting training for {self.epochs} epochs.")
        self.model.train()  # Put model in training mode

        total_batches = len(self.data)
        global_step = 0

        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            print(f"Epoch {epoch}/{self.epochs}")

            tqdm_bar = tqdm(self.data, desc=f"Training Epoch {epoch}")
            for step, batch in enumerate(tqdm_bar, start=1):
                # Move data to the appropriate device
                input_ids = batch["input_ids"].to(self._get_device())
                attention_mask = batch["attention_mask"].to(self._get_device())
                labels = batch.get("labels", input_ids).to(self._get_device())

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(input_ids, attention_mask=attention_mask)
                loss = self.loss_fn(outputs.view(-1, self.model.hidden_size), labels.view(-1))

                # Backward pass and optimizer step
                loss.backward()

                # Gradient clipping
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
                self.optimizer.step()
                self.scheduler.step()

                # Update loss
                epoch_loss += loss.item()
                global_step += 1

                # Log progress to tqdm
                tqdm_bar.set_postfix(
                    {
                        "batch_loss": loss.item(),
                        "epoch_avg_loss": epoch_loss / step,
                        "lr": self.scheduler.get_last_lr()[0],
                    }
                )

                # Save checkpoint every N steps
                if global_step % 5000 == 0:
                    self.save_model(os.path.join(self.output_dir, f"checkpoint_step_{global_step}.pt"))

            # Epoch-wise logging
            print(f"Epoch {epoch} completed. Avg Loss: {epoch_loss / total_batches:.4f}")

            # Save epoch checkpoint
            self.save_model(os.path.join(self.output_dir, f"model_epoch_{epoch}.pt"))

        print("Training complete! Model ready for evaluation.")

    def save_model(self, output_path: str) -> None:
        """
        Save the model and optimizer state to a checkpoint file.

        Args:
            output_path (str): Path to save the model checkpoint.
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "config": self.config,
            },
            output_path,
        )
        print(f"Model checkpoint saved to {output_path}")

    def _get_device(self) -> torch.device:
        """
        Determine whether to use GPU or CPU for training.

        Returns:
            torch.device: The device ('cuda' or 'cpu') to use.
        """
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
