import os
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from tqdm.auto import tqdm
from typing import Optional, Dict, Any, Tuple

# Local imports (assuming these files are in their respective paths)
try:
    from config import Config
    from model.gated_transformer import GatedTransformer
    from data_loader import DataLoader
    from utils import get_lr_scheduler
except ImportError as e:
    print(f"Failed to import local modules: {e}")
    print("Ensure config.py, model/gated_transformer.py, data_loader.py, and utils.py are accessible.")
    # Dummy classes for standalone testing/IDE syntax checking.
    # In a real run, these must be properly imported.
    class Config:
        def __init__(self):
            self.experiment_name = "dummy"
            self.output_dir = "./dummy_experiments"
            self.model = type('ModelConfig', (object,), {
                'type': 'dense',
                'max_seq_len': 4096,
                'vocab_size': 32000,
            })()
            self.moe = type('MoEConfig', (object,), {
                'load_balancing_loss_coeff': 0.01,
            })()
            self.training = type('TrainingConfig', (object,), {
                'total_train_tokens': 1000000.0,
                'max_learning_rate': 1e-4,
                'min_learning_rate': 1e-5,
                'warmup_steps': 100,
                'global_batch_size': 16,
                'gradient_accumulation_steps': 1,
                'optimizer': 'adamw',
                'adam_beta1': 0.9,
                'adam_beta2': 0.999,
                'adam_epsilon': 1e-8,
                'weight_decay': 0.1,
                'mixed_precision': 'no',
                'checkpoint_interval_steps': 1000,
                'eval_interval_steps': 100,
            })()
            self.num_training_steps = 1000 # Placeholder, would be calculated in real config

    class GatedTransformer(nn.Module):
        def __init__(self, config: Config):
            super().__init__()
            self.lm_head = nn.Linear(config.model.vocab_size, config.model.vocab_size)
            self.dummy_param = nn.Parameter(torch.randn(10)) # Ensure model has params

        def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
            # Dummy forward pass for testing Trainer logic
            dummy_logits = torch.randn(input_ids.shape[0], input_ids.shape[1], 32000, device=input_ids.device)
            dummy_loss = None
            if labels is not None:
                dummy_loss = torch.randn(1, device=input_ids.device) # Simulate some loss
            return dummy_logits, dummy_loss, None

    class DataLoader:
        def __init__(self, config: Config):
            self.config = config
            self.max_seq_len = config.model.max_seq_len
            self.global_batch_size = config.training.global_batch_size
            self.effective_batch_size = config.training.global_batch_size // config.training.gradient_accumulation_steps
            print(f"DataLoader dummy: effective_batch_size={self.effective_batch_size}")

        def get_train_dataloader(self) -> torch.utils.data.DataLoader:
            # Dummy DataLoader returning random data
            input_ids = torch.randint(0, 32000, (self.effective_batch_size, self.max_seq_len))
            attention_mask = torch.ones((self.effective_batch_size, self.max_seq_len))
            labels = torch.randint(0, 32000, (self.effective_batch_size, self.max_seq_len))
            return torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(input_ids, attention_mask, labels),
                batch_size=self.effective_batch_size,
                shuffle=True,
            )

        def get_eval_dataloader(self) -> torch.utils.data.DataLoader:
            # Dummy DataLoader returning random data
            input_ids = torch.randint(0, 32000, (self.effective_batch_size, self.max_seq_len))
            attention_mask = torch.ones((self.effective_batch_size, self.max_seq_len))
            labels = torch.randint(0, 32000, (self.effective_batch_size, self.max_seq_len))
            return torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(input_ids, attention_mask, labels),
                batch_size=self.effective_batch_size,
                shuffle=False,
            )

    def get_lr_scheduler(optimizer: optim.Optimizer, config: Config) -> Any:
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0) # Dummy scheduler


class Trainer:
    """
    Manages the entire training process for the Gated Attention LLM.
    Orchestrates initialization, the main training loop, gradient accumulation,
    mixed-precision training, periodic evaluation, and checkpointing.
    """

    def __init__(self, model: GatedTransformer, data_loader: DataLoader, config: Config):
        """
        Initializes the Trainer instance.

        Args:
            model: The GatedTransformer model to be trained.
            data_loader: The DataLoader instance for managing training and evaluation data.
            config: The global configuration object.
        """
        self.model: GatedTransformer = model
        self.data_loader: DataLoader = data_loader
        self.config: Config = config

        # 1. Initialize Accelerator for distributed training and mixed precision
        self.accelerator: Accelerator = Accelerator(
            mixed_precision=self.config.training.mixed_precision,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            log_with="tensorboard", # Or other logging tools
            project_dir=self.config.output_dir
        )

        # 2. Optimizer Setup
        # Using max_learning_rate as initial LR, it will be overridden by scheduler
        self.optimizer: optim.Optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.max_learning_rate,
            betas=(self.config.training.adam_beta1, self.config.training.adam_beta2),
            eps=self.config.training.adam_epsilon,
            weight_decay=self.config.training.weight_decay,
        )

        # 3. Learning Rate Scheduler Setup
        # num_training_steps is pre-calculated in config.py based on total_train_tokens
        num_optimizer_steps = self.config.num_training_steps
        if num_optimizer_steps is None:
             raise ValueError("num_training_steps must be calculated and set in the config.")
        
        self.scheduler: Any = get_lr_scheduler(
            optimizer=self.optimizer,
            config=self.config,
        )

        # 4. Prepare for Distributed Training using Accelerator
        self.train_dataloader = self.data_loader.get_train_dataloader()
        self.eval_dataloader = self.data_loader.get_eval_dataloader()

        self.model, self.optimizer, self.scheduler, self.train_dataloader, self.eval_dataloader = \
            self.accelerator.prepare(
                self.model,
                self.optimizer,
                self.scheduler,
                self.train_dataloader,
                self.eval_dataloader,
            )

        # Create output directory if it doesn't exist
        os.makedirs(self.config.output_dir, exist_ok=True)
        self.accelerator.print(f"Training will save to: {self.config.output_dir}")

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Performs a single forward pass and loss calculation for a given batch.

        Args:
            batch: A dictionary containing 'input_ids', 'attention_mask', and 'labels'.

        Returns:
            A tuple containing:
                - loss: The language modeling loss for the batch.
                - moe_loss: The MoE regularization loss for the batch (or None if not an MoE model).
        """
        self.model.train() # Ensure model is in training mode

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        # Forward pass
        _, lm_loss, moe_loss = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        
        if lm_loss is None:
            raise ValueError("Language modeling loss was not computed. Check model forward method and labels.")

        return lm_loss, moe_loss

    def _eval_ppl(self, dataloader: torch.utils.data.DataLoader) -> float:
        """
        Evaluates the model's perplexity on a given dataset.

        Args:
            dataloader: The DataLoader for the evaluation dataset.

        Returns:
            The computed perplexity (float).
        """
        self.model.eval() # Set model to evaluation mode
        
        total_loss_lm: torch.Tensor = torch.tensor(0.0, device=self.accelerator.device)
        num_batches: int = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating PPL", disable=not self.accelerator.is_main_process):
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]

                # Forward pass - only language modeling loss is typically used for PPL
                _, lm_loss, _ = self.model(input_ids, attention_mask=attention_mask, labels=labels)

                if lm_loss is not None:
                    total_loss_lm += lm_loss # Sum loss for later averaging
                num_batches += 1

        # Gather losses from all processes and compute mean
        all_lm_losses = self.accelerator.gather(total_loss_lm).sum()
        all_num_batches = self.accelerator.gather(torch.tensor(num_batches, device=self.accelerator.device)).sum()

        avg_lm_loss = all_lm_losses / all_num_batches
        ppl = torch.exp(avg_lm_loss)

        self.model.train() # Restore model to training mode
        return ppl.item()

    def _save_checkpoint(self, step: int) -> None:
        """
        Saves the current state of the model, optimizer, and scheduler.

        Args:
            step: The current training step, used for naming the checkpoint directory.
        """
        output_dir = os.path.join(self.config.output_dir, f"checkpoint_{step}")
        os.makedirs(output_dir, exist_ok=True)

        self.accelerator.wait_for_everyone() # Ensure all processes are ready for saving
        self.accelerator.save_state(output_dir)
        
        # Save unwrapped model weights for easier loading later (e.g., for inference)
        # This is typically done only by the main process
        if self.accelerator.is_main_process:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            torch.save(unwrapped_model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
            self.accelerator.print(f"Checkpoint saved to {output_dir}")

    def train(self) -> None:
        """
        Executes the main training loop, including periodic evaluation and checkpointing.
        """
        # num_training_steps is pre-calculated in config.py
        num_optimizer_steps = self.config.num_training_steps
        if num_optimizer_steps is None:
            raise ValueError("num_training_steps must be calculated and set in the config before training.")

        optimizer_step_idx: int = 0
        progress_bar = tqdm(
            range(num_optimizer_steps),
            desc=f"Training {self.config.experiment_name}",
            disable=not self.accelerator.is_main_process,
        )

        # Loop indefinitely until `optimizer_step_idx` reaches `num_optimizer_steps`
        # We use an infinite loop and break explicitly to handle dataloader exhaustion
        # while respecting gradient accumulation steps and total training steps.
        
        # Use an iterable for the dataloader to restart it if it exhausts before
        # the total number of steps is reached (e.g., small dataset, many steps).
        train_dataloader_iter = iter(self.train_dataloader)

        while optimizer_step_idx < num_optimizer_steps:
            try:
                batch = next(train_dataloader_iter)
            except StopIteration:
                self.accelerator.print("Train dataloader exhausted, restarting iterator.")
                train_dataloader_iter = iter(self.train_dataloader)
                batch = next(train_dataloader_iter) # Get the first batch from the new epoch

            # Gradient accumulation context
            with self.accelerator.accumulate(self.model):
                lm_loss, moe_loss = self._train_step(batch)
                
                # Total loss calculation
                total_loss = lm_loss
                if self.config.model.type == "moe" and moe_loss is not None:
                    total_loss += self.config.moe.load_balancing_loss_coeff * moe_loss

                # Backward pass
                self.accelerator.backward(total_loss)

                # Check if it's time to sync gradients and update parameters
                if self.accelerator.sync_gradients:
                    # Gradient clipping (standard practice for LLM stability)
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad() # Clear gradients after update

                    optimizer_step_idx += 1 # Increment optimizer step counter

                    # Log training metrics
                    current_lr = self.scheduler.get_last_lr()[0]
                    self.accelerator.log({
                        "train_loss": total_loss.item(),
                        "lr": current_lr,
                        "lm_loss": lm_loss.item(),
                        "moe_loss": moe_loss.item() if moe_loss is not None else 0.0,
                    }, step=optimizer_step_idx)
                    
                    progress_bar.update(1)
                    progress_bar.set_postfix({"loss": f"{total_loss.item():.4f}", "lr": f"{current_lr:.1e}"})

                    # Periodic evaluation
                    if (self.config.training.eval_interval_steps > 0 and 
                        optimizer_step_idx % self.config.training.eval_interval_steps == 0):
                        
                        self.accelerator.print(f"Step {optimizer_step_idx}: Starting evaluation...")
                        eval_ppl = self._eval_ppl(self.eval_dataloader)
                        self.accelerator.log({"eval_ppl": eval_ppl}, step=optimizer_step_idx)
                        self.accelerator.print(f"Step {optimizer_step_idx}: Evaluation PPL = {eval_ppl:.4f}")

                    # Checkpointing
                    if (self.config.training.checkpoint_interval_steps > 0 and
                        optimizer_step_idx % self.config.training.checkpoint_interval_steps == 0):
                        self.accelerator.print(f"Step {optimizer_step_idx}: Saving checkpoint...")
                        self._save_checkpoint(optimizer_step_idx)

            # Break if target number of optimizer steps reached
            if optimizer_step_idx >= num_optimizer_steps:
                break

        self.accelerator.print("Training complete.")
        # Final evaluation and checkpoint
        self.accelerator.wait_for_everyone()
        final_eval_ppl = self._eval_ppl(self.eval_dataloader)
        self.accelerator.log({"final_eval_ppl": final_eval_ppl}, step=optimizer_step_idx)
        self.accelerator.print(f"Final evaluation PPL = {final_eval_ppl:.4f}")
        self._save_checkpoint("final")
        progress_bar.close()

