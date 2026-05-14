
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import random
from tqdm import tqdm

from config import Config
from model import MaskedDiffusionModel, MDMConfig
from data import get_dataloader

class MDMTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.device = config.device

        self.mdm_config = MDMConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_layers=config.num_layers,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            max_sequence_length=config.max_sequence_length,
            initializer_range=config.initializer_range,
            layer_norm_eps=config.layer_norm_eps,
            use_learnable_pos_embeddings=config.use_learnable_pos_embeddings,
            pad_token_id=config.mask_token_id
        )
        self.model = MaskedDiffusionModel(self.mdm_config).to(self.device)

        # Optimizer (AdamW)
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=config.learning_rate, 
            betas=(config.beta1, config.beta2), 
            weight_decay=config.weight_decay
        )
        
        # Cosine learning rate schedule
        self.scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=config.num_train_epochs, # This needs to be carefully set based on total steps or epochs
            eta_min=config.min_learning_rate
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=config.mask_token_id) # Only compute loss on unmasked tokens

        # Noise schedule for masking (linear schedule for masking probability)
        # alpha_t decreases from ~1 to ~0, so 1-alpha_t (masking prob) increases from ~0 to ~1
        self.mask_probs = np.linspace(0.0, 1.0, config.num_diffusion_steps)

    def _apply_mask(self, x0: torch.Tensor, t_idx: int):
        """
        Applies masking to x0 based on the noise schedule at a given time index.
        x_t ~ q_{t|0}(. | x_0)
        """
        mask_prob = self.mask_probs[t_idx]
        
        x_t = x0.clone()
        
        # Randomly mask tokens
        # Create a mask for positions to be masked
        probability_matrix = torch.full(x0.shape, mask_prob, device=self.device)
        # Ensure that special tokens (like actual padding, if any, which is not mask_token_id here) are not masked.
        # But for MDM, mask_token_id means it *is* masked.
        
        # This implementation masks *some* tokens. A more faithful implementation of q_t|0 would be:
        # q_t|0(x_t^i | x_0^i) = Cat(alpha_t * e_{x_0^i} + (1-alpha_t) * e_0)
        # meaning, with prob 1-alpha_t, token becomes mask_token_id, otherwise it stays x_0^i.
        
        # Here, we simplify: we randomly select tokens to be masked, and ensure they are not already masked.
        # This mimics a snapshot of x_t at time t.
        
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # Do not mask if the original token is already mask_token_id (e.g., padding in Sudoku)
        # But per paper, mask_token_id is the "mask" token, so it can be changed from x0.
        # It's better to think of x0 as the "clean" data, and then we introduce mask_token_id.
        
        # Important: the labels should be x0, not x_t
        labels = x0.clone() 
        
        # For tokens that become masked in x_t, their original values (from x0) are the targets.
        # We also need to make sure we don't try to predict padding tokens (if padding_idx is used for them)
        
        x_t[masked_indices] = self.config.mask_token_id

        # The loss is computed only for tokens that were originally *not* mask_token_id and *became* mask_token_id in x_t
        # To compute loss over masked tokens, we need to know which ones were masked.
        # The `ignore_index` in CrossEntropyLoss takes care of positions where labels are `mask_token_id`.
        # So we should set labels for unmasked positions in x_t to `mask_token_id` to ignore them.

        # Create target for loss: 
        # For positions that were masked in x_t, target is x0[pos]
        # For positions that were NOT masked in x_t, target is mask_token_id (to be ignored by loss)
        target_for_loss = torch.full_like(x0, self.config.mask_token_id)
        target_for_loss[masked_indices] = labels[masked_indices] # Only predict original tokens at masked positions
        
        return x_t, target_for_loss, masked_indices # x_t is the input to the model, target_for_loss for CE, masked_indices for other uses

    def train_epoch(self, dataloader: DataLoader):
        self.model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc="Training"):
            input_ids = batch["input_ids"].to(self.device)
            # labels are the original clean tokens (x0)
            original_x0 = batch["labels"].to(self.device) 

            # Sample a diffusion timestep t
            t_idx = random.randint(0, self.config.num_diffusion_steps - 1)
            
            # Apply masking to get x_t and the corresponding labels for loss computation
            # The labels here are original x0 values for tokens that were just masked to form x_t
            x_t_input, labels_for_loss, masked_positions_in_xt = self._apply_mask(original_x0, t_idx)

            self.optimizer.zero_grad()
            
            # Model predicts logits for all positions
            logits = self.model(x_t_input) # (batch_size, sequence_length, vocab_size)

            # Reshape for CrossEntropyLoss: (batch_size * sequence_length, vocab_size)
            logits_reshaped = logits.view(-1, logits.size(-1))
            labels_reshaped = labels_for_loss.view(-1)
            
            # Compute loss only for the positions that were actually masked
            loss = self.criterion(logits_reshaped, labels_reshaped)
            
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        self.scheduler.step() # Step the learning rate scheduler
        return total_loss / len(dataloader)

    def train(self, train_dataloader: DataLoader, eval_dataloader: DataLoader = None):
        print("Starting training...")
        for epoch in range(self.config.num_train_epochs):
            avg_train_loss = self.train_epoch(train_dataloader)
            print(f"Epoch {epoch+1}/{self.config.num_train_epochs}, Train Loss: {avg_train_loss:.4f}")

            if eval_dataloader:
                avg_eval_loss = self.evaluate(eval_dataloader)
                print(f"Epoch {epoch+1}/{self.config.num_train_epochs}, Eval Loss: {avg_eval_loss:.4f}")
        print("Training complete.")

    def evaluate(self, dataloader: DataLoader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                original_x0 = batch["labels"].to(self.device)

                # Sample a diffusion timestep t
                t_idx = random.randint(0, self.config.num_diffusion_steps - 1)
                
                x_t_input, labels_for_loss, masked_positions_in_xt = self._apply_mask(original_x0, t_idx)

                logits = self.model(x_t_input)
                logits_reshaped = logits.view(-1, logits.size(-1))
                labels_reshaped = labels_for_loss.view(-1)
                
                loss = self.criterion(logits_reshaped, labels_reshaped)
                total_loss += loss.item()
        return total_loss / len(dataloader)

if __name__ == "__main__":
    # Example usage:
    config = Config()
    
    # Update config for specific experiment, e.g., Sudoku
    config.dataset_name = "Sudoku"
    config.vocab_size = 10 # 0-9
    config.max_sequence_length = 81
    config.num_train_epochs = 10 # Reduced for a quick test

    print(f"Using device: {config.device}")
    print(f"Training on {config.dataset_name} dataset.")

    # Create dataloaders
    train_dataloader = get_dataloader(
        config.dataset_name, 
        config.batch_size, 
        shuffle=True, 
        num_samples=10000
    )
    eval_dataloader = get_dataloader(
        config.dataset_name, 
        config.batch_size, 
        shuffle=False, 
        num_samples=1000
    )

    trainer = MDMTrainer(config)
    trainer.train(train_dataloader, eval_dataloader)

