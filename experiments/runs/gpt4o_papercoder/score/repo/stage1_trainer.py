# stage1_trainer.py

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from typing import Dict, Any
from model import Model
from reward_shaper import RewardShaper
import logging

class Stage1Trainer:
    """
    Stage1Trainer trains the model using reinforcement learning (REINFORCE policy gradient)
    to optimize the second-turn accuracy while regularizing first-turn responses.
    """

    def __init__(self, model: Model, train_dataset: torch.utils.data.Dataset, config: Dict[str, Any]):
        """
        Initializes the Stage1Trainer with the model, dataset, and training configuration.

        Args:
            model (Model): Pre-trained model to be fine-tuned in Stage I.
            train_dataset (torch.utils.data.Dataset): Preprocessed training dataset.
            config (Dict[str, Any]): Configuration dictionary loaded from 'config.yaml'.
        """
        self.model = model
        self.train_dataset = train_dataset
        self.config = config

        # Training hyperparameters
        self.learning_rate: float = config['training']['stage1']['learning_rate']
        self.batch_size: int = config['training']['stage1']['batch_size']
        self.steps: int = config['training']['stage1']['steps']
        self.kl_penalty: float = config['training']['stage1']['kl_penalty']
        self.sampling_temperature: float = config['training']['stage1']['sampling_temperature']

        # Optimizer for training
        self.optimizer = Adam(self.model.model.parameters(), lr=self.learning_rate)

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Reward shaper (used for debugging rewards if needed)
        self.reward_shaper = RewardShaper(config)

        # DataLoader for batching
        self.dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Setting up logging for the trainer
        self.logger = logging.getLogger("Stage1Trainer")
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    def train(self) -> Model:
        """
        Main training loop for Stage I. Performs forward passes, computes KL divergence,
        applies REINFORCE policy updates, and saves the trained model periodically.

        Returns:
            Model: The trained model after Stage I fine-tuning.
        """
        self.model.model.train()  # Set model to training mode
        total_steps = 0  # Keeps track of total training steps

        # Training loop
        for epoch in range(int(self.steps / len(self.dataloader)) + 1):  # Loop over epochs
            for batch in self.dataloader:
                if total_steps >= self.steps:
                    break  # Stop training after specified steps

                # Move batch to correct device
                inputs_1 = batch['input_ids_1'].to(self.device)
                attention_mask_1 = batch['attention_mask_1'].to(self.device)
                inputs_2 = batch['input_ids_2'].to(self.device)
                attention_mask_2 = batch['attention_mask_2'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Stage I: Forward pass for first-turn prediction (y1)
                first_turn_outputs = self.model.model(
                    input_ids=inputs_1, attention_mask=attention_mask_1, labels=labels
                )
                first_turn_logits = first_turn_outputs.logits
                first_turn_loss = first_turn_outputs.loss

                # KL divergence regularization for first-turn responses
                with torch.no_grad():
                    base_model_outputs = self.model.model(
                        input_ids=inputs_1, attention_mask=attention_mask_1, labels=labels
                    )
                base_model_probs = F.softmax(base_model_outputs.logits, dim=-1)
                first_turn_probs = F.softmax(first_turn_logits, dim=-1)

                kl_divergence = F.kl_div(
                    first_turn_probs.log(),  # Model probabilities
                    base_model_probs,  # Base model probabilities
                    reduction='batchmean',
                )

                # Stage I: Generate second-turn responses (y2)
                second_turn_outputs = self.model.model(
                    input_ids=inputs_2, attention_mask=attention_mask_2, labels=labels
                )
                second_turn_logits = second_turn_outputs.logits
                second_turn_loss = second_turn_outputs.loss

                # Reward shaping for REINFORCE training
                rewards = self.reward_shaper.compute_rewards(
                    first_response=first_turn_logits.argmax(dim=-1),
                    second_response=second_turn_logits.argmax(dim=-1),
                    ground_truth=labels,
                )

                # Total loss: Include second-turn reward and KL regularization penalty
                loss = second_turn_loss - self.kl_penalty * kl_divergence

                # Backpropagation and optimization step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Logging metrics
                total_steps += 1
                if total_steps % 100 == 0 or total_steps == 1:  # Log every 100 steps
                    self.logger.info(
                        f"Step {total_steps}/{self.steps}: Loss = {loss.item()}, "
                        f"KL Penalty = {kl_divergence.item()}, Rewards = {rewards.mean().item()}"
                    )

                # Save checkpoints periodically
                if total_steps % 1000 == 0:
                    checkpoint_path = f"stage1_checkpoint_step{total_steps}.pth"
                    self.model.save_model_weights(checkpoint_path)
                    self.logger.info(f"Model checkpoint saved at {checkpoint_path}")

        # Final checkpoint after training
        final_checkpoint_path = "stage1_final_checkpoint.pth"
        self.model.save_model_weights(final_checkpoint_path)
        self.logger.info(f"Final Stage I model checkpoint saved at {final_checkpoint_path}")
        return self.model
