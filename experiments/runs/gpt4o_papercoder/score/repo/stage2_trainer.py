# stage2_trainer.py

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple
from model import Model
from reward_shaper import RewardShaper
import logging


class Stage2Trainer:
    """
    Implements Stage II of the SCoRe framework, where the model is trained using reinforcement learning
    with reward shaping to optimize self-correction behavior across turns.
    """

    def __init__(self, model: Model, train_dataset: torch.utils.data.Dataset, config: Dict[str, Any]):
        """
        Initializes the Stage2Trainer with model, data, and configuration.

        Args:
            model (Model): Pre-trained model initialized from Stage I.
            train_dataset (torch.utils.data.Dataset): Preprocessed training dataset.
            config (dict): Configuration parameters from config.yaml.
        """
        self.model = model
        self.train_dataset = train_dataset
        self.config = config

        # Hyperparameters for training
        self.learning_rate: float = config['training']['stage2']['learning_rate']
        self.batch_size: int = config['training']['stage2']['batch_size']
        self.total_steps: int = config['training']['stage2']['steps']
        self.kl_penalty: float = config['training']['stage2']['kl_penalty']
        self.reward_shaping_alpha: float = config['training']['stage2']['reward_shaping_alpha']
        self.sampling_temperature: float = config['training']['stage2']['sampling_temperature']

        # Optimizer
        self.optimizer = Adam(self.model.model.parameters(), lr=self.learning_rate)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Reward shaper
        self.reward_shaper = RewardShaper(config)

        # DataLoader
        self.dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

        # Logging setup
        self.logger = logging.getLogger("Stage2Trainer")
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    def train(self) -> Model:
        """
        Train the model using multi-turn RL with reward shaping.

        Returns:
            Model: The trained model after completing Stage II.
        """
        self.model.model.train()  # Set model to training mode
        total_steps_completed = 0

        # Main training loop
        for epoch in range(int(self.total_steps / len(self.dataloader)) + 1):
            for batch in self.dataloader:
                if total_steps_completed >= self.total_steps:
                    break

                # Move data to the correct device
                inputs_1, attention_mask_1, inputs_2, attention_mask_2, labels = self._move_to_device(batch)

                # Forward pass for first-turn responses
                first_outputs = self.model.model(
                    input_ids=inputs_1,
                    attention_mask=attention_mask_1,
                    labels=labels
                )
                first_logits = first_outputs.logits
                first_loss = first_outputs.loss

                # KL-divergence regularization for first-turn responses
                kl_loss_1 = self._compute_kl_penalty(inputs_1, attention_mask_1, first_logits)

                # Forward pass for second-turn responses
                second_outputs = self.model.model(
                    input_ids=inputs_2,
                    attention_mask=attention_mask_2,
                    labels=labels
                )
                second_logits = second_outputs.logits
                second_loss = second_outputs.loss

                # Compute RL reward signal with reward shaping
                shaped_rewards = self.reward_shaper.compute_rewards(
                    first_response=first_logits.argmax(dim=-1),
                    second_response=second_logits.argmax(dim=-1),
                    ground_truth=labels
                )

                # Combine losses for Stage II
                total_loss = -torch.mean(shaped_rewards) + self.kl_penalty * (kl_loss_1 + second_loss)

                # Backpropagation and optimization
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                # Logging
                total_steps_completed += 1
                if total_steps_completed % 100 == 0 or total_steps_completed == 1:
                    self.logger.info(
                        f"Step {total_steps_completed}/{self.total_steps}: "
                        f"Loss = {total_loss.item():.4f}, KL Loss = {kl_loss_1.item():.4f}, "
                        f"Shaped Rewards Mean = {torch.mean(shaped_rewards).item():.4f}"
                    )

                # Save model checkpoints periodically
                if total_steps_completed % 1000 == 0:
                    checkpoint_path = f"stage2_checkpoint_step{total_steps_completed}.pth"
                    self.model.save_model_weights(checkpoint_path)
                    self.logger.info(f"Checkpoint saved: {checkpoint_path}")

        # Final model save
        final_checkpoint_path = "stage2_final_checkpoint.pth"
        self.model.save_model_weights(final_checkpoint_path)
        self.logger.info(f"Final Stage II model checkpoint saved: {final_checkpoint_path}")
        return self.model

    def _move_to_device(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        """
        Moves a batch of data to the appropriate computation device.

        Args:
            batch (dict): Batch of input data containing tokenized tensors.

        Returns:
            Tuple[torch.Tensor, ...]: Batch components on the correct device.
        """
        inputs_1 = batch['input_ids_1'].to(self.device)
        attention_mask_1 = batch['attention_mask_1'].to(self.device)
        inputs_2 = batch['input_ids_2'].to(self.device)
        attention_mask_2 = batch['attention_mask_2'].to(self.device)
        labels = batch['labels'].to(self.device)
        return inputs_1, attention_mask_1, inputs_2, attention_mask_2, labels

    def _compute_kl_penalty(
        self, inputs: torch.Tensor, attention_mask: torch.Tensor, logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes KL-divergence penalty between model output and the reference model output.

        Args:
            inputs (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask for inputs.
            logits (torch.Tensor): Logits from the current model.

        Returns:
            torch.Tensor: KL divergence as a scalar tensor.
        """
        with torch.no_grad():
            # Reference (base model) outputs
            ref_outputs = self.model.model(input_ids=inputs, attention_mask=attention_mask)
            ref_probs = F.softmax(ref_outputs.logits, dim=-1)

        # Current (policy) output probabilities
        current_probs = F.softmax(logits, dim=-1)

        # Compute KL divergence
        kl_divergence = F.kl_div(
            current_probs.log(),
            ref_probs,
            reduction='batchmean'
        )
        return kl_divergence
