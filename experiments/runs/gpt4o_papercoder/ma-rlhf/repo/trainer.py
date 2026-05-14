# trainer.py

import os
import yaml
import torch
from typing import Dict, List
from torch.utils.tensorboard import SummaryWriter
from termination import Termination
from model import Model
from ppo import PPO
from dataset_loader import DatasetLoader


class Trainer:
    def __init__(self, model: Model, dataset: DatasetLoader, termination: Termination, optimizer: PPO, config: Dict):
        """
        Initialize the Trainer with all dependencies.

        Args:
            model (Model): Gemma-based pre-trained model instance.
            dataset (DatasetLoader): DatasetLoader instance for loading and processing data.
            termination (Termination): Termination handler for macro-action generation.
            optimizer (PPO): PPO-based optimizer for MA-RLHF training.
            config (Dict): Configuration dictionary loaded from 'config.yaml'.
        """
        self.model = model
        self.dataset = dataset
        self.termination = termination
        self.optimizer = optimizer
        self.config = config

        # Logging setup
        self.log_dir = config.get("training", {}).get("log_dir", "./logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # Checkpoint setup
        self.checkpoint_dir = config.get("training", {}).get("checkpoint_dir", "./checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Training parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = config.get("training", {}).get("batch_size", 32)
        self.num_epochs = config.get("training", {}).get("num_epochs", 3)

        # Task-specific configuration
        self.task_type = config.get("training", {}).get("task_type", "tl_dr")

    def train(self) -> None:
        """
        Manage the complete training pipeline: SFT stage, RM stage, and MA-PPO optimization.
        """
        print("Starting Training Pipeline.")
        # Stage 1: Supervised Fine-Tuning
        self.run_supervised_finetuning()

        # Stage 2: Reward Modeling
        self.run_reward_modeling()

        # Stage 3: MA-PPO Optimization
        self.run_ppo_training()

    def run_supervised_finetuning(self) -> None:
        """
        Supervised Fine-Tuning stage: Training the model on task-specific input-output pairs.
        """
        print("Executing Supervised Fine-Tuning...")
        raw_train_data, metadata = self.dataset.load_data()
        sft_data = self.dataset.format_supervised_finetuning(raw_train_data)

        dataloader = self.dataset.batchify(sft_data, self.batch_size)

        optimizer = torch.optim.Adam(self.model.trainable_parameters(), lr=self.config.get("training", {}).get("supervised_finetuning_lr", 5e-5))

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                output_ids = batch["output_ids"].to(self.device)

                self.model.model.train()  # Set model to training mode
                outputs = self.model.forward(input_ids)
                
                loss_fn = torch.nn.CrossEntropyLoss()
                loss = loss_fn(outputs.view(-1, outputs.size(-1)), output_ids.view(-1))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            average_loss = epoch_loss / len(dataloader)
            self.log_metrics(metric_name="sft_loss", value=average_loss, step=epoch)
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {average_loss:.4f}")
        
        self.save_checkpoint(stage="sft")

    def run_reward_modeling(self) -> None:
        """
        Reward Modeling stage: Training the reward model to rank preference pairs.
        """
        print("Executing Reward Modeling...")
        raw_train_data, metadata = self.dataset.load_data()
        rm_data = self.dataset.format_reward_model(raw_train_data)

        dataloader = self.dataset.batchify(rm_data, self.batch_size)

        optimizer = torch.optim.Adam(self.model.trainable_parameters(), lr=self.config.get("training", {}).get("reward_modeling_lr", 1e-6))

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                prompt_ids = batch["prompt_ids"].to(self.device)
                chosen_token_ids = batch["chosen_token_ids"].to(self.device)
                rejected_token_ids = batch["rejected_token_ids"].to(self.device)

                self.model.model.train()  # Set model to training mode
                chosen_logits = self.model.predict_reward(prompt_ids)
                rejected_logits = self.model.predict_reward(prompt_ids)

                criterion = torch.nn.CrossEntropyLoss()
                loss = -criterion(chosen_logits, rejected_logits)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
            
            average_loss = epoch_loss / len(dataloader)
            self.log_metrics(metric_name="rm_loss", value=average_loss, step=epoch)
            print(f"Epoch {epoch + 1}/{self.num_epochs}, RM Loss: {average_loss:.4f}")
        
        self.save_checkpoint(stage="rm")

    def run_ppo_training(self) -> None:
        """
        PPO stage: Optimize policy using macro-action termination strategies and MA-PPO loss.
        """
        print("Executing PPO Training...")
        raw_train_data, metadata = self.dataset.load_data()
        rlhf_data = self.dataset.format_rlhf(raw_train_data)
        dataloader = self.dataset.batchify(rlhf_data, self.batch_size)

        for epoch in range(self.num_epochs):
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0

            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                rewards = torch.tensor([batch["output_ids"]], device=self.device)

                macro_actions = self.termination.get_macro_actions(input_ids.tolist())
                
                policy_loss, value_loss = self.optimizer.compute_loss(batch, macro_actions)
                self.optimizer.optimize(policy_loss, value_loss)
                
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
            
            average_policy_loss = epoch_policy_loss / len(dataloader)
            average_value_loss = epoch_value_loss / len(dataloader)
            self.log_metrics(metric_name="policy_loss", value=average_policy_loss, step=epoch)
            self.log_metrics(metric_name="value_loss", value=average_value_loss, step=epoch)
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Policy Loss: {average_policy_loss:.4f}, Value Loss: {average_value_loss:.4f}")
        
        self.save_checkpoint(stage="ppo")

    def save_checkpoint(self, stage: str) -> None:
        """
        Save model and optimizer state during different stages of training.

        Args:
            stage (str): Current stage of training (e.g., "sft", "rm", "ppo").
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{stage}_checkpoint.pth")
        torch.save({
            "model_state_dict": self.model.model.state_dict(),
            "optimizer_state_dict": self.optimizer.optimizer_policy.state_dict(),
        }, checkpoint_path)
        print(f"Checkpoint saved for stage: {stage}")

    def load_checkpoint(self, stage: str) -> None:
        """
        Load saved checkpoint for resuming training.

        Args:
            stage (str): Stage-specific checkpoint to load (e.g., "sft", "rm", "ppo").
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{stage}_checkpoint.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.model.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.optimizer_policy.load_state_dict(checkpoint["optimizer_state_dict"])
            print(f"Checkpoint loaded for stage: {stage}")
        else:
            print(f"No checkpoint found for stage: {stage}")

    def log_metrics(self, metric_name: str, value: float, step: int) -> None:
        """
        Log training metrics to TensorBoard.

        Args:
            metric_name (str): Name of the metric being logged.
            value (float): Value of the metric.
            step (int): Current training step/epoch.
        """
        self.writer.add_scalar(metric_name, value, step)

