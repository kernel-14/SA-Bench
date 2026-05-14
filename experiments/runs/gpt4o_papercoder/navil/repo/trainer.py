"""
trainer.py: Implements the Trainer class responsible for executing pre-training and fine-tuning workflows using PyTorch Lightning.

Dependencies:
- torch: For optimization and tensor operations.
- pytorch_lightning: For scalable, distributed training workflows.
- model: Provides the NaViL model architecture combining VisualEncoder and LLM.
- dataset_loader: Facilitates loading pretraining and fine-tuning datasets.
- config: Supplies training configurations from the YAML file.
- utils: Provides shared utility functions such as logging metrics.

Design:
- Integrates tightly with the modular components in the NaViL system.
- Ensures reproducibility by adhering to the paper's methodology and the provided config.yaml directives.
"""

import torch
from typing import Dict
from pytorch_lightning import Trainer as LightningTrainer, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from model import Model
from dataset_loader import DatasetLoader
from utils import log_metrics  # Utility function for custom logging.

class Trainer:
    """
    Trainer class for managing the training and fine-tuning workflows of NaViL models.
    
    Attributes:
        config (dict): Parsed configuration dictionary from config.yaml.
        model (Model): The NaViL model integrating VisualEncoder and LLM components.
        data_loader (DatasetLoader): Facilitates dataset loading based on training phase.
        precision (str): Numerical precision format fetched from hardware configuration.
        gpus (int): Number of GPUs allocated for training.
        gradient_accumulation (int): Number of gradient accumulation steps per update.
        lightning_trainer (LightningTrainer): PyTorch Lightning trainer instance managing training loops.
    """
    
    def __init__(self, config: Dict, model: Model, data_loader: DatasetLoader):
        """
        Initializes the Trainer class with configurations, datasets, and model setup.

        Args:
            config (dict): Configuration dictionary for the training pipeline.
            model (Model): NaViL model, combining VisualEncoder and LLM.
            data_loader (DatasetLoader): DatasetLoader instance for managing phase-specific dataset loading.
        """
        self.config = config
        self.model = model
        self.data_loader = data_loader
        self.precision = config["hardware"]["precision"]
        self.gpus = config["hardware"]["num_gpus"]
        self.gradient_accumulation = config["hardware"]["gradient_accumulation"]

        # Initialize Lightning Trainer
        self.lightning_trainer = LightningTrainer(
            precision=self.precision,
            accelerator="gpu",
            devices=self.gpus,
            strategy="ddp",  # Distributed Data Parallel for multi-node training
            accumulate_grad_batches=self.gradient_accumulation,
            log_every_n_steps=config["logging"]["log_interval"],
        )
        self.optimizer = None
        self.lr_scheduler = None

        # Define checkpoint mechanism
        checkpoint_cb = ModelCheckpoint(
            save_top_k=1,
            monitor="validation_loss",
            mode="min",
            dirpath="./checkpoints/",
            filename="{phase}_{epoch}-{step}"
        )
        self.lightning_trainer.callbacks.append(checkpoint_cb)

    def train_pretraining_stage(self):
        """
        Executes the pre-training workflow, comprising two sub-phases:
        Phase 1: Training with noisy datasets (self-supervised learning).
        Phase 2: Refining multimodal alignment with high-quality datasets.
        """
        # Sub-phase 1: Pre-training with noisy datasets
        print("Starting Pre-training Phase 1...")
        self._setup_optimizer_and_scheduler("pretraining:phase_1")
        pretrain_data = self.data_loader.load_pretraining_data()
        self._log_and_train(pretrain_data, phase="pretraining:phase_1")

        # Sub-phase 2: Refining with high-quality datasets
        print("Starting Pre-training Phase 2...")
        self._setup_optimizer_and_scheduler("pretraining:phase_2")
        high_quality_data = self.data_loader.load_finetuning_data()  # Reuse fine-tuning data for phase 2
        self._log_and_train(high_quality_data, phase="pretraining:phase_2")

    def train_finetuning_stage(self):
        """
        Executes the fine-tuning workflow using high-quality multimodal datasets.
        """
        print("Starting Fine-tuning...")
        self._setup_optimizer_and_scheduler("finetuning")
        fine_tune_data = self.data_loader.load_finetuning_data()
        self._log_and_train(fine_tune_data, phase="finetuning")

    def _log_and_train(self, dataset, phase: str):
        """
        Handles logging and execution of the training loop within a specific phase.

        Args:
            dataset: PyTorch Dataset object prepared for the current phase.
            phase (str): Specifies the current training phase (e.g., "pretraining:phase_1").
        """
        print(f"Training Phase: {phase}")
        from pytorch_lightning import datamodule
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config["training"][phase]["batch_size"],
            shuffle=True
        )
        self.lightning_trainer.fit(self.model, train_dataloader=dataloader)
        self._log_training_metrics({"phase": phase, "status": "completed"})

    def _setup_optimizer_and_scheduler(self, phase: str):
        """
        Configures optimizer and learning rate scheduler dynamically based on training phase.

        Args:
            phase (str): Current phase for optimization configuration ("pretraining:phase_1", etc.).
        """
        params = self.config["training"][phase]

        # Instantiate optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=params["learning_rate"],
            weight_decay=params["weight_decay"],
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        # Configure scheduler
        if params["scheduler"] == "constant_with_warmup":
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, 
                lr_lambda=lambda step: min(1.0, step / params["warmup_steps"])
            )
        elif params["scheduler"] == "cosine_decay":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=params["steps"]
            )

    def _log_training_metrics(self, metrics: Dict):
        """
        Logs training metrics if logging is enabled.

        Args:
            metrics (Dict): Dictionary containing training metrics to be logged.
        """
        if self.config.get("logging", {}).get("log_metrics", False):
            log_metrics(metrics, self.config["logging"]["log_interval"])


if __name__ == "__main__":
    # Example use of Trainer
    from config import Config
    from model import Model
    from dataset_loader import DatasetLoader

    try:
        config = Config("config.yaml").get_config()
        dataset_loader = DatasetLoader(config)

        # Initialize NaViL Model
        visual_encoder_config = config["model"]["visual_encoder"]
        llm_config = config["model"]["llm"]

        visual_encoder = VisualEncoder(
            depth=visual_encoder_config["depth"],
            width=visual_encoder_config["width"],
            patch_size=visual_encoder_config["patch_size"],
            mlp_width=visual_encoder_config["mlp_width"],
            attention_heads=visual_encoder_config["attention_heads"]
        )

        llm = LLM(
            model_name=llm_config["model_name"],
            use_moe=llm_config["use_moe"],
            num_experts=llm_config["num_experts"],
            depth=llm_config["depth"],
            width=llm_config["width"],
            mlp_width=llm_config["mlp_width"],
            attention_heads=llm_config["attention_heads"]
        )

        model = Model(visual_encoder, llm)
        trainer = Trainer(config, model, dataset_loader)

        print("Starting training process...")
        trainer.train_pretraining_stage()
        trainer.train_finetuning_stage()
        print("Training completed successfully!")

    except Exception as e:
        print(f"Error during training: {e}")
