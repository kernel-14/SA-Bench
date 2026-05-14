## trainer.py
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.functional import cross_entropy
from typing import Tuple, Dict
from model import Model
from utilities import Utilities

class Trainer:
    """
    Trainer class responsible for orchestrating the training process of NGPT.
    Includes forward passes, backward passes, normalization after each batch, and scheduling.
    """

    def __init__(
        self, 
        model: Model, 
        optimizer: Adam, 
        data: Tuple[torch.Tensor, torch.Tensor], 
        config: Dict
    ):
        """
        Initializes the Trainer with the model, optimizer, data, and configuration parameters.

        Args:
            model (Model): The NGPT model instance.
            optimizer (Adam): The optimizer instance for parameter updates.
            data (Tuple[torch.Tensor, torch.Tensor]): Training data (inputs, targets).
            config (Dict): Config dictionary loaded from config.yaml.
        """
        self.model = model
        self.optimizer = optimizer
        self.train_data = data[0]
        self.validation_data = data[1]
        self.config = config
        self.batch_size = config["training"]["batch_size"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.dtype = torch.bfloat16 if config["hardware"]["dtype"] == "bfloat16" else torch.float

        # Learning rate scheduler based on cosine annealing
        self.scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=config["training"]["epochs"], 
            eta_min=config.get("training", {}).get("final_learning_rate", 0)
        )

        # Scaling factor initialization
        scaling_factors = config["model"]["scaling_factors"]
        self.alpha_a_scale = scaling_factors["alpha_a_scale"]
        self.alpha_m_scale = scaling_factors["alpha_m_scale"]
        self.sqk_scale = scaling_factors["sqk_scale"]
        self.sz_scale = scaling_factors["sz_scale"]
        
        self.current_epoch = 0

    def setup_optimizer(self) -> torch.optim.Optimizer:
        """
        Configures the optimizer for training. Uses Adam optimizer with weight decay set to 0.

        Returns:
            torch.optim.Optimizer: Configured optimizer.
        """
        # Ensure weight decay = 0 as per NGPT requirements
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=0.0
        )
        return self.optimizer

    def train(self, epochs: int = None):
        """
        Executes the training loop over the specified number of epochs.

        Args:
            epochs (int): Number of epochs to train. Defaults to the value in config.
        """
        self.model.train()
        epochs = epochs or self.config["training"]["epochs"]
        total_steps = len(self.train_data) // self.batch_size  # Total steps per epoch

        for epoch in range(epochs):
            self.current_epoch = epoch
            epoch_loss = 0.0

            # Batch-wise Training
            for step, (inputs, targets) in enumerate(self._create_batches(self.train_data)):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                logits = self.model(inputs)
                loss = cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

                # Backward pass and optimizer step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Post-batch normalization
                self.normalize_after_batch()

                # Update metrics
                epoch_loss += loss.item()

                # Logging progress
                if step % 10 == 0:
                    print(f"Epoch [{epoch + 1}/{epochs}], Step [{step + 1}/{total_steps}], Loss: {loss.item():.4f}")

            # Learning rate schedule update
            self.scheduler.step()

            # Validation and epoch summary
            validation_loss = self.evaluate()
            print(f"Epoch {epoch + 1}/{epochs} -> Training Loss: {epoch_loss / total_steps:.4f}, \
Validation Loss: {validation_loss:.4f}")

    def _create_batches(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Helper function to create batches from data.

        Args:
            data (torch.Tensor): Raw data tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Batched input and target tensors.
        """
        batch_size = self.batch_size
        for i in range(0, len(data[0]), batch_size):
            inputs = data[0][i:i + batch_size]
            targets = data[1][i:i + batch_size]
            yield inputs, targets

    def normalize_after_batch(self):
        """
        Normalizes model parameters after each training batch to enforce hypersphere constraints.
        This includes embeddings, attention matrices, and MLP matrices.
        """
        self.model.normalize_parameters()

    def evaluate(self) -> float:
        """
        Evaluates the model on validation data and computes the validation loss.

        Returns:
            float: Validation loss.
        """
        self.model.eval()
        validation_loss = 0.0
        total_steps = len(self.validation_data[0]) // self.batch_size

        with torch.no_grad():
            for step, (inputs, targets) in enumerate(self._create_batches(self.validation_data)):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                logits = self.model(inputs)
                loss = cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                validation_loss += loss.item()

        return validation_loss / total_steps
