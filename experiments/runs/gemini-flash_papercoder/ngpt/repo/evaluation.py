import torch
import torch.distributed as dist
from torch.cuda.amp import autocast
from typing import Dict

# Avoid circular import by importing Config and model/data types for type hinting
# and accessing their members rather than inheriting or directly depending on
# their full implementation logic which might lead to circular dependencies.
from config import Config
# Assuming NGPTModel is defined in model.py and DataModule in data.py
# We will use these for type hints, but direct class access is fine too.
# For actual instantiation, these should be imported in main.py or trainer.py
# and passed as objects.
# For evaluation.py, we only need to know their interfaces for type hinting.
# To prevent circular imports if model.py or data.py also import config.py,
# we define dummy classes for type hinting here, or rely on forward references if needed.
# For this specific setup, it's safer to directly import for clarity, assuming
# model.py and data.py do not import evaluation.py.
from model import NGPTModel
from data import DataModule


class NGPTEvaluator:
    """
    Evaluates the NGPT model's performance on validation and (placeholder for) downstream tasks.
    """

    def __init__(self, config: Config, model: NGPTModel, data_module: DataModule):
        """
        Initializes the NGPTEvaluator.

        Args:
            config: An instance of the Config dataclass containing all
                    hyperparameters and evaluation settings.
            model: The NGPTModel instance to be evaluated.
            data_module: The DataModule instance providing access to datasets.
        """
        self.config = config
        self.model = model
        self.data_module = data_module

        # Determine the device. For DDP, each process gets its own GPU.
        if self.config.system_config.num_gpus > 1 and dist.is_initialized():
            self.device = torch.device(f'cuda:{dist.get_rank()}')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        self.model.to(self.device)
        print(f"Evaluator initialized. Model moved to device: {self.device}")

    def evaluate_validation_loss(self) -> float:
        """
        Computes the average validation loss (perplexity) over the entire validation dataset.

        Returns:
            The global average validation loss across all distributed processes.
        """
        self.model.eval()  # Set model to evaluation mode
        total_loss = 0.0
        num_batches = 0

        # Get the validation dataloader
        val_dataloader = self.data_module.val_dataloader()

        # Disable gradient calculations for efficiency during inference
        with torch.no_grad():
            # Use autocast for mixed-precision inference if configured
            with autocast(enabled=self.config.training_config.precision == "bfloat16",
                          dtype=torch.bfloat16 if self.config.training_config.precision == "bfloat16" else torch.float32):
                for batch in val_dataloader:
                    input_ids = batch['input_ids'].to(self.device)
                    targets = batch['labels'].to(self.device) # Using 'labels' as per DataModule

                    loss, _ = self.model(input_ids, targets)
                    total_loss += loss.item()
                    num_batches += 1

        if num_batches == 0:
            local_avg_loss = 0.0
        else:
            local_avg_loss = total_loss / num_batches

        # Aggregate losses across distributed processes if applicable
        if self.config.system_config.num_gpus > 1 and dist.is_initialized():
            # Create a tensor for the local average loss
            loss_tensor = torch.tensor(local_avg_loss, device=self.device)
            # Reduce all losses to sum on all ranks
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            # Global average loss is the sum divided by the number of GPUs
            global_avg_loss = loss_tensor.item() / dist.get_world_size()
        else:
            global_avg_loss = local_avg_loss

        self.model.train()  # Set model back to training mode
        return global_avg_loss

    def evaluate_downstream_tasks(self) -> Dict[str, float]:
        """
        Placeholder for evaluating the model on various downstream tasks.
        The paper mentions "a set of standard downstream tasks" and specifically
        WMT14-FR-EN (BLEU) and PG19 (perplexity for length extrapolation).

        For initial reproduction, this method returns the validation perplexity
        and includes placeholder entries for other tasks.

        Returns:
            A dictionary where keys are task names and values are their scores.
        """
        print("Evaluating downstream tasks (placeholder implementation)...")
        results: Dict[str, float] = {}

        # The validation loss (perplexity) is an intrinsic evaluation and often
        # a good proxy for general performance.
        val_loss = self.evaluate_validation_loss()
        results["validation_perplexity"] = val_loss

        # Placeholder for other specific downstream tasks mentioned in the paper
        # Exact implementation would require specific datasets, metrics, and evaluation pipelines.
        # WMT14-FR-EN (BLEU score)
        results["wmt14_fr_en_bleu"] = 0.0 # To be implemented
        # PG19 perplexity (for length extrapolation)
        results["pg19_perplexity"] = 0.0 # To be implemented
        # Average Accuracy on downstream tasks (Table 4, 5 in paper)
        results["average_downstream_accuracy"] = 0.0 # To be implemented

        if dist.is_initialized() and dist.get_rank() == 0:
            print(f"Downstream evaluation results: {results}")

        return results

