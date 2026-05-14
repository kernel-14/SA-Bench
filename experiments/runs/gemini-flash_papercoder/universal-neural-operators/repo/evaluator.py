import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, List

from config import Config
from dataset_manager import DatasetManager
from models.neural_operator import NeuralOperatorModel
from utils import get_device, denormalize_data


class Evaluator:
    """
    Handles the evaluation of trained models using specified metrics.
    """

    def __init__(self, config: Config, dataset_manager: DatasetManager):
        """
        Initializes the Evaluator.

        Args:
            config (Config): The global configuration object.
            dataset_manager (DatasetManager): An instance of the DatasetManager for data access.
        """
        if not isinstance(config, Config):
            raise TypeError(f"Expected 'config' to be an instance of Config, but got {type(config)}.")
        if not isinstance(dataset_manager, DatasetManager):
            raise TypeError(f"Expected 'dataset_manager' to be an instance of DatasetManager, but got {type(dataset_manager)}.")

        self.config = config
        self.dataset_manager = dataset_manager
        self.device = get_device(self.config.device)
        self.metrics_to_compute = self.config.evaluation_settings.get('metrics', ['mse', 'nmae'])
        self.epsilon = 1e-8 # A small constant to prevent division by zero in NMAE

    def _calculate_mse(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Calculates the Mean Squared Error (MSE).

        Args:
            predictions (torch.Tensor): Predicted tensor.
            targets (torch.Tensor): Ground truth tensor.

        Returns:
            float: The computed MSE value.
        """
        if not isinstance(predictions, torch.Tensor):
            raise TypeError(f"Expected 'predictions' to be a torch.Tensor, but got {type(predictions)}.")
        if not isinstance(targets, torch.Tensor):
            raise TypeError(f"Expected 'targets' to be a torch.Tensor, but got {type(targets)}.")

        return F.mse_loss(predictions, targets, reduction='mean').item()

    def _calculate_nmae(self, predictions: torch.Tensor, targets: torch.Tensor,
                        data_min_val: float, data_max_val: float) -> float:
        """
        Calculates the Range-Normalized Mean Absolute Error (NMAE) as defined in the paper.

        The formula is: NMAE = ( ||predictions - targets||_1 ) / ( max(targets) - min(targets) + epsilon )
        Predictions and targets are denormalized using data_min_val and data_max_val before error calculation.
        The max/min in the denominator are calculated per sample from the denormalized targets.

        Args:
            predictions (torch.Tensor): Normalized predicted tensor.
            targets (torch.Tensor): Normalized ground truth tensor.
            data_min_val (float): Minimum value of the original (unnormalized) dataset output range.
            data_max_val (float): Maximum value of the original (unnormalized) dataset output range.

        Returns:
            float: The computed NMAE value averaged across the batch.
        """
        if not isinstance(predictions, torch.Tensor):
            raise TypeError(f"Expected 'predictions' to be a torch.Tensor, but got {type(predictions)}.")
        if not isinstance(targets, torch.Tensor):
            raise TypeError(f"Expected 'targets' to be a torch.Tensor, but got {type(targets)}.")
        if not isinstance(data_min_val, (int, float)):
            raise TypeError(f"Expected 'data_min_val' to be a float, but got {type(data_min_val)}.")
        if not isinstance(data_max_val, (int, float)):
            raise TypeError(f"Expected 'data_max_val' to be a float, but got {type(data_max_val)}.")

        # 1. Denormalize predictions and targets to their original scale
        denormalized_predictions = denormalize_data(predictions, data_min_val, data_max_val)
        denormalized_targets = denormalize_data(targets, data_min_val, data_max_val)

        # 2. Calculate L1 error per sample
        # Reshape to (batch_size, -1) to flatten all spatial/feature dimensions for L1 norm calculation
        abs_diff = torch.abs(denormalized_predictions - denormalized_targets)
        l1_error_per_sample = torch.mean(abs_diff.view(abs_diff.shape[0], -1), dim=1) # Mean over flattened elements per sample

        # 3. Calculate target range per sample (max_G u - min_G u)
        # Also reshape to (batch_size, -1) to find min/max for each sample over its grid
        targets_flat = denormalized_targets.view(denormalized_targets.shape[0], -1)
        max_target_per_sample = targets_flat.max(dim=1).values
        min_target_per_sample = targets_flat.min(dim=1).values
        
        # Add epsilon to denominator to prevent division by zero
        range_per_sample = max_target_per_sample - min_target_per_sample + self.epsilon

        # 4. Calculate NMAE per sample and then average over the batch
        nmae_per_sample = l1_error_per_sample / range_per_sample
        
        return torch.mean(nmae_per_sample).item()

    def evaluate(self, model: NeuralOperatorModel, dataloader: DataLoader,
                 dataset_key: str) -> Dict[str, float]:
        """
        Runs inference on the provided dataloader and computes all configured metrics.

        Args:
            model (NeuralOperatorModel): The trained neural operator model.
            dataloader (DataLoader): DataLoader for the test set.
            dataset_key (str): A unique string key for the dataset (e.g., "burgers_finetune_dataset").
                               Used to retrieve min/max values for denormalization.

        Returns:
            Dict[str, float]: A dictionary containing computed metric names and their values.
        """
        if not isinstance(model, NeuralOperatorModel):
            raise TypeError(f"Expected 'model' to be an instance of NeuralOperatorModel, but got {type(model)}.")
        if not isinstance(dataloader, DataLoader):
            raise TypeError(f"Expected 'dataloader' to be an instance of DataLoader, but got {type(dataloader)}.")
        if not isinstance(dataset_key, str) or not dataset_key:
            raise ValueError("dataset_key must be a non-empty string.")

        model.eval()  # Set model to evaluation mode
        model.to(self.device)

        total_metrics: Dict[str, float] = {metric: 0.0 for metric in self.metrics_to_compute}
        num_samples = 0

        # Retrieve min/max values for denormalization from DatasetManager
        min_max_vals = self.dataset_manager.dataset_min_max_vals.get(dataset_key)
        if min_max_vals is None:
            raise ValueError(f"Min/max values for dataset '{dataset_key}' not found in DatasetManager. "
                             "Ensure get_dataloaders was called for this dataset.")
        data_min_val, data_max_val = min_max_vals

        with torch.no_grad():  # Disable gradient computation
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                predictions = model(inputs)

                batch_size = inputs.size(0)
                num_samples += batch_size

                if 'mse' in self.metrics_to_compute:
                    total_metrics['mse'] += self._calculate_mse(predictions, targets) * batch_size

                if 'nmae' in self.metrics_to_compute:
                    total_metrics['nmae'] += self._calculate_nmae(predictions, targets,
                                                                   data_min_val, data_max_val) * batch_size

        if num_samples == 0:
            return {metric: 0.0 for metric in self.metrics_to_compute} # Return 0 for all if no samples
        
        # Calculate average metrics
        avg_metrics = {metric: total_value / num_samples for metric, total_value in total_metrics.items()}
        return avg_metrics

