```python
## trainers/sc_fno_pinn_trainer.py
import torch
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any, Tuple
import numpy as np # For random choice if needed

# Import project-specific modules
from config import Config
from losses import Losses
from trainers.base_trainer import BaseTrainer
from models.sc_fno_pinn import SCFNO_PINN  # Specific SCFNO_PINN model
from data_generation.pde_solver import PDESolver # For equation_fn

# Third-party library for progress bar
from tqdm import tqdm


class SCFNO_PINNTrainer(BaseTrainer):
    """
    Concrete trainer for Sensitivity-Constrained FNO with PINN regularization (SC-FNO-PINN) models.
    This trainer computes data loss (L_u), sensitivity loss (L_s), and equation loss (L_Eq)
    for model optimization, as described in the paper's Algorithm 3.
    """

    def __init__(
        self,
        model: SCFNO_PINN,  # Expecting an SCFNO_PINN model specifically
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        config: Config,
        losses: Losses,
        device: str
    ) -> None:
        """
        Initializes the SCFNO_PINNTrainer.

        Args:
            model (SCFNO_PINN): The SC-FNO_PINN model instance to be trained.
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader): DataLoader for the validation dataset.
            optimizer (torch.optim.Optimizer): PyTorch optimizer instance.
            scheduler (Optional[Any]): Optional learning rate scheduler.
            config (Config): Configuration object.
            losses (Losses): Instance of the Losses class.
            device (str): The computational device ('cuda' or 'cpu').
        """
        # Ensure the model is indeed an SCFNO_PINN instance
        if not isinstance(model, SCFNO_PINN):
            raise TypeError("SCFNO_PINNTrainer expects a model of type SCFNO_PINN.")
        
        super().__init__(model, train_loader, val_loader, optimizer, scheduler, config, losses, device)
        
        # Get sensitivity sampling percentage from config
        self.sensitivity_sampling_percentage: float = self.config.get(
            "dataset_generation.sensitivity_sampling_percentage", 0.1
        )
        self.pinn_num_collocation_points: int = self.config.get(
            "dataset_generation.pinn_num_collocation_points", 10000
        )
        
        self.equation_id: str = self.config.get("experiment.equation_id")