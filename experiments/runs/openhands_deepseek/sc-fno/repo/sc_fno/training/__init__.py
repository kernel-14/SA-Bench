"""Training utilities for SC-FNO."""
from .losses import data_loss, sensitivity_loss, pde_loss, ic_loss, bc_loss
from .trainer import (
    FNOTrainer,
    SCFNOTrainer,
    FNOTrainerPINN,
    SCFNOTrainerPINN,
    train_model,
)
from .inversion import invert_parameters, invert_parameters_multi
