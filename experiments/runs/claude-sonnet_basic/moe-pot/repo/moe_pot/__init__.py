"""MoE-POT: Mixture-of-Experts Operator Transformer for Large-Scale PDE Pre-Training"""

from .model import (
    MoEPOT,
    MoEPOTBlock,
    PatchEmbedding,
    TemporalAggregation,
    create_moe_pot_tiny,
    create_moe_pot_small,
    create_moe_pot_medium,
)
from .fourier_layer import FourierLayer
from .moe_layer import MoELayer, ConvExpert, RouterGatingNetwork
from .trainer import MoEPOTTrainer, l2_relative_error
from .datasets import PDEDataset, MixedPDEDataset, create_dataloaders

__all__ = [
    "MoEPOT",
    "MoEPOTBlock",
    "PatchEmbedding",
    "TemporalAggregation",
    "FourierLayer",
    "MoELayer",
    "ConvExpert",
    "RouterGatingNetwork",
    "MoEPOTTrainer",
    "l2_relative_error",
    "PDEDataset",
    "MixedPDEDataset",
    "create_dataloaders",
    "create_moe_pot_tiny",
    "create_moe_pot_small",
    "create_moe_pot_medium",
]
