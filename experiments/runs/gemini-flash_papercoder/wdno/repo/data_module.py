```python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import h5py # Potentially for PDEBench or custom datasets
import random
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union, Optional

# Local imports
from config import Config
from wavelet_utils import WaveletTransformManager
from pde_solvers import PdeSolver # Base class, specific solvers will be instantiated by main.py and passed
from utils import normalize_data, denormalize_data # Assuming logger is not explicitly needed here or handled by a global setup
from scipy.ndimage import zoom


class SingleResolutionDataset(Dataset):
    """
    Standard PyTorch dataset for single-resolution data, used for BRM training/evaluation
    and SRM evaluation.
    """
    def __init__(self, processed_data_list: List[Dict]):
        """
        Args:
            processed_data_list: A list of dictionaries, where each dictionary represents
                                 a single processed sample (wavelet coefficients, metadata, raw_gt).
        """
        self.processed_data_list = processed_data_list

    def __len__(self) -> int:
        return len(self.processed_data_list)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, Dict]]:
        """
        Returns a dictionary containing processed wavelet coefficients and raw ground truth data.
        """
        return self.processed_data_list[idx]


class MultiResolutionDataset(Dataset):
    """
    PyTorch dataset for multi-resolution training for the Super-Resolution Model (SRM).
    It provides pairs of (high_res_x_0_wavelets, low_res_x_wavelets_upsampled, high_res_conditions_wavelets).
    """
    def __init__(self, srm_training_pairs: List[Dict]):
        """
        Args:
            srm_training_pairs: A list of dictionaries, each representing a training pair for SRM.
                                Each dict contains:
                                - 'high_res_x_0_wavelets' (W_h for target)
                                - 'high_res_x_0_wavelets_metadata'
                                - 'low_res_x_wavelets_upsampled' (W_l as condition, already upsampled)
                                - 'high_res_conditions_wavelets' (W_a_h as condition)
                                - 'high_res_conditions_wavelets_metadata'
                                - 'raw_gt_data_for_eval'
        """
        self.srm_training_pairs = srm_training_pairs

    def __len__(self) -> int:
        return len(self.srm_training_pairs)

    def __getitem__(self, idx: int) -> Dict[str, Union[