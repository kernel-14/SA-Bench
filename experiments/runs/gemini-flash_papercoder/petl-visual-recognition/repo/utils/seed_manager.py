## utils/seed_manager.py

import random
import numpy as np
import torch


class SeedManager:
    """
    Manages the setting of random seeds across various libraries to ensure
    reproducibility of experiments.
    """

    def set_seed(self, seed: int) -> None:
        """
        Sets the random seed for Python's random module, NumPy, PyTorch (CPU and CUDA),
        and configures PyTorch's CuDNN for deterministic behavior.

        Args:
            seed (int): The integer value to use as the random seed.
        """
        if not isinstance(seed, int):
            raise TypeError(f"Seed must be an integer, but got type {type(seed).__name__}")

        # Set seed for Python's built-in random module
        random.seed(seed)

        # Set seed for NumPy operations
        np.random.seed(seed)

        # Set seed for PyTorch CPU operations
        torch.manual_seed(seed)

        # Set seed for PyTorch CUDA (GPU) operations if available
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

        # Configure CuDNN for deterministic behavior
        # This can sometimes lead to a performance decrease, but ensures reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
