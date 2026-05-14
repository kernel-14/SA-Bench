# dataset_loader.py
"""
This module handles the loading and preprocessing of datasets for the Wavelet Diffusion Neural Operator (WDNO).
It integrates wavelet decomposition for simulation and control tasks, supports multi-resolution data generation,
and ensures compatibility with training and evaluation pipelines.
"""

import os
from typing import Any, Dict, List, Tuple
import torch
from torch import Tensor
import h5py
from wavelet_transform import WaveletTransform

class DatasetLoader:
    """
    A class to load and preprocess datasets for the Wavelet Diffusion Neural Operator (WDNO).
    Includes functionalities for wavelet decomposition, creating multi-resolution datasets,
    and retrieving data splits for training and testing.

    Attributes:
        config (dict): Configuration dictionary loaded from the `config.yaml` file.
        wavelet_transformer (WaveletTransform): Instance of the `WaveletTransform` class.
        data_cache (Dict[str, Any]): Temporary storage for loaded datasets.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the DatasetLoader with configurations for wavelet decomposition and dataset settings.

        Args:
            config (dict): Configuration dictionary from `config.yaml`. Includes dataset paths,
                           wavelet settings, and training/testing splits.

        Raises:
            ValueError: If essential keys are missing in the configuration.
        """
        self.config = config

        # Initialize wavelet transformer based on configuration
        self.wavelet_transformer_1d = WaveletTransform(
            wavelet_type=config["wavelet"]["basis_1d"],
            mode=config["wavelet"]["mode_1d"]
        )
        self.wavelet_transformer_2d = WaveletTransform(
            wavelet_type=config["wavelet"]["basis_2d"],
            mode=config["wavelet"]["mode_2d"]
        )

        # Data cache for loaded datasets
        self.data_cache = {}

    def load_data(self) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """
        Load datasets for simulation and control tasks from disk and organize them into raw tensors.
        
        Returns:
            Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
                - Raw state data (\( u \)) for simulation tasks and control sequences (\( f \)) for control tasks.
                - Organized as nested dictionaries keyed by dataset name and split type.
        """
        datasets = self.config["data"]
        loaded_data_raw = {}
        loaded_data_preprocessed = {}

        for dataset_name, settings in datasets.items():
            file_path = settings.get("source")
            if not file_path or not os.path.exists(file_path):
                raise FileNotFoundError(f"Dataset file `{file_path}` not found for `{dataset_name}`.")

            print(f"Loading dataset: {dataset_name} from {file_path}")
            # Specific handling for PDEBench datasets (HDF5 format)
            if dataset_name in ["burgers_equation", "navier_stokes", "fluid_2D"]:
                dataset_raw, dataset_preprocessed = self._load_pdebench_data(file_path, settings)
                loaded_data_raw[dataset_name] = dataset_raw
                loaded_data_preprocessed[dataset_name] = dataset_preprocessed

        return loaded_data_raw, loaded_data_preprocessed

    def _load_pdebench_data(self, file_path: str, settings: Dict[str, Any]) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """
        Load PDEBench datasets (.h5 format) and preprocess them into tensors.

        Args:
            file_path (str): Path to the HDF5 file containing PDEBench dataset.
            settings (dict): Dataset-specific settings (e.g., resolution, splits).

        Returns:
            Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
                - Raw data organized as split dictionaries (training, testing datasets).
                - Preprocessed wavelet-transformed data as split dictionaries.
        """
        with h5py.File(file_path, "r") as h5_data:
            splits = ["train_size", "test_size", "super_resolution_test_size"]
            raw_data = {}
            preprocessed_data = {}

            for split in splits:
                split_size = settings.get(split)
                if not split_size:
                    continue

                print(f"Loading split: {split}")
                states_raw = torch.tensor(h5_data[f"{split}/states"])
                controls_raw = torch.tensor(h5_data[f"{split}/controls"])
                raw_data[split] = {"states": states_raw, "controls": controls_raw}

                # Apply wavelet transform
                wavelet_preprocessed = self.prepare_wavelet_data(states_raw)
                preprocessed_data[split] = wavelet_preprocessed

        return raw_data, preprocessed_data

    def prepare_wavelet_data(self, data: Tensor) -> Dict[str, Any]:
        """
        Perform wavelet decomposition on raw data and return wavelet coefficients.

        Args:
            data (Tensor): Raw simulation/control data as PyTorch tensor.

        Returns:
            Dict[str, Any]: Dictionary containing low-frequency and detail coefficients.
        """
        print("Applying wavelet decomposition...")
        return {
            "wavelet_coefficients": self.wavelet_transformer_1d.apply_transform(data),
        }

    def prepare_super_resolution_data(self, data: Tensor) -> List[Dict[str, Tensor]]:
        """
        Prepare multi-resolution datasets for Super-Resolution Model (SRM) training.

        Args:
            data (Tensor): High-resolution datasets to downsample.

        Returns:
            List[Dict[str, Tensor]]: Multi-resolution paired datasets.
        """
        print(f"Preparing multi-resolution dataset for SRM...")
        resolutions = [
            (data.size(1) // 2, data.size(2) // 2),
            (data.size(1) // 4, data.size(2) // 4),
        ]
        multi_res_data = []

        for resolution in resolutions:
            print(f"Downsampling to resolution: {resolution}")
            # Downsample spatially and temporally
            low_res_data = self._downsample_tensor(data, resolution)
            low_res_wavelet = self.wavelet_transformer_1d.apply_transform(low_res_data)

            multi_res_data.append({
                "low_resolution_wavelet": low_res_wavelet,
                "high_resolution_wavelet": self.wavelet_transformer_1d.apply_transform(data),
            })

        return multi_res_data

    def _downsample_tensor(self, data: Tensor, resolution: Tuple[int, int]) -> Tensor:
        """
        Perform spatial and temporal downsampling on tensor data.

        Args:
            data (Tensor): Input data tensor.
            resolution (Tuple[int, int]): Target resolution (temporal, spatial).

        Returns:
            Tensor: Downsampled tensor.
        """
        return torch.nn.functional.interpolate(
            data, size=resolution, mode="bilinear", align_corners=False
        )

    def get_data_split(self, dataset_name: str, split: str) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Retrieve training/testing splits for a specific dataset.

        Args:
            dataset_name (str): Name of the dataset (e.g., `burgers`, `navier_stokes`).
            split (str): Split type (`train`, `test`, etc.).

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Corresponding split tensors for states, controls, and optionally target (control tasks).
        """
        if dataset_name not in self.data_cache:
            raise ValueError(f"Dataset `{dataset_name}` not loaded in cache.")

        if split not in self.data_cache[dataset_name]:
            raise ValueError(f"Split `{split}` not found for dataset `{dataset_name}`.")

        data = self.data_cache[dataset_name][split]
        return data.get("states"), data.get("controls"), data.get("targets", None)


# Usage Example
# config = {'wavelet': ..., 'data': ...} [PLACE YAML STRUCTURE FROM config.yaml]
# loader = DatasetLoader(config)
# raw_data, wavelet_preprocessed = loader.load_data()
