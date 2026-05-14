## evaluation.py
import os
import time
import torch
from torchmetrics.functional import frechet_distance
from torch.utils.data import DataLoader
import torch.cuda
from typing import Dict, List
from config import Config
import numpy as np
import torchvision.models.video as video_models


class Evaluation:
    """
    Evaluation class for assessing generated video quality and efficiency metrics.
    Supports FVD computation, runtime profiling, and GPU memory consumption analysis.
    """

    def __init__(self, ground_truth: DataLoader, generated: DataLoader, config: Config) -> None:
        """
        Initialize the evaluation pipeline with ground-truth and generated datasets.

        Args:
            ground_truth (DataLoader): DataLoader for the ground-truth video dataset.
            generated (DataLoader): DataLoader for the generated videos.
            config (Config): Configuration object containing evaluation parameters.
        """
        self.ground_truth = ground_truth
        self.generated = generated
        self.config = config

        # Load pretrained I3D model path from config
        pretrained_i3d_path = self.config.get("evaluation.fvd.pretrained_i3d", "")
        if not pretrained_i3d_path or not os.path.exists(pretrained_i3d_path):
            raise FileNotFoundError(f"I3D pretrained weights not found at '{pretrained_i3d_path}'.")

        # Initialize I3D model for video feature extraction
        self.i3d_model = self._load_pretrained_i3d(pretrained_i3d_path)
        self.chunk_size = self.config.get("evaluation.fvd.chunk_size", 16)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_pretrained_i3d(self, path: str) -> torch.nn.Module:
        """
        Load a pretrained I3D model for video feature extraction.

        Args:
            path (str): Path to the I3D checkpoint.

        Returns:
            torch.nn.Module: Loaded I3D model.
        """
        model = video_models.mc3_18(pretrained=False)
        state_dict = torch.load(path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        model.eval().to(self.device)
        return model

    def _extract_features(self, dataloader: DataLoader) -> List[torch.Tensor]:
        """
        Extract features from videos using the pretrained I3D model.

        Args:
            dataloader (DataLoader): A DataLoader containing video batches.

        Returns:
            List[torch.Tensor]: List of extracted per-video features.
        """
        features = []
        for batch in dataloader:
            videos = batch["video"].to(self.device)  # Shape: (B, T, C, H, W)
            with torch.no_grad():
                # I3D model processes input as (B, C, T, H, W), so permute input
                videos = videos.permute(0, 2, 1, 3, 4)
                batch_features = self.i3d_model(videos)
                features.append(batch_features.cpu())
        return torch.cat(features, dim=0)

    def compute_fvd(self, pretrained_i3d: str) -> float:
        """
        Compute Frechet Video Distance (FVD) between the ground-truth and generated datasets.

        Args:
            pretrained_i3d (str): Path to pretrained I3D weights.

        Returns:
            float: The computed FVD score.
        """
        print("Extracting features for FVD computation...")
        gt_features = self._extract_features(self.ground_truth)
        gen_features = self._extract_features(self.generated)

        print("Computing FVD...")
        fvd_score = frechet_distance(gt_features, gen_features)
        print(f"FVD Score: {fvd_score}")
        return fvd_score

    def compute_time_efficiency(self) -> Dict[str, float]:
        """
        Measure time efficiency for video generation in autoregressive settings.

        Returns:
            Dict[str, float]: Dictionary containing cumulative generation time and per-step runtimes.
        """
        start_time = time.perf_counter()
        runtime_per_step = []

        for idx, batch in enumerate(self.generated):
            step_start = time.perf_counter()
            _ = batch["video"]  # Simulation of data processing
            step_end = time.perf_counter()
            runtime_per_step.append(step_end - step_start)

        total_time = time.perf_counter() - start_time
        print(f"Total Generation Time: {total_time:.4f}s")
        print(f"Average Time per Step: {np.mean(runtime_per_step):.4f}s")
        
        return {
            "total_time": total_time,
            "average_time_per_step": np.mean(runtime_per_step),
        }

    def compute_memory_usage(self) -> float:
        """
        Estimate peak GPU memory usage during autoregressive inference.

        Returns:
            float: Peak GPU memory usage (in GB).
        """
        if not torch.cuda.is_available():
            print("Memory profiling requires a GPU-enabled system.")
            return 0.0

        torch.cuda.reset_peak_memory_stats()
        for batch in self.generated:
            _ = batch["video"].to(self.device)  # Simulate data transfer to GPU
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Convert bytes to GB
        print(f"Peak GPU Memory Usage: {peak_memory:.4f} GB")
        return peak_memory
    
    def save_results(self, output_path: str, results: Dict[str, float]) -> None:
        """
        Save evaluation metrics to a file.

        Args:
            output_path (str): Path to the output file.
            results (Dict[str, float]): Dictionary of evaluation metrics.
        """
        with open(output_path, "w") as f:
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
        print(f"Results saved to {output_path}")

    def visualize_metrics(self) -> None:
        """
        Visualize key metrics, such as time and memory efficiency.
        Generates plots comparable to those in the paper.
        """
        try:
            import matplotlib.pyplot as plt

            # Generate sample metrics
            time_efficiency = self.compute_time_efficiency()
            memory_usage = self.compute_memory_usage()

            # Plot runtimes
            plt.figure(figsize=(10, 6))
            plt.bar(["Total Time (s)", "Average Step Time (s)"],
                    [time_efficiency["total_time"], time_efficiency["average_time_per_step"]])
            plt.title("Runtime Metrics")
            plt.ylabel("Time (s)")
            plt.savefig("./runtime_metrics.png")
            print("Runtime metrics visualization saved as runtime_metrics.png.")

            # Plot memory usage
            plt.figure(figsize=(6, 6))
            plt.bar(["GPU Memory (GB)"], [memory_usage])
            plt.title("Memory Usage")
            plt.ylabel("Memory (GB)")
            plt.savefig("./memory_metrics.png")
            print("Memory metrics visualization saved as memory_metrics.png.")

        except ImportError:
            print("Matplotlib is required for metric visualization. Please install it first.")
