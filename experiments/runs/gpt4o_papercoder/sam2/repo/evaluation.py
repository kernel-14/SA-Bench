"""
evaluation.py

Module to implement the Evaluation class for SAM 2. Handles zero-shot and interactive evaluation modes,
computes metrics like T&F, mIoU, and occlusion accuracy, and aggregates results across multiple datasets.
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
from tqdm import tqdm
from utils import compute_metrics, generate_prompts
import os


class Evaluation:
    """
    Provides functionality for evaluating SAM 2 through zero-shot and interactive segmentation tasks.
    """

    def __init__(self, model, test_loader: DataLoader, config: Dict):
        """
        Initialize the Evaluation module.

        Args:
            model (Model): SAM 2 model instance for evaluation.
            test_loader (DataLoader): PyTorch DataLoader containing the test dataset.
            config (dict): Configuration dictionary from config.yaml.
        """
        self.model = model.to(config['device'])
        self.test_loader = test_loader
        self.config = config

        self.device = torch.device(config['device'])
        self.metrics = config['evaluation']['metrics']
        self.result_dir = config['logging']['log_dir']
        os.makedirs(self.result_dir, exist_ok=True)

    def evaluate_zero_shot(self) -> Dict[str, Dict]:
        """
        Perform zero-shot evaluation on selected datasets.

        Returns:
            Dict[str, Dict]: Dictionary mapping dataset names to metrics (e.g., T&F, mIoU, occlusion accuracy).
        """
        self.model.eval()
        results = {}
        print("Starting zero-shot evaluation...")

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating datasets"):
                dataset_name = batch["dataset_name"]  # Assume dataset name is part of batch metadata
                video_frames = batch["frames"].to(self.device)
                annotations = batch["annotations"].to(self.device)
                memory_states = None  # Initialize memory states as None
                
                metrics = {"T&F": 0.0, "mIoU": 0.0, "occlusion_accuracy": 0.0}
                total_frames = 0

                for t in range(video_frames.size(1)):  # Process video frame-by-frame
                    frame = video_frames[:, t]
                    annotation = annotations[:, t]

                    # Zero-shot evaluation: No prompts provided
                    mask_logits, iou_logits, occlusion_logits = self.model(frame, memory=memory_states)

                    # Compute metrics
                    frame_metrics = compute_metrics(mask_logits, annotation)
                    metrics["T&F"] += frame_metrics["T&F"]
                    metrics["mIoU"] += frame_metrics["mIoU"]

                    # Compute occlusion accuracy if relevant
                    occlusion_labels = annotation["occlusion"].to(self.device)  # Placeholder for occlusion labels
                    occlusion_predictions = torch.sigmoid(occlusion_logits).round()
                    metrics["occlusion_accuracy"] += (occlusion_predictions == occlusion_labels).float().mean().item()

                    total_frames += 1

                    # Update memory states
                    self.model.update_memory(mask_logits, frame)

                # Aggregate frame-level metrics
                metrics["T&F"] /= total_frames
                metrics["mIoU"] /= total_frames
                metrics["occlusion_accuracy"] /= total_frames

                results[dataset_name] = metrics
                self._log_results(dataset_name, metrics, mode="zero-shot")

        return results

    def evaluate_interactive(self) -> Dict[str, Dict]:
        """
        Perform interactive evaluation, simulating corrective prompts to refine predictions.

        Returns:
            Dict[str, Dict]: Dictionary mapping dataset names to interactive metrics (e.g., average T&F).
        """
        self.model.eval()
        results = {}
        print("Starting interactive evaluation...")

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating datasets interactively"):
                dataset_name = batch["dataset_name"]  # Assume dataset name is part of batch metadata
                video_frames = batch["frames"].to(self.device)
                annotations = batch["annotations"].to(self.device)
                memory_states = None  # Initialize memory states as None
                
                metrics = {"avg_T&F": 0.0, "avg_prompts": 0.0, "occlusion_accuracy": 0.0}
                total_frames = 0
                prompt_counter = 0

                for t in range(video_frames.size(1)):  # Process video frame-by-frame
                    frame = video_frames[:, t]
                    annotation = annotations[:, t]

                    # Generate corrective prompts based on previous predictions
                    prompts = generate_prompts(annotation, strategy="error_based") if t > 0 else None

                    # Forward pass with corrective prompts
                    mask_logits, iou_logits, occlusion_logits = self.model(frame, prompts=prompts, memory=memory_states)

                    # Compute metrics
                    frame_metrics = compute_metrics(mask_logits, annotation)
                    metrics["avg_T&F"] += frame_metrics["T&F"]

                    # Track the number of prompts used
                    prompt_counter += len(prompts) if prompts is not None else 0

                    # Compute occlusion accuracy if relevant
                    occlusion_labels = annotation["occlusion"].to(self.device)  # Placeholder for occlusion labels
                    occlusion_predictions = torch.sigmoid(occlusion_logits).round()
                    metrics["occlusion_accuracy"] += (occlusion_predictions == occlusion_labels).float().mean().item()

                    total_frames += 1

                    # Update memory states
                    self.model.update_memory(mask_logits, frame)

                # Aggregate metrics across frames
                metrics["avg_T&F"] /= total_frames
                metrics["avg_prompts"] = prompt_counter / total_frames
                metrics["occlusion_accuracy"] /= total_frames

                results[dataset_name] = metrics
                self._log_results(dataset_name, metrics, mode="interactive")

        return results

    def _log_results(self, dataset_name: str, metrics: Dict[str, float], mode: str) -> None:
        """
        Log evaluation results to file for reproducibility.

        Args:
            dataset_name (str): Name of the dataset being evaluated.
            metrics (Dict[str, float]): Computed metrics for the evaluation.
            mode (str): Evaluation mode ("zero-shot" or "interactive").
        """
        log_file = os.path.join(self.result_dir, f"{dataset_name}_{mode}_results.txt")
        with open(log_file, "w") as f:
            for metric_name, value in metrics.items():
                f.write(f"{metric_name}: {value:.4f}\n")
        print(f"Logged {mode} results for {dataset_name} to {log_file}.")
