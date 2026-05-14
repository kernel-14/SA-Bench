"""
evaluation.py: Implements the Evaluation class for benchmarking the NaViL model on multimodal tasks,
computing metrics, and generating qualitative visualizations.

Dependencies:
- torch: For tensor operations and computations.
- dataset_loader: Provides dataset interfaces for evaluation datasets.
- model: Includes the trained NaViL model.
- utils: Contains shared utility functions for logging and visualizations.
- matplotlib: Used for generating attention maps and other visual outputs.
"""

import torch
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import os
from utils import log_metrics, generate_visualizations  # Utility functions for logging and visualizations


class Evaluation:
    """
    Evaluation class for performing multimodal task benchmarks and generating qualitative visualizations.

    Attributes:
        model (torch.nn.Module): Trained NaViL model for evaluation tasks.
        dataset_loader (DatasetLoader): DatasetLoader instance for accessing evaluation datasets.
        config (dict): Configuration dictionary loaded from config.yaml.
        results (Dict[str, Any]): Dictionary to store evaluation results (e.g., accuracy, loss).
    """

    def __init__(self, model: torch.nn.Module, dataset_loader: Any, config: Dict[str, Any]):
        """
        Initialize the Evaluation class with model, dataset, and configuration.

        Args:
            model (torch.nn.Module): Trained NaViL model.
            dataset_loader (DatasetLoader): DatasetLoader instance for fetching evaluation datasets.
            config (dict): Configuration dictionary containing evaluation settings.
        """
        self.model = model
        self.dataset_loader = dataset_loader
        self.config = config
        self.results = {}

    def evaluate_multimodal(self, tasks: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate the model on a list of multimodal tasks and compute metrics.

        Args:
            tasks (List[str]): List of task names (e.g., ["OCRBench", "MMVet"]).

        Returns:
            Dict[str, Dict[str, float]]: Dictionary containing evaluation metrics for each task.
        """
        self.model.eval()  # Set model to evaluation mode
        evaluation_results = {}

        with torch.no_grad():
            for task in tasks:
                print(f"Evaluating task: {task}")
                try:
                    # Load the dataset for the specific task
                    dataset = self.dataset_loader.load_finetuning_data()  # Placeholder for task-specific loading
                    dataloader = torch.utils.data.DataLoader(
                        dataset,
                        batch_size=self.config['logging'].get('evaluation_batch_size', 64),
                        shuffle=False
                    )

                    task_metrics = self._evaluate_task(dataloader, task)
                    evaluation_results[task] = task_metrics
                    log_metrics(task_metrics, self.config['logging'].get('evaluation_interval', 5000))

                except Exception as e:
                    print(f"Error evaluating task {task}: {str(e)}")
        
        self.results = evaluation_results
        return evaluation_results

    def _evaluate_task(self, dataloader: torch.utils.data.DataLoader, task_name: str) -> Dict[str, float]:
        """
        Evaluates the model on a specific evaluation task.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader for the evaluation dataset.
            task_name (str): Name of the evaluation task.

        Returns:
            Dict[str, float]: Evaluation metrics for the task.
        """
        total_samples = 0
        correct_predictions = 0
        cumulative_loss = 0.0
        criterion = torch.nn.CrossEntropyLoss()  # Generic loss function

        for idx, batch in enumerate(dataloader):
            # Assuming batch contains "image" and "text" fields
            images = batch["image"].to(self.config['hardware']['precision'])
            text_tokens = batch["text"].to(self.config['hardware']['precision'])
            labels = batch["labels"].to(self.config['hardware']['precision'])  # Assuming labeled dataset

            # Forward pass through the model
            outputs = self.model(images, text_tokens)
            logits = outputs[:, :-1, :]  # Ignore last token (teacher forcing settings)

            # Compute loss
            loss = criterion(logits.contiguous(), labels.contiguous())
            cumulative_loss += loss.item() * len(images)

            # Compute accuracy
            predictions = torch.argmax(logits, dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += len(images)

        # Calculate metrics
        avg_loss = cumulative_loss / total_samples
        accuracy = correct_predictions / total_samples * 100

        print(f"[{task_name}] Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        return {
            "validation_loss": avg_loss,
            "accuracy": accuracy
        }

    def generate_visualization(self, inputs: Dict[str, torch.Tensor], layer: int = 12):
        """
        Generate visualizations of attention maps or cross-modal interactions for qualitative analysis.

        Args:
            inputs (Dict[str, torch.Tensor]): Dictionary containing multimodal inputs (images, text).
            layer (int): Layer index (e.g., LLM or visual encoder layer) to visualize attention maps.

        Returns:
            None: Saves visualizations to disk as specified in the configuration.
        """
        print(f"Generating visualization for layer {layer}...")

        images = inputs["image"].to(self.config['hardware']['precision'])
        text_tokens = inputs["text"].to(self.config['hardware']['precision'])

        # Forward pass with visualization capture
        self.model.eval()
        with torch.no_grad():
            # Assuming the model supports extracting attention maps
            outputs, attention_maps = self.model.forward_with_attention(images, text_tokens, layer=layer)

        # Save attention visualizations
        output_dir = self.config['logging'].get("visualization_output_dir", "./visualizations/")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        visualization_path = os.path.join(output_dir, f"attention_map_layer_{layer}.png")
        generate_visualizations(attention_maps, visualization_path)
        print(f"Visualization saved to {visualization_path}")

    def log_evaluation_results(self, results: Dict[str, Dict[str, float]]) -> None:
        """
        Log evaluation results into a structured JSON or CSV for reproducibility.

        Args:
            results (Dict[str, Dict[str, float]]): Evaluation results to log.

        Returns:
            None: Logs results to console or file.
        """
        output_path = self.config['logging'].get("evaluation_output_path", "./evaluation_results.json")
        try:
            import json
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=4)

            print(f"Evaluation results logged to {output_path}")
        except Exception as e:
            print(f"Failed to log evaluation results: {str(e)}")


if __name__ == "__main__":
    # Example usage of Evaluation class
    from config import Config
    from model import Model
    from dataset_loader import DatasetLoader

    try:
        config = Config("config.yaml").get_config()
        dataset_loader = DatasetLoader(config)
        model = Model(...)

        evaluator = Evaluation(model, dataset_loader, config)

        tasks_to_evaluate = ["OCRBench", "MMVet"]
        evaluation_results = evaluator.evaluate_multimodal(tasks_to_evaluate)

        # Generate Visualization Example
        sample_inputs = {
            "image": torch.rand(1, 3, 512, 512),
            "text": torch.randint(0, 100, (1, 128))
        }
        evaluator.generate_visualization(sample_inputs, layer=12)

        evaluator.log_evaluation_results(evaluation_results)

    except Exception as e:
        print(f"Error during evaluation: {e}")

