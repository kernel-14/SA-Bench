## evaluation.py
import torch
import numpy as np
from typing import Dict, List
from utils import calculate_entropy, calculate_perplexity
from adaptive_inference import AdaptiveInference

class Evaluation:
    """
    Evaluation class for computing various metrics including perplexity, accuracy, and entropy.
    """
    def __init__(self, model, test_data: Dict[str, torch.utils.data.Dataset], metrics: List[str] = None):
        """
        Initializes the Evaluation class.

        Args:
            model: Trained Masked Diffusion Model (MDM) instance.
            test_data (Dict[str, torch.utils.data.Dataset]): Preprocessed test datasets.
            metrics (List[str]): List of metrics to compute. Defaults to ['accuracy', 'entropy', 'perplexity'].
        """
        self.model = model
        self.test_data = test_data
        self.metrics = metrics if metrics else ['accuracy', 'entropy', 'perplexity']

        # Device setup for evaluation (GPU/CPU based on availability)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def evaluate_text_data(self) -> Dict[str, float]:
        """
        Evaluates the text-based datasets (e.g., SlimPajama) using perplexity and entropy.

        Returns:
            Dict[str, float]: Dictionary containing perplexity and entropy metrics.
        """
        results = {}
        if 'perplexity' in self.metrics:
            results['perplexity'] = self._calculate_perplexity(self.test_data.get('slim_pajama', None))
        if 'entropy' in self.metrics:
            results['entropy'] = self._calculate_entropy(self.test_data.get('slim_pajama', None))
        return results

    def evaluate_logic_puzzles(self) -> Dict[str, Dict[str, float]]:
        """
        Evaluates logic puzzles like Sudoku and Zebra puzzles based on accuracy.

        Returns:
            Dict[str, Dict[str, float]]: Dictionary containing accuracy metrics for puzzles.
        """
        results = {}
        if 'accuracy' in self.metrics:
            if 'sudoku_test' in self.test_data:
                results['Sudoku'] = self._evaluate_puzzle_accuracy('sudoku_test', 'adaptive_top_probability_margin')
            if 'zebra_test' in self.test_data:
                results['Zebra'] = self._evaluate_puzzle_accuracy('zebra_test', 'adaptive_top_probability')
        return results

    def evaluate_task_specific(self, task_type: str, dataset_key: str) -> Dict[str, float]:
        """
        Evaluates tasks like HumanEval or Math based on their specific requirements.

        Args:
            task_type (str): The task type to evaluate ('infill', 'answer', or 'reasoning').
            dataset_key (str): Key for the dataset in test_data.

        Returns:
            Dict[str, float]: Evaluation metrics for the specific task type.
        """
        results = {}
        if dataset_key not in self.test_data:
            raise ValueError(f"Dataset key '{dataset_key}' not found in test data.")
        
        if task_type == 'infill':
            results['perplexity'] = self._calculate_perplexity(self.test_data[dataset_key])
            results['entropy'] = self._calculate_entropy(self.test_data[dataset_key])
        elif task_type == 'reasoning' or task_type == 'answer':
            results['accuracy'] = self._evaluate_puzzle_accuracy(dataset_key, 'adaptive_top_probability_margin')
        else:
            raise ValueError(f"Unknown task type '{task_type}'. Supported: 'infill', 'answer', 'reasoning'.")

        return results

    def compute_metrics(self, predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        Computes modular metrics (accuracy, entropy, perplexity) for any type of data.

        Args:
            predictions (torch.Tensor): Generated predictions from the model.
            labels (torch.Tensor): Ground truth sequence or solutions.

        Returns:
            Dict[str, float]: Computed metrics.
        """
        results = {}
        if 'accuracy' in self.metrics:
            results['accuracy'] = self._compute_accuracy(predictions, labels)
        if 'entropy' in self.metrics:
            results['entropy'] = calculate_entropy(predictions)
        if 'perplexity' in self.metrics:
            results['perplexity'] = calculate_perplexity(predictions, "huggingface/llama-7b")
        return results

    def _calculate_entropy(self, dataset: torch.utils.data.Dataset) -> float:
        """
        Calculates the entropy of the generated sequences.

        Args:
            dataset (torch.utils.data.Dataset): Dataset to evaluate.

        Returns:
            float: Average entropy of the generated outputs.
        """
        entropy_values = []
        for batch in dataset:
            batch_sequences = batch["input_sequences"].to(self.device)
            predictions = self.model.forward(batch_sequences)
            entropy_values.append(calculate_entropy(predictions))
        return np.mean(entropy_values)

    def _calculate_perplexity(self, dataset: torch.utils.data.Dataset) -> float:
        """
        Calculates the perplexity of the generated sequences using a baseline language model.

        Args:
            dataset (torch.utils.data.Dataset): Dataset to evaluate.

        Returns:
            float: Average perplexity of the generated outputs.
        """
        perplexity_values = []
        for batch in dataset:
            batch_sequences = batch["input_sequences"].to(self.device)
            predictions = self.model.forward(batch_sequences)
            perplexity_values.append(calculate_perplexity(predictions, "huggingface/llama-7b"))
        return np.mean(perplexity_values)

    def _evaluate_puzzle_accuracy(self, dataset_key: str, strategy: str) -> float:
        """
        Evaluates the accuracy of logic puzzles using an adaptive inference strategy.

        Args:
            dataset_key (str): Key for the dataset in test_data.
            strategy (str): Adaptive inference strategy ('adaptive_top_probability' or 'adaptive_top_probability_margin').

        Returns:
            float: Percentage accuracy of correctly solved puzzles.
        """
        if dataset_key not in self.test_data:
            raise ValueError(f"Dataset '{dataset_key}' not found in test data.")
        
        dataset = self.test_data[dataset_key]
        correct_count = 0
        total_count = 0

        adaptive_inference = AdaptiveInference(self.model, {'inference': {'strategy': strategy}})
        
        for batch in dataset:
            input_sequences = batch["input_sequences"].to(self.device)
            labels = batch["targets"].to(self.device)
            
            # Perform adaptive inference
            predictions = adaptive_inference.apply_adaptive_strategy(input_sequences, strategy)

            # Compute accuracy
            correct_count += (predictions == labels).sum().item()
            total_count += labels.size(0)

        return (correct_count / total_count) * 100

    def _compute_accuracy(self, predictions: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Computes exact-match accuracy for predictions and labels.

        Args:
            predictions (torch.Tensor): Generated predictions from the model.
            labels (torch.Tensor): Ground truth labels.

        Returns:
            float: Exact match accuracy.
        """
        return (predictions == labels).float().mean().item()

