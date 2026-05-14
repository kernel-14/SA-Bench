## evaluation.py

import torch
from typing import Dict, List, Callable, Tuple, Any
from datasets import Dataset
from model import Model
from utils import set_random_seed, log_metrics
from transformers import AutoTokenizer


class Evaluation:
    """
    Handles evaluation of the trained Mixture-of-Experts (MoE) model on specified benchmarks.
    """

    def __init__(self, model: Model, test_data: Dataset, config: dict) -> None:
        """
        Initialize the Evaluation class with the model, test data, and configuration.

        Args:
            model (Model): Trained Mixture-of-Experts model.
            test_data (Dataset): Dataset object containing evaluation data loaded from DatasetLoader.
            config (dict): Configuration dictionary loaded from config.yaml.
        """
        self.model = model
        self.test_data = test_data
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        set_random_seed(self.config.get("global_seed", 42))
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")  # GPT-NeoX-compatible tokenizer

    def evaluate(self) -> Dict[str, Any]:
        """
        Perform evaluation across specified benchmarks and return the computed metrics.

        Returns:
            Dict[str, Any]: Dictionary containing metrics for each benchmark.
        """
        benchmarks = self.config["evaluation"]["benchmarks"]
        metrics = {}

        for benchmark_name in benchmarks:
            print(f"Evaluating on benchmark: {benchmark_name}")

            try:
                # Prepare benchmark-specific dataset and evaluation logic
                benchmark_data, postprocess_fn = self._prepare_evaluation_data(benchmark_name)

                # Perform evaluation
                predictions, ground_truth = self._generate_predictions(benchmark_data, postprocess_fn)

                # Compute evaluation metrics
                benchmark_metrics = self.compute_metrics(predictions, ground_truth, benchmark_name)
                metrics[benchmark_name] = benchmark_metrics

                # Log metrics for the benchmark
                self.log_results({f"{benchmark_name}_{key}": value for key, value in benchmark_metrics.items()})

            except Exception as e:
                print(f"Failed to evaluate benchmark {benchmark_name}: {e}")

        return metrics

    def _prepare_evaluation_data(self, benchmark_name: str) -> Tuple[Dataset, Callable]:
        """
        Prepare dataset and postprocessing logic for a specific benchmark.

        Args:
            benchmark_name (str): Name of the evaluation benchmark.

        Returns:
            Tuple[Dataset, Callable]: The benchmark dataset and a callable to postprocess predictions.
        """
        if benchmark_name not in self.test_data:
            raise ValueError(f"Benchmark dataset for {benchmark_name} not found.")

        # Select dataset and postprocessing methods based on benchmark name
        benchmark_data = self.test_data[benchmark_name]

        def mmlu_postprocess(predictions: List[torch.Tensor]) -> List[str]:
            return [self.tokenizer.decode(pred.argmax(dim=-1), skip_special_tokens=True) for pred in predictions]

        def hellaswag_postprocess(predictions: List[torch.Tensor]) -> List[str]:
            return [self.tokenizer.decode(pred.argmax(dim=-1), skip_special_tokens=True) for pred in predictions]

        def human_eval_postprocess(predictions: List[torch.Tensor]) -> List[str]:
            return [self.tokenizer.decode(pred.argmax(dim=-1), skip_special_tokens=True) for pred in predictions]

        # Define postprocessing function lookup
        postprocess_lookup = {
            "mmlu": mmlu_postprocess,
            "hellaswag": hellaswag_postprocess,
            "human_eval": human_eval_postprocess,
        }

        # Get the appropriate postprocessing function for the benchmark
        postprocess_fn = postprocess_lookup.get(benchmark_name, lambda x: x)
        return benchmark_data, postprocess_fn

    def _generate_predictions(self, benchmark_data: Dataset, postprocess_fn: Callable) -> Tuple[List[Any], List[Any]]:
        """
        Generate model predictions for a given benchmark dataset.

        Args:
            benchmark_data (Dataset): Dataset for the benchmark.
            postprocess_fn (Callable): Function to postprocess raw predictions.

        Returns:
            Tuple[List[Any], List[Any]]: Predictions and ground truth labels.
        """
        self.model.eval()
        predictions = []
        ground_truth = []

        for batch in benchmark_data:
            inputs = batch["input_ids"].to(self.device)
            targets = batch["labels"]

            with torch.no_grad():
                # Forward pass
                outputs = self.model(inputs)

                # Collect predictions and ground truth
                raw_predictions = outputs.cpu()
                predictions.extend(postprocess_fn(raw_predictions))
                ground_truth.extend([self.tokenizer.decode(label, skip_special_tokens=True) for label in targets])

        return predictions, ground_truth

    def compute_metrics(self, predictions: List[Any], ground_truth: List[Any], metric_name: str) -> Dict[str, Any]:
        """
        Compute evaluation metrics for the benchmark.

        Args:
            predictions (List[Any]): Model's predicted outputs.
            ground_truth (List[Any]): Reference outputs for comparison.
            metric_name (str): Name of the benchmark being evaluated.

        Returns:
            Dict[str, Any]: Metrics computed for the benchmark.
        """
        if metric_name == "mmlu":
            accuracy = self._compute_accuracy(predictions, ground_truth)
            return {"accuracy": accuracy}
        elif metric_name == "hellaswag":
            perplexity = self._compute_perplexity(predictions, ground_truth)
            return {"perplexity": perplexity}
        elif metric_name == "human_eval":
            pass_at_10 = self._compute_pass_at_k(predictions, ground_truth, k=10)
            return {"pass@10": pass_at_10}
        else:
            raise ValueError(f"Unsupported metric computation for benchmark {metric_name}")

    def _compute_accuracy(self, predictions: List[str], ground_truth: List[str]) -> float:
        """
        Compute accuracy by comparing predictions and ground truth.

        Args:
            predictions (List[str]): Predicted outputs.
            ground_truth (List[str]): Actual outputs.

        Returns:
            float: Accuracy score.
        """
        correct = sum([1 for pred, gt in zip(predictions, ground_truth) if pred == gt])
        return correct / len(ground_truth) if ground_truth else 0.0

    def _compute_perplexity(self, predictions: List[str], ground_truth: List[str]) -> float:
        """
        Compute perplexity for language modeling tasks.

        Args:
            predictions (List[str]): Predicted outputs (log probabilities).
            ground_truth (List[str]): Actual outputs.

        Returns:
            float: Perplexity score.
        """
        total_log_prob = sum([np.log(float(pred == gt)) for pred, gt in zip(predictions, ground_truth)])
        return np.exp(-total_log_prob / len(ground_truth)) if ground_truth else float("inf")

    def _compute_pass_at_k(self, predictions: List[str], ground_truth: List[str], k: int) -> float:
        """
        Compute Pass@k metric for coding tasks like HumanEval.

        Args:
            predictions (List[str]): Predicted outputs.
            ground_truth (List[str]): Actual outputs.
            k (int): Number of attempts considered.

        Returns:
            float: Pass@k metric.
        """
        correct = sum([1 for pred, gt in zip(predictions[:k], ground_truth) if pred == gt])
        return correct / k

    def log_results(self, metrics: Dict[str, float]) -> None:
        """
        Log results to Wandb for performance tracking.

        Args:
            metrics (Dict[str, float]): Dictionary containing metric names and computed values.
        """
        try:
            log_metrics(metrics, step=0, log_to_wandb=True, project_name="OLMoE")
        except Exception as e:
            print(f"Wandb logging failed: {e}")
