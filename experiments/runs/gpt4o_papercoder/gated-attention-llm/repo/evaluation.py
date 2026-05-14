# evaluation.py

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Any, List
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer
from tqdm import tqdm


class Evaluation:
    """
    Handles the evaluation of trained Transformer models with Gated Attention.
    Evaluates different metrics such as perplexity for language modeling
    and benchmark-specific accuracies.
    """

    def __init__(self, model: torch.nn.Module, data: Dict[str, DataLoader], metrics: List[str], config: Dict[str, Any]):
        """
        Initialize the Evaluation instance.

        Args:
            model (torch.nn.Module): Trained Transformer model with Gated Attention.
            data (dict): Dictionary of evaluation data loaders for benchmarks.
            metrics (list): List of metrics to compute (e.g., perplexity, accuracy).
            config (dict): Configuration dictionary loaded from `config.yaml`.
        """
        self.model = model
        self.data = data  # Dict containing DataLoaders for benchmarks
        self.metrics = metrics
        self.config = config

        # General settings from configuration
        self.batch_size = config["training"].get("batch_size", 1024)
        self.context_length = config["dataset"].get("context_length", 4096)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to evaluation device
        self.model.to(self.device)
        self.model.eval()

        # Tokenizer for handling untokenized benchmark datasets
        self.tokenizer = AutoTokenizer.from_pretrained(config["model"]["type"])

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the model on the given tasks. Includes perplexity computation for
        language modeling datasets and task-specific benchmarks.

        Returns:
            dict: Dictionary of evaluation results with metrics as keys.
        """
        results = {}

        # Evaluate Perplexity for language modeling
        if "perplexity" in self.metrics:
            perplexity = self._evaluate_perplexity()
            results["perplexity"] = perplexity

        # Evaluate Benchmarks
        for benchmark_name, data_loader in self.data.items():
            if "accuracy" in self.metrics:  # Task-specific accuracy evaluation
                accuracy = self._evaluate_benchmark(data_loader)
                results[benchmark_name] = {"accuracy": accuracy}

        return results

    def _evaluate_perplexity(self) -> float:
        """
        Compute perplexity over the language modeling dataset.

        Returns:
            float: The model's perplexity on the evaluation dataset.
        """
        total_loss = 0.0
        total_tokens = 0

        criterion = torch.nn.CrossEntropyLoss(reduction="sum")  # Summed loss for averaging

        print("Computing Perplexity...")
        for batch in tqdm(self.data["language_modeling"], desc="Perplexity Evaluation"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            with torch.no_grad():
                logits = self.model(input_ids, attention_mask=attention_mask)

            # Shift logits and labels for language modeling loss
            logits = logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()

            # Compute loss
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item()
            total_tokens += labels.numel()

        perplexity = np.exp(total_loss / total_tokens)
        print(f"Perplexity: {perplexity:.4f}")
        return perplexity

    def _evaluate_benchmark(self, data_loader: DataLoader) -> float:
        """
        Evaluate accuracy on a benchmark dataset.

        Args:
            data_loader (DataLoader): DataLoader for the specific benchmark.

        Returns:
            float: Accuracy score for the benchmark.
        """
        all_preds = []
        all_labels = []

        print("Evaluating Benchmark...")
        for batch in tqdm(data_loader, desc="Benchmark Evaluation"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            with torch.no_grad():
                logits = self.model(input_ids, attention_mask=attention_mask)

            # Get predictions
            preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.detach().cpu().numpy())

        # Compute accuracy
        accuracy = accuracy_score(all_labels, all_preds)
        print(f"Accuracy: {accuracy:.4f}")
        return accuracy

    def _tokenize_and_batch(self, text_data: List[str], max_length: int) -> DataLoader:
        """
        Tokenize raw text data and prepare DataLoader.

        Args:
            text_data (list): List of raw text strings.
            max_length (int): Maximum sequence length for tokenization.

        Returns:
            DataLoader: Batches of tokenized inputs for model evaluation.
        """
        print("Tokenizing and batching input data...")

        encodings = self.tokenizer(
            text_data,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        dataset = torch.utils.data.TensorDataset(
            encodings["input_ids"], encodings["attention_mask"]
        )
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return data_loader
