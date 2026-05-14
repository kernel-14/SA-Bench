## evaluation.py
import torch
from torch.nn.functional import cross_entropy
from transformers import PreTrainedTokenizer
from typing import Tuple, Dict
from utilities import Utilities
import nltk
from nltk.translate.bleu_score import sentence_bleu


class Evaluation:
    """
    Responsible for evaluating a trained NGPT or GPT model on various metrics, including
    validation loss, BLEU score for translation tasks, perplexity for language modeling,
    and extrapolation performance across different context lengths.
    """

    def __init__(self, model: torch.nn.Module, data: Tuple[torch.Tensor, torch.Tensor], config: dict):
        """
        Initialize the evaluation class.

        Args:
            model (torch.nn.Module): Trained NGPT or GPT model instance.
            data (Tuple[torch.Tensor, torch.Tensor]): Evaluation dataset as (inputs, targets).
            config (dict): Configuration dictionary from `config.yaml`.
        """
        self.model = model
        self.data = data
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Initialize tokenizer from configuration settings
        dataset_config = config["dataset"]
        self.tokenizer = PreTrainedTokenizer.from_pretrained(dataset_config["tokenizer"])

        # Metrics and other settings
        self.metrics_config = config["evaluation"]["metrics"]
        self.context_length_tests = config["evaluation"]["context_length_test"]
        self.logger = Utilities()

    def evaluate(self) -> dict:
        """
        Evaluates the model on validation loss, perplexity, downstream tasks (e.g., BLEU score),
        and extrapolation ability. Aggregates all results into a structured report.

        Returns:
            dict: A dictionary containing evaluation results.
        """
        self.model.eval()  # Set the model to evaluation mode
        results = {}

        # Validation loss and perplexity
        validation_loss, validation_perplexity = self.compute_validation_loss()
        results["validation_loss"] = validation_loss
        results["validation_perplexity"] = validation_perplexity

        # Downstream task evaluation
        downstream_results = self.evaluate_downstream_tasks()
        results.update(downstream_results)

        # Length extrapolation evaluation
        extrapolation_results = self.evaluate_length_extrapolation()
        results.update(extrapolation_results)

        self.logger.log_metrics(results, step=0, log_file="evaluation_metrics.log")
        return results

    def compute_validation_loss(self) -> Tuple[float, float]:
        """
        Compute the validation loss and perplexity over the dataset.

        Returns:
            Tuple[float, float]: Validation loss and perplexity.
        """
        validation_loss = 0.0
        batch_count = 0
        inputs, targets = self.data
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        with torch.no_grad():
            for i in range(0, len(inputs), self.config["training"]["batch_size"]):
                input_batch = inputs[i:i + self.config["training"]["batch_size"]]
                target_batch = targets[i:i + self.config["training"]["batch_size"]]

                # Forward pass through the model
                logits = self.model(input_batch)
                loss = cross_entropy(logits.view(-1, logits.size(-1)), target_batch.view(-1))
                validation_loss += loss.item()
                batch_count += 1

        validation_loss /= batch_count
        validation_perplexity = torch.exp(torch.tensor(validation_loss))
        return validation_loss, validation_perplexity.item()

    def evaluate_downstream_tasks(self) -> dict:
        """
        Evaluate performance on downstream NLP tasks, including BLEU score for translation
        and other task-specific metrics.

        Returns:
            dict: Dictionary containing the downstream task results.
        """
        results = {}

        # BLEU score evaluation for translation tasks
        if "WMT14-FR-EN" in self.metrics_config["downstream_tasks"]:
            bleu_score = self.compute_bleu_score()
            results["bleu_score"] = bleu_score

        # Example: Add other downstream tasks (e.g., accuracy) if required
        # if "some_task" in self.metrics_config["downstream_tasks"]:
        #     accuracy = self.compute_task_accuracy()
        #     results["some_task_accuracy"] = accuracy

        return results

    def compute_bleu_score(self) -> float:
        """
        Compute the BLEU score for translation tasks (e.g., WMT14-FR-EN).

        Returns:
            float: BLEU score for the specified translation task.
        """
        inputs, targets = self.data
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        predictions = []
        references = []

        with torch.no_grad():
            for i in range(0, len(inputs), self.config["training"]["batch_size"]):
                input_batch = inputs[i:i + self.config["training"]["batch_size"]]
                target_batch = targets[i:i + self.config["training"]["batch_size"]]

                logits = self.model(input_batch)
                predicted_tokens = torch.argmax(logits, dim=-1)  # Get predicted indices
                predictions.extend(predicted_tokens.tolist())
                references.extend(target_batch.tolist())

        # Tokenize predictions and references for BLEU computation
        predictions = [self.tokenizer.decode(p, skip_special_tokens=True) for p in predictions]
        references = [[self.tokenizer.decode(r, skip_special_tokens=True)] for r in references]

        # Use NLTK BLEU implementation
        bleu_scores = [
            sentence_bleu([reference], prediction, weights=(0.25, 0.25, 0.25, 0.25))
            for prediction, reference in zip(predictions, references)
        ]
        return sum(bleu_scores) / len(bleu_scores)

    def evaluate_length_extrapolation(self) -> dict:
        """
        Evaluate model performance for context lengths outside its pretraining range.
        For example, test perplexity on lengths longer than 8k tokens.

        Returns:
            dict: Dictionary containing extrapolation results.
        """
        results = {}
        for length in self.context_length_tests:
            test_loss, test_perplexity = self._length_extrapolation_test(length)
            results[f"extrapolation_loss_{length}"] = test_loss
            results[f"extrapolation_perplexity_{length}"] = test_perplexity
        return results

    def _length_extrapolation_test(self, context_length: int) -> Tuple[float, float]:
        """
        Helper function to compute validation loss and perplexity for a specific context length.

        Args:
            context_length (int): Test context length.

        Returns:
            Tuple[float, float]: Loss and perplexity for the specified context length.
        """
        evaluation_dataset = self._truncate_dataset(context_length)
        inputs, targets = evaluation_dataset

        inputs, targets = inputs.to(self.device), targets.to(self.device)

        loss_sum = 0.0
        batch_count = 0

        with torch.no_grad():
            for i in range(0, len(inputs), self.config["training"]["batch_size"]):
                input_batch = inputs[i:i + self.config["training"]["batch_size"]]
                target_batch = targets[i:i + self.config["training"]["batch_size"]]

                logits = self.model(input_batch)
                loss = cross_entropy(logits.view(-1, logits.size(-1)), target_batch.view(-1))
                loss_sum += loss.item()
                batch_count += 1

        average_loss = loss_sum / batch_count
        perplexity = torch.exp(torch.tensor(average_loss))
        return average_loss, perplexity.item()

    def _truncate_dataset(self, context_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Truncate the evaluation dataset to match the specified context length.

        Args:
            context_length (int): Desired context length.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Truncated input and target tensors.
        """
        inputs, targets = self.data
        truncated_inputs = inputs[:, :context_length]
        truncated_targets = targets[:, :context_length]
        return truncated_inputs, truncated_targets
