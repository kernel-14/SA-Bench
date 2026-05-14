## dataset_loader.py
from typing import Dict, Any
from datasets import load_dataset, Dataset
import numpy as np
from transformers import AutoTokenizer
from utils import set_random_seed
from config_loader import ConfigLoader


class DatasetLoader:
    """
    Class responsible for loading, filtering, and preprocessing pretraining and adaptation datasets.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize DatasetLoader with the provided configuration dictionary.
        :param config: Configuration dictionary loaded from 'config.yaml'.
        """
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")  # GPT-NeoX tokenizer
        self.ngram_span = config["data"]["pretraining_dataset"]["filters"]["ngram_span"]
        self.repeated_ngram_threshold = config["data"]["pretraining_dataset"]["filters"]["repeated_ngram_threshold"]
        self.min_repository_stars = config["data"]["pretraining_dataset"]["filters"]["min_repository_stars"]
        self.code_word_threshold_1 = config["data"]["pretraining_dataset"]["filters"]["code_word_threshold"]  # 30%
        self.code_word_threshold_2 = config["data"]["pretraining_dataset"]["filters"].get("top_two_word_threshold", 50)
        self.max_seq_length = config["data"]["adaptation_dataset"]["max_token_length"]

    def load_pretraining_data(self) -> Dataset:
        """
        Load and preprocess the pretraining dataset according to the configuration.
        :return: Filtered and shuffled pretraining dataset.
        """
        sources = self.config["data"]["pretraining_dataset"]["sources"]
        filtered_datasets = []

        for source in sources:
            dataset = load_dataset(source)
            dataset = self.filter_ngram_repetition(dataset)
            if source == "dolma_1.7/starcoder":
                dataset = self.filter_code_repositories(dataset)
            filtered_datasets.append(dataset)

        combined_dataset = self._merge_datasets(filtered_datasets)
        combined_dataset = self.shuffle_data(combined_dataset)
        return combined_dataset

    def load_adaptation_data(self) -> Dict[str, Dataset]:
        """
        Load and preprocess the adaptation dataset for SFT and DPO.
        :return: A dictionary containing SFT and DPO datasets.
        """
        sft_sources = self.config["data"]["adaptation_dataset"]["sft_sources"]
        dpo_sources = self.config["data"]["adaptation_dataset"]["dpo_sources"]

        sft_dataset = self._load_filter_sources(sft_sources)
        dpo_dataset = self._load_filter_sources(dpo_sources)

        sft_dataset = self.truncate_length(sft_dataset, self.max_seq_length)
        dpo_dataset = self.truncate_length(dpo_dataset, self.max_seq_length)

        return {"sft": sft_dataset, "dpo": dpo_dataset}

    def filter_ngram_repetition(self, dataset: Dataset) -> Dataset:
        """
        Filter documents with consecutive repeated n-grams exceeding the threshold.
        :param dataset: Dataset to filter.
        :return: Filtered dataset.
        """
        def ngram_repetition_filter(sample):
            tokens = self.tokenizer.tokenize(sample["text"])
            n_grams = [tuple(tokens[i:i + n]) for n in range(1, self.ngram_span + 1) for i in range(len(tokens) - n + 1)]
            counts = {ngram: n_grams.count(ngram) for ngram in set(n_grams)}
            exceeds_threshold = any(count >= self.repeated_ngram_threshold for count in counts.values())
            return not exceeds_threshold

        dataset = dataset.filter(ngram_repetition_filter)
        return dataset

    def filter_code_repositories(self, dataset: Dataset) -> Dataset:
        """
        Filter documents from code repositories using GitHub-specific heuristics.
        :param dataset: Dataset to filter.
        :return: Filtered dataset.
        """
        def code_repo_filter(sample):
            tokens = self.tokenizer.tokenize(sample["text"])
            unique_tokens, counts = np.unique(tokens, return_counts=True)
            dominant_word_percentage = counts.max() / len(tokens) * 100
            top_two_word_percentage = sum(list(counts[np.argsort(counts)[-2:]]) / len(tokens) * 100)
            return (
                sample["stars"] >= self.min_repository_stars and
                dominant_word_percentage <= self.code_word_threshold_1 and
                top_two_word_percentage <= self.code_word_threshold_2
            )

        dataset = dataset.filter(code_repo_filter)
        return dataset

    def shuffle_data(self, dataset: Dataset) -> Dataset:
        """
        Shuffle the dataset entries randomly.
        :param dataset: Dataset to shuffle.
        :return: Shuffled dataset.
        """
        set_random_seed(self.config.get("global_seed", 42))
        return dataset.shuffle(seed=self.config.get("global_seed", 42))

    def truncate_length(self, dataset: Dataset, max_seq_length: int) -> Dataset:
        """
        Truncate sequences longer than the specified maximum token length.
        :param dataset: Dataset to truncate.
        :param max_seq_length: Maximum allowable token length.
        :return: Truncated dataset.
        """
        def truncate(sample):
            tokens = self.tokenizer.encode(sample["text"], truncation=True, max_length=max_seq_length)
            return {"input_ids": tokens}

        dataset = dataset.map(truncate)
        return dataset

    def _merge_datasets(self, datasets: list[Dataset]) -> Dataset:
        """
        Merge multiple datasets into one.
        :param datasets: List of datasets to merge.
        :return: Combined dataset.
        """
        return Dataset.from_dict(sum([dataset.to_dict() for dataset in datasets], []))

    def _load_filter_sources(self, sources: list[str]) -> Dataset:
        """
        Load and filter datasets from the given sources list.
        :param sources: List of dataset sources.
        :return: Processed dataset.
        """
        filtered_datasets = []
        for source in sources:
            dataset = load_dataset(source)
            filtered_datasets.append(dataset)
        return self._merge_datasets(filtered_datasets)

