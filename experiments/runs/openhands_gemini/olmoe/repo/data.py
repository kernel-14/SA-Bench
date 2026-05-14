
import random
from typing import List, Dict, Any, Iterator
import numpy as np
from collections import Counter
import torch

# Placeholder for Hugging Face datasets and tokenizers
# In a real scenario, these would be imported:
# from datasets import load_dataset, Dataset
# from transformers import AutoTokenizer

class DummyTokenizer:
    """A dummy tokenizer for simulation purposes."""
    def encode(self, text: str) -> List[int]:
        # Simple split by space and map to arbitrary integers
        return [hash(word) % 1000 for word in text.split()]

    def decode(self, tokens: List[int]) -> str:
        # Not used for filtering, so a simple placeholder is fine
        return " ".join([str(t) for t in tokens])

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.tokenizer = DummyTokenizer() # Replace with actual tokenizer

    def _n_gram_filter(self, text: str) -> bool:
        """
        Removes documents with a sequence of `ngram_filter_length` or more repeated n-grams.
        An n-gram is any span of 1 to 13 tokens.
        """
        tokens = self.tokenizer.encode(text)
        seq_len = len(tokens)
        
        for n in range(self.config.ngram_filter_min_ngram_size, self.config.ngram_filter_max_ngram_size + 1):
            if seq_len < n:
                continue
            n_grams = [tuple(tokens[i : i + n]) for i in range(seq_len - n + 1)]
            
            # Check for repeated n-gram sequences
            for i in range(len(n_grams) - self.config.ngram_filter_length + 1):
                # Check if the n-gram repeats itself 'ngram_filter_length' times consecutively
                current_ngram = n_grams[i]
                is_repeated = True
                for j in range(1, self.config.ngram_filter_length):
                    if i + j >= len(n_grams) or n_grams[i + j] != current_ngram:
                        is_repeated = False
                        break
                if is_repeated:
                    return False # Document contains repeated n-grams, filter it

        return True # Document passes the filter

    def _starcoder_filter(self, doc: Dict[str, Any]) -> bool:
        """
        Applies StarCoder specific filters:
        - Removes documents from a repository with fewer than `starcoder_min_github_stars`.
        - Removes documents whose most frequent word constitutes over `starcoder_max_most_frequent_word_ratio` of the document.
        - Removes documents whose top-2 most frequent words constitute over `starcoder_max_top2_frequent_words_ratio` of the document.
        """
        if doc.get("github_stars", 0) < self.config.starcoder_min_github_stars:
            return False

        text = doc.get("text", "")
        if not text:
            return False

        words = text.lower().split()
        if not words:
            return False

        word_counts = Counter(words)
        total_words = len(words)
        
        most_common_words = word_counts.most_common(2)

        if most_common_words:
            most_frequent_word_ratio = most_common_words[0][1] / total_words
            if most_frequent_word_ratio > self.config.starcoder_max_most_frequent_word_ratio:
                return False

        if len(most_common_words) >= 2:
            top2_frequent_words_ratio = (most_common_words[0][1] + most_common_words[1][1]) / total_words
            if top2_frequent_words_ratio > self.config.starcoder_max_top2_frequent_words_ratio:
                return False
        
        return True

    def preprocess_pretraining_data(self, raw_data_iterator: Iterator[Dict[str, Any]]) -> Iterator[str]:
        """
        Applies filtering and preprocessing for pretraining data.
        In a real scenario, `raw_data_iterator` would yield samples from loaded datasets.
        """
        for i, doc in enumerate(raw_data_iterator):
            text = doc.get("text", "")
            if not text:
                continue

            # Apply general n-gram filter
            if not self._n_gram_filter(text):
                continue
            
            # Apply StarCoder specific filters if source is StarCoder
            if doc.get("source") == "starcoder": # Assuming a 'source' key in doc
                if not self._starcoder_filter(doc):
                    continue
            
            # Further tokenization and formatting would happen here
            # For this reproduction, we'll yield the filtered text
            yield text
    
    def get_pretraining_dataloader(self, batch_size: int, seq_len: int, num_batches: int) -> Iterator[torch.Tensor]:
        """
        Simulates a pretraining dataloader.
        In a real scenario, this would load data from Hugging Face, apply processing,
        and yield batches of tokenized input_ids and labels.
        """
        # Placeholder for actual data loading and processing
        print("Simulating pretraining data loading and preprocessing...")
        
        dummy_texts = [
            "This is a sample document for testing the n-gram filter.",
            "This document has repeated n-grams: test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test.",
            "Another document for testing. It has varied content.",
            "starcoder code example with few stars", # github_stars = 1
            "def hello_world(): print('hello world') # A code snippet. Word print appears often.",
            "This is a very long document with many words. The quick brown fox jumps over the lazy dog. This is another sentence.",
            "This document contains python code. Python is a programming language. Python is very popular. Python code is used everywhere. Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python."
        ]

        # Simulate raw data with 'source' and 'github_stars' for starcoder filter
        raw_data = []
        for i, text in enumerate(dummy_texts):
            doc = {"text": text, "source": "dummy"}
            if "starcoder" in text:
                doc["source"] = "starcoder"
                doc["github_stars"] = 1 if "few stars" in text else 5 # Simulate star count
            if "python" in text.lower():
                 doc["source"] = "starcoder"
                 doc["github_stars"] = 5
            raw_data.append(doc)

        processed_texts = list(self.preprocess_pretraining_data(iter(raw_data)))
        print(f"Processed {len(processed_texts)} documents after filtering.")

        # Simulate tokenization and batching
        all_tokens = []
        for text in processed_texts:
            encoded = self.tokenizer.encode(text)
            all_tokens.extend(encoded)
        
        # Ensure enough tokens for a batch
        if len(all_tokens) < batch_size * seq_len:
            print("Warning: Not enough dummy tokens for a full batch. Repeating data.")
            all_tokens = (all_tokens * (batch_size * seq_len // len(all_tokens) + 1))[:batch_size * seq_len]

        # Simulate yielding batches
        for i in range(num_batches):
            start_idx = i * batch_size * seq_len
            if start_idx + batch_size * seq_len > len(all_tokens):
                start_idx = 0 # Loop back or handle end of data
            
            batch_tokens = all_tokens[start_idx : start_idx + batch_size * seq_len]
            input_ids = torch.tensor(batch_tokens, dtype=torch.long).view(batch_size, seq_len)
            labels = input_ids.clone() # For language modeling, labels are usually next token
            
            yield {"input_ids": input_ids, "labels": labels}

    def get_adaptation_dataloader(self, batch_size: int, seq_len: int, is_dpo: bool = False, num_batches: int = 10):
        """
        Simulates an adaptation dataloader.
        In a real scenario, this would load instruction/preference tuning data.
        """
        print(f"Simulating adaptation data loading for {'DPO' if is_dpo else 'SFT'}...")
        
        dummy_instruction_data = [
            {"instruction": "Tell me a joke.", "response": "Why don't scientists trust atoms? Because they make up everything!"},
            {"instruction": "Write a python function to add two numbers.", "response": "def add(a, b): return a + b"},
        ]
        dummy_dpo_data = [
            {"prompt": "What is the capital of France?", "chosen": "Paris", "rejected": "Berlin"},
            {"prompt": "Tell me about large language models.", "chosen": "LLMs are powerful...", "rejected": "They are small..."},
        ]

        data = dummy_dpo_data if is_dpo else dummy_instruction_data
        
        for i in range(num_batches):
            # Simulate tokenization and batching
            batch_data = random.sample(data, min(batch_size, len(data)))
            
            input_ids_list = []
            labels_list = []
            
            for item in batch_data:
                if is_dpo:
                    # DPO typically requires tokenizing prompt, chosen, and rejected
                    # For simplicity, we'll tokenize chosen/rejected as separate sequences
                    prompt_tokens = self.tokenizer.encode(item["prompt"])
                    chosen_tokens = self.tokenizer.encode(item["chosen"])
                    rejected_tokens = self.tokenizer.encode(item["rejected"])
                    
                    # Pad or truncate to seq_len for chosen and rejected
                    chosen_input_ids = (chosen_tokens * (seq_len // len(chosen_tokens) + 1))[:seq_len] if len(chosen_tokens) < seq_len else chosen_tokens[:seq_len]
                    rejected_input_ids = (rejected_tokens * (seq_len // len(rejected_tokens) + 1))[:seq_len] if len(rejected_tokens) < seq_len else rejected_tokens[:seq_len]

                    input_ids_list.append(torch.tensor(chosen_input_ids, dtype=torch.long))
                    labels_list.append(torch.tensor(rejected_input_ids, dtype=torch.long)) # Placeholder for rejected labels
                else:
                    # SFT: tokenize instruction + response
                    full_text = item["instruction"] + " " + item["response"]
                    tokens = self.tokenizer.encode(full_text)
                    
                    # Pad or truncate to seq_len
                    input_ids = (tokens * (seq_len // len(tokens) + 1))[:seq_len] if len(tokens) < seq_len else tokens[:seq_len]
                    
                    input_ids_list.append(torch.tensor(input_ids, dtype=torch.long))
                    labels_list.append(torch.tensor(input_ids, dtype=torch.long)) # For SFT, labels are usually next token
            
            # Pad to uniform sequence length if necessary
            max_len = max(len(ids) for ids in input_ids_list)
            padded_input_ids = torch.stack([torch.cat([ids, torch.zeros(seq_len - len(ids), dtype=torch.long)]) if len(ids) < seq_len else ids for ids in input_ids_list])
            padded_labels = torch.stack([torch.cat([ids, torch.zeros(seq_len - len(ids), dtype=torch.long)]) if len(ids) < seq_len else ids for ids in labels_list])

            if is_dpo:
                # For DPO, we usually yield chosen_input_ids, rejected_input_ids, and their attention masks
                yield {
                    "chosen_input_ids": padded_input_ids,
                    "rejected_input_ids": padded_labels, # This is a simplification; DPO is more complex
                    "attention_mask_chosen": (padded_input_ids != 0).long(),
                    "attention_mask_rejected": (padded_labels != 0).long(),
                }
            else:
                yield {
                    "input_ids": padded_input_ids,
                    "labels": padded_labels,
                    "attention_mask": (padded_input_ids != 0).long(),
                }

