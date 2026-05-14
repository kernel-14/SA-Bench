# termination.py

from typing import List, Dict, Union
import random

class Termination:
    def __init__(self, termination_type: str, termination_params: Dict):
        """
        Initialize the Termination class.

        Args:
            termination_type (str): Type of termination ("fixed_ngram", "randomized_ngram", "parsing", "perplexity").
            termination_params (Dict): Parameters specific to the termination strategy.
        """
        self.termination_type = termination_type
        self.params = termination_params

        # Validate supported termination types
        self.supported_types = ["fixed_ngram", "randomized_ngram", "parsing", "perplexity"]
        if self.termination_type not in self.supported_types:
            raise ValueError(f"Termination type '{termination_type}' is not supported. Must be one of {self.supported_types}.")
        
        # Default values for unsupported or missing parameters can be initialized here
        self.default_fixed_ngram = 5
        self.default_randomized_sizes = [2, 3, 5, 10]
        self.max_tree_token_limit = self.params.get("max_tree_token_limit", 5)
        self.ppl_threshold = self.params.get("ppl_threshold", 1.5)

    def get_macro_actions(self, sequence: List[str], **kwargs) -> List[List[str]]:
        """
        Determine macro-actions based on the termination strategy.

        Args:
            sequence (List[str]): Token sequence.
            **kwargs: Additional metadata (e.g., perplexity scores, parsed trees).

        Returns:
            List[List[str]]: Macro-actions segmented from the sequence.
        """
        if self.termination_type == "fixed_ngram":
            n = self.params.get("fixed_ngram_size", self.default_fixed_ngram)
            return self.fixed_ngram(sequence, n)
        elif self.termination_type == "randomized_ngram":
            return self.randomized_ngram(sequence, self.default_randomized_sizes)
        elif self.termination_type == "parsing":
            tree = kwargs.get("parsed_tree")
            if tree is None:
                raise ValueError("Parsed tree is required for parsing-based termination.")
            return self.parsing_based(sequence, tree)
        elif self.termination_type == "perplexity":
            perplexity_scores = kwargs.get("perplexity_scores", [])
            if not perplexity_scores:
                raise ValueError("Perplexity scores are required for perplexity-based termination.")
            return self.perplexity_based(sequence, perplexity_scores)
        else:
            raise ValueError(f"Unknown termination type: {self.termination_type}")

    def fixed_ngram(self, sequence: List[str], n: int) -> List[List[str]]:
        """
        Fixed n-gram termination strategy.

        Args:
            sequence (List[str]): Token sequence.
            n (int): Fixed size for n-gram segmentation.

        Returns:
            List[List[str]]: Macro-actions segmented into fixed-sized chunks.
        """
        macro_actions = [sequence[i:i + n] for i in range(0, len(sequence), n)]
        return macro_actions

    def randomized_ngram(self, sequence: List[str], sizes: List[int]) -> List[List[str]]:
        """
        Randomized n-gram termination strategy.

        Args:
            sequence (List[str]): Token sequence.
            sizes (List[int]): List of possible n-gram sizes.

        Returns:
            List[List[str]]: Macro-actions segmented based on randomized sizes.
        """
        random.shuffle(sizes)  # Shuffle sizes to introduce randomness
        macro_actions, i = [], 0
        while i < len(sequence):
            size = random.choice(sizes)
            macro_actions.append(sequence[i:min(i + size, len(sequence))])
            i += size
        return macro_actions

    def parsing_based(self, sequence: List[str], tree: Dict) -> List[List[str]]:
        """
        Parsing-based termination strategy using a constituent tree.

        Args:
            sequence (List[str]): Token sequence.
            tree (Dict): Parsed constituent tree.

        Returns:
            List[List[str]]: Macro-actions segmented using the tree structure.
        """
        def dfs(node, macro_actions):
            if "tokens" in node and len(node["tokens"]) <= self.max_tree_token_limit:
                macro_actions.append(node["tokens"])
            for child in node.get("children", []):
                dfs(child, macro_actions)

        macro_actions = []
        dfs(tree, macro_actions)
        return macro_actions

    def perplexity_based(self, sequence: List[str], perplexity_scores: List[float]) -> List[List[str]]:
        """
        Perplexity-based termination strategy.

        Args:
            sequence (List[str]): Token sequence.
            perplexity_scores (List[float]): Perplexity scores for each token in the sequence.

        Returns:
            List[List[str]]: Macro-actions segmented using perplexity thresholds.
        """
        macro_actions, current_action = [], []
        for i, token in enumerate(sequence):
            current_action.append(token)
            if i + 1 < len(perplexity_scores) and perplexity_scores[i + 1] > self.ppl_threshold:
                macro_actions.append(current_action)
                current_action = []
        if current_action:  # Append remaining tokens as a macro-action
            macro_actions.append(current_action)
        return macro_actions
