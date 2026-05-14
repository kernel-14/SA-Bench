## data_structures.py
"""Core data container classes for the adversarial leaderboard manipulation project.

This module defines two foundational data structures used throughout the pipeline:

1. ResponseDataset: A three-level nested dict container for storing all collected
   LLM API responses, with JSON serialization for persistence across runs.

2. SimulationResult: A dataclass capturing the outcome of a single leaderboard
   manipulation simulation run, including vote/interaction counts and rank history.

This module has zero internal project dependencies and must remain importable
without any installed third-party packages (only stdlib is used).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


class ResponseDataset:
    """Three-level nested dict container for LLM API responses.

    Stores responses in the structure:
        data[category][prompt][model_name] = List[str]

    This mirrors the natural access pattern of the experiments:
    - Training-based detector iterates over (category, prompt, model) triples.
    - PCA visualization needs all model responses for a specific (category, prompt).
    - Identity-probing uses the same structure with category="identity_probing".

    The class is intentionally simple with no write-time validation — callers
    (DataCollector) are responsible for providing well-formed inputs.

    Example:
        >>> dataset = ResponseDataset()
        >>> dataset.add_responses("english", "Who are you?", "gpt-4o", ["I am GPT-4o."])
        >>> dataset.get_responses("english", "Who are you?", "gpt-4o")
        ['I am GPT-4o.']
        >>> dataset.get_all_models()
        ['gpt-4o']
        >>> dataset.get_all_categories()
        ['english']
    """

    def __init__(self) -> None:
        """Initialize an empty ResponseDataset.

        The internal data dict is empty; all nested levels are created lazily
        on first write via add_responses().
        """
        # Three-level nested dict: category -> prompt -> model_name -> responses.
        self.data: Dict[str, Dict[str, Dict[str, List[str]]]] = {}

    def add_responses(
        self,
        category: str,
        prompt: str,
        model_name: str,
        responses: List[str],
    ) -> None:
        """Store a list of responses for a (category, prompt, model_name) triple.

        Creates intermediate dict levels if they do not already exist. Calling
        this method twice for the same triple silently overwrites the previous
        list — the DataCollector is responsible for avoiding redundant writes
        by checking the cache first.

        Args:
            category: Prompt category name, e.g. "english", "math",
                "identity_probing". Must be a non-empty string.
            prompt: The prompt text sent to the model. May be arbitrarily long.
            model_name: Exact model identifier, e.g. "gpt-4o-2024-05-13".
            responses: List of response strings collected from the model.
                Typically 50 strings for training-based detector or 1000 for
                identity-probing, but no length constraint is enforced here.
        """
        # Use setdefault to create intermediate levels lazily without overwriting
        # existing data at higher levels.
        category_data: Dict[str, Dict[str, List[str]]] = self.data.setdefault(
            category, {}
        )
        prompt_data: Dict[str, List[str]] = category_data.setdefault(prompt, {})
        prompt_data[model_name] = responses

    def get_responses(
        self,
        category: str,
        prompt: str,
        model_name: str,
    ) -> List[str]:
        """Retrieve stored responses for a (category, prompt, model_name) triple.

        Returns an empty list rather than raising KeyError when any level of
        the key path is missing. This defensive pattern prevents crashes in
        partial-collection scenarios where data collection was interrupted.

        Args:
            category: Prompt category name.
            prompt: The prompt text.
            model_name: Exact model identifier.

        Returns:
            List of response strings, or [] if the key path does not exist.

        Example:
            >>> dataset = ResponseDataset()
            >>> dataset.get_responses("english", "Hello", "gpt-4o")
            []
            >>> dataset.add_responses("english", "Hello", "gpt-4o", ["Hi!"])
            >>> dataset.get_responses("english", "Hello", "gpt-4o")
            ['Hi!']
        """
        return (
            self.data
            .get(category, {})
            .get(prompt, {})
            .get(model_name, [])
        )

    def get_all_responses_for_prompt(
        self,
        category: str,
        prompt: str,
    ) -> Dict[str, List[str]]:
        """Return all model responses for a specific (category, prompt) pair.

        This is the primary access pattern for the training-based detector when
        building binary datasets — it needs all models' responses for a given
        prompt to sample negative examples from the pool of non-target models.

        Args:
            category: Prompt category name.
            prompt: The prompt text.

        Returns:
            Dict mapping model_name -> List[str] of responses for all models
            that have responses stored for this (category, prompt) pair.
            Returns {} if the category or prompt is not found.

        Example:
            >>> dataset = ResponseDataset()
            >>> dataset.add_responses("math", "2+2=?", "gpt-4o", ["4"])
            >>> dataset.add_responses("math", "2+2=?", "claude-3", ["4"])
            >>> dataset.get_all_responses_for_prompt("math", "2+2=?")
            {'gpt-4o': ['4'], 'claude-3': ['4']}
        """
        return self.data.get(category, {}).get(prompt, {})

    def get_all_models(self) -> List[str]:
        """Return a sorted list of all unique model names across all categories.

        Collects model names by iterating the full nested structure and taking
        the union across all (category, prompt) pairs. Sorting ensures
        deterministic ordering, which matters for reproducibility when
        constructing win matrices or iterating models in classifiers.

        Returns:
            Sorted list of unique model name strings. Returns [] if the dataset
            is empty.

        Example:
            >>> dataset = ResponseDataset()
            >>> dataset.add_responses("english", "Hi", "gpt-4o", ["Hello!"])
            >>> dataset.add_responses("math", "2+2", "claude-3", ["4"])
            >>> dataset.get_all_models()
            ['claude-3', 'gpt-4o']
        """
        models: set = set()
        for category_data in self.data.values():
            for prompt_data in category_data.values():
                models.update(prompt_data.keys())
        return sorted(list(models))

    def get_all_categories(self) -> List[str]:
        """Return a sorted list of all category names present in the dataset.

        Sorting ensures deterministic ordering across runs, which is important
        for reproducibility when iterating categories in the training-based
        detector evaluation loop.

        Returns:
            Sorted list of category name strings. Returns [] if the dataset
            is empty.

        Example:
            >>> dataset = ResponseDataset()
            >>> dataset.add_responses("math", "2+2", "gpt-4o", ["4"])
            >>> dataset.add_responses("english", "Hi", "gpt-4o", ["Hello!"])
            >>> dataset.get_all_categories()
            ['english', 'math']
        """
        return sorted(list(self.data.keys()))

    def get_all_prompts(self, category: str) -> List[str]:
        """Return a sorted list of all prompt strings for a given category.

        Convenience method used by TrainingBasedDetector when iterating over
        all prompts within a category to train per-prompt classifiers.

        Args:
            category: Prompt category name.

        Returns:
            Sorted list of prompt strings for the category. Returns [] if the
            category is not found.

        Example:
            >>> dataset = ResponseDataset()
            >>> dataset.add_responses("math", "2+2", "gpt-4o", ["4"])
            >>> dataset.add_responses("math", "3+3", "gpt-4o", ["6"])
            >>> dataset.get_all_prompts("math")
            ['2+2', '3+3']
        """
        return sorted(list(self.data.get(category, {}).keys()))

    def save(self, path: str) -> None:
        """Serialize the dataset to a JSON file at the given path.

        The nested dict structure (category -> prompt -> model_name -> responses)
        is JSON-native, so no custom serialization is needed. Creates parent
        directories if they do not exist. Writes with indent=2 for
        human-readability and debuggability.

        Args:
            path: File path for the output JSON file, e.g.
                "outputs/responses/dataset.json".

        Raises:
            OSError: If the file cannot be written (e.g., permission denied).

        Example:
            >>> dataset = ResponseDataset()
            >>> dataset.add_responses("english", "Hi", "gpt-4o", ["Hello!"])
            >>> dataset.save("/tmp/test_dataset.json")
        """
        # Ensure parent directory exists before attempting to write.
        parent_dir: str = os.path.dirname(path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.data, fh, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "ResponseDataset":
        """Deserialize a ResponseDataset from a JSON file.

        Returns a fresh empty ResponseDataset if the file does not exist,
        enabling resumable data collection — the caller can detect "no data
        yet" by checking len(dataset.get_all_categories()) == 0.

        Args:
            path: File path of the JSON file to load, e.g.
                "outputs/responses/dataset.json".

        Returns:
            A ResponseDataset populated with the loaded data, or an empty
            ResponseDataset if the file does not exist.

        Raises:
            json.JSONDecodeError: If the file exists but contains invalid JSON.
            OSError: If the file exists but cannot be read (e.g., permission denied).

        Example:
            >>> dataset = ResponseDataset.load("/nonexistent/path.json")
            >>> dataset.get_all_categories()
            []
        """
        if not os.path.exists(path):
            # Return empty dataset for resumable collection.
            return cls()

        with open(path, "r", encoding="utf-8") as fh:
            raw_data: Dict[str, Dict[str, Dict[str, List[str]]]] = json.load(fh)

        instance: ResponseDataset = cls()
        # Assign the loaded dict directly — the JSON structure maps exactly to
        # the internal nested dict format, so no transformation is needed.
        instance.data = raw_data
        return instance

    def __len__(self) -> int:
        """Return the total number of (category, prompt, model) triples stored.

        Useful for progress reporting and sanity checks during data collection.

        Returns:
            Total count of stored (category, prompt, model_name) entries.

        Example:
            >>> dataset = ResponseDataset()
            >>> len(dataset)
            0
            >>> dataset.add_responses("english", "Hi", "gpt-4o", ["Hello!"])
            >>> len(dataset)
            1
        """
        count: int = 0
        for category_data in self.data.values():
            for prompt_data in category_data.values():
                count += len(prompt_data)
        return count

    def __repr__(self) -> str:
        """Return a concise string representation for debugging.

        Returns:
            String showing category count, total prompt-model pairs, and
            the list of categories present.
        """
        categories: List[str] = self.get_all_categories()
        return (
            f"ResponseDataset("
            f"categories={categories}, "
            f"total_entries={len(self)})"
        )


@dataclass
class SimulationResult:
    """Outcome of a single leaderboard manipulation simulation run.

    Captures all metrics needed to reproduce Tables 4, 5, 8, and 9 from the
    paper, as well as the rank trajectory over time for visualization.

    Attributes:
        target_model: The model being attacked, e.g. "llama-13b". Used as the
            row label in the paper's simulation tables.
        target_rank: The absolute rank the attacker is trying to achieve,
            e.g. 79 (not the delta ↑50). Used as the column label in tables.
        achieved: Whether target_rank was reached before max_interactions.
            When False, n_votes and n_interactions reflect values at
            max_interactions. The Metrics class renders achieved=False as
            N/A in output tables, matching the paper's table format.
        n_votes: Total adversarial votes cast when the simulation stopped
            (either goal achieved or max_interactions reached). This is the
            primary metric in Tables 4(a) and 5(a).
        n_interactions: Total interactions (votes + abstentions) when the
            simulation stopped. This is the primary metric in Tables 4(b)
            and 5(b). Always >= n_votes since abstentions are counted.
        rank_history: List of target model ranks recorded at each
            eval_interval=1000 interaction checkpoint. Length equals
            n_interactions // eval_interval. Used by Visualizer.plot_rank_history.
        vote_history: List of cumulative vote counts at each eval_interval
            checkpoint, parallel to rank_history. Used for plotting vote
            efficiency over time.
        detection_accuracy: The detector accuracy used in this simulation run.
            Stored for ablation study bookkeeping (Table 8).
        non_target_strategy: The strategy used when the target model was not
            detected. One of "do_nothing", "random_upvote", "vote_tie",
            "vote_both_bad". Stored for ablation study bookkeeping (Table 9).
        direction: Attack direction, either "up" (boost model) or "down"
            (suppress model). Stored for result filtering and table generation.

    Example:
        >>> result = SimulationResult(
        ...     target_model="llama-13b",
        ...     target_rank=79,
        ...     achieved=True,
        ...     n_votes=1304,
        ...     n_interactions=85000,
        ... )
        >>> result.achieved
        True
        >>> result.n_votes
        1304
    """

    target_model: str
    target_rank: int
    achieved: bool
    n_votes: int
    n_interactions: int
    # Mutable list fields must use field(default_factory=list) to avoid the
    # shared mutable default argument pitfall in dataclasses.
    rank_history: List[int] = field(default_factory=list)
    vote_history: List[int] = field(default_factory=list)
    # Metadata fields for ablation study bookkeeping.
    detection_accuracy: float = 0.95
    non_target_strategy: str = "do_nothing"
    direction: str = "up"

    def to_dict(self) -> Dict[str, object]:
        """Serialize this SimulationResult to a plain dict for JSON output.

        Used by AttackSimulator when checkpointing simulation results to disk.
        The dataclasses.asdict() function would also work, but this explicit
        implementation avoids deep-copying the potentially large rank_history
        and vote_history lists.

        Returns:
            Dict with all fields as JSON-serializable Python primitives.

        Example:
            >>> result = SimulationResult("llama-13b", 79, True, 1304, 85000)
            >>> d = result.to_dict()
            >>> d["target_model"]
            'llama-13b'
            >>> d["achieved"]
            True
        """
        return {
            "target_model": self.target_model,
            "target_rank": self.target_rank,
            "achieved": self.achieved,
            "n_votes": self.n_votes,
            "n_interactions": self.n_interactions,
            "rank_history": self.rank_history,
            "vote_history": self.vote_history,
            "detection_accuracy": self.detection_accuracy,
            "non_target_strategy": self.non_target_strategy,
            "direction": self.direction,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "SimulationResult":
        """Deserialize a SimulationResult from a plain dict.

        Inverse of to_dict(). Used when loading checkpointed simulation results
        from JSON files written by AttackSimulator.

        Args:
            d: Dict produced by to_dict(), typically loaded via json.load().

        Returns:
            A SimulationResult instance with all fields populated from d.

        Raises:
            KeyError: If any required field is missing from d.
            TypeError: If a field value has an incompatible type.

        Example:
            >>> d = {"target_model": "llama-13b", "target_rank": 79,
            ...      "achieved": True, "n_votes": 1304, "n_interactions": 85000,
            ...      "rank_history": [], "vote_history": [],
            ...      "detection_accuracy": 0.95, "non_target_strategy": "do_nothing",
            ...      "direction": "up"}
            >>> result = SimulationResult.from_dict(d)
            >>> result.target_model
            'llama-13b'
        """
        return cls(
            target_model=str(d["target_model"]),
            target_rank=int(d["target_rank"]),
            achieved=bool(d["achieved"]),
            n_votes=int(d["n_votes"]),
            n_interactions=int(d["n_interactions"]),
            rank_history=list(d.get("rank_history", [])),  # type: ignore[arg-type]
            vote_history=list(d.get("vote_history", [])),  # type: ignore[arg-type]
            detection_accuracy=float(d.get("detection_accuracy", 0.95)),  # type: ignore[arg-type]
            non_target_strategy=str(d.get("non_target_strategy", "do_nothing")),
            direction=str(d.get("direction", "up")),
        )

    def n_votes_or_na(self) -> Optional[int]:
        """Return n_votes if the simulation achieved its goal, else None.

        Convenience method for Metrics.summarize_simulation_table, which
        renders None as "N/A" in output tables to match the paper's format.

        Returns:
            n_votes if achieved is True, None otherwise.

        Example:
            >>> SimulationResult("m", 1, True, 500, 10000).n_votes_or_na()
            500
            >>> SimulationResult("m", 1, False, 500, 10000).n_votes_or_na()
            # Returns None
        """
        return self.n_votes if self.achieved else None

    def n_interactions_or_na(self) -> Optional[int]:
        """Return n_interactions if the simulation achieved its goal, else None.

        Convenience method for Metrics.summarize_simulation_table, which
        renders None as "N/A" in output tables to match the paper's format.

        Returns:
            n_interactions if achieved is True, None otherwise.

        Example:
            >>> SimulationResult("m", 1, True, 500, 10000).n_interactions_or_na()
            10000
            >>> SimulationResult("m", 1, False, 500, 10000).n_interactions_or_na()
            # Returns None
        """
        return self.n_interactions if self.achieved else None
