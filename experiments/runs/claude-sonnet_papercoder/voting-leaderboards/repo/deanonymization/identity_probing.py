## deanonymization/identity_probing.py
"""Identity-probing detector for the adversarial leaderboard manipulation paper.

This module implements the identity-probing detector described in Section 2.2
and evaluated in Section 2.4.1 of the paper. The detector crafts prompts that
ask a model to reveal its identity, then checks whether the response contains
the model's name or organization via case-insensitive substring matching.

Paper alignment:
  - Section 2.2: "The attacker crafts a prompt P designed to elicit identifying
    information about the target model, e.g., its name."
  - Section 2.3: "The classifier predicts the model as a positive match if its
    name (e.g., 'Llama') or organization (e.g., 'Meta') appears anywhere in
    the response."
  - Section 2.4.1: "we report the average accuracy across 1,000 model queries."
  - Table 2: 7 selected models × 5 prompts, accuracy percentages.
  - Table 7 (Appendix B.1): All 22 models × 5 prompts, accuracy percentages.

The module-level constant IDENTITY_PROMPTS is imported by data_collector.py
and main.py and must not be redefined elsewhere in the project.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd

from config import Config, ModelConfig
from utils.logger import get_logger

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger: logging.Logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# IDENTITY_PROMPTS: The five identity-probing prompts from Section 2.3.
# Defined at module level (not inside the class) so that data_collector.py
# and main.py can import this constant directly without instantiating the
# detector class. The order determines column ordering in the output DataFrame
# and must match the paper's table column order exactly.
# ---------------------------------------------------------------------------
IDENTITY_PROMPTS: List[str] = [
    "Who are you?",
    "Which model are you?",
    "What is your model name?",
    "How should I refer to you as an AI?",
    "How would you define your role or identity as an AI?",
]


class IdentityProbingDetector:
    """Detects target model identity via keyword matching on identity-probing responses.

    Implements the identity-probing detector from Section 2.2 of the paper.
    Given a model's response to an identity-probing prompt, the detector checks
    whether the response contains any of the model's identifying keywords
    (name or organization) via case-insensitive substring matching.

    This class is stateless beyond the config reference — no model training or
    vocabulary learning is required. Each call to detect() or evaluate_model()
    is independent.

    Attributes:
        config: The global Config object from config.py. Used to access
            config.random_state and the model list for logging purposes.

    Example:
        >>> from config import Config, ModelConfig
        >>> config = Config.from_yaml("config.yaml")
        >>> detector = IdentityProbingDetector(config)
        >>> model_cfg = config.get_model_by_name("claude-3-5-sonnet-20240620")
        >>> detector.detect("I am Claude, made by Anthropic.", model_cfg)
        True
        >>> detector.detect("I am a helpful AI assistant.", model_cfg)
        False
    """

    def __init__(self, config: Config) -> None:
        """Initialize the IdentityProbingDetector with a Config instance.

        Stores the config reference for access to model configurations and
        random state. No heavy initialization is performed — the detector is
        stateless and ready to use immediately after construction.

        Args:
            config: The global Config object. Provides the model list
                (config.models) and random state (config.random_state).
                The model keyword lists (ModelConfig.keywords) are used
                during detection.
        """
        self.config: Config = config
        logger.info(
            "IdentityProbingDetector initialized with %d models and %d prompts.",
            len(config.models),
            len(IDENTITY_PROMPTS),
        )

    def _keyword_match(self, response: str, keywords: List[str]) -> bool:
        """Check whether any keyword appears as a substring in the response.

        Performs case-insensitive substring matching. Returns True on the first
        keyword match (short-circuit evaluation). This implements the paper's
        detection rule from Section 2.3: "The classifier predicts the model as
        a positive match if its name (e.g., 'Llama') or organization (e.g.,
        'Meta') appears anywhere in the response."

        No regex or word-boundary matching is used — plain Python 'in' operator
        on lowercased strings is correct per the paper's description.

        Args:
            response: The model's response string to check. May be empty (e.g.,
                if the model was safety-blocked by Google's API). An empty
                response always returns False.
            keywords: List of keyword strings to search for. Each keyword is
                checked as a case-insensitive substring of the response.
                Typically contains the model family name and organization name,
                e.g., ["Claude", "Anthropic"] or ["Llama", "Meta"].
                An empty keywords list always returns False.

        Returns:
            True if any keyword (case-insensitive) appears as a substring
            anywhere in the response. False if no keyword matches or if either
            the response or keywords list is empty.

        Example:
            >>> detector._keyword_match("I am Claude, made by Anthropic.", ["Claude", "Anthropic"])
            True
            >>> detector._keyword_match("I am a helpful AI assistant.", ["Claude", "Anthropic"])
            False
            >>> detector._keyword_match("", ["Claude", "Anthropic"])
            False
            >>> detector._keyword_match("I am Claude.", [])
            False
            >>> detector._keyword_match("I use llama architecture.", ["Llama", "Meta"])
            True
        """
        # Guard against empty inputs — no match possible.
        if not response or not keywords:
            return False

        # Lowercase the response once to avoid repeated lowercasing in the loop.
        response_lower: str = response.lower()

        for keyword in keywords:
            # Lowercase each keyword for case-insensitive comparison.
            kw_lower: str = keyword.lower()
            if kw_lower in response_lower:
                # Short-circuit: return True on first match.
                return True

        return False

    def detect(self, response: str, model_config: ModelConfig) -> bool:
        """Detect whether a response was produced by the target model.

        Thin wrapper around _keyword_match that accepts a ModelConfig object
        rather than a raw keyword list. This is the primary public detection
        method used during live attack simulation.

        Args:
            response: The model's response string to check for identifying
                keywords.
            model_config: Configuration for the target model. Provides the
                keywords list (model_config.keywords) used for matching.

        Returns:
            True if the response contains any of the model's identifying
            keywords (case-insensitive substring match). False otherwise.

        Example:
            >>> model_cfg = ModelConfig(
            ...     name="claude-3-5-sonnet-20240620",
            ...     organization="Anthropic",
            ...     api_provider="anthropic",
            ...     keywords=["Claude", "Anthropic"],
            ...     max_tokens=512,
            ... )
            >>> detector.detect("I am Claude, an AI assistant by Anthropic.", model_cfg)
            True
            >>> detector.detect("I'm a helpful AI. How can I assist you?", model_cfg)
            False
        """
        return self._keyword_match(response, model_config.keywords)

    def evaluate_model(
        self,
        model_config: ModelConfig,
        responses_by_prompt: Dict[str, List[str]],
    ) -> Dict[str, float]:
        """Compute detection accuracy for each identity-probing prompt for one model.

        For each of the five identity-probing prompts, computes the fraction of
        responses where the model's keywords were detected. This implements the
        per-model evaluation described in Section 2.4.1: "we report the average
        accuracy across 1,000 model queries."

        Args:
            model_config: Configuration for the target model. Provides the
                keywords list used for detection.
            responses_by_prompt: Dict mapping each identity-probing prompt
                string to a list of response strings. Typically produced by
                DataCollector.collect_identity_probing_responses(). Each list
                should contain n_identity_queries (1,000) responses per the
                paper's experimental setup. If a prompt is missing from the
                dict or maps to an empty list, accuracy is recorded as 0.0.

        Returns:
            Dict mapping each identity-probing prompt string (from
            IDENTITY_PROMPTS) to a detection accuracy as a fraction in [0.0, 1.0].
            For example, {"Who are you?": 0.993, "Which model are you?": 1.0, ...}.
            Values are fractions, not percentages — conversion to percentages
            (×100) is performed in evaluate_all() when building the DataFrame.

        Example:
            >>> responses = {
            ...     "Who are you?": ["I am Claude.", "I'm Claude, an AI."] * 500,
            ...     "Which model are you?": ["I'm an AI assistant."] * 1000,
            ... }
            >>> accuracies = detector.evaluate_model(model_cfg, responses)
            >>> accuracies["Who are you?"]  # All 1000 responses matched
            1.0
            >>> accuracies["Which model are you?"]  # No responses matched
            0.0
        """
        accuracies: Dict[str, float] = {}

        for prompt in IDENTITY_PROMPTS:
            responses: List[str] = responses_by_prompt.get(prompt, [])

            if not responses:
                # No responses available for this prompt — record 0.0 and
                # continue. This handles incomplete data collection gracefully.
                logger.debug(
                    "No responses for model='%s', prompt='%s'. "
                    "Recording accuracy=0.0.",
                    model_config.name,
                    prompt,
                )
                accuracies[prompt] = 0.0
                continue

            # Count how many responses triggered a positive detection.
            n_detected: int = sum(
                1 for response in responses
                if self.detect(response, model_config)
            )

            # Accuracy = fraction of responses where the model was identified.
            accuracy: float = n_detected / len(responses)
            accuracies[prompt] = accuracy

            logger.debug(
                "model='%s', prompt='%s': %d/%d detected (accuracy=%.3f).",
                model_config.name,
                prompt,
                n_detected,
                len(responses),
                accuracy,
            )

        return accuracies

    def evaluate_all(
        self,
        identity_responses: Dict[str, Dict[str, List[str]]],
        model_configs: List[ModelConfig],
    ) -> pd.DataFrame:
        """Evaluate all models across all identity-probing prompts.

        Calls evaluate_model() for each model and assembles the results into
        a pandas DataFrame matching the format of Table 2 and Table 7 in the
        paper. Values are accuracy percentages (0.0–100.0), rounded to 1
        decimal place.

        Args:
            identity_responses: Nested dict with structure:
                identity_responses[model_name][prompt_string] = List[str]
                This matches the output of
                DataCollector.collect_all_identity_responses(). Models not
                present in this dict are evaluated with empty response lists
                (all accuracies = 0.0).
            model_configs: List of ModelConfig objects for all models to
                evaluate. Typically config.models (all 22 models). The order
                of this list determines the row order in the output DataFrame.

        Returns:
            pandas DataFrame with:
              - Index: model name strings (e.g., "claude-3-5-sonnet-20240620")
              - Columns: the 5 identity-probing prompt strings in IDENTITY_PROMPTS
                order
              - Values: detection accuracy percentages (float, 0.0–100.0),
                rounded to 1 decimal place
            This format matches Table 2 (7 selected models) and Table 7
            (all 22 models) in the paper.

        Example:
            >>> df = detector.evaluate_all(identity_responses, config.models)
            >>> df.loc["claude-3-5-sonnet-20240620", "Who are you?"]
            99.3
            >>> df.loc["gemini-1.5-flash", "Who are you?"]
            0.0
            >>> df.shape
            (22, 5)
        """
        logger.info(
            "Evaluating identity-probing detector for %d models × %d prompts.",
            len(model_configs),
            len(IDENTITY_PROMPTS),
        )

        # Accumulate per-model accuracy rows as a dict of dicts.
        # Structure: rows[model_name][prompt] = accuracy_percentage
        rows: Dict[str, Dict[str, float]] = {}

        for model_config in model_configs:
            model_name: str = model_config.name

            # Retrieve this model's responses from the nested dict.
            # Use empty dict as fallback if the model has no collected responses.
            responses_by_prompt: Dict[str, List[str]] = identity_responses.get(
                model_name, {}
            )

            if not responses_by_prompt:
                logger.warning(
                    "No identity-probing responses found for model='%s'. "
                    "All accuracies will be 0.0.",
                    model_name,
                )

            # Compute per-prompt accuracies as fractions [0.0, 1.0].
            accuracies_fraction: Dict[str, float] = self.evaluate_model(
                model_config, responses_by_prompt
            )

            # Convert fractions to percentages and round to 1 decimal place
            # to match the paper's table format (e.g., 99.3, 100.0, 0.0).
            rows[model_name] = {
                prompt: round(accuracy * 100.0, 1)
                for prompt, accuracy in accuracies_fraction.items()
            }

            logger.info(
                "model='%s': accuracies = %s",
                model_name,
                {p: f"{v:.1f}%" for p, v in rows[model_name].items()},
            )

        # Construct the DataFrame with models as rows and prompts as columns.
        # orient='index' means each key in rows becomes a row index.
        # Specifying columns=IDENTITY_PROMPTS ensures consistent column ordering
        # regardless of dict insertion order (Python 3.7+ preserves insertion
        # order, but explicit is safer for reproducibility).
        df: pd.DataFrame = pd.DataFrame.from_dict(
            rows,
            orient="index",
            columns=IDENTITY_PROMPTS,
        )

        # Ensure the DataFrame index name is set for clarity in CSV output.
        df.index.name = "Model"

        logger.info(
            "Identity-probing evaluation complete. "
            "DataFrame shape: %s. "
            "Mean accuracy across all models and prompts: %.1f%%.",
            df.shape,
            df.values.mean(),
        )

        return df

    def get_best_prompt_per_model(
        self,
        results_df: pd.DataFrame,
    ) -> Dict[str, str]:
        """Return the most effective identity-probing prompt for each model.

        Identifies the prompt that achieves the highest detection accuracy for
        each model. This corresponds to the boldfaced entries in Table 2 and
        Table 7 of the paper: "We highlight the most effective identity-probing
        prompt(s) for each model in boldface."

        Args:
            results_df: DataFrame produced by evaluate_all(), with model names
                as index and identity-probing prompts as columns. Values are
                accuracy percentages.

        Returns:
            Dict mapping model name strings to the prompt string that achieved
            the highest accuracy for that model. In case of ties, returns the
            first prompt (leftmost column) with the maximum value, which
            corresponds to "Who are you?" — the paper's finding that this is
            "the most effective prompt among the five options."

        Example:
            >>> best = detector.get_best_prompt_per_model(df)
            >>> best["claude-3-5-sonnet-20240620"]
            'Who are you?'
        """
        best_prompts: Dict[str, str] = {}

        for model_name in results_df.index:
            row: pd.Series = results_df.loc[model_name]
            # idxmax() returns the column label (prompt string) with the
            # maximum value. In case of ties, returns the first occurrence.
            best_prompt: str = str(row.idxmax())
            best_prompts[model_name] = best_prompt

        return best_prompts

    def filter_table_for_paper(
        self,
        results_df: pd.DataFrame,
        selected_models: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Filter the full results DataFrame to the subset shown in Table 2.

        Table 2 in the paper shows only 7 selected models (a representative
        subset of the full 22-model evaluation in Table 7). This method
        filters the full DataFrame to the requested subset of models.

        Args:
            results_df: Full DataFrame produced by evaluate_all() with all
                22 models as rows.
            selected_models: List of model name strings to include in the
                filtered output. If None, uses the 7 models shown in Table 2
                of the paper:
                  - claude-3-5-sonnet-20240620
                  - gemini-1.5-pro
                  - gpt-4o-mini-2024-07-18
                  - gemma-2-27b-it
                  - llama-3.1-70b-instruct
                  - mixtral-8x7b-instruct-v0.1
                  - qwen2-72b-instruct

        Returns:
            Filtered DataFrame containing only the rows for the selected models,
            in the order specified by selected_models. Models not present in
            results_df are silently skipped.

        Example:
            >>> table2_df = detector.filter_table_for_paper(full_df)
            >>> table2_df.shape
            (7, 5)
        """
        # Default to the 7 models shown in Table 2 of the paper.
        if selected_models is None:
            selected_models = [
                "claude-3-5-sonnet-20240620",
                "gemini-1.5-pro",
                "gpt-4o-mini-2024-07-18",
                "gemma-2-27b-it",
                "llama-3.1-70b-instruct",
                "mixtral-8x7b-instruct-v0.1",
                "qwen2-72b-instruct",
            ]

        # Filter to models that are actually present in the DataFrame.
        # This prevents KeyError if some models were not evaluated.
        available_models: List[str] = [
            m for m in selected_models if m in results_df.index
        ]

        if len(available_models) < len(selected_models):
            missing: List[str] = [
                m for m in selected_models if m not in results_df.index
            ]
            logger.warning(
                "filter_table_for_paper: %d requested models not found in "
                "results DataFrame: %s. Returning %d available models.",
                len(missing),
                missing,
                len(available_models),
            )

        return results_df.loc[available_models]
