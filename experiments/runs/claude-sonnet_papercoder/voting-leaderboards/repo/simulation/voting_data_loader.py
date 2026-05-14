```python
## simulation/voting_data_loader.py
"""Voting data loader for the Chatbot Arena leaderboard manipulation simulation.

This module provides the VotingDataLoader class, which ingests Chatbot Arena
voting records and exposes them in the formats required by the Bradley-Terry
fitting pipeline (simulation/bradley_terry.py) and the attack simulator
(simulation/attack_simulator.py).

Two data sources are supported:
  1. Local file (JSON or CSV) with columns ['model_a', 'model_b', 'winner'].
  2. HuggingFace dataset ID (e.g., 'lmsys/chatbot_arena_conversations') as a
     fallback when the full private dataset is unavailable.

Paper alignment:
  - Appendix A.4: "anonymized and deduplicated dataset of voting records from
    Chatbot Arena. The dataset includes 1,670,250 votes from 477,322 unique
    users, with 1,093,875 votes resulting in wins and 576,375 in ties. These
    votes cover 6,895 unique combinations of side-by-side model comparisons."
  - Section 3.1: "Chatbot Arena ranks models using Bradley-Terry coefficients
    derived from user interactions."
  - Section 3.1: "we iteratively simulate attacker interactions and adversarial
    votes with the system."

The pair sampling distribution is pre-computed once in __init__ to support
the high-frequency sample_pair() calls in the simulation inner loop
(up to 500,000 iterations per experiment).
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.logger import get_logger

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger: logging.Logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Expected column names in the normalized votes DataFrame.
# ---------------------------------------------------------------------------
_COL_MODEL_A: str = "model_a"
_COL_MODEL_B: str = "model_b"
_COL_WINNER: str = "winner"
_REQUIRED_COLUMNS: frozenset = frozenset({_COL_MODEL_A, _COL_MODEL_B, _COL_WINNER})

# ---------------------------------------------------------------------------
# Valid winner values after normalization.
# ---------------------------------------------------------------------------
_WINNER_MODEL_A: str = "model_a"
_WINNER_MODEL_B: str = "model_b"
_WINNER_TIE: str = "tie"
_VALID_WINNERS: frozenset = frozenset({_WINNER_MODEL_A, _WINNER_MODEL_B, _WINNER_TIE})

# ---------------------------------------------------------------------------
# Tie variant strings that should be normalized to "tie".
# ---------------------------------------------------------------------------
_TIE_VARIANTS: frozenset = frozenset({"tie (bothbad)", "tie(bothbad)", "both bad"})

# ---------------------------------------------------------------------------
# Minimum number of rows required to proceed without a warning.
# ---------------------------------------------------------------------------
_MIN_ROWS_WARNING_THRESHOLD: int = 1000


class VotingDataLoader:
    """Loads and preprocesses Chatbot Arena voting data for Bradley-Terry simulation.

    Supports two data sources:
      1. Local JSON or CSV file with columns ['model_a', 'model_b', 'winner'].
      2. HuggingFace dataset ID as a fallback (e.g., 'lmsys/chatbot_arena_conversations').

    Pre-computes the empirical pair sampling distribution once during
    initialization to support high-frequency sample_pair() calls in the
    simulation inner loop.

    Attributes:
        data_path: Path to the local data file or HuggingFace dataset ID.

    Example:
        >>> loader = VotingDataLoader("data/chatbot_arena_votes.json")
        >>> votes_df = loader.load_votes()
        >>> model_list = loader.get_model_list()
        >>> win_matrix, model_names = loader.get_win_matrix()
        >>> rng = np.random.default_rng(42)
        >>> pair = loader.sample_pair(rng)
        >>> isinstance(pair, tuple) and len(pair) == 2
        True
    """

    def __init__(self, data_path: str = "data/chatbot_arena_votes.json") -> None:
        """Initialize the VotingDataLoader and load voting data.

        Attempts to load from a local file first. If the local file does not
        exist, treats data_path as a HuggingFace dataset ID and loads via the
        datasets library. After loading, normalizes the schema, validates
        required columns, and pre-computes the pair sampling distribution.

        Args:
            data_path: Path to a local JSON/CSV file or a HuggingFace dataset
                ID string (e.g., 'lmsys/chatbot_arena_conversations'). Defaults
                to 'data/chatbot_arena_votes.json' per config.yaml
                simulation.voting_data_path.

        Raises:
            FileNotFoundError: If data_path is a local file path that does not
                exist and cannot be interpreted as a HuggingFace dataset ID.
            ValueError: If the loaded data is empty after normalization, or if
                required columns are missing from a local file.
            ImportError: If data_path is a HuggingFace ID but the 'datasets'
                library is not installed.
        """
        self.data_path: str = data_path

        # Internal state — populated during __init__.
        self._votes_df: pd.DataFrame = pd.DataFrame()
        self._model_list: List[str] = []
        self._pair_counts: Dict[Tuple[str, str], int] = {}
        self._pair_list: List[Tuple[str, str]] = []
        self._pair_probs: np.ndarray = np.array([])

        # Load and normalize the voting data.
        logger.info("VotingDataLoader: loading data from '%s'.", data_path)
        self._votes_df = self._load_and_normalize(data_path)

        # Validate that the DataFrame is non-empty after normalization.
        if self._votes_df.empty:
            raise ValueError(
                f"VotingDataLoader: loaded DataFrame is empty after normalization "
                f"from source '{data_path}'. Cannot proceed with simulation."
            )

        n_rows: int = len(self._votes_df)
        if n_rows < _MIN_ROWS_WARNING_THRESHOLD:
            logger.warning(
                "VotingDataLoader: only %d rows loaded from '%s'. "
                "Simulation results may not match the paper's scale "
                "(paper uses 1,670,250 votes).",
                n_rows,
                data_path,
            )

        # Pre-compute derived data structures for fast access during simulation.
        self._model_list = self._compute_model_list()
        self._pair_counts = self._compute_pair_counts()
        self._pair_list, self._pair_probs = self._compute_pair_sampling_distribution()

        logger.info(
            "VotingDataLoader initialized: %d votes, %d unique models, "
            "%d unique pairs.",
            n_rows,
            len(self._model_list),
            len(self._pair_list),
        )

        # Log pair frequency statistics for debugging.
        if len(self._pair_probs) > 0:
            logger.debug(
                "Pair sampling distribution: min_prob=%.6f, max_prob=%.6f, "
                "mean_prob=%.6f.",
                float(self._pair_probs.min()),
                float(self._pair_probs.max()),
                float(self._pair_probs.mean()),
            )

    # -----------------------------------------------------------------------
    # Public interface methods
    # -----------------------------------------------------------------------

    def load_votes(self) -> pd.DataFrame:
        """Return the normalized votes DataFrame.

        Returns the cleaned and schema-normalized voting records loaded during
        __init__. The DataFrame has exactly three columns:
          - 'model_a': name of the first model in the comparison
          - 'model_b': name of the second model in the comparison
          - 'winner': one of 'model_a', 'model_b', 'tie'

        This is the primary accessor for downstream consumers that need the
        raw voting history (e.g., MaliciousUserDetector.fit_benign_distribution).

        Returns:
            pd.DataFrame with columns ['model_a', 'model_b', 'winner'].
            All winner values are normalized to {'model_a', 'model_b', 'tie'}.
            The DataFrame is a copy to prevent accidental mutation of internal state.

        Example:
            >>> loader = VotingDataLoader("data/votes.json")
            >>> df = loader.load_votes()
            >>> set(df.columns) == {'model_a', 'model_b', 'winner'}
            True
            >>> set(df['winner'].unique()).issubset({'model_a', 'model_b', 'tie'})
            True
        """
        return self._votes_df.copy()

    def get_model_list(self) -> List[str]:
        """Return a sorted list of all unique model names in the voting data.

        Collects all model names appearing in either the 'model_a' or 'model_b'
        columns and returns them sorted alphabetically. Sorting ensures
        deterministic ordering, which is critical for consistent win matrix
        indexing across calls.

        Returns:
            Sorted list of unique model name strings. The order of this list
            defines the row/column indices of the win matrix returned by
            get_win_matrix().

        Example:
            >>> loader = VotingDataLoader("data/votes.json")
            >>> models = loader.get_model_list()
            >>> models == sorted(models)  # Always sorted
            True
            >>> len(models) > 0
            True
        """
        return list(self._model_list)

    def get_win_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """Construct the N×N win count matrix for Bradley-Terry model fitting.

        Builds a float matrix where win_matrix[i][j] is the number of times
        model i beat model j. Ties are split 0.5/0.5 per Chatbot Arena
        convention: a tie between models i and j adds 0.5 to win_matrix[i][j]
        and 0.5 to win_matrix[j][i].

        This matrix is consumed by BradleyTerryModel.fit() via the choix
        library's ilsr_pairwise function, which expects pairwise comparison
        counts in this format.

        Returns:
            Tuple of (win_matrix, model_names) where:
              - win_matrix: np.ndarray of shape (N, N) with dtype float64.
                win_matrix[i][j] = number of times model i beat model j
                (including 0.5 contributions from ties).
              - model_names: List[str] of length N giving the model name for
                each row/column index. Identical to get_model_list().

        Example:
            >>> loader = VotingDataLoader("data/votes.json")
            >>> matrix, names = loader.get_win_matrix()
            >>> matrix.shape == (len(names), len(names))
            True
            >>> matrix.dtype == np.float64
            True
            >>> np.all(np.diag(matrix) == 0.0)  # No self-comparisons
            True
        """
        model_names: List[str] = self.get_model_list()
        n_models: int = len(model_names)

        if n_models == 0:
            logger.warning("get_win_matrix: no models found. Returning empty matrix.")
            return np.zeros((0, 0), dtype=np.float64), []

        # Build index lookup for O(1) model name -> index mapping.
        model_to_idx: Dict[str, int] = {
            name: idx for idx, name in enumerate(model_names)
        }

        # Initialize win matrix with zeros.
        win_matrix: np.ndarray = np.zeros((n_models, n_models), dtype=np.float64)

        # Use vectorized groupby aggregation for performance on 1.67M rows.
        # This avoids slow row-by-row iteration with .iterrows().
        df: pd.DataFrame = self._votes_df

        # --- Process model_a wins ---
        wins_a: pd.Series = (
            df[df[_COL_WINNER] == _WINNER_MODEL_A]
            .groupby([_COL_MODEL_A, _COL_MODEL_B])
            .size()
        )
        for (ma, mb), count in wins_a.items():
            idx_a: Optional[int] = model_to_idx.get(str(ma))
            idx_b: Optional[int] = model_to_idx.get(str(mb))
            if idx_a is not None and idx_b is not None:
                win_matrix[idx_a, idx_b] += float(count)

        # --- Process model_b wins ---
        wins_b: pd.Series = (
            df[df[_COL_WINNER] == _WINNER_MODEL_B]
            .groupby([_COL_MODEL_A, _COL_MODEL_B])
            .size()
        )
        for (ma, mb), count in wins_b.items():
            idx_a = model_to_idx.get(str(ma))
            idx_b = model_to_idx.get(str(mb))
            if idx_a is not None and idx_b is not None:
                # model_b won: add to win_matrix[idx_b, idx_a]
                win_matrix[idx_b, idx_a] += float(count)

        # --- Process ties (split 0.5/0.5) ---
        ties: pd.Series = (
            df[df[_COL_WINNER] == _WINNER_TIE]
            .groupby([_COL_MODEL_A, _COL_MODEL_B])
            .size()
        )
        for (ma, mb), count in ties.items():
            idx_a = model_to_idx.get(str(ma))
            idx_b = model_to_idx.get(str(mb))
            if idx_a is not None and idx_b is not None:
                win_matrix[idx_a, idx_b] += 0.5 * float(count)
                win_matrix[idx_b, idx_a] += 0.5 * float(count)

        total_wins: float = float(win_matrix.sum())
        logger.info(
            "get_win_matrix: shape=(%d, %d), total_win_counts=%.1f.",
            n_models,
            n_models,
            total_wins,
        )

        return win_matrix, model_names

    def get_model_vote_counts(self) -> Dict[str, int]:
        """Return the total number of appearances (votes) per model.

        Counts how many times each model appeared in any comparison, regardless
        of outcome. A model appearing as model_a in one row and model_b in
        another row is counted twice (once per appearance).

        This matches the paper's reported vote counts in Tables 4 and 5, e.g.,
        'chatgpt-4o-latest' has 14,514 votes (appearances).

        Returns:
            Dict mapping model name strings to their total appearance count
            (int). Models with zero appearances are not included.

        Example:
            >>> loader = VotingDataLoader("data/votes.json")
            >>> counts = loader.get_model_vote_counts()
            >>> all(isinstance(v, int) for v in counts.values())
            True
            >>> all(v > 0 for v in counts.values())
            True
        """
        df: pd.DataFrame = self._votes_df

        # Count appearances in model_a column.
        counts_a: pd.Series = df[_COL_MODEL_A].value_counts()
        # Count appearances in model_b column.
        counts_b: pd.Series = df[_COL_MODEL_B].value_counts()

        # Sum both counts, filling missing models with 0.
        combined: pd.Series = counts_a.add(counts_b, fill_value=0).astype(int)

        return combined.to_dict()

    def get_pair_counts(self) -> Dict[Tuple[str, str], int]:
        """Return the empirical frequency of each unordered model pair.

        Counts how many times each unordered pair of models appeared together
        in a comparison. Pairs are normalized so that the alphabetically smaller
        model name is always first (model_a < model_b), ensuring each pair is
        counted once regardless of which model was assigned to position A or B.

        Returns:
            Dict mapping (model_a, model_b) tuples (with model_a < model_b
            alphabetically) to their appearance count (int). The paper reports
            6,895 unique pairs in Appendix A.4.

        Example:
            >>> loader = VotingDataLoader("data/votes.json")
            >>> pair_counts = loader.get_pair_counts()
            >>> all(k[0] < k[1] for k in pair_counts.keys())  # Normalized order
            True
            >>> all(v > 0 for v in pair_counts.values())
            True
        """
        return dict(self._pair_counts)

    def sample_pair(self, rng: np.random.Generator) -> Tuple[str, str]:
        """Sample a model pair proportional to empirical pair frequencies.

        Uses the pre-computed normalized probability distribution over all
        observed model pairs to sample one pair. This simulates the arena's
        non-uniform model selection (popular/newer models appear more often).

        This method is called in the inner simulation loop (up to 500,000
        times per experiment), so it uses pre-computed arrays for O(1) sampling.

        Args:
            rng: A numpy.random.Generator instance (created with
                np.random.default_rng(seed)) for reproducible sampling.
                The caller (AttackSimulator) is responsible for creating and
                managing this RNG to ensure reproducibility across runs.

        Returns:
            Tuple (model_a, model_b) where model_a < model_b alphabetically
            (normalized pair order). The caller should treat this as an
            unordered pair — either model could be the attack target.

        Raises:
            RuntimeError: If no pairs are available (empty pair list), which
                indicates the voting data was not loaded correctly.

        Example:
            >>> loader = VotingDataLoader("data/votes.json")
            >>> rng = np.random.default_rng(42)
            >>> pair = loader.sample_pair(rng)
            >>> isinstance(pair, tuple) and len(pair) == 2
            True
            >>> pair[0] < pair[1]  # Normalized alphabetical order
            True
        """
        if len(self._pair_list) == 0:
            raise RuntimeError(
                "VotingDataLoader.sample_pair: no pairs available. "
                "Ensure voting data was loaded correctly."
            )

        # Sample one index from the pair list using pre-computed probabilities.
        # rng.choice with p= is O(N) where N = len(_pair_list) = ~6,895.
        idx: int = int(rng.choice(len(self._pair_list), p=self._pair_probs))
        return self._pair_list[idx]

    # -----------------------------------------------------------------------
    # Private loading and normalization methods
    # -----------------------------------------------------------------------

    def _load_and_normalize(self, data_path: str) -> pd.DataFrame:
        """Load voting data from a local file or HuggingFace dataset and normalize schema.

        Dispatches to the appropriate loader based on whether data_path points
        to an existing local file or should be treated as a HuggingFace dataset ID.

        Args:
            data_path: Local file path (JSON/CSV) or HuggingFace dataset ID.

        Returns:
            Normalized pd.DataFrame with columns ['model_a', 'model_b', 'winner']
            and winner values in {'model_a', 'model_b', 'tie'}.

        Raises:
            FileNotFoundError: If data_path looks like a local file path but
                the file does not exist.
            ValueError: If required columns are missing from a local file.
        """
        # Determine whether to load from local file or HuggingFace.
        is_local_file: bool = self._is_local_file_path(data_path)

        if is_local_file:
            if not os.path.exists(data_path):
                raise FileNotFoundError(
                    f"VotingDataLoader: local file not found at '{data_path}'. "
                    f"Set simulation.voting_data_path in config.yaml to a valid "
                    f"path or use the HuggingFace fallback ID "
                    f"'lmsys/chatbot_arena_conversations'."
                )
            logger.info(
                "VotingDataLoader: loading from local file '%s'.", data_path
            )
            df: pd.DataFrame = self._load_local_file(data_path)
        else:
            # Treat data_path as a HuggingFace dataset ID.
            logger.info(
                "VotingDataLoader: '%s' is not a local file. "
                "Attempting to load as HuggingFace dataset ID.",
                data_path,
            )
            df = self._load_huggingface_dataset(data_path)

        # Normalize the winner column values.
        df = self._normalize_winner_column(df)

        # Drop rows with invalid winner values after normalization.
        n_before: int = len(df)
        df = df[df[_COL_WINNER].isin(_VALID_WINNERS)].copy()
        n_dropped: int = n_before - len(df)
        if n_dropped > 0:
            logger.warning(
                "VotingDataLoader: dropped %d rows with invalid winner values "
                "after normalization.",
                n_dropped,
            )

        # Reset index after filtering.
        df = df.reset_index(drop=True)

        logger.info(
            "VotingDataLoader: loaded and normalized %d rows from '%s'.",
            len(df),
            data_path,
        )

        return df

    def _is_local_file_path(self, path: str) -> bool:
        """Determine whether a path string refers to a local file.

        A path is treated as a local file path if it:
          - Contains a path separator (/ or \\), OR
          - Ends with a known file extension (.json, .csv, .jsonl), OR
          - Starts with './' or '../'

        HuggingFace dataset IDs typically look like 'org/dataset-name' with
        exactly one '/' and no file extension.

        Args:
            path: The path string to classify.

        Returns:
            True if the path should be treated as a local file path.
            False if it should be treated as a HuggingFace dataset ID.
        """
        # Check for file extensions that indicate a local file.
        lower_path: str = path.lower()
        if any(lower_path.endswith(ext) for ext in (".json", ".csv", ".jsonl")):
            return True

        # Check for relative path indicators.
        if path.startswith("./") or path.startswith("../"):
            return True

        # Check for absolute path indicators.
        if os.path.isabs(path):
            return True

        # Check if the file actually exists on disk (handles paths without extensions).
        if os.path.exists(path):
            return True

        # Default: treat as HuggingFace dataset ID.
        return False

    def _load_local_file(self, data_path: str) -> pd.DataFrame:
        """Load a local JSON or CSV file into a DataFrame.

        Validates that the required columns ['model_a', 'model_b', 'winner']
        are present after loading.

        Args:
            data_path: Path to a local JSON or CSV file.

        Returns:
            pd.DataFrame with at least the required columns.

        Raises:
            ValueError: If required columns are missing.
        """
        lower_path: str = data_path.lower()

        if lower_path.endswith(".json") or lower_path.endswith(".jsonl"):
            try:
                df: pd.DataFrame = pd.read_json(data_path)
            except ValueError:
                # Try reading as JSON Lines format.
                df = pd.read_json(data_path, lines=True)
        elif lower_path.endswith(".csv"):
            df = pd.read_csv(data_path)
        else:
            # Try JSON first, then CSV as fallback.
            try:
                df = pd.read_json(data_path)
            except ValueError:
                df = pd.read_csv(data_path)

        # Validate required columns.
        missing_cols: set = _REQUIRED_COLUMNS - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"VotingDataLoader: local file '{data_path}' is missing "
                f"required columns: {sorted(missing_cols)}. "
                f"Found columns: {sorted(df.columns.tolist())}."
            )

        # Keep only the required columns to reduce memory usage.
        df = df[[_COL_MODEL_A, _COL_MODEL_B, _COL_WINNER]].copy()

        # Ensure string types for model name columns.
        df[_COL_MODEL_A] = df[_COL_MODEL_A].astype(str)
        df[_COL_MODEL_B] = df[_COL_MODEL_B].astype(str)
        df[_COL_WINNER] = df[_COL_WINNER].astype(str)

        logger.info(
            "_load_local_file: loaded %d rows from '%s'.", len(df), data_path
        )
        return df

    def _load_huggingface_dataset(self, dataset_id: str) -> pd.DataFrame:
        """Load voting data from a HuggingFace dataset.

        Handles the schema differences between the expected format and the
        lmsys/chatbot_arena_conversations dataset. Maps the HuggingFace
        schema to the expected ['model_a', 'model_b', 'winner'] format.

        Args:
            dataset_id: HuggingFace dataset identifier, e.g.
                'lmsys/chatbot_arena_conversations'.

        Returns:
            pd.DataFrame with columns ['model_a', 'model_b', 'winner'].

        Raises:
            ImportError: If the 'datasets' library is not installed.
            Exception: If the dataset cannot be loaded from HuggingFace.
        """
        try:
            import datasets as hf_datasets  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "VotingDataLoader: the 'datasets' library is required to load "
                "HuggingFace datasets. Install it with: pip install datasets"
            ) from exc

        logger.info(
            "_load_huggingface_dataset: loading dataset '%s' from HuggingFace.",
            dataset_id,
        )

        try:
            dataset = hf_datasets.load_dataset(
                dataset_id,
                split="train",
                trust_remote_code=True,
            )
        except Exception as exc:
            logger.error(
                "_load_huggingface_dataset: failed to load '%s': %s",
                dataset_id,
                exc,
            )
            raise

        # Convert to pandas DataFrame.
        df: pd.DataFrame = dataset.to_pandas()

        logger.info(
            "_load_huggingface_dataset: loaded %d rows with columns: %s.",
            len(df),
            sorted(df.columns.tolist()),
        )

        # Map HuggingFace schema to expected schema.
        # The lmsys/chatbot_arena_conversations dataset has fields:
        #   - 'model_a': name of model A (may already exist)
        #   - 'model_b': name of model B (may already exist)
        #   - 'winner': 'model_a', 'model_b', 'tie', 'tie (bothbad)'
        # Check if the required columns already exist.
        existing_cols: set = set(df.columns)

        if _REQUIRED_COLUMNS.issubset(existing_cols):
            # All required columns present — use them directly.
            logger.info(
                "_load_huggingface_dataset: required columns already present."
            )
            df = df[[_COL_MODEL_A, _COL_MODEL_B, _COL_WINNER]].copy()
        else:
            # Attempt to map from alternative column names.
            df = self._map_hf_schema(df)

        # Ensure string types.
        df[_COL_MODEL_A] = df[_COL_MODEL_A].astype(str)
        df[_COL_MODEL_B] = df[_COL_MODEL_B].astype(str)
        df[_COL_WINNER] = df[_COL_WINNER].astype(str)

        # Drop duplicate rows to match the paper's "deduplicated" dataset.
        n_before: int = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        n_deduped: int = n_before - len(df)
        if n_deduped > 0:
            logger.info(
                "_load_huggingface_dataset: removed %d duplicate rows.",
                n_deduped,
            )

        logger.info(
            "_load_huggingface_dataset: final DataFrame has %d rows.", len(df)
        )
        return df

    def _map_hf_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map alternative HuggingFace column names to the expected schema.

        Handles cases where the HuggingFace dataset uses different column names
        than the expected ['model_a', 'model_b', 'winner'] format.

        Args:
            df: Raw DataFrame from HuggingFace with potentially different columns.

        Returns:
            DataFrame with columns ['model_a', 'model_b', 'winner'].

        Raises:
            ValueError: If the required columns cannot be mapped from the
                available columns.
        """
        cols: set = set(df.columns)
        result: Dict[str, pd.Series] = {}

        # Map model_a column.
        if _COL_MODEL_A in cols:
            result[_COL_MODEL_A] = df[_COL_MODEL_A]
        else:
            # Try alternative names.
            for alt in ("model_a_name", "model1", "model_1"):
                if alt in cols:
                    result[_COL_MODEL_A] = df[alt]
                    logger.info(
                        "_map_hf_schema: mapped '%s' -> 'model_a'.", alt
                    )
                    break
            else:
                raise ValueError(
                    f"_map_hf_schema: cannot find 'model_a' column. "
                    f"Available columns: {sorted(cols)}."
                )

        # Map model_b column.
        if _COL_MODEL_B in cols:
            result[_COL_MODEL_B] = df[_COL_MODEL_B]
        else:
            for alt in ("model_b_name", "model2", "model_2"):
                if alt in cols:
                    result[_COL_MODEL_B] = df[alt]
                    logger.info(
                        "_map_hf_schema: mapped '%s' -> 'model_b'.", alt
                    )
                    break
            else:
                raise ValueError(
                    f"_map_hf_schema: cannot find 'model_b' column. "
                    f"Available columns: {sorted(cols)}."
                )

        # Map winner column.
        if _COL_WINNER in cols:
            result[_COL_WINNER] = df[_COL_WINNER]
        else:
            for alt in ("winner_model", "result", "preference"):
                if alt in cols:
                    result[_COL_WINNER] = df[alt]
                    logger.info(
                        "_map_hf_schema: mapped '%s' -> 'winner'.", alt
                    )
                    break
            else:
                raise ValueError(
                    f"_map_h