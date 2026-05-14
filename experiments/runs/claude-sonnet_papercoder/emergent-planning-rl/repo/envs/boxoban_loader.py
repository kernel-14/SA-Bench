## envs/boxoban_loader.py
"""Boxoban dataset loader for the emergent planning interpretability pipeline.

This module provides the BoxobanLoader class, which is the data access layer
for all Sokoban levels used throughout the pipeline. It parses the Boxoban
text-format level files (Guez et al., 2018a) into numpy integer-coded grids
that SokobanEnv can consume directly.

The Boxoban dataset is available at:
    https://github.com/deepmind/boxoban-levels

Directory layout expected under data_dir:
    data/boxoban/
    ├── unfiltered/
    │   ├── train/   (*.txt files, ~900k levels total)
    │   ├── valid/
    │   └── test/
    ├── medium/
    │   └── train/
    └── hard/
        └── train/

Cell code encoding (matches SokobanEnv one-hot encoding, paper Section E.2):
    0 = wall       ('#')
    1 = empty      (' ')
    2 = box        ('$')
    3 = agent      ('@')
    4 = box+target ('*')
    5 = agent+tgt  ('+')
    6 = target     ('.')

All levels are stored as np.ndarray of shape (8, 8) with dtype np.int8.
The loader is eager: all levels are parsed at __init__ time and held in memory
(~57 MB for the full unfiltered training set of ~900k levels).

Example:
    >>> loader = BoxobanLoader("data/boxoban", split="unfiltered_train")
    >>> len(loader)
    900000
    >>> level = loader.get_random_level()
    >>> level.shape
    (8, 8)
    >>> level.dtype
    dtype('int8')
"""

from __future__ import annotations

import pathlib
import random
import warnings
from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Cell code constants — shared with SokobanEnv for consistent one-hot encoding.
# The integer codes 0-6 correspond to the 7 channels of the symbolic observation
# x_t ∈ R^{8×8×7} described in paper Section 2.2 and E.2.
# ---------------------------------------------------------------------------

#: Mapping from Boxoban text characters to integer cell codes.
#: Index k in the one-hot observation vector corresponds to cell code k.
CELL_CODES: Dict[str, int] = {
    "#": 0,  # wall
    " ": 1,  # empty square
    "$": 2,  # box on empty square
    "@": 3,  # agent on empty square
    "*": 4,  # box on target square
    "+": 5,  # agent on target square
    ".": 6,  # target with nothing on it
}

#: Inverse mapping from integer cell code to character (for debugging).
CODE_TO_CHAR: Dict[int, str] = {v: k for k, v in CELL_CODES.items()}

#: Expected grid dimensions matching config.yaml agent.grid_h and agent.grid_w.
_GRID_H: int = 8
_GRID_W: int = 8

#: Default cell code for unknown characters (treat as wall).
_DEFAULT_CELL_CODE: int = 0  # wall

# ---------------------------------------------------------------------------
# Split name mapping: logical config names → filesystem subdirectory paths.
# Accepts both short names ('train', 'val') and full config names
# ('unfiltered_train', 'unfiltered_val') for flexibility.
# ---------------------------------------------------------------------------

#: Maps split name strings to relative subdirectory paths under data_dir.
_SPLIT_TO_SUBDIR: Dict[str, str] = {
    # Full config-style names (from config.yaml data section)
    "unfiltered_train": "unfiltered/train",
    "unfiltered_val": "unfiltered/valid",
    "unfiltered_test": "unfiltered/test",
    "medium": "medium/train",
    "hard": "hard/train",
    # Short aliases for convenience
    "train": "unfiltered/train",
    "val": "unfiltered/valid",
    "valid": "unfiltered/valid",
    "test": "unfiltered/test",
    "medium_train": "medium/train",
    "medium_val": "medium/valid",
    "hard_train": "hard/train",
}


class BoxobanLoader:
    """Loads and provides access to Boxoban Sokoban levels from text files.

    Parses all levels from the specified split at construction time (eager
    loading) and stores them as a list of numpy integer-coded grids. Provides
    random and indexed access for use by SokobanEnv during training and
    evaluation.

    The loader is the single point of contact with the filesystem for level
    data. All downstream components (SokobanEnv, IMPALATrainer, ConceptLabeler,
    ThinkingStepsAnalyzer, TrainingEmergenceAnalyzer) access levels through
    this class.

    Attributes:
        data_dir: Root directory of the Boxoban dataset.
        split: Logical split name (e.g., 'unfiltered_train', 'medium').
        levels: List of parsed levels, each as np.ndarray of shape (8, 8)
            with dtype np.int8. Populated by load_levels() at __init__ time.

    Example:
        >>> loader = BoxobanLoader("data/boxoban", split="unfiltered_train")
        >>> print(f"Loaded {len(loader)} levels")
        Loaded 900000 levels
        >>> level = loader.get_level(42)
        >>> level.shape, level.dtype
        ((8, 8), dtype('int8'))
        >>> random_level = loader.get_random_level()
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "unfiltered_train",
    ) -> None:
        """Initialize the loader and eagerly parse all levels for the split.

        Resolves the split name to a filesystem path, validates the directory
        exists, and calls load_levels() to populate self.levels.

        Args:
            data_dir: Root directory of the Boxoban dataset. Should contain
                subdirectories 'unfiltered/', 'medium/', 'hard/'. If the
                directory does not exist, a FileNotFoundError is raised with
                a helpful message pointing to the Boxoban GitHub repository.
            split: Logical split name. Accepted values:
                - 'unfiltered_train' or 'train': unfiltered training set (~900k levels)
                - 'unfiltered_val' or 'val' or 'valid': unfiltered validation set
                - 'unfiltered_test' or 'test': unfiltered test set
                - 'medium': medium difficulty training set
                - 'hard': hard difficulty training set
                Defaults to 'unfiltered_train'.

        Raises:
            ValueError: If split is not a recognized split name.
            FileNotFoundError: If data_dir or the split subdirectory does not
                exist. The error message includes the Boxoban GitHub URL.
            ValueError: If no levels were successfully parsed from the split
                directory (empty or all files malformed).
        """
        self.data_dir: str = data_dir
        self.split: str = split
        self.levels: List[np.ndarray] = []

        # Resolve split name to filesystem subdirectory.
        if split not in _SPLIT_TO_SUBDIR:
            raise ValueError(
                f"Unknown split '{split}'. Valid split names are: "
                f"{sorted(_SPLIT_TO_SUBDIR.keys())}. "
                f"See the Boxoban dataset at "
                f"https://github.com/deepmind/boxoban-levels"
            )

        subdir: str = _SPLIT_TO_SUBDIR[split]
        self._split_dir: pathlib.Path = pathlib.Path(data_dir) / subdir

        # Validate that the data directory root exists.
        data_root = pathlib.Path(data_dir)
        if not data_root.exists():
            raise FileNotFoundError(
                f"Boxoban data directory not found: '{data_dir}'. "
                f"Please download the Boxoban dataset from "
                f"https://github.com/deepmind/boxoban-levels "
                f"and place it at '{data_dir}'."
            )

        # Validate that the split subdirectory exists.
        if not self._split_dir.exists():
            raise FileNotFoundError(
                f"Boxoban split directory not found: '{self._split_dir}'. "
                f"Expected split '{split}' to be at subdirectory '{subdir}' "
                f"under data_dir='{data_dir}'. "
                f"Please verify the Boxoban dataset structure at "
                f"https://github.com/deepmind/boxoban-levels"
            )

        # Eagerly load all levels at construction time.
        self.levels = self.load_levels()

        if len(self.levels) == 0:
            raise ValueError(
                f"No valid levels found in split '{split}' at '{self._split_dir}'. "
                f"The directory exists but contains no parseable Sokoban levels. "
                f"Please verify the Boxoban dataset files."
            )

    def load_levels(self) -> List[np.ndarray]:
        """Parse all levels from all .txt files in the split directory.

        Iterates over all .txt files in self._split_dir in sorted (deterministic)
        order, parses each file using _parse_file(), and concatenates all
        returned level lists into a single flat list.

        Sorting by filename ensures reproducible level ordering across runs,
        which is important for reproducibility of training and evaluation.

        Returns:
            List of all parsed levels as np.ndarray of shape (8, 8) with
            dtype np.int8. The list is also stored as self.levels. Returns
            an empty list if no .txt files are found or all files are empty.

        Note:
            This method is called once at __init__ time. Calling it again
            will re-parse all files and overwrite self.levels.
        """
        txt_files: List[pathlib.Path] = sorted(
            self._split_dir.glob("*.txt")
        )

        if not txt_files:
            warnings.warn(
                f"No .txt files found in split directory '{self._split_dir}'. "
                f"The BoxobanLoader will have zero levels.",
                UserWarning,
                stacklevel=2,
            )
            return []

        all_levels: List[np.ndarray] = []

        for filepath in txt_files:
            file_levels: List[np.ndarray] = self._parse_file(filepath)
            all_levels.extend(file_levels)

        self.levels = all_levels
        return all_levels

    def get_level(self, idx: int) -> np.ndarray:
        """Return a copy of the level at the given index.

        Returns a copy (not a view) to prevent the caller from accidentally
        mutating the loader's internal level storage. SokobanEnv modifies
        the grid during gameplay, so this protection is essential.

        Args:
            idx: Zero-based index into self.levels. Valid range: [0, len(self)).

        Returns:
            A copy of the level as np.ndarray of shape (8, 8) with dtype np.int8.

        Raises:
            IndexError: If idx is out of range [0, len(self)), with an
                informative message including the valid range.

        Example:
            >>> level = loader.get_level(0)
            >>> level.shape
            (8, 8)
        """
        n: int = len(self.levels)
        if idx < 0 or idx >= n:
            raise IndexError(
                f"Level index {idx} is out of range. "
                f"Valid range: [0, {n}). "
                f"This loader has {n} levels for split '{self.split}'."
            )
        # Return a copy to prevent mutation of the internal level storage.
        return self.levels[idx].copy()

    def get_random_level(self) -> np.ndarray:
        """Return a copy of a uniformly random level from the dataset.

        Uses Python's random module (not numpy.random) to avoid interfering
        with numpy's global random state, which may be used elsewhere for
        reproducibility of training or evaluation.

        Returns:
            A copy of a randomly selected level as np.ndarray of shape (8, 8)
            with dtype np.int8.

        Raises:
            ValueError: If the loader has no levels (should not happen after
                successful __init__, but guards against edge cases).

        Example:
            >>> level = loader.get_random_level()
            >>> level.shape
            (8, 8)
        """
        n: int = len(self.levels)
        if n == 0:
            raise ValueError(
                f"Cannot get a random level: BoxobanLoader for split "
                f"'{self.split}' has no levels."
            )
        idx: int = random.randint(0, n - 1)
        return self.get_level(idx)

    def __len__(self) -> int:
        """Return the total number of levels in this split.

        Returns:
            Number of successfully parsed levels. Typically ~900k for
            unfiltered_train, ~1k for medium/hard evaluation sets.

        Example:
            >>> len(loader)
            900000
        """
        return len(self.levels)

    # ------------------------------------------------------------------
    # Private parsing methods
    # ------------------------------------------------------------------

    def _parse_file(self, filepath: pathlib.Path) -> List[np.ndarray]:
        """Parse all Sokoban levels from a single Boxoban .txt file.

        Boxoban files contain multiple levels separated by blank lines.
        Comment lines starting with ';' (level headers like '; level 0')
        are skipped. Each level is a sequence of grid rows containing
        Sokoban characters (#, space, $, @, *, +, .).

        The parser accumulates grid rows into a buffer and finalizes a level
        whenever a blank line is encountered (or at EOF). Malformed levels
        (too few rows, invalid characters) are silently skipped with a warning.

        Args:
            filepath: Path to the .txt file to parse.

        Returns:
            List of parsed levels from this file, each as np.ndarray of
            shape (8, 8) with dtype np.int8. Returns an empty list if the
            file contains no valid levels.
        """
        levels: List[np.ndarray] = []
        current_rows: List[str] = []

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines: List[str] = f.readlines()
        except (OSError, UnicodeDecodeError) as exc:
            warnings.warn(
                f"Could not read Boxoban file '{filepath}': {exc}. Skipping.",
                UserWarning,
                stacklevel=3,
            )
            return []

        for line in lines:
            # Strip only the trailing newline, preserving spaces within rows.
            stripped: str = line.rstrip("\n").rstrip("\r")

            # Skip comment/header lines (e.g., '; level 0').
            if stripped.startswith(";"):
                continue

            # Blank line signals end of current level.
            if stripped.strip() == "":
                if current_rows:
                    grid: Optional[np.ndarray] = self._finalize_grid(current_rows)
                    if grid is not None:
                        levels.append(grid)
                    current_rows = []
                continue

            # Only accumulate lines that look like Sokoban grid rows.
            # A valid grid row must contain at least one '#' (wall character).
            if "#" in stripped:
                current_rows.append(stripped)

        # Handle the last level in the file (no trailing blank line).
        if current_rows:
            grid = self._finalize_grid(current_rows)
            if grid is not None:
                levels.append(grid)

        return levels

    def _finalize_grid(self, rows: List[str]) -> Optional[np.ndarray]:
        """Convert a list of string rows into an 8×8 numpy integer-coded grid.

        Handles the following edge cases robustly:
        - Rows shorter than 8 characters: padded with walls (code 0) on the right.
        - Rows longer than 8 characters: truncated to 8 characters.
        - Grids with more than 8 rows: first 8 rows used (after filtering).
        - Grids with fewer than 8 rows: padded with all-wall rows.
        - 10×10 grids (with surrounding wall border): inner 8×8 extracted.
        - Unknown characters: treated as walls (code 0) with a warning.

        The paper (Section E.2) states: "our version of Sokoban forgoes the
        layer of wall squares that is sometimes appended to the edge of Sokoban
        boards in previous work." This means 10×10 grids must be cropped to 8×8
        by removing the outer wall border.

        Args:
            rows: List of string rows forming a single Sokoban level. Each
                string is one row of the grid. Must contain at least 1 row.

        Returns:
            np.ndarray of shape (8, 8) with dtype np.int8 containing integer
            cell codes, or None if the rows cannot form a valid 8×8 grid
            (e.g., fewer than 4 rows, indicating a malformed level).
        """
        if not rows:
            return None

        # Filter to only rows that contain '#' (valid grid rows).
        grid_rows: List[str] = [r for r in rows if "#" in r]

        if len(grid_rows) < 4:
            # Too few rows to be a valid Sokoban level; skip silently.
            return None

        # Determine the actual grid dimensions from the parsed rows.
        n_rows: int = len(grid_rows)
        n_cols: int = max(len(r) for r in grid_rows)

        # Build the raw integer grid at the parsed dimensions.
        raw_grid: np.ndarray = np.zeros((n_rows, n_cols), dtype=np.int8)

        for row_idx, row_str in enumerate(grid_rows):
            for col_idx in range(n_cols):
                if col_idx < len(row_str):
                    char: str = row_str[col_idx]
                    if char in CELL_CODES:
                        raw_grid[row_idx, col_idx] = CELL_CODES[char]
                    else:
                        # Unknown character: treat as wall (code 0).
                        raw_grid[row_idx, col_idx] = _DEFAULT_CELL_CODE
                else:
                    # Row is shorter than max width: pad with walls.
                    raw_grid[row_idx, col_idx] = _DEFAULT_CELL_CODE

        # Handle the case where the grid has a surrounding wall border
        # (10×10 instead of 8×8). The paper explicitly removes this border.
        # We detect this by checking if the grid is larger than 8×8 and
        # the outer ring is all walls.
        final_grid: np.ndarray = self._crop_to_target_size(raw_grid)

        return final_grid

    def _crop_to_target_size(self, grid: np.ndarray) -> np.ndarray:
        """Crop or pad a parsed grid to exactly (_GRID_H, _GRID_W) = (8, 8).

        Handles three cases:
        1. Grid is already 8×8: return as-is (copy with correct dtype).
        2. Grid is 10×10 (with surrounding wall border): strip the outer
           1-cell border to get the inner 8×8. This matches the paper's
           statement that "our version of Sokoban forgoes the layer of wall
           squares that is sometimes appended to the edge of Sokoban boards."
        3. Grid is any other size: center-crop if larger, or pad with walls
           if smaller, to produce an 8×8 output.

        Args:
            grid: Raw parsed grid as np.ndarray of shape (H, W) with dtype
                np.int8. H and W may differ from _GRID_H and _GRID_W.

        Returns:
            np.ndarray of shape (_GRID_H, _GRID_W) = (8, 8) with dtype np.int8.
        """
        h, w = grid.shape

        # Case 1: Already the correct size.
        if h == _GRID_H and w == _GRID_W:
            return grid.astype(np.int8)

        # Case 2: 10×10 grid with surrounding wall border → strip outer ring.
        if h == _GRID_H + 2 and w == _GRID_W + 2:
            return grid[1:_GRID_H + 1, 1:_GRID_W + 1].astype(np.int8)

        # Case 3: General case — crop or pad to (_GRID_H, _GRID_W).
        result: np.ndarray = np.zeros((_GRID_H, _GRID_W), dtype=np.int8)

        # Determine how many rows/cols to copy.
        copy_h: int = min(h, _GRID_H)
        copy_w: int = min(w, _GRID_W)

        result[:copy_h, :copy_w] = grid[:copy_h, :copy_w].astype(np.int8)

        # Any remaining rows/cols in result are already 0 (wall), which is
        # the correct padding for out-of-bounds grid positions.

        return result
