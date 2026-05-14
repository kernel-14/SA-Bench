"""
Loader for the Boxoban dataset of Sokoban levels.
Reference: Guez et al. (2018a) - https://github.com/deepmind/boxoban-levels

The Boxoban dataset contains Sokoban levels in text format.
Each level is an 8x8 grid with the following characters:
  '#' = wall
  ' ' = empty floor
  '$' = box
  '@' = agent
  '.' = target
  '*' = box on target
  '+' = agent on target
"""

import os
import numpy as np
from typing import List, Optional, Tuple


# Character to integer mapping
CHAR_TO_INT = {
    '#': 0,  # wall
    ' ': 1,  # empty
    '$': 2,  # box
    '@': 3,  # agent
    '*': 4,  # box on target
    '+': 5,  # agent on target
    '.': 6,  # target
}

GRID_SIZE = 8


class BoxobanLoader:
    """
    Loads Sokoban levels from the Boxoban dataset.
    
    The Boxoban dataset is organized as:
    - train/: training levels (unfiltered)
    - valid/: validation levels (unfiltered)
    - test/: test levels (unfiltered)
    - medium/: medium difficulty levels
    - hard/: hard difficulty levels
    
    Each split contains multiple .txt files, each with multiple levels.
    """
    
    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: Path to the Boxoban dataset directory
        """
        self.data_dir = data_dir
    
    def load_levels(self, split: str = 'train', max_levels: Optional[int] = None) -> List[np.ndarray]:
        """
        Load levels from a specific split.
        
        Args:
            split: One of 'train', 'valid', 'test', 'medium', 'hard'
            max_levels: Maximum number of levels to load (None = all)
            
        Returns:
            List of 8x8 integer arrays representing levels
        """
        split_dir = os.path.join(self.data_dir, split)
        
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        
        levels = []
        
        # Get all .txt files in the split directory
        txt_files = sorted([f for f in os.listdir(split_dir) if f.endswith('.txt')])
        
        for filename in txt_files:
            filepath = os.path.join(split_dir, filename)
            file_levels = self._parse_level_file(filepath)
            levels.extend(file_levels)
            
            if max_levels is not None and len(levels) >= max_levels:
                levels = levels[:max_levels]
                break
        
        return levels
    
    def _parse_level_file(self, filepath: str) -> List[np.ndarray]:
        """
        Parse a Boxoban level file.
        
        Args:
            filepath: Path to the level file
            
        Returns:
            List of 8x8 integer arrays
        """
        levels = []
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Split into individual levels
        # Levels are separated by blank lines or ';' characters
        current_level_lines = []
        
        for line in content.split('\n'):
            line = line.rstrip()
            
            # Skip comment lines and level separators
            if line.startswith(';') or line.startswith('Level'):
                if current_level_lines:
                    level = self._parse_level_lines(current_level_lines)
                    if level is not None:
                        levels.append(level)
                    current_level_lines = []
                continue
            
            if line == '':
                if current_level_lines:
                    level = self._parse_level_lines(current_level_lines)
                    if level is not None:
                        levels.append(level)
                    current_level_lines = []
            else:
                current_level_lines.append(line)
        
        # Don't forget the last level
        if current_level_lines:
            level = self._parse_level_lines(current_level_lines)
            if level is not None:
                levels.append(level)
        
        return levels
    
    def _parse_level_lines(self, lines: List[str]) -> Optional[np.ndarray]:
        """
        Parse lines of a single level into an 8x8 grid.
        
        Args:
            lines: List of strings representing rows of the level
            
        Returns:
            8x8 integer array or None if parsing fails
        """
        if len(lines) < GRID_SIZE:
            return None
        
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
        
        for i, line in enumerate(lines[:GRID_SIZE]):
            # Pad or truncate to GRID_SIZE
            line = line.ljust(GRID_SIZE)[:GRID_SIZE]
            
            for j, char in enumerate(line):
                if char in CHAR_TO_INT:
                    grid[i, j] = CHAR_TO_INT[char]
                else:
                    grid[i, j] = CHAR_TO_INT[' ']  # treat unknown as empty
        
        # Validate: must have exactly one agent and at least one box/target
        agent_count = np.sum((grid == 3) | (grid == 5))
        box_count = np.sum((grid == 2) | (grid == 4))
        target_count = np.sum((grid == 4) | (grid == 5) | (grid == 6))
        
        if agent_count != 1 or box_count == 0 or target_count == 0:
            return None
        
        return grid
    
    def get_level_count(self, split: str) -> int:
        """Get the number of levels in a split."""
        return len(self.load_levels(split))
    
    @staticmethod
    def create_simple_level() -> np.ndarray:
        """
        Create a simple test level for debugging.
        
        Returns:
            8x8 integer array
        """
        # Simple level: agent, one box, one target
        level = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 3, 1, 1, 1, 0],  # agent at (3,3)
            [0, 1, 1, 2, 1, 1, 1, 0],  # box at (4,3)
            [0, 1, 1, 6, 1, 1, 1, 0],  # target at (5,3)
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=np.int32)
        return level
