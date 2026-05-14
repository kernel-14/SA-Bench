"""
Data Utilities for Boxoban Dataset and Level Management.

Provides functions to:
- Load Boxoban levels (unfiltered training/validation/test, medium, hard)
- Convert between different level formats
- Generate training/testing splits for probe datasets
"""

import numpy as np
import os
import pickle
from typing import List, Tuple, Optional, Dict
from src.sokoban import SquareType


def parse_boxoban_level(lines: List[str]) -> np.ndarray:
    """
    Parse a single Boxoban level from text lines.
    
    Boxoban format uses specific characters:
    # = wall, ' ' = empty, $ = box, @ = agent, . = target, * = box on target, + = agent on target
    
    Args:
        lines: List of strings representing the level rows
    
    Returns:
        (8, 8) integer array in our symbolic format
    """
    level = np.full((8, 8), SquareType.EMPTY, dtype=np.int32)
    
    char_to_square = {
        '#': SquareType.WALL,
        ' ': SquareType.EMPTY,
        '$': SquareType.BOX,
        '@': SquareType.AGENT,
        '.': SquareType.TARGET,
        '*': SquareType.BOX_ON_TARGET,
        '+': SquareType.AGENT_ON_TARGET,
    }
    
    for y, line in enumerate(lines):
        if y >= 8:
            break
        for x, ch in enumerate(line):
            if x >= 8:
                break
            if ch in char_to_square:
                level[y, x] = char_to_square[ch]
    
    return level


def load_boxoban_levels(filepath: str, max_levels: Optional[int] = None) -> List[np.ndarray]:
    """
    Load Boxoban levels from a .txt file.
    
    Boxoban format: levels are separated by empty lines, each level is 8-10 rows.
    
    Args:
        filepath: Path to Boxoban level file
        max_levels: Maximum number of levels to load (None = all)
    
    Returns:
        List of (8, 8) level arrays
    """
    levels = []
    current_lines = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            
            if line.strip() == '':
                if current_lines:
                    # Filter out lines that are just comments or metadata
                    level_lines = [l for l in current_lines if not l.startswith(';')]
                    if level_lines:
                        level = parse_boxoban_level(level_lines)
                        levels.append(level)
                        
                        if max_levels and len(levels) >= max_levels:
                            break
                    current_lines = []
            else:
                current_lines.append(line)
        
        # Handle last level
        if current_lines and (not max_levels or len(levels) < max_levels):
            level_lines = [l for l in current_lines if not l.startswith(';')]
            if level_lines:
                level = parse_boxoban_level(level_lines)
                levels.append(level)
    
    return levels


def generate_simple_levels(num_levels: int = 100, seed: int = 42) -> List[np.ndarray]:
    """
    Generate simple handcrafted Sokoban levels for testing.
    
    These are NOT the Boxoban levels but simple levels for development/testing.
    
    Args:
        num_levels: Number of levels to generate
        seed: Random seed
    
    Returns:
        List of level arrays
    """
    rng = np.random.RandomState(seed)
    levels = []
    
    for _ in range(num_levels):
        level = np.full((8, 8), SquareType.EMPTY, dtype=np.int32)
        
        # Border walls
        level[0, :] = SquareType.WALL
        level[-1, :] = SquareType.WALL
        level[:, 0] = SquareType.WALL
        level[:, -1] = SquareType.WALL
        
        # Random interior walls
        for y in range(1, 7):
            for x in range(1, 7):
                if rng.random() < 0.1:
                    level[y, x] = SquareType.WALL
        
        # Place targets
        target_positions = set()
        while len(target_positions) < 4:
            y, x = rng.randint(1, 7), rng.randint(1, 7)
            if level[y, x] == SquareType.EMPTY:
                level[y, x] = SquareType.TARGET
                target_positions.add((y, x))
        
        # Place boxes
        box_positions = set()
        while len(box_positions) < 4:
            y, x = rng.randint(1, 7), rng.randint(1, 7)
            if level[y, x] == SquareType.EMPTY:
                level[y, x] = SquareType.BOX
                box_positions.add((y, x))
        
        # Place agent
        while True:
            y, x = rng.randint(1, 7), rng.randint(1, 7)
            if level[y, x] == SquareType.EMPTY:
                level[y, x] = SquareType.AGENT
                break
        
        levels.append(level)
    
    return levels


def create_probe_dataset_splits(
    levels: List[np.ndarray],
    train_ratio: float = 0.75,
    seed: int = 42,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Split levels into training and test sets for probe training.
    
    As described in Section 4.1:
    - Training: 3000 episodes on Boxoban unfiltered training
    - Test: 1000 episodes on Boxoban unfiltered validation
    
    Args:
        levels: All levels
        train_ratio: Fraction for training
        seed: Random seed
    
    Returns:
        (train_levels, test_levels)
    """
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(levels))
    split = int(len(levels) * train_ratio)
    train_indices = indices[:split]
    test_indices = indices[split:]
    
    return [levels[i] for i in train_indices], [levels[i] for i in test_indices]


def level_to_string(level: np.ndarray) -> str:
    """Convert a level array back to string representation."""
    square_to_char = {
        SquareType.WALL: '#',
        SquareType.EMPTY: ' ',
        SquareType.BOX: '$',
        SquareType.AGENT: '@',
        SquareType.TARGET: '.',
        SquareType.BOX_ON_TARGET: '*',
        SquareType.AGENT_ON_TARGET: '+',
    }
    lines = []
    for y in range(8):
        line = ''.join(square_to_char.get(level[y, x], '?') for x in range(8))
        lines.append(line)
    return '\n'.join(lines)


def save_levels(levels: List[np.ndarray], filepath: str):
    """Save levels to a file."""
    with open(filepath, 'w') as f:
        for i, level in enumerate(levels):
            f.write(level_to_string(level))
            f.write('\n\n')


def load_levels(filepath: str) -> List[np.ndarray]:
    """Load levels saved by save_levels."""
    return load_boxoban_levels(filepath)
