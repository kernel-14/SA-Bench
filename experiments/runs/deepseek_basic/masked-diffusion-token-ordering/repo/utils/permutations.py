"""
Permutation utilities for π-learner training and analysis
==========================================================
Implements permutation sampling strategies described in Section 3.2:
- Uniform random permutations from Unif(S_L)
- Interpolating distributions between identity and uniform
  using random transpositions (Bormashenko, 2011)

Used for:
- Section 3.2: Measuring hardness via π-learner scaling laws
- Section 3.3: Error imbalance across masking subproblems
"""

import numpy as np
from typing import Optional, Tuple, List
import torch


def random_permutation(L: int, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """Sample a uniformly random permutation from S_L."""
    if rng is None:
        rng = np.random.RandomState()
    return rng.permutation(L)


def identity_permutation(L: int) -> np.ndarray:
    """Return the identity permutation [0, 1, ..., L-1]."""
    return np.arange(L)


def random_transposition(L: int, rng: Optional[np.random.RandomState] = None) -> Tuple[int, int]:
    """Randomly select two distinct indices for a transposition."""
    if rng is None:
        rng = np.random.RandomState()
    i, j = rng.choice(L, size=2, replace=False)
    return i, j


def apply_swaps(perm: np.ndarray, num_swaps: int,
                rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """
    Apply num_swaps random transpositions to a permutation.
    
    After L*log(L) swaps, the resulting distribution is very close to
    uniform over S_L (Bormashenko, 2011).
    
    Args:
        perm: Starting permutation
        num_swaps: Number of random swaps
        rng: Random state
        
    Returns:
        Modified permutation
    """
    if rng is None:
        rng = np.random.RandomState()
    
    L = len(perm)
    perm = perm.copy()
    
    for _ in range(num_swaps):
        i, j = rng.choice(L, size=2, replace=False)
        perm[i], perm[j] = perm[j], perm[i]
    
    return perm


def interpolated_permutation(L: int, num_swaps: int,
                              rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """
    Create a permutation by applying num_swaps random transpositions
    to the identity permutation.
    
    This interpolates between identity (0 swaps) and uniform (L*log(L) swaps).
    
    Used in Section 3.2:
    - "Much-closer": num_swaps = sqrt(L)
    - "Closer": num_swaps = L/10
    - "Uniform": num_swaps >= L * log(L)
    
    Args:
        L: Sequence length
        num_swaps: Number of swaps to apply
        rng: Random state
        
    Returns:
        Interpolated permutation
    """
    return apply_swaps(identity_permutation(L), num_swaps, rng)


def sample_permutations_for_interpolation(
    L: int,
    n_samples: int = 3,
    rng: Optional[np.random.RandomState] = None
) -> dict:
    """
    Sample permutations from different interpolation distributions.
    
    Returns permutations from:
    - Identity
    - Much-closer (sqrt(L) swaps)
    - Closer (L/10 swaps)
    - Uniform (L*log(L) swaps)
    
    As used in Section 3.2 experiments.
    """
    if rng is None:
        rng = np.random.RandomState()
    
    results = {
        'identity': [identity_permutation(L)],
        'much_closer': [],
        'closer': [],
        'uniform': [],
    }
    
    L_log_L = int(L * np.log(L))
    
    for _ in range(n_samples):
        results['much_closer'].append(
            interpolated_permutation(L, int(np.sqrt(L)), rng)
        )
        results['closer'].append(
            interpolated_permutation(L, L // 10, rng)
        )
        results['uniform'].append(
            interpolated_permutation(L, L_log_L, rng)
        )
    
    return results


def permutation_to_mask_set(pi: np.ndarray, i: int) -> np.ndarray:
    """
    Convert permutation and position to mask set.
    
    For a π-learner predicting position π(i), the mask set is:
    M = {π(i), π(i+1), ..., π(L-1)}
    
    This maps between the π-learner formulation and the MDM masking formulation.
    
    Args:
        pi: Permutation array of length L
        i: Current position in the permutation
        
    Returns:
        mask_set: Boolean array indicating masked positions
    """
    L = len(pi)
    mask_set = np.zeros(L, dtype=bool)
    mask_set[pi[i:]] = True
    return mask_set


def mask_set_to_permutation_order(mask_set: np.ndarray) -> np.ndarray:
    """
    Given a mask set M, return a permutation order that could produce it.
    
    This is the inverse operation: given masked positions M, find a permutation
    π such that M = {π(i), ..., π(L-1)} for some i.
    
    This is not unique, but we can canonically order masked positions first.
    """
    L = len(mask_set)
    masked_positions = np.where(mask_set)[0]
    unmasked_positions = np.where(~mask_set)[0]
    
    # Permutation: unmasked first, then masked
    pi = np.concatenate([unmasked_positions, masked_positions])
    return pi


def compute_permutation_distance(pi: np.ndarray, identity: Optional[np.ndarray] = None) -> float:
    """
    Compute a distance metric between permutation π and the identity.
    
    Uses Kendall's tau distance (number of inversions) normalized by max possible.
    
    Args:
        pi: Permutation
        identity: Optional identity permutation (computed if None)
        
    Returns:
        Normalized distance in [0, 1]
    """
    L = len(pi)
    if identity is None:
        identity = np.arange(L)
    
    # Count inversions
    inv_count = 0
    for i in range(L):
        for j in range(i + 1, L):
            if pi[i] > pi[j]:
                inv_count += 1
    
    max_inv = L * (L - 1) / 2
    return inv_count / max_inv if max_inv > 0 else 0.0


def generate_all_masks(L: int) -> List[Tuple[np.ndarray, int]]:
    """
    Generate all possible mask sets and a target position for each.
    
    Returns list of (mask_set, target_position) tuples covering all
    2^L masking patterns.
    
    This enumerates the exponentially many subproblems that MDMs solve.
    """
    all_masks = []
    for mask_int in range(1, 2**L):  # Skip empty mask
        mask = np.array([bool(mask_int >> i & 1) for i in range(L)])
        for i in np.where(mask)[0]:
            all_masks.append((mask.copy(), i))
    return all_masks


def count_subproblems(L: int) -> int:
    """
    Count total number of masking subproblems that MDM solves.
    
    Returns Θ(L * 2^L) as stated in Section 3.2.
    """
    total = 0
    for k in range(1, L + 1):
        # choose(L, k) mask sets of size k, each with k target positions
        from math import comb
        total += k * comb(L, k)
    return total
