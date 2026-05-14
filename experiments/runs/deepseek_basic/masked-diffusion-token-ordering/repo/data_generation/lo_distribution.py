"""
Latents-and-Observations (L&O) Distribution
============================================
Implementation of Definition 3.1 from the paper:
'Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions'

An L&O distribution is a data distribution p_data over sequences of length L with
alphabet size m, specified by:
- A permutation π over indices {1, 2, ..., L}
- Number of latent tokens N
- Number of observation tokens P (N + P = L)
- Prior distribution p_prior of latent variables over {1, ..., m}
- Efficiently learnable observation functions O_1, ..., O_P
"""

import numpy as np
from typing import Tuple, List, Callable, Optional
import itertools


class LODistribution:
    """
    Latents-and-Observations (L&O) distribution.
    
    Samples sequences with two types of tokens:
    - Latent tokens: independently sampled from prior
    - Observation tokens: deterministic/randomized functions of latent tokens
    
    Args:
        N: Number of latent tokens
        P: Number of observation tokens
        m: Alphabet/vocabulary size (1-indexed, 0 reserved for mask)
        prior: Prior distribution over {1,...,m} for latent tokens
        observation_funcs: List of P observation functions, each mapping 
                          latent tokens to a distribution over {1,...,m}
        pi: Permutation over indices; if None, uses identity permutation
    """
    
    def __init__(
        self,
        N: int,
        P: int,
        m: int,
        prior: np.ndarray,
        observation_funcs: List[Callable],
        pi: Optional[np.ndarray] = None,
    ):
        self.N = N
        self.P = P
        self.L = N + P
        self.m = m
        self.prior = prior  # shape (m,)
        
        assert len(observation_funcs) == P, f"Expected {P} observation functions, got {len(observation_funcs)}"
        self.observation_funcs = observation_funcs
        
        if pi is None:
            self.pi = np.arange(self.L)
        else:
            assert len(pi) == self.L
            self.pi = np.array(pi)
        
        # Inverse permutation for recovering positions
        self.pi_inv = np.argsort(self.pi)
    
    def sample(self, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """
        Sample a sequence from the L&O distribution.
        
        Returns:
            x: Array of length L, with values in {1,...,m}
        """
        if rng is None:
            rng = np.random.RandomState()
        
        # Step 1: Sample latent tokens
        latent_tokens = rng.choice(self.m, size=self.N, p=self.prior) + 1  # 1-indexed
        
        # Step 2: Sample observation tokens
        obs_tokens = np.zeros(self.P, dtype=int)
        for j in range(self.P):
            obs_dist = self.observation_funcs[j](latent_tokens)
            obs_tokens[j] = rng.choice(self.m, p=obs_dist) + 1
        
        # Step 3: Arrange by permutation
        # π(1)...π(N) are latent positions, π(N+1)...π(N+P) are observation positions
        x = np.zeros(self.L, dtype=int)
        for i in range(self.N):
            x[self.pi[i]] = latent_tokens[i]
        for j in range(self.P):
            x[self.pi[self.N + j]] = obs_tokens[j]
        
        return x
    
    def sample_batch(self, batch_size: int, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """Sample a batch of sequences."""
        if rng is None:
            rng = np.random.RandomState()
        return np.stack([self.sample(rng) for _ in range(batch_size)])
    
    def oracle_predictor(self, x_masked: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Compute the Bayes-optimal predictor for masked positions.
        
        Given a partially masked sequence x_masked and mask indicator,
        returns the posterior marginal distribution over values for each masked position.
        
        This is computationally intensive and enumerates over all possible latent assignments.
        
        Args:
            x_masked: Length L array with 0 for masked positions
            mask: Boolean array, True for masked positions
            
        Returns:
            posteriors: (L, m) array of posterior probabilities for each position
        """
        posteriors = np.zeros((self.L, self.m))
        
        # For unmasked positions, the posterior is a one-hot
        for i in range(self.L):
            if not mask[i]:
                posteriors[i, x_masked[i] - 1] = 1.0
        
        # For masked positions, we need to marginalize over latent assignments
        # This is exponential in N, so only practical for small N
        if self.N > 15:
            raise ValueError(f"N={self.N} too large for exact enumeration")
        
        # Enumerate all possible latent assignments
        total_prob = 0.0
        for latent_assign in itertools.product(range(self.m), repeat=self.N):
            latent_tokens = np.array(latent_assign) + 1
            
            # Check consistency with unmasked observation tokens
            prior_prob = np.prod([self.prior[lt - 1] for lt in latent_tokens])
            obs_prob = 1.0
            
            for j in range(self.P):
                obs_pos = self.pi[self.N + j]
                if not mask[obs_pos]:
                    obs_dist = self.observation_funcs[j](latent_tokens)
                    if obs_dist[x_masked[obs_pos] - 1] == 0:
                        obs_prob = 0.0
                        break
                    obs_prob *= obs_dist[x_masked[obs_pos] - 1]
            
            if obs_prob == 0:
                continue
            
            joint_prob = prior_prob * obs_prob
            total_prob += joint_prob
            
            # Accumulate for latent positions
            for i in range(self.N):
                latent_pos = self.pi[i]
                if mask[latent_pos]:
                    posteriors[latent_pos, latent_tokens[i] - 1] += joint_prob
        
        if total_prob > 0:
            for i in range(self.L):
                if mask[i] and posteriors[i].sum() > 0:
                    posteriors[i] /= posteriors[i].sum()
        
        return posteriors


def make_nae_observation(triple_indices, m):
    """
    Create an observation function for the Not-All-Equal (NAE) predicate.
    
    NAE(x_i1, x_i2, x_i3) = 1 - 1[x_i1 = x_i2 = x_i3]
    
    Each observation function returns a deterministic distribution (one-hot)
    corresponding to the NAE evaluation.
    
    Args:
        triple_indices: Tuple (i1, i2, i3) of indices into latent tokens
        m: Alphabet size
        
    Returns:
        Callable observation function
    """
    i1, i2, i3 = triple_indices
    
    def obs_func(latent_tokens):
        dist = np.zeros(m)
        # Check if all three are equal
        if latent_tokens[i1] == latent_tokens[i2] == latent_tokens[i3]:
            # NAE returns 0 (but our vocabulary is 1-indexed)
            # We map 0 -> token 1, 1 -> token 2
            dist[0] = 1.0  # token value 1
        else:
            dist[1] = 1.0  # token value 2
        return dist
    
    return obs_func


def create_lo_nae_sat(N: int, P: int, m: int = 2, rng: Optional[np.random.RandomState] = None) -> LODistribution:
    """
    Create an L&O-NAE-SAT distribution as defined in Sections 3.3 and 4.2.
    
    The observation tokens are determined by NAE predicates over randomly 
    chosen triples of latent tokens.
    
    Args:
        N: Number of latent tokens
        P: Number of observation tokens
        m: Vocabulary size (default 2)
        rng: Random state for choosing triples
        
    Returns:
        LODistribution instance
    """
    if rng is None:
        rng = np.random.RandomState(42)
    
    # Uniform prior over {1,...,m}
    prior = np.ones(m) / m
    
    # Generate random triples for observations
    observation_funcs = []
    for _ in range(P):
        triple = tuple(rng.choice(N, size=3, replace=False))
        observation_funcs.append(make_nae_observation(triple, m))
    
    return LODistribution(N, P, m, prior, observation_funcs)


def make_sparse_predicate_observation(subset_indices, g_func, m):
    """
    Create an observation function for sparse predicate observations (Example 3.2).
    
    Each observation corresponds to a k-sized subset S of {1,...,N}, and the
    observation function is given by g({x^{π(i)}}_{i in S}).
    
    Args:
        subset_indices: Tuple of k indices
        g_func: Predicate function mapping m^k -> {0,1}
        m: Alphabet size
        
    Returns:
        Callable observation function
    """
    def obs_func(latent_tokens):
        values = tuple(latent_tokens[i] for i in subset_indices)
        result = g_func(values)
        dist = np.zeros(m)
        dist[0] = 1.0 if result == 0 else 0.0
        dist[1] = 0.0 if result == 0 else 1.0
        return dist
    
    return obs_func


def create_sparse_predicate_lo(N: int, k: int, m: int, g_func: Callable,
                                rng: Optional[np.random.RandomState] = None) -> LODistribution:
    """
    Create an L&O distribution with sparse predicate observations (Example 3.2).
    
    Args:
        N: Number of latent tokens
        k: Arity of predicate
        m: Alphabet size
        g_func: Predicate function
        rng: Random state
        
    Returns:
        LODistribution instance
    """
    if rng is None:
        rng = np.random.RandomState(42)
    
    prior = np.ones(m) / m
    
    # Generate all k-sized ordered subsets
    from itertools import permutations
    all_subsets = list(permutations(range(N), k))
    P = len(all_subsets)
    
    observation_funcs = []
    for subset in all_subsets:
        observation_funcs.append(make_sparse_predicate_observation(subset, g_func, m))
    
    return LODistribution(N, P, m, prior, observation_funcs)
