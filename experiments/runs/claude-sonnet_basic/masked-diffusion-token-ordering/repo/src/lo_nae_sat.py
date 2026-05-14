"""
L&O-NAE-SAT Distribution
========================
Implements the Latents-and-Observations (L&O) distribution with
Not-All-Equal (NAE) SAT observations, as described in Section 3.3 of the paper.

The distribution has:
- N latent tokens sampled uniformly from {1, ..., m}
- P observation tokens, each determined by NAE(x_i1, x_i2, x_i3) for random triples
  NAE(a, b, c) = 1 - 1[a == b == c]
"""

import numpy as np
import torch
from torch.utils.data import Dataset


def nae(a, b, c):
    """Not-All-Equal predicate: returns 1 if not all equal, 0 if all equal."""
    return 1 - int(a == b == c)


class LONAESATDistribution:
    """
    L&O-NAE-SAT distribution generator.
    
    Args:
        N: number of latent tokens
        P: number of observation tokens
        m: vocabulary size (tokens in {1, ..., m})
        seed: random seed for reproducibility
    """
    
    def __init__(self, N: int, P: int, m: int = 3, seed: int = 42):
        self.N = N
        self.P = P
        self.m = m
        self.L = N + P  # total sequence length
        
        rng = np.random.RandomState(seed)
        # Pre-generate random triples for observation functions
        # Each observation j uses a random triple (i1, i2, i3) from [N]
        self.triples = []
        for _ in range(P):
            triple = rng.choice(N, size=3, replace=True)
            self.triples.append(tuple(triple))
    
    def sample(self, n_samples: int, rng: np.random.RandomState = None) -> np.ndarray:
        """
        Sample n_samples sequences from the L&O-NAE-SAT distribution.
        
        Returns:
            Array of shape (n_samples, L) with values in {1, ..., m} for latents
            and {1, 2} for observations (NAE outputs mapped to avoid mask token 0).
        """
        if rng is None:
            rng = np.random.RandomState()
        
        sequences = np.zeros((n_samples, self.L), dtype=np.int64)
        
        for idx in range(n_samples):
            # Sample latent tokens from {1, ..., m}
            latents = rng.randint(1, self.m + 1, size=self.N)
            sequences[idx, :self.N] = latents
            
            # Compute observation tokens
            for j, (i1, i2, i3) in enumerate(self.triples):
                obs = nae(latents[i1], latents[i2], latents[i3])
                # Map to {1, 2} to avoid 0 (mask token)
                sequences[idx, self.N + j] = obs + 1  # 1 or 2
        
        return sequences


class LONAESATDataset(Dataset):
    """PyTorch Dataset for L&O-NAE-SAT distribution."""
    
    def __init__(self, N: int, P: int, m: int = 3, n_samples: int = 10000, 
                 seed: int = 42, pad_to: int = None):
        """
        Args:
            N: number of latent tokens
            P: number of observation tokens
            m: vocabulary size
            n_samples: number of samples to generate
            seed: random seed
            pad_to: if set, pad sequences to this length with a padding token
        """
        self.dist = LONAESATDistribution(N, P, m, seed)
        self.N = N
        self.P = P
        self.m = m
        self.L = N + P
        self.pad_to = pad_to
        
        rng = np.random.RandomState(seed)
        self.data = self.dist.sample(n_samples, rng)
        
        if pad_to is not None and pad_to > self.L:
            pad_len = pad_to - self.L
            # Pad with token value m+2 (a special padding token)
            padding = np.full((n_samples, pad_len), m + 2, dtype=np.int64)
            self.data = np.concatenate([self.data, padding], axis=1)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long)
    
    @property
    def vocab_size(self):
        """Vocabulary size including mask token (0) and padding."""
        if self.pad_to is not None:
            return self.m + 3  # {0=mask, 1..m=latent tokens, m+1=obs, m+2=pad}
        return self.m + 2  # {0=mask, 1..m=latent tokens, m+1=obs}


def create_lo_nae_sat_datasets(N: int, P: int, m: int = 3,
                                n_train: int = 50000, n_val: int = 5000,
                                n_test: int = 5000, pad_to: int = None,
                                seed: int = 42):
    """Create train/val/test splits for L&O-NAE-SAT."""
    train_ds = LONAESATDataset(N, P, m, n_train, seed=seed, pad_to=pad_to)
    val_ds = LONAESATDataset(N, P, m, n_val, seed=seed + 1, pad_to=pad_to)
    test_ds = LONAESATDataset(N, P, m, n_test, seed=seed + 2, pad_to=pad_to)
    return train_ds, val_ds, test_ds
