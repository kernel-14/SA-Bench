"""
Scheduling functions for improved consistency training (iCT) from Song & Dhariwal (2024).

Key schedules:
1. Noise schedule: sigma_i = (sigma_0^(1/rho) + i/N * (sigma_N^(1/rho) - sigma_0^(1/rho)))^rho
2. Timestep schedule: N(k) = min(s0 * 2^floor(k/K'), s1) + 1
3. Loss weighting: lambda(sigma_i) = 1 / (sigma_{i+1} - sigma_i)
4. Timestep sampling: p(sigma_i) proportional to erf differences
"""

import math
import numpy as np
import torch
from scipy.special import erf


class NoiseSchedule:
    """
    Noise schedule from Karras et al. (2022) / Song & Dhariwal (2024).
    
    sigma_i = (sigma_0^(1/rho) + i/N * (sigma_N^(1/rho) - sigma_0^(1/rho)))^rho
    """
    def __init__(self, sigma_min=0.002, sigma_max=80.0, rho=7.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def get_sigmas(self, N):
        """
        Get N+1 noise levels from sigma_min to sigma_max.
        
        Args:
            N: Number of timesteps
            
        Returns:
            Array of N+1 sigma values
        """
        i = np.arange(N + 1)
        sigmas = (
            self.sigma_min ** (1 / self.rho) +
            i / N * (self.sigma_max ** (1 / self.rho) - self.sigma_min ** (1 / self.rho))
        ) ** self.rho
        return sigmas

    def get_loss_weights(self, sigmas):
        """
        Compute loss weights lambda(sigma_i) = 1 / (sigma_{i+1} - sigma_i).
        
        Args:
            sigmas: Array of sigma values
            
        Returns:
            Array of loss weights (length N)
        """
        return 1.0 / (sigmas[1:] - sigmas[:-1])


class TimestepSchedule:
    """
    Timestep schedule from Song & Dhariwal (2024).
    
    N(k) = min(s0 * 2^floor(k/K'), s1) + 1
    where K' = floor(K / (log2(s1/s0) + 1))
    """
    def __init__(self, s0=10, s1=1280, total_steps=100000):
        self.s0 = s0
        self.s1 = s1
        self.total_steps = total_steps
        # K' = floor(K / (log2(s1/s0) + 1))
        self.K_prime = math.floor(total_steps / (math.log2(s1 / s0) + 1))

    def get_N(self, k):
        """
        Get number of timesteps at training step k.
        
        Args:
            k: Current training step
            
        Returns:
            Number of timesteps N(k)
        """
        return min(self.s0 * 2 ** math.floor(k / self.K_prime), self.s1) + 1


class TimestepSampler:
    """
    Timestep sampler from Song & Dhariwal (2024).
    
    p(sigma_i) proportional to erf((log(sigma_{i+1}) - P_mean) / (sqrt(2) * P_std)) 
                              - erf((log(sigma_i) - P_mean) / (sqrt(2) * P_std))
    """
    def __init__(self, P_mean=-1.1, P_std=2.0):
        self.P_mean = P_mean
        self.P_std = P_std

    def get_probabilities(self, sigmas):
        """
        Compute sampling probabilities for each timestep.
        
        Args:
            sigmas: Array of N+1 sigma values
            
        Returns:
            Array of N probabilities (normalized)
        """
        log_sigmas = np.log(sigmas)
        probs = (
            erf((log_sigmas[1:] - self.P_mean) / (math.sqrt(2) * self.P_std)) -
            erf((log_sigmas[:-1] - self.P_mean) / (math.sqrt(2) * self.P_std))
        )
        probs = np.maximum(probs, 0)
        probs = probs / probs.sum()
        return probs

    def sample_indices(self, sigmas, batch_size, device='cpu'):
        """
        Sample timestep indices according to the probability distribution.
        
        Args:
            sigmas: Array of N+1 sigma values
            batch_size: Number of indices to sample
            device: Device for the output tensor
            
        Returns:
            Tensor of sampled indices of shape (batch_size,)
        """
        probs = self.get_probabilities(sigmas)
        N = len(sigmas) - 1
        indices = np.random.choice(N, size=batch_size, p=probs)
        return torch.tensor(indices, device=device)
