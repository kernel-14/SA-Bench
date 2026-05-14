
import torch
from torch import nn
import math
import numpy as np
from modules import UNet

class ConsistencyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_channels = 3 # Assuming RGB images
        self.unet = UNet(
            image_size=config.image_resolution,
            in_channels=self.image_channels,
            out_channels=self.image_channels,
            model_channels=config.model_channels,
            num_blocks=config.num_blocks,
            channel_multiplicative_factor=config.channel_multiplicative_factor,
            attn_resolutions=config.attn_resolutions,
            dropout=config.dropout,
            time_emb_dim=config.model_channels # Typically same as model_channels for simplicity, adjust if paper specifies otherwise
        )

        self.sigma_data = config.sigma_data
        self.sigma_0 = config.sigma_0 # Smallest noise level
        self.sigma_N = config.sigma_1 # Largest noise level (T in the paper, but here sigma_1 is max sigma)
        self.rho = config.rho

    def _c_skip(self, sigma):
        return self.sigma_data**2 / (self.sigma_data**2 + (sigma - self.sigma_0)**2)

    def _c_out(self, sigma):
        # Paper's formula: sigma_d * (sigma - sigma_0) / sqrt(sigma_d^2 + sigma^2)
        # Note: There might be a typo in the paper for c_out.
        # Following the text exactly, which states (sigma^2 + sigma_d^2)^0.5 for the denominator
        # For consistency with c_skip's denominator structure, it might be (sigma_data**2 + (sigma - sigma_0)**2)**0.5
        # However, to be faithful to the provided formula:
        return sigma * self.sigma_data / (sigma**2 + self.sigma_data**2)**0.5


    def forward(self, x_t, sigma):
        # x_t: noisy input (batch_size, channels, H, W)
        # sigma: noise level (batch_size,) or (1,)
        if sigma.dim() == 0:
            sigma = sigma.expand(x_t.shape[0])
        elif sigma.dim() == 1 and sigma.shape[0] != x_t.shape[0]:
            sigma = sigma.expand(x_t.shape[0])

        c_skip = self._c_skip(sigma)[:, None, None, None] # Expand for broadcasting
        c_out = self._c_out(sigma)[:, None, None, None]   # Expand for broadcasting

        # F_theta is the UNet output
        F_theta_output = self.unet(x_t, sigma)

        # Consistency Model prediction
        # f_theta(x_t, sigma_t) = c_skip(sigma_t) * x_t + c_out(sigma_t) * F_theta(x_t, sigma_t)
        return c_skip * x_t + c_out * F_theta_output

    def generate_noise_schedule(self, N):
        """
        Generates the noise schedule {sigma_i} as defined in Appendix D.
        sigma_i = (sigma_0^(1/rho) + (i/N) * (sigma_N^(1/rho) - sigma_0^(1/rho)))^rho
        """
        if N == 0:
            return torch.tensor([self.sigma_0], dtype=torch.float32) # Only sigma_0 for 1-step inference

        i = torch.arange(N + 1, dtype=torch.float32)
        term1 = self.sigma_0**(1/self.rho)
        term2 = (i / N) * (self.sigma_N**(1/self.rho) - self.sigma_0**(1/self.rho))
        sigmas = (term1 + term2)**self.rho
        return sigmas

    def _get_discrete_timestep_weights(self, sigmas):
        """
        Calculates discrete probability distribution on timesteps, mimicking continuous distribution.
        p(sigma_i) proportional to erf((log(sigma_i+1) - P_mean) / sqrt(2)P_std) - erf((log(sigma_i) - P_mean) / sqrt(2)P_std)
        Using parameters from Appendix D: P_mean = -1.1, P_std = 2.0
        """
        P_mean = -1.1
        P_std = 2.0
        
        # Calculate erf values for sigma_i and sigma_i+1
        erf_sigmas_plus_1 = torch.erf((torch.log(sigmas[1:]) - P_mean) / (math.sqrt(2) * P_std))
        erf_sigmas = torch.erf((torch.log(sigmas[:-1]) - P_mean) / (math.sqrt(2) * P_std))
        
        # Differences in erf values
        probs_unnormalized = erf_sigmas_plus_1 - erf_sigmas
        
        # Normalize to get a probability distribution
        probs = probs_unnormalized / probs_unnormalized.sum()
        
        # For weighting the loss, lambda(sigma_i) = 1 / (sigma_i+1 - sigma_i)
        lambda_weights = 1.0 / (sigmas[1:] - sigmas[:-1])
        
        return probs, lambda_weights

    def get_training_timesteps(self, N, batch_size, device):
        """
        Samples an index i and returns the corresponding sigma_t_i and sigma_t_i+1
        based on the progressive and weighted sampling strategy from the paper.
        """
        sigmas = self.generate_noise_schedule(N).to(device)
        
        # Get probability distribution for sampling i
        probs, lambda_weights = self._get_discrete_timestep_weights(sigmas)
        
        # Sample index i from {0, ..., N-1} based on probabilities
        # multinomial returns indices
        i_indices = torch.multinomial(probs, num_samples=batch_size, replacement=True)
        
        sigma_ti_plus_1 = sigmas[i_indices + 1]
        sigma_ti = sigmas[i_indices]
        
        # Get loss weighting for the sampled timesteps
        lambda_t_i = lambda_weights[i_indices]
        
        return sigma_ti, sigma_ti_plus_1, lambda_t_i

