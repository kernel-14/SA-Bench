"""
Masked Diffusion Model (MDM)
=============================
Implementation of the Masked Diffusion Model as described in Section 2 of the paper:
'Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions'

Based on the framework from Shi et al. (2024), Sahoo et al. (2025):
- Forward process: coordinate-independent masking
- Reverse process: learned categorical denoising
- Training: ELBO-based loss equivalent to masked language modeling

Key components:
1. Forward masking process with noise schedule α_t
2. Denoising network p_θ(·|x_t) (time-embedding-free)
3. Training loss L_θ (Equation 1)
4. Vanilla sampling (Algorithm 1)
5. Adaptive sampling (Section 4.1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, List, Callable


class NoiseSchedule:
    """Noise schedule α_t for the forward masking process."""
    
    def __init__(self, schedule_type: str = 'cosine', T: int = 1000):
        """
        Args:
            schedule_type: 'cosine', 'linear', or 'log_linear'
            T: Number of discretization steps
        """
        self.T = T
        
        if schedule_type == 'cosine':
            # Cosine schedule as in Nichol & Dhariwal (2021), adapted for discrete
            t = torch.linspace(1, 0, T + 1)
            self.alpha = torch.cos((1 - t) * math.pi / 2) ** 2
            self.alpha = self.alpha / self.alpha[0]  # Normalize so α_0 = 1
            self.alpha[-1] = 0.0  # α_1 = 0
            
        elif schedule_type == 'linear':
            self.alpha = torch.linspace(1.0, 0.0, T + 1)
            
        elif schedule_type == 'log_linear':
            # α_t = 1 - t for log-linear as in Sahoo et al. (2025)
            self.alpha = 1.0 - torch.linspace(0.0, 1.0, T + 1)
            self.alpha = self.alpha / self.alpha[0]
            self.alpha[-1] = 0.0
            
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        # Precompute for fast access
        self.alpha_t = self.alpha  # α_t for t in [0, 1] discretized
    
    def get_alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Get α_t for a batch of time indices."""
        return self.alpha[t]


class MDMConfig:
    """Configuration for Masked Diffusion Model."""
    
    def __init__(
        self,
        vocab_size: int,
        seq_length: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 512,
        noise_schedule: str = 'cosine',
        T: int = 1000,
        mask_token_id: int = 0,
    ):
        self.vocab_size = vocab_size  # Including mask token (0)
        self.seq_length = seq_length
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.max_seq_length = max_seq_length
        self.noise_schedule = noise_schedule
        self.T = T
        self.mask_token_id = mask_token_id


class SinusoidalPositionalEncoding(nn.Module):
    """Learnable positional embeddings (not RoPE, to avoid left-to-right bias)."""
    
    def __init__(self, d_model: int, max_seq_length: int):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_seq_length, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model) or (batch, seq_len)
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        return self.pos_embedding(positions)


class MDMTransformer(nn.Module):
    """
    Transformer-based denoising network for MDM.
    
    This is a bidirectional transformer (BERT-style) that takes a partially masked
    sequence and predicts the clean token for each masked position.
    
    The network is time-embedding-free: information about the noise level is
    implicitly contained in the number of masked tokens.
    """
    
    def __init__(self, config: MDMConfig):
        super().__init__()
        self.config = config
        
        # Token embedding (vocab includes mask token)
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Positional encoding (learnable, no left-to-right bias)
        self.pos_encoding = SinusoidalPositionalEncoding(config.d_model, config.max_seq_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        
        # Output projection
        self.output_proj = nn.Linear(config.d_model, config.vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the denoising network.
        
        Args:
            x: (batch, seq_len) token ids, with mask_token_id for masked positions
            attention_mask: Optional attention mask
            
        Returns:
            logits: (batch, seq_len, vocab_size) logits for each position
        """
        # Embed tokens
        h = self.token_embedding(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        h = h + self.pos_encoding(h)
        
        # Transformer
        h = self.transformer(h, src_key_padding_mask=attention_mask)
        
        # Output projection
        logits = self.output_proj(h)  # (batch, seq_len, vocab_size)
        
        return logits
    
    def get_log_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Get log probabilities from logits."""
        logits = self.forward(x)
        return F.log_softmax(logits, dim=-1)
    
    def get_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Get probabilities from logits."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)


class MaskedDiffusionModel:
    """
    Masked Diffusion Model combining forward process, reverse process,
    training, and sampling.
    
    Implements:
    - Forward masking process q_{t|0}
    - Training loss L_θ (Equation 1 in paper)
    - Vanilla MDM inference (Algorithm 1)
    - Adaptive MDM inference (Section 4.1)
    """
    
    def __init__(self, denoiser: MDMTransformer, config: MDMConfig):
        self.denoiser = denoiser
        self.config = config
        self.noise_schedule = NoiseSchedule(config.noise_schedule, config.T)
        self.mask_id = config.mask_token_id
    
    def forward_process(self, x_0: torch.Tensor, t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply forward masking process at noise level t.
        
        q_{t|0}(x_t^i | x_0^i) = Cat(α_t e_{x_0^i} + (1-α_t) e_0)
        
        Args:
            x_0: (batch, seq_len) clean sequence
            t: Noise level index
            
        Returns:
            x_t: Masked sequence
            mask: Boolean mask indicating which positions are masked
        """
        alpha_t = self.noise_schedule.get_alpha(torch.tensor([t])).item()
        
        # Sample mask for each position
        mask = torch.rand_like(x_0.float()) > alpha_t  # True where masked
        
        # Apply mask
        x_t = x_0.clone()
        x_t[mask] = self.mask_id
        
        return x_t, mask
    
    def reverse_step(self, x_t: torch.Tensor, s: int, t: int,
                     adaptive: bool = False,
                     oracle_fn: Optional[Callable] = None,
                     gumbel_temp: float = 0.0) -> torch.Tensor:
        """
        Single reverse step from noise level t to s.
        
        Args:
            x_t: Current partially masked sequence
            s: Target noise level
            t: Current noise level
            adaptive: Whether to use adaptive selection
            oracle_fn: Function F(θ, x_t) to select positions to unmask
            gumbel_temp: Temperature for Gumbel noise in oracle
            
        Returns:
            x_s: Sequence after unmasking
        """
        batch_size, seq_len = x_t.shape
        alpha_s = self.noise_schedule.get_alpha(torch.tensor([s])).item()
        alpha_t = self.noise_schedule.get_alpha(torch.tensor([t])).item()
        
        # Get model probabilities
        with torch.no_grad():
            probs = self.denoiser.get_probs(x_t)  # (batch, seq_len, vocab_size)
        
        # Identify currently masked positions
        is_masked = (x_t == self.mask_id)  # (batch, seq_len)
        
        # For each sequence in batch, determine which positions to unmask
        if adaptive and oracle_fn is not None:
            # Use oracle to select positions
            selected_mask = oracle_fn(probs, is_masked, alpha_s, alpha_t, gumbel_temp)
        else:
            # Vanilla: randomly select positions
            prob_unmask = (alpha_s - alpha_t) / (1.0 - alpha_t + 1e-8)
            selected_mask = torch.rand_like(x_t.float()) < prob_unmask
            selected_mask = selected_mask & is_masked
        
        # Sample new tokens for selected positions
        x_s = x_t.clone()
        for b in range(batch_size):
            unmask_positions = selected_mask[b].nonzero(as_tuple=True)[0]
            if len(unmask_positions) > 0:
                sampled = torch.multinomial(probs[b, unmask_positions], 1).squeeze(-1)
                x_s[b, unmask_positions] = sampled
        
        return x_s
    
    def vanilla_sample(self, batch_size: int, num_steps: int = 50,
                       device: str = 'cpu') -> torch.Tensor:
        """
        Vanilla MDM inference (Algorithm 1 in paper).
        
        Starts from fully masked sequence and iteratively unmasks tokens.
        
        Args:
            batch_size: Number of sequences to generate
            num_steps: Number of reverse steps
            device: Device for computation
            
        Returns:
            Generated sequences
        """
        seq_len = self.config.seq_length
        
        # Start from fully masked
        x_t = torch.full((batch_size, seq_len), self.mask_id, dtype=torch.long, device=device)
        
        # Reverse process
        step_size = self.config.T // num_steps
        for step in range(num_steps):
            t = self.config.T - step * step_size
            s = max(0, self.config.T - (step + 1) * step_size)
            x_t = self.reverse_step(x_t, s, t, adaptive=False)
        
        return x_t
    
    def adaptive_sample(self, batch_size: int, num_steps: int = 50,
                        oracle: str = 'top_probability',
                        gumbel_temp: float = 0.0,
                        device: str = 'cpu') -> torch.Tensor:
        """
        Adaptive MDM inference (Section 4.1).
        
        Uses an oracle to select which tokens to unmask at each step.
        
        Args:
            batch_size: Number of sequences to generate
            num_steps: Number of reverse steps
            oracle: 'top_probability' or 'top_probability_margin'
            gumbel_temp: Temperature for Gumbel noise
            device: Device for computation
            
        Returns:
            Generated sequences
        """
        if oracle == 'top_probability':
            oracle_fn = top_probability_oracle
        elif oracle == 'top_probability_margin':
            oracle_fn = top_probability_margin_oracle
        else:
            raise ValueError(f"Unknown oracle: {oracle}")
        
        seq_len = self.config.seq_length
        
        # Start from fully masked
        x_t = torch.full((batch_size, seq_len), self.mask_id, dtype=torch.long, device=device)
        
        # Reverse process
        step_size = self.config.T // num_steps
        for step in range(num_steps):
            t = self.config.T - step * step_size
            s = max(0, self.config.T - (step + 1) * step_size)
            x_t = self.reverse_step(x_t, s, t, adaptive=True, oracle_fn=oracle_fn,
                                    gumbel_temp=gumbel_temp)
        
        return x_t
    
    def compute_loss(self, x_0: torch.Tensor) -> torch.Tensor:
        """
        Compute the MDM training loss L_θ (Equation 1 in paper).
        
        L_θ = ∫_0^1 (α_t' / (1-α_t)) E_{x_t} Σ_{i: x_t^i=0} -log p_θ(x_0^i | x_t, t) dt
        
        Args:
            x_0: (batch, seq_len) clean sequences
            
        Returns:
            Scalar loss value
        """
        batch_size, seq_len = x_0.shape
        device = x_0.device
        
        # Sample random noise level t uniformly from [0, T-1]
        t = torch.randint(1, self.config.T, (1,), device=device).item()
        
        # Apply forward process
        x_t, mask = self.forward_process(x_0, t)
        
        # Get model predictions
        log_probs = self.denoiser.get_log_probs(x_t)  # (batch, seq_len, vocab_size)
        
        # Compute loss only on masked positions
        loss = 0.0
        for b in range(batch_size):
            masked_positions = mask[b].nonzero(as_tuple=True)[0]
            if len(masked_positions) > 0:
                # Gather log probabilities for the true tokens at masked positions
                log_p = log_probs[b, masked_positions, x_0[b, masked_positions]]
                loss -= log_p.mean()
        
        # Weight by α_t' / (1-α_t) ≈ discretization weight
        alpha_t = self.noise_schedule.get_alpha(torch.tensor([t])).item()
        alpha_t_next = self.noise_schedule.get_alpha(torch.tensor([t + 1])).item()
        weight = abs(alpha_t_next - alpha_t) / (1.0 - alpha_t + 1e-8)
        
        return loss * weight / batch_size
    
    def compute_pi_learner_loss(self, x_0: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
        """
        Compute the π-learner loss (Equation 3 in paper).
        
        For a given permutation π, this is the autoregressive loss
        when predicting tokens in order π.
        
        Args:
            x_0: (batch, seq_len) clean sequences
            pi: (batch, seq_len) permutation or (seq_len,) shared permutation
            
        Returns:
            π-learner loss
        """
        batch_size, seq_len = x_0.shape
        device = x_0.device
        
        total_loss = 0.0
        
        for i in range(seq_len):
            # Mask positions π(i), ..., π(L-1)
            if pi.dim() == 1:
                mask_positions = pi[i:]  # All positions from i onwards
            else:
                mask_positions = pi[:, i:]  # (batch, ...)
            
            # Create masked input
            x_input = x_0.clone()
            if pi.dim() == 1:
                x_input[:, mask_positions] = self.mask_id
                target_pos = pi[i:i+1]
            else:
                for b in range(batch_size):
                    x_input[b, mask_positions[b]] = self.mask_id
                target_pos = pi[:, i:i+1]
            
            # Get prediction for position π(i)
            log_probs = self.denoiser.get_log_probs(x_input)
            
            if pi.dim() == 1:
                log_p = log_probs[:, target_pos[0], x_0[:, target_pos[0]]]
            else:
                log_p = log_probs[torch.arange(batch_size), target_pos[:, 0], 
                                  x_0[torch.arange(batch_size), target_pos[:, 0]]]
            
            total_loss -= log_p.mean()
        
        return total_loss / seq_length


# ─── Oracle Functions for Adaptive Inference ───

def top_probability_oracle(probs: torch.Tensor, is_masked: torch.Tensor,
                           alpha_s: float, alpha_t: float,
                           gumbel_temp: float = 0.0) -> torch.Tensor:
    """
    Top probability oracle (Section 4.1).
    
    Selects positions based on max probability: max_j p_θ(x^i = j | x_t).
    
    Args:
        probs: (batch, seq_len, vocab_size) model probabilities
        is_masked: (batch, seq_len) boolean mask
        alpha_s, alpha_t: Noise schedule values
        gumbel_temp: Gumbel noise temperature
        
    Returns:
        selected: (batch, seq_len) boolean mask of positions to unmask
    """
    batch_size, seq_len, vocab_size = probs.shape
    device = probs.device
    
    # Compute certainty scores
    max_probs, _ = probs.max(dim=-1)  # (batch, seq_len)
    
    # Mask out already unmasked positions
    max_probs = max_probs.masked_fill(~is_masked, -float('inf'))
    
    # Number of tokens to unmask
    num_masked = is_masked.sum(dim=-1).float()  # (batch,)
    prob_unmask = (alpha_s - alpha_t) / (1.0 - alpha_t + 1e-8)
    K = (num_masked * prob_unmask).long()
    
    # Add Gumbel noise if needed
    if gumbel_temp > 0:
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(max_probs) + 1e-8) + 1e-8)
        scores = max_probs + gumbel_temp * gumbel_noise
    else:
        scores = max_probs
    
    # Select top K positions for each sequence
    selected = torch.zeros_like(is_masked)
    for b in range(batch_size):
        if K[b] > 0:
            _, top_indices = scores[b].topk(min(K[b].item(), is_masked[b].sum().item()))
            selected[b, top_indices] = True
    
    return selected


def top_probability_margin_oracle(probs: torch.Tensor, is_masked: torch.Tensor,
                                   alpha_s: float, alpha_t: float,
                                   gumbel_temp: float = 0.0) -> torch.Tensor:
    """
    Top probability margin oracle (Section 4.1).
    
    Selects positions based on: |p_θ(x^i = j_1 | x_t) - p_θ(x^i = j_2 | x_t)|
    where j_1, j_2 are the two most probable values.
    
    Args:
        probs: (batch, seq_len, vocab_size) model probabilities
        is_masked: (batch, seq_len) boolean mask
        alpha_s, alpha_t: Noise schedule values
        gumbel_temp: Gumbel noise temperature
        
    Returns:
        selected: (batch, seq_len) boolean mask of positions to unmask
    """
    batch_size, seq_len, vocab_size = probs.shape
    device = probs.device
    
    # Get top 2 probabilities
    top2_probs, _ = probs.topk(2, dim=-1)  # (batch, seq_len, 2)
    margins = top2_probs[:, :, 0] - top2_probs[:, :, 1]  # (batch, seq_len)
    
    # Mask out already unmasked positions
    margins = margins.masked_fill(~is_masked, -float('inf'))
    
    # Number of tokens to unmask
    num_masked = is_masked.sum(dim=-1).float()
    prob_unmask = (alpha_s - alpha_t) / (1.0 - alpha_t + 1e-8)
    K = (num_masked * prob_unmask).long()
    
    # Add Gumbel noise if needed
    if gumbel_temp > 0:
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(margins) + 1e-8) + 1e-8)
        scores = margins + gumbel_temp * gumbel_noise
    else:
        scores = margins
    
    # Select top K positions
    selected = torch.zeros_like(is_masked)
    for b in range(batch_size):
        if K[b] > 0:
            _, top_indices = scores[b].topk(min(K[b].item(), is_masked[b].sum().item()))
            selected[b, top_indices] = True
    
    return selected
