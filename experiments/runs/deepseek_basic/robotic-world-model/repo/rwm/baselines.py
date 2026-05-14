"""
Baseline world model architectures for comparison with RWM.

Based on Table S8 from the paper:
- MLP: hidden shape (256, 256), ReLU activation
- RSSM: GRU type, hidden size 256, 2 layers, latent dim 64, categorical 32
- Transformer: decoder type, dim 64, 8 heads, 2 layers, context 32, sinusoidal PE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math


# ============================================================
# MLP Baseline
# ============================================================
class MLPWorldModel(nn.Module):
    """
    MLP-based world model baseline.
    Architecture: (256, 256) hidden layers, ReLU activation.
    """
    
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        history_horizon: int = 32,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.history_horizon = history_horizon
        
        input_dim = history_horizon * (obs_dim + act_dim)
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, obs_dim * 2),  # mean and log_std
        )
        
    def forward(self, obs_history: torch.Tensor, act_history: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            obs_history: (batch, M, obs_dim)
            act_history: (batch, M, act_dim)
        Returns:
            dict with 'obs_means', 'obs_stds' for next step prediction
        """
        batch = obs_history.shape[0]
        x = torch.cat([obs_history, act_history], dim=-1)  # (batch, M, input_dim)
        x = x.reshape(batch, -1)  # Flatten
        
        out = self.net(x)
        mean = out[..., :self.obs_dim]
        log_std = out[..., self.obs_dim:]
        log_std = torch.clamp(log_std, min=-10, max=2)
        std = torch.exp(log_std)
        
        return {
            'obs_means': mean.unsqueeze(1),  # (batch, 1, obs_dim)
            'obs_stds': std.unsqueeze(1),
        }
    
    def autoregressive_forward(
        self,
        obs_history: torch.Tensor,
        act_history: torch.Tensor,
        act_future: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Autoregressive rollout for N steps."""
        batch = obs_history.shape[0]
        N = act_future.shape[1]
        device = obs_history.device
        
        obs_means_list = []
        obs_stds_list = []
        
        obs_seq = obs_history.clone()
        act_seq = act_history.clone()
        
        for k in range(N):
            pred = self.forward(obs_seq[:, -self.history_horizon:, :], 
                               act_seq[:, -self.history_horizon:, :])
            obs_mean = pred['obs_means'][:, 0, :]  # (batch, obs_dim)
            obs_std = pred['obs_stds'][:, 0, :]
            
            obs_means_list.append(obs_mean)
            obs_stds_list.append(obs_std)
            
            # Append prediction and action for next step
            obs_seq = torch.cat([obs_seq, obs_mean.unsqueeze(1)], dim=1)
            act_seq = torch.cat([act_seq, act_future[:, k:k+1, :]], dim=1)
        
        return {
            'obs_means': torch.stack(obs_means_list, dim=1),
            'obs_stds': torch.stack(obs_stds_list, dim=1),
        }


# ============================================================
# RSSM Baseline (Recurrent State-Space Model)
# ============================================================
class RSSM(nn.Module):
    """
    Recurrent State-Space Model as used in PlaNet/Dreamer.
    
    Architecture from Table S8:
    - GRU type, hidden size 256, 2 layers
    - Latent dimension 64
    - Categorical latent with 32 categories
    """
    
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_size: int = 256,
        latent_dim: int = 64,
        num_categories: int = 32,
        num_layers: int = 2,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.num_categories = num_categories
        self.class_dim = latent_dim * num_categories  # 64 * 32 = 2048
        
        # Encoder: observation -> latent posterior
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.class_dim),
        )
        
        # Recurrent dynamics: deterministic state
        self.rnn = nn.GRU(
            input_size=self.class_dim + act_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        
        # Prior: predict latent from deterministic state
        self.prior = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.class_dim),
        )
        
        # Decoder: predict next observation
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size + self.class_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, obs_dim * 2),
        )
        
    def _compute_latent(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute categorical latent from logits."""
        # logits: (batch, latent_dim, num_categories) 
        # or (batch, seq_len, latent_dim, num_categories)
        shape = logits.shape
        logits_reshaped = logits.reshape(-1, self.latent_dim, self.num_categories)
        
        # Straight-through Gumbel-softmax for training
        probs = F.softmax(logits_reshaped, dim=-1)
        latent = F.gumbel_softmax(logits_reshaped, tau=1.0, hard=True)
        
        latent_flat = latent.reshape(*shape[:-1], self.class_dim)
        probs_flat = probs.reshape(*shape[:-1], self.class_dim)
        
        return latent_flat, probs_flat
    
    def forward(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for a sequence.
        
        Args:
            obs: (batch, seq_len, obs_dim)
            act: (batch, seq_len, act_dim)
            h: initial GRU hidden state
        """
        batch, seq_len, _ = obs.shape
        
        if h is None:
            h = torch.zeros(self.rnn.num_layers, batch, self.hidden_size, device=obs.device)
        
        outputs = {'obs_means': [], 'obs_stds': []}
        
        for t in range(seq_len):
            o_t = obs[:, t, :]
            a_t = act[:, t, :]
            
            # Encoder: posterior
            h_t = h[-1]  # Last layer hidden state
            posterior_logits = self.encoder(torch.cat([o_t, h_t], dim=-1))
            posterior_logits = posterior_logits.reshape(batch, self.latent_dim, self.num_categories)
            z_t, _ = self._compute_latent(posterior_logits)
            
            # RNN step
            rnn_input = torch.cat([z_t, a_t], dim=-1).unsqueeze(1)
            _, h = self.rnn(rnn_input, h)
            
            # Prior from new hidden state
            h_next = h[-1]
            prior_logits = self.prior(h_next).reshape(batch, self.latent_dim, self.num_categories)
            
            # Decoder: predict next observation
            dec_input = torch.cat([h_next, z_t], dim=-1)
            dec_out = self.decoder(dec_input)
            mean = dec_out[..., :self.obs_dim]
            log_std = dec_out[..., self.obs_dim:]
            log_std = torch.clamp(log_std, min=-10, max=2)
            std = torch.exp(log_std)
            
            outputs['obs_means'].append(mean)
            outputs['obs_stds'].append(std)
        
        outputs['obs_means'] = torch.stack(outputs['obs_means'], dim=1)
        outputs['obs_stds'] = torch.stack(outputs['obs_stds'], dim=1)
        
        return outputs


# ============================================================
# Transformer Baseline
# ============================================================
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * 
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1)].unsqueeze(0)


class TransformerWorldModel(nn.Module):
    """
    Transformer-based world model baseline.
    
    Architecture from Table S8:
    - Decoder type, dim 64, 8 heads, 2 layers
    - Context length 32, sinusoidal positional encoding
    """
    
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 2,
        context_length: int = 32,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.d_model = d_model
        self.context_length = context_length
        
        # Input projections
        self.obs_proj = nn.Linear(obs_dim, d_model)
        self.act_proj = nn.Linear(act_dim, d_model)
        
        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, max_len=context_length * 2)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, obs_dim * 2),
        )
        
    def forward(
        self,
        obs_history: torch.Tensor,
        act_history: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            obs_history: (batch, seq_len, obs_dim)
            act_history: (batch, seq_len, act_dim)
        """
        batch, seq_len, _ = obs_history.shape
        
        # Project inputs
        obs_emb = self.obs_proj(obs_history)  # (batch, seq_len, d_model)
        act_emb = self.act_proj(act_history)
        
        # Combine: interleave or add? Paper implies sequence modeling of (obs, act) pairs
        # We'll use obs embeddings as the main sequence, adding action information
        combined = obs_emb + act_emb
        
        # Add positional encoding
        combined = self.pos_enc(combined)
        
        # Transformer forward (decode from "memory" of past)
        # Use causal mask for autoregressive prediction
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=obs_history.device) * float('-inf'),
            diagonal=1
        )
        
        out = self.transformer(combined, combined, tgt_mask=causal_mask)
        
        # Predict next observation
        pred = self.output_head(out)
        mean = pred[..., :self.obs_dim]
        log_std = pred[..., self.obs_dim:]
        log_std = torch.clamp(log_std, min=-10, max=2)
        std = torch.exp(log_std)
        
        return {
            'obs_means': mean,
            'obs_stds': std,
        }


# ============================================================
# Factory functions for standard configurations
# ============================================================
def create_mlp_baseline(obs_dim: int, act_dim: int) -> MLPWorldModel:
    return MLPWorldModel(obs_dim=obs_dim, act_dim=act_dim, history_horizon=32)


def create_rssm_baseline(obs_dim: int, act_dim: int) -> RSSM:
    return RSSM(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_size=256,
        latent_dim=64,
        num_categories=32,
        num_layers=2,
    )


def create_transformer_baseline(obs_dim: int, act_dim: int) -> TransformerWorldModel:
    return TransformerWorldModel(
        obs_dim=obs_dim,
        act_dim=act_dim,
        d_model=64,
        nhead=8,
        num_layers=2,
        context_length=32,
    )
