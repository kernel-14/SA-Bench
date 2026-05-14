"""
world_model.py

Implements the core world model architecture (GRU + MLP heads) for the Robotic World Model
(RWM) paper.  Contains two classes:

- WorldModel (base): teacher‑forcing forward pass, loss computation (Gaussian NLL + BCE/MSE),
  normalisation utilities, and shared architecture construction.
- RWM (subclass): dual‑autoregressive forward pass (inner history + outer forecast with
  reparameterisation), autoregressive training (`pretrain` / `finetune_online`), and a
  helper to perform autoregressive rollouts for imagination.

All hyperparameters are sourced from `config.yaml` (passed via constructor or method
arguments).  The module does not depend on simulation code and can be tested in isolation.
"""

import math
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ----------------------------------------------------------------------
# Helper: split privileged dimensions into binary / continuous per robot
# ----------------------------------------------------------------------
def get_priv_split(robot: str) -> Tuple[int, int]:
    """
    Return (num_binary_priv, num_cont_priv) for a given robot identifier.

    These match the privileged information described in Table S3 of the paper.
    """
    mapping = {
        "anymal_d":    (8, 0),    # 4 knee contact + 4 foot contact (all binary)
        "unitree_g1":  (26, 4),   # 26 body contacts (binary) + 2 foot height + 2 foot velocity
    }
    if robot not in mapping:
        raise ValueError(f"Unknown robot '{robot}'. Cannot determine priv split.")
    return mapping[robot]


# ----------------------------------------------------------------------
# Base World Model
# ----------------------------------------------------------------------
class WorldModel(nn.Module):
    """
    Base world model with a GRU encoder and two MLP heads (observation, privileged).

    This class provides:
    - Architecture construction from configuration.
    - Teacher‑forcing forward pass (for baseline comparisons).
    - Loss computation (Gaussian NLL for observations, BCE for binary privileged,
      MSE for continuous privileged).
    - Normalisation / denormalisation utilities.

    Subclass `RWM` overrides the forward pass to implement autoregressive prediction.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        priv_dim: int,
        num_binary_priv: int,
        num_cont_priv: int,
        history_len: int,
        forecast_len: int,
        gru_hidden_size: List[int] = (256, 256),
        head_hidden_size: List[int] = (128,),
        activation: str = "relu",
        std_min: float = 1e-4,
    ):
        """
        Args:
            obs_dim:           Dimensionality of the (world‑model) observation.
            act_dim:           Dimensionality of the action.
            priv_dim:          Total dimensionality of privileged information.
            num_binary_priv:   How many of the first priv_dim dimensions are binary.
            num_cont_priv:     How many of the remaining are continuous.
            history_len:       Number of historical steps (M) used as context.
            forecast_len:      Number of future steps (N) to predict.
            gru_hidden_size:   Hidden size(s) for the GRU.  If a list, must have equal
                               values (standard PyTorch GRU limitation).
            head_hidden_size:  List of hidden sizes for the MLP heads.
            activation:        Activation function name ("relu" or "elu").
            std_min:           Minimum standard deviation for numerical stability.
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.priv_dim = priv_dim
        self.num_binary_priv = num_binary_priv
        self.num_cont_priv = num_cont_priv
        self.history_len = history_len
        self.forecast_len = forecast_len
        self.std_min = std_min

        # ---------- GRU ----------
        num_layers = len(gru_hidden_size)
        if num_layers == 0:
            raise ValueError("gru_hidden_size must not be empty")
        hidden_size = gru_hidden_size[-1]
        # PyTorch GRU requires all layers to have the same hidden size.
        # We trust the configuration to provide equal values (as in the paper: [256,256]).
        if not all(h == hidden_size for h in gru_hidden_size):
            raise ValueError(
                f"All elements of gru_hidden_size must be equal for standard GRU. "
                f"Got {gru_hidden_size}."
            )

        self.gru = nn.GRU(
            input_size=obs_dim + act_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # ---------- Activation ----------
        if activation.lower() not in ("relu", "elu"):
            raise ValueError(f"Unsupported activation: {activation}")
        act_fn = nn.ReLU if activation.lower() == "relu" else nn.ELU

        # ---------- Observation head ----------
        obs_head_layers = []
        in_features = hidden_size
        for h in head_hidden_size:
            obs_head_layers.append(nn.Linear(in_features, h))
            obs_head_layers.append(act_fn())
            in_features = h
        obs_head_layers.append(nn.Linear(in_features, obs_dim * 2))
        self.obs_head = nn.Sequential(*obs_head_layers)

        # ---------- Privileged head ----------
        priv_head_layers = []
        in_features = hidden_size
        for h in head_hidden_size:
            priv_head_layers.append(nn.Linear(in_features, h))
            priv_head_layers.append(act_fn())
            in_features = h
        priv_head_layers.append(nn.Linear(in_features, priv_dim))
        self.priv_head = nn.Sequential(*priv_head_layers)

        # ---------- Normalisation buffers ----------
        # Initialised to zero mean / unit std; must be set later.
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))
        self.register_buffer("act_mean", torch.zeros(act_dim))
        self.register_buffer("act_std", torch.ones(act_dim))
        if num_cont_priv > 0:
            self.register_buffer("priv_cont_mean", torch.zeros(num_cont_priv))
            self.register_buffer("priv_cont_std", torch.ones(num_cont_priv))
        else:
            # Dummy buffers for compatibility
            self.register_buffer("priv_cont_mean", torch.zeros(0))
            self.register_buffer("priv_cont_std", torch.ones(0))

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------
    def _normalize(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return (x - mean) / (std + 1e-8)

    def _denormalize(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean

    def normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return self._normalize(obs, self.obs_mean, self.obs_std)

    def denormalize_obs(self, obs_norm: torch.Tensor) -> torch.Tensor:
        return self._denormalize(obs_norm, self.obs_mean, self.obs_std)

    def normalize_act(self, act: torch.Tensor) -> torch.Tensor:
        return self._normalize(act, self.act_mean, self.act_std)

    def normalize_priv_cont(self, priv_cont: torch.Tensor) -> torch.Tensor:
        """Normalize the continuous part of privileged information."""
        if self.num_cont_priv == 0:
            return priv_cont
        return self._normalize(priv_cont, self.priv_cont_mean, self.priv_cont_std)

    def denormalize_priv_cont(self, priv_cont_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize the continuous part of privileged information."""
        if self.num_cont_priv == 0:
            return priv_cont_norm
        return self._denormalize(priv_cont_norm, self.priv_cont_mean, self.priv_cont_std)

    # ------------------------------------------------------------------
    # Set normalisation statistics (e.g., after computing from dataset)
    # ------------------------------------------------------------------
    def set_normalization_stats(
        self,
        obs_mean: torch.Tensor,
        obs_std: torch.Tensor,
        act_mean: torch.Tensor,
        act_std: torch.Tensor,
        priv_cont_mean: Optional[torch.Tensor] = None,
        priv_cont_std: Optional[torch.Tensor] = None,
    ) -> None:
        """Copy externally computed normalisation statistics."""
        self.obs_mean.copy_(obs_mean)
        self.obs_std.copy_(obs_std)
        self.act_mean.copy_(act_mean)
        self.act_std.copy_(act_std)
        if priv_cont_mean is not None and self.num_cont_priv > 0:
            self.priv_cont_mean.copy_(priv_cont_mean)
        if priv_cont_std is not None and self.num_cont_priv > 0:
            self.priv_cont_std.copy_(priv_cont_std)

    def compute_statistics(self, buffer: "TrajectoryBuffer") -> None:
        """
        Compute per‑dimension mean and standard deviation from a `TrajectoryBuffer`
        and store them as the model’s normalisation statistics.
        """
        # Collect all episodes
        all_obs = []
        all_act = []
        all_priv_cont = []

        for ep in buffer.episodes:
            all_obs.append(ep["obs"])        # (T, obs_dim)
            all_act.append(ep["act"])        # (T, act_dim)
            if self.num_cont_priv > 0:
                priv = ep["priv"]            # (T, priv_dim)
                # Assume binary first, then continuous
                priv_cont = priv[..., self.num_binary_priv:]
                all_priv_cont.append(priv_cont)

        obs = torch.cat(all_obs, dim=0)      # (total_transitions, obs_dim)
        act = torch.cat(all_act, dim=0)

        obs_mean = obs.mean(dim=0)
        obs_std = obs.std(dim=0, unbiased=False).clamp(min=1e-6)
        act_mean = act.mean(dim=0)
        act_std = act.std(dim=0, unbiased=False).clamp(min=1e-6)

        priv_cont_mean = None
        priv_cont_std = None
        if self.num_cont_priv > 0 and all_priv_cont:
            priv_cont = torch.cat(all_priv_cont, dim=0)
            priv_cont_mean = priv_cont.mean(dim=0)
            priv_cont_std = priv_cont.std(dim=0, unbiased=False).clamp(min=1e-6)

        self.set_normalization_stats(
            obs_mean, obs_std, act_mean, act_std,
            priv_cont_mean, priv_cont_std,
        )

    # ------------------------------------------------------------------
    # Teacher‑forcing forward pass (used by baselines)
    # ------------------------------------------------------------------
    def forward(
        self,
        obs_seq: torch.Tensor,          # (B, T, obs_dim)
        act_seq: torch.Tensor,          # (B, T, act_dim)
        h0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Teacher‑forcing forward pass.  Assumes the full ground‑truth sequences are
        provided (length T = history_len + forecast_len).

        Returns:
            obs_mean:   (B, T, obs_dim)
            obs_log_std:(B, T, obs_dim)
            priv_params:(B, T, priv_dim)   (raw logits for binary part, raw values for continuous)
        """
        B, T, _ = obs_seq.shape

        # Normalise inputs
        obs_norm = self.normalize_obs(obs_seq)
        act_norm = self.normalize_act(act_seq)

        # Concatenate along feature axis
        gru_input = torch.cat([obs_norm, act_norm], dim=-1)   # (B, T, in_dim)

        # GRU
        gru_out, _ = self.gru(gru_input, h0)                  # (B, T, hidden_size)

        # Observation head
        obs_params = self.obs_head(gru_out)                    # (B, T, obs_dim*2)
        obs_mean, obs_log_std = torch.chunk(obs_params, 2, dim=-1)

        # Stable range for log‑std
        obs_log_std = torch.clamp(obs_log_std, min=-20.0, max=2.0)

        # Privileged head
        priv_params = self.priv_head(gru_out)                  # (B, T, priv_dim)

        return obs_mean, obs_log_std, priv_params

    # ------------------------------------------------------------------
    # Loss computation (on the forecast horizon N only)
    # ------------------------------------------------------------------
    def compute_loss(
        self,
        obs_targets: torch.Tensor,      # (B, N, obs_dim)  normalised observations
        priv_targets: torch.Tensor,     # (B, N, priv_dim)  continuous part normalised; binary raw
        obs_mean: torch.Tensor,         # (B, N, obs_dim)
        obs_log_std: torch.Tensor,      # (B, N, obs_dim)
        priv_params: torch.Tensor,      # (B, N, priv_dim) raw logits
    ) -> torch.Tensor:
        """
        Compute the total loss (Gaussian NLL + BCE + MSE) over the forecast horizon.

        Returns a scalar.
        """
        # Observation loss (negative log‑likelihood of Gaussian)
        dist = Normal(obs_mean, torch.exp(obs_log_std) + self.std_min)
        log_prob = dist.log_prob(obs_targets)           # (B, N, obs_dim)
        L_o = -log_prob.sum(dim=-1).mean()              # mean over batch, time, features

        # Privileged loss
        # Split binary / continuous
        if self.num_binary_priv > 0 and self.num_cont_priv > 0:
            priv_bin_pred = priv_params[..., :self.num_binary_priv]
            priv_cont_pred = priv_params[..., self.num_binary_priv:]
            priv_bin_target = priv_targets[..., :self.num_binary_priv]
            priv_cont_target = priv_targets[..., self.num_binary_priv:]
        elif self.num_binary_priv > 0:
            priv_bin_pred = priv_params
            priv_cont_pred = None
            priv_bin_target = priv_targets
            priv_cont_target = None
        elif self.num_cont_priv > 0:
            priv_bin_pred = None
            priv_cont_pred = priv_params
            priv_bin_target = None
            priv_cont_target = priv_targets
        else:
            priv_bin_pred = None
            priv_cont_pred = None
            priv_bin_target = None
            priv_cont_target = None

        L_bin = 0.0
        if priv_bin_pred is not None and priv_bin_target is not None:
            L_bin = F.binary_cross_entropy_with_logits(
                priv_bin_pred, priv_bin_target, reduction="mean"
            )

        L_cont = 0.0
        if priv_cont_pred is not None and priv_cont_target is not None:
            L_cont = F.mse_loss(priv_cont_pred, priv_cont_target)

        return L_o + L_bin + L_cont


# ----------------------------------------------------------------------
# RWM: Robotic World Model with dual‑autoregressive training
# ----------------------------------------------------------------------
class RWM(WorldModel):
    """
    Robotic World Model.

    Overrides the forward pass to implement the dual‑autoregressive mechanism
    described in Sec. 3.2 of the paper:

    - Inner autoregression: teacher‑forcing over the history horizon M.
    - Outer autoregression: autoregressive generation over the forecast horizon N,
      using reparameterised samples as inputs for subsequent steps.

    Also provides a training loop (`pretrain`) and an online fine‑tuning step
    (`finetune_online`), as well as a method for producing denormalised rollout
    predictions.
    """

    def forward(
        self,
        obs_seq: torch.Tensor,          # (B, T, obs_dim) with T >= M
        act_seq: torch.Tensor,          # (B, T, act_dim)
        h0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Autoregressive forward pass.

        Only the first M = `history_len` observations are used as ground‑truth input;
        the rest are ignored and instead generated autoregressively.  The full action
        sequence is used (ground‑truth during training, policy actions during imagination).
        Returns predicted distributions for all T steps (history teacher‑forced, forecast
        autoregressive).

        Returns:
            obs_mean:    (B, T, obs_dim)
            obs_log_std: (B, T, obs_dim)
            priv_params: (B, T, priv_dim)
        """
        B, T, _ = obs_seq.shape
        M = self.history_len
        if T < M:
            raise ValueError(f"Sequence length T={T} must be >= history_len M={M}")

        # Normalise full sequences (we only need act later, but normalise all for simplicity)
        obs_norm = self.normalize_obs(obs_seq)    # (B, T, obs_dim)
        act_norm = self.normalize_act(act_seq)    # (B, T, act_dim)

        # ---------- Inner autoregression (teacher‑forcing over history) ----------
        hist_obs = obs_norm[:, :M]                # (B, M, obs_dim)
        hist_act = act_norm[:, :M]
        hist_input = torch.cat([hist_obs, hist_act], dim=-1)   # (B, M, in_dim)

        gru_hist_out, h = self.gru(hist_input, h0)             # h: (num_layers, B, hidden_size)

        # Head outputs for history (not used in loss, but returned for completeness)
        hist_obs_params = self.obs_head(gru_hist_out)           # (B, M, obs_dim*2)
        hist_obs_mean, hist_obs_log_std = torch.chunk(hist_obs_params, 2, dim=-1)
        hist_obs_log_std = torch.clamp(hist_obs_log_std, -20.0, 2.0)
        hist_priv = self.priv_head(gru_hist_out)                # (B, M, priv_dim)

        # ---------- Outer autoregression (forecast) ----------
        forecast_steps = T - M
        forecast_obs_mean = []
        forecast_obs_log_std = []
        forecast_priv = []

        if forecast_steps > 0:
            for k in range(forecast_steps):
                # Use the last layer’s hidden state for prediction
                last_hidden = h[-1]                              # (B, hidden_size)

                # Predict next observation distribution
                pred_params = self.obs_head(last_hidden)         # (B, obs_dim*2)
                pred_mean, pred_log_std = torch.chunk(pred_params, 2, dim=-1)
                pred_log_std = torch.clamp(pred_log_std, -20.0, 2.0)

                # Privileged information
                pred_priv = self.priv_head(last_hidden)          # (B, priv_dim)

                # Reparameterisation: sample next observation
                std = torch.exp(pred_log_std) + self.std_min
                eps = torch.randn_like(std)
                next_obs = pred_mean + std * eps                 # (B, obs_dim)

                # Store predictions
                forecast_obs_mean.append(pred_mean.unsqueeze(1))   # (B, 1, obs_dim)
                forecast_obs_log_std.append(pred_log_std.unsqueeze(1))
                forecast_priv.append(pred_priv.unsqueeze(1))       # (B, 1, priv_dim)

                # Prepare next input: sampled observation + action from act_seq
                next_act = act_norm[:, M + k]                    # (B, act_dim)
                next_input = torch.cat([next_obs, next_act], dim=-1).unsqueeze(1)  # (B, 1, in_dim)

                # Feed to GRU for next step
                _, h = self.gru(next_input, h)

            # Concatenate forecast predictions
            forecast_obs_mean = torch.cat(forecast_obs_mean, dim=1)       # (B, N, obs_dim)
            forecast_obs_log_std = torch.cat(forecast_obs_log_std, dim=1)
            forecast_priv = torch.cat(forecast_priv, dim=1)
        else:
            forecast_obs_mean = torch.empty(B, 0, self.obs_dim, device=obs_seq.device)
            forecast_obs_log_std = torch.empty_like(forecast_obs_mean)
            forecast_priv = torch.empty(B, 0, self.priv_dim, device=obs_seq.device)

        # Concatenate history and forecast
        obs_mean = torch.cat([hist_obs_mean, forecast_obs_mean], dim=1)         # (B, T, obs_dim)
        obs_log_std = torch.cat([hist_obs_log_std, forecast_obs_log_std], dim=1)
        priv_params = torch.cat([hist_priv, forecast_priv], dim=1)              # (B, T, priv_dim)

        return obs_mean, obs_log_std, priv_params

    # ------------------------------------------------------------------
    # Autoregressive training step (one batch)
    # ------------------------------------------------------------------
    def train_autoregressive(
        self,
        obs_seq: torch.Tensor,   # (B, M+N, obs_dim)  ground‑truth observations
        act_seq: torch.Tensor,   # (B, M+N, act_dim)  ground‑truth actions
        priv_seq: torch.Tensor,  # (B, M+N, priv_dim) ground‑truth privileged
    ) -> torch.Tensor:
        """
        Perform one autoregressive training step on a batch of length M+N.
        Returns the scalar loss.
        """
        M = self.history_len
        N = self.forecast_len

        # Normalise inputs (full sequence)
        obs_norm_full = self.normalize_obs(obs_seq)
        act_norm_full = self.normalize_act(act_seq)

        # Normalise the continuous part of privileged information in‑place
        priv_norm_full = priv_seq.clone()
        if self.num_cont_priv > 0:
            priv_cont = priv_norm_full[..., self.num_binary_priv:]
            priv_cont_norm = self.normalize_priv_cont(priv_cont)
            priv_norm_full[..., self.num_binary_priv:] = priv_cont_norm

        # Forward pass (autoregressive)
        obs_mean, obs_log_std, priv_params = self.forward(obs_norm_full, act_norm_full)

        # Slice forecast horizon (last N steps)
        obs_targets = obs_norm_full[:, M:]        # (B, N, obs_dim)
        priv_targets = priv_norm_full[:, M:]      # (B, N, priv_dim)
        obs_mean_fc = obs_mean[:, M:]
        obs_log_std_fc = obs_log_std[:, M:]
        priv_params_fc = priv_params[:, M:]

        # Compute loss (forecast decay α is 1.0 in the configuration, so no temporal weighting)
        loss = self.compute_loss(
            obs_targets, priv_targets, obs_mean_fc, obs_log_std_fc, priv_params_fc
        )
        return loss

    # ------------------------------------------------------------------
    # Pretraining loop
    # ------------------------------------------------------------------
    def pretrain(
        self,
        buffer: "TrajectoryBuffer",
        batch_size: int = 1024,
        max_iterations: int = 2500,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        log_every: int = 100,
        device: str = "cuda",
    ) -> None:
        """
        Pretrain the RWM using data from a `TrajectoryBuffer`.

        This method:
          1. Computes normalisation statistics from the buffer.
          2. Runs `max_iterations` of autoregressive training.

        Args:
            buffer:          TrajectoryBuffer containing pretraining data.
            batch_size:      Mini‑batch size (Table S10: 1024).
            max_iterations:  Number of training iterations (Table S10: 2500).
            learning_rate:   Adam learning rate (Table S10: 1e-4).
            weight_decay:    Adam weight decay (Table S10: 1e-5).
            log_every:       Print loss every `log_every` iterations.
            device:          Torch device to use.
        """
        # 1. Compute and set normalisation statistics
        self.compute_statistics(buffer)

        # 2. Move model to device
        self.to(device)

        # 3. Optimiser
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.train()
        print(f"Starting pretraining for {max_iterations} iterations...")
        for it in range(1, max_iterations + 1):
            # Sample a batch of windows (M+N length)
            batch = buffer.sample_batch(
                batch_size=batch_size,
                history_len=self.history_len,
                forecast_len=self.forecast_len,
            )
            obs_seq = batch["obs_seq"].to(device)
            act_seq = batch["act_seq"].to(device)
            priv_seq = batch["priv_seq"].to(device)

            # Compute autoregressive loss
            loss = self.train_autoregressive(obs_seq, act_seq, priv_seq)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
            optimizer.step()

            if it % log_every == 0:
                print(f"  iter {it:5d}/{max_iterations}  loss: {loss.item():.6f}")

        print("Pretraining finished.")

    # ------------------------------------------------------------------
    # Online fine‑tuning step (used in MBPO‑PPO, Algorithm 1, line 4)
    # ------------------------------------------------------------------
    def finetune_online(
        self,
        buffer: "TrajectoryBuffer",
        batch_size: int = 1024,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        device: str = "cuda",
    ) -> float:
        """
        Perform a **single** gradient‑step update on a mini‑batch sampled from the
        real‑experience replay buffer (capacity limited).

        Returns the loss value for logging.
        """
        self.train()
        self.to(device)

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        batch = buffer.sample_batch(
            batch_size=batch_size,
            history_len=self.history_len,
            forecast_len=self.forecast_len,
        )
        obs_seq = batch["obs_seq"].to(device)
        act_seq = batch["act_seq"].to(device)
        priv_seq = batch["priv_seq"].to(device)

        loss = self.train_autoregressive(obs_seq, act_seq, priv_seq)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
        optimizer.step()

        return loss.item()

    # ------------------------------------------------------------------
    # Autoregressive rollout for imagination (denormalised outputs)
    # ------------------------------------------------------------------
    def rollout_autoregressive(
        self,
        init_obs: torch.Tensor,          # (B, M, obs_dim)  initial history observations
        init_act: torch.Tensor,          # (B, M, act_dim)  corresponding actions
        policy_actions: torch.Tensor,    # (B, T, act_dim)  actions for the imagined rollout
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform an autoregressive rollout of length T, using a provided action
        sequence (from a policy).  Returns denormalised outputs suitable for
        reward computation.

        Args:
            init_obs:        History of observations (M steps).
            init_act:        History of actions (M steps).
            policy_actions:  Future actions of length T (from the policy).

        Returns:
            obs_denorm:   (B, T, obs_dim)  denormalised predicted observations.
            priv_denorm:  (B, T, priv_dim) continuous part denormalised, binary raw.
            obs_mean:     (B, T, obs_dim)  predicted mean (normalised).
            obs_log_std:  (B, T, obs_dim)  predicted log‑std (normalised).
        """
        B, M, _ = init_obs.shape
        T = policy_actions.shape[1]

        # Build full action and dummy observation sequences
        act_seq = torch.cat([init_act, policy_actions], dim=1)   # (B, M+T, act_dim)
        # The observations beyond M are never used; fill with zeros.
        obs_seq = torch.cat(
            [init_obs,
             torch.zeros(B, T, self.obs_dim, device=init_obs.device, dtype=init_obs.dtype)],
            dim=1,
        )                                                         # (B, M+T, obs_dim)

        # Autoregressive forward pass
        obs_mean_full, obs_log_std_full, priv_full = self.forward(obs_seq, act_seq)

        # Extract forecast predictions
        obs_mean = obs_mean_full[:, M:]          # (B, T, obs_dim)
        obs_log_std = obs_log_std_full[:, M:]
        priv = priv_full[:, M:]                  # (B, T, priv_dim)

        # Denormalise observations
        obs_denorm = self.denormalize_obs(obs_mean)

        # Denormalise continuous privileged part if any
        if self.num_cont_priv > 0:
            priv_cont = priv[..., self.num_binary_priv:]                  # continuous slice
            priv_cont_denorm = self.denormalize_priv_cont(priv_cont)
            priv_denorm = priv.clone()
            priv_denorm[..., self.num_binary_priv:] = priv_cont_denorm
        else:
            priv_denorm = priv

        return obs_denorm, priv_denorm, obs_mean, obs_log_std
