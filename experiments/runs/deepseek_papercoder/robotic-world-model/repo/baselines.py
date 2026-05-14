"""
baselines.py

Implements three baseline dynamics models used in the comparative
prediction‑error experiments of the RWM paper:

- MLPBaseline      (flat MLP, Table S8)
- RSSMBaseline     (Recurrent State‑Space Model, Dreamer‑style, Table S8)
- TransformerBaseline (causal transformer decoder, Table S8)

All models inherit from the `WorldModel` base class defined in `world_model.py`
so that they share the same interface and can be evaluated uniformly.  They are
trained with **teacher forcing** (forecast horizon N=1) and are intended for
off‑line accuracy comparisons; they are **not** used in the online MBPO‑PPO
training pipeline.

Each class provides:
- `forward(obs_seq, act_seq, h0=None) -> (obs_mean, obs_log_std, priv_params)`
  where the returned tensors have shape (batch, 1, …), i.e. the prediction for
  the **next** time step after the given history.
- `compute_loss(…)` is inherited from `WorldModel` and works with 1‑step outputs.
- `train_teacher_forcing(buffer, …)` runs the teacher‑forcing training loop
  using sliding windows of length history_len+1 drawn from a `TrajectoryBuffer`.

Hyper‑parameters follow the paper’s Table S8 and the `config.yaml` where applicable.
"""

from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# Use the base class and the helper to split privileged dimensions from world_model.
# world_model.py must be in the same package.
from world_model import WorldModel, get_priv_split


# ------------------------------------------------------------------------------
# MLPBaseline
# ------------------------------------------------------------------------------
class MLPBaseline(WorldModel):
    """
    Flat multi‑layer perceptron baseline.

    Architecture:
      Input: flattened concatenation of M history (obs, act) pairs.
      Hidden: two fully‑connected layers of 256 units, ReLU activation.
      Output: two linear heads → (obs_mean, obs_log_std) and priv values.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        priv_dim: int,
        history_len: int = 32,
        hidden_dims: Tuple[int, ...] = (256, 256),
        activation: str = "relu",
        std_min: float = 1e-4,
        **kwargs
    ):
        # Determine the split of privileged dimensions into binary / continuous.
        # This is needed by the parent WorldModel for loss computation.
        # Use the robot name passed via a keyword or default to 'anymal_d'.
        robot = kwargs.pop("robot", "anymal_d")
        num_binary_priv, num_cont_priv = get_priv_split(robot)

        # Call the base WorldModel constructor.  It initialises normalisation buffers
        # and the heads that we will replace.  The forecast_len=1 reflects teacher forcing.
        super().__init__(
            obs_dim=obs_dim,
            act_dim=act_dim,
            priv_dim=priv_dim,
            num_binary_priv=num_binary_priv,
            num_cont_priv=num_cont_priv,
            history_len=history_len,
            forecast_len=1,          # never used in this baseline
            gru_hidden_size=[256],   # dummy; not used
            head_hidden_size=[128],  # dummy
            activation=activation,
            std_min=std_min,
        )

        self.history_len = history_len
        self.std_min = std_min

        # Build the MLP body.
        act_fn = nn.ReLU if activation == "relu" else nn.ELU
        layers = []
        in_features = history_len * (obs_dim + act_dim)
        for h in hidden_dims:
            layers.append(nn.Linear(in_features, h))
            layers.append(act_fn())
            in_features = h
        self.mlp = nn.Sequential(*layers)

        # Replace heads: final linear layers from the last hidden size to the required
        # output dimensions.  The parent had already created self.obs_head and self.priv_head;
        # we overwrite them with new modules that match our flattened architecture.
        hidden_size = hidden_dims[-1]
        self.obs_head = nn.Linear(hidden_size, obs_dim * 2)
        self.priv_head = nn.Linear(hidden_size, priv_dim)

    def forward(
        self,
        obs_seq: torch.Tensor,       # (B, M, obs_dim)
        act_seq: torch.Tensor,       # (B, M, act_dim)
        h0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return the distribution parameters for the **next** step only.
        Output tensors have shape (B, 1, *) to be compatible with the base
        WorldModel interface that expects a time dimension.
        """
        B, M, _ = obs_seq.shape

        # Flatten the sequence of observation‑action pairs.
        x = torch.cat([obs_seq, act_seq], dim=-1)          # (B, M, obs_dim+act_dim)
        x = x.reshape(B, -1)                               # (B, M * in_dim)

        # MLP forward
        h = self.mlp(x)                                    # (B, hidden_size)

        # Heads
        obs_params = self.obs_head(h)                      # (B, obs_dim*2)
        obs_mean, obs_log_std = torch.chunk(obs_params, 2, dim=-1)
        obs_log_std = torch.clamp(obs_log_std, min=-20.0, max=2.0)

        priv_params = self.priv_head(h)                    # (B, priv_dim)

        # Unsqueeze time dimension (B, 1, ...)
        return (
            obs_mean.unsqueeze(1),
            obs_log_std.unsqueeze(1),
            priv_params.unsqueeze(1),
        )

    def train_teacher_forcing(
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
        Run teacher‑forcing training on data from the given TrajectoryBuffer.
        """
        self.to(device)
        optimizer = torch.optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.train()
        print(f"MLPBaseline: starting teacher‑forcing training for {max_iterations} iters")

        for it in range(1, max_iterations + 1):
            # Sample a batch of sequences of length M+1 (history + target)
            batch = buffer.sample_batch(batch_size, self.history_len, 1)
            obs_seq = batch["obs_seq"].to(device)      # (B, M+1, obs_dim)
            act_seq = batch["act_seq"].to(device)      # (B, M+1, act_dim)
            priv_seq = batch["priv_seq"].to(device)    # (B, M+1, priv_dim)

            obs_history = obs_seq[:, : self.history_len]        # (B, M, obs_dim)
            act_history = act_seq[:, : self.history_len]
            obs_target = obs_seq[:, self.history_len]            # (B, obs_dim)
            priv_target = priv_seq[:, self.history_len]          # (B, priv_dim)

            # Normalise inputs (the model has statistics set from pretraining data)
            obs_history_norm = self.normalize_obs(obs_history)
            act_history_norm = self.normalize_act(act_history)
            obs_target_norm = self.normalize_obs(obs_target)

            # Normalise the continuous part of the privileged target
            priv_target_norm = priv_target.clone()
            if self.num_cont_priv > 0:
                priv_cont = priv_target_norm[..., self.num_binary_priv:]
                priv_cont_norm = self.normalize_priv_cont(priv_cont)
                priv_target_norm[..., self.num_binary_priv:] = priv_cont_norm

            # Forward pass (teacher forcing)
            obs_mean, obs_log_std, priv_params = self.forward(
                obs_history_norm, act_history_norm
            )

            # Compute loss (expects target and prediction with time dimension 1)
            loss = self.compute_loss(
                obs_target_norm.unsqueeze(1),
                priv_target_norm.unsqueeze(1),
                obs_mean,
                obs_log_std,
                priv_params,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
            optimizer.step()

            if it % log_every == 0:
                print(f"  iter {it:5d}/{max_iterations}  loss: {loss.item():.6f}")

        print("MLPBaseline training finished.")


# ------------------------------------------------------------------------------
# RSSMBaseline
# ------------------------------------------------------------------------------
class RSSMBodel(nn.Module):
    """
    Recurrent State‑Space Model (RSSM) as used in Dreamer‑style world models.

    This internal helper implements the deterministic (GRU) and stochastic
    (discrete categorical) states, along with the prior, posterior, and decoder.
    It is used inside `RSSMBaseline`.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        priv_dim: int,
        rssm_hidden: int = 256,
        rssm_layers: int = 2,
        latent_dim: int = 32,
        categories: int = 64,
        std_min: float = 1e-4,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.priv_dim = priv_dim
        self.latent_dim = latent_dim
        self.categories = categories
        self.std_min = std_min

        # Encoder for observations → features for posterior
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, rssm_hidden),
            nn.ReLU(),
            nn.Linear(rssm_hidden, rssm_hidden),
            nn.ReLU(),
        )

        # GRU for deterministic state: input = [z_proj, action]
        z_proj_dim = latent_dim * categories   # one‑hot → projected
        self.z_proj = nn.Linear(z_proj_dim, rssm_hidden)
        gru_input_dim = rssm_hidden + act_dim
        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=rssm_hidden,
            num_layers=rssm_layers,
            batch_first=False,  # we will iterate over time manually
        )

        # Prior: from deterministic state to logits of categorical distributions
        self.prior_net = nn.Linear(rssm_hidden, latent_dim * categories)

        # Posterior: from [deterministic state, encoded observation] to logits
        self.posterior_net = nn.Linear(rssm_hidden + rssm_hidden, latent_dim * categories)

        # Decoder: from [deterministic state, latent projection] to observation/priv
        dec_in_dim = rssm_hidden + z_proj_dim
        self.decoder_obs = nn.Sequential(
            nn.Linear(dec_in_dim, rssm_hidden),
            nn.ReLU(),
            nn.Linear(rssm_hidden, obs_dim * 2),   # mean, log_std
        )
        self.decoder_priv = nn.Sequential(
            nn.Linear(dec_in_dim, rssm_hidden),
            nn.ReLU(),
            nn.Linear(rssm_hidden, priv_dim),
        )

    def _sample_z(self, logits: torch.Tensor, straight_through: bool = True) -> torch.Tensor:
        """
        Sample discrete latent from categorical(logits) and optionally apply
        straight‑through gradient estimator.
        Returns:
            sample:      (batch, latent_dim, categories) one‑hot
            sample_proj: (batch, z_proj_dim) projected vector
            probs:       softmax probabilities (for KL calculation)
        """
        B = logits.shape[0]
        logits = logits.view(B, self.latent_dim, self.categories)
        probs = F.softmax(logits, dim=-1)
        # Gumbel‑softmax sampling (straight‑through)
        sample_onehot = F.gumbel_softmax(logits, tau=1.0, hard=straight_through, dim=-1)
        # Flatten for projection
        proj_input = sample_onehot.view(B, -1)           # (B, z_proj_dim)
        proj = self.z_proj(proj_input)                   # (B, rssm_hidden)
        return sample_onehot, proj, probs

    def forward(
        self,
        obs_seq: torch.Tensor,         # (B, M, obs_dim)  history observations
        act_seq: torch.Tensor,         # (B, M, act_dim)
        h0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Unroll the RSSM for the given history and return the prediction for the
        **next** step, using the prior (no additional observation).

        Returns:
            obs_mean:    (B, 1, obs_dim)
            obs_log_std: (B, 1, obs_dim)
            priv_out:    (B, 1, priv_dim)
        """
        B, M, _ = obs_seq.shape
        device = obs_seq.device

        # Initial deterministic state
        if h0 is None:
            h = torch.zeros(self.gru.num_layers, B, self.gru.hidden_size, device=device)
        else:
            h = h0

        # Initial stochastic state (zero)
        z_proj = torch.zeros(B, self.latent_dim * self.categories, device=device)
        z_proj = self.z_proj(z_proj)                     # (B, rssm_hidden)

        # Iterate over time steps
        for t in range(M):
            # GRU input: concatenate latent projection and action
            gru_input = torch.cat([z_proj, act_seq[:, t, :]], dim=-1).unsqueeze(0)  # (1, B, in)
            gru_out, h = self.gru(gru_input, h)          # gru_out: (1, B, hidden)
            h_det = gru_out.squeeze(0)                   # (B, hidden)

            # Posterior (only if we have an observation, i.e., all history steps)
            if t < M:
                obs_enc = self.encoder(obs_seq[:, t, :])       # (B, rssm_hidden)
                post_input = torch.cat([h_det, obs_enc], dim=-1)
                post_logits = self.posterior_net(post_input)
                # Sample latent from posterior
                _, z_proj, _ = self._sample_z(post_logits, straight_through=True)
            else:
                # For the final step (target), we use the prior
                break

        # After processing all history, compute prior for the next step
        # h_det is the deterministic state after the last step
        prior_logits = self.prior_net(h_det)                 # (B, latent_dim*categories)
        # Sample from prior (straight‑through during training, but here we use soft for stability)
        sample_onehot, z_proj_next, prior_probs = self._sample_z(prior_logits, straight_through=False)

        # Decoder input: concatenate h_det and sampled latent projection
        dec_input = torch.cat([h_det, sample_onehot.view(B, -1)], dim=-1)
        obs_params = self.decoder_obs(dec_input)             # (B, obs_dim*2)
        obs_mean, obs_log_std = torch.chunk(obs_params, 2, dim=-1)
        obs_log_std = torch.clamp(obs_log_std, min=-20.0, max=2.0)
        priv_output = self.decoder_priv(dec_input)

        # Return with time dimension 1
        return (
            obs_mean.unsqueeze(1),
            obs_log_std.unsqueeze(1),
            priv_output.unsqueeze(1),
        )


class RSSMBaseline(WorldModel):
    """
    Recurrent State‑Space Model baseline (Dreamer‑style).

    Implements teacher‑forcing training with the KL loss between posterior and prior
    over the history, and a reconstruction loss for the next observation.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        priv_dim: int,
        history_len: int = 32,
        rssm_hidden: int = 256,
        rssm_layers: int = 2,
        latent_dim: int = 32,
        categories: int = 64,
        std_min: float = 1e-4,
        **kwargs
    ):
        robot = kwargs.pop("robot", "anymal_d")
        num_binary_priv, num_cont_priv = get_priv_split(robot)

        super().__init__(
            obs_dim=obs_dim,
            act_dim=act_dim,
            priv_dim=priv_dim,
            num_binary_priv=num_binary_priv,
            num_cont_priv=num_cont_priv,
            history_len=history_len,
            forecast_len=1,
            gru_hidden_size=[rssm_hidden],  # dummy
            head_hidden_size=[128],
            activation="relu",
            std_min=std_min,
        )

        self.rssm = RSSMBodel(
            obs_dim=obs_dim,
            act_dim=act_dim,
            priv_dim=priv_dim,
            rssm_hidden=rssm_hidden,
            rssm_layers=rssm_layers,
            latent_dim=latent_dim,
            categories=categories,
            std_min=std_min,
        )

    def forward(
        self,
        obs_seq: torch.Tensor,
        act_seq: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.rssm.forward(obs_seq, act_seq, h0)

    def train_teacher_forcing(
        self,
        buffer: "TrajectoryBuffer",
        batch_size: int = 1024,
        max_iterations: int = 2500,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        log_every: int = 100,
        device: str = "cuda",
    ) -> None:
        self.to(device)
        optimizer = torch.optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.train()
        print(f"RSSMBaseline: starting teacher‑forcing training for {max_iterations} iters")

        for it in range(1, max_iterations + 1):
            batch = buffer.sample_batch(batch_size, self.history_len, 1)
            obs_seq = batch["obs_seq"].to(device)
            act_seq = batch["act_seq"].to(device)
            priv_seq = batch["priv_seq"].to(device)

            obs_history = obs_seq[:, : self.history_len]
            act_history = act_seq[:, : self.history_len]
            obs_target = obs_seq[:, self.history_len]
            priv_target = priv_seq[:, self.history_len]

            # Normalise (use the model's buffers)
            obs_history_norm = self.normalize_obs(obs_history)
            act_history_norm = self.normalize_act(act_history)
            obs_target_norm = self.normalize_obs(obs_target)
            priv_target_norm = priv_target.clone()
            if self.num_cont_priv > 0:
                priv_cont = priv_target_norm[..., self.num_binary_priv:]
                priv_cont_norm = self.normalize_priv_cont(priv_cont)
                priv_target_norm[..., self.num_binary_priv:] = priv_cont_norm

            # ---- Manually unroll to compute RSSM losses ----
            B, M, _ = obs_history_norm.shape
            # Initial states
            h = torch.zeros(
                self.rssm.gru.num_layers, B, self.rssm.gru.hidden_size, device=device
            )
            z_proj = torch.zeros(B, self.rssm.latent_dim * self.rssm.categories, device=device)
            z_proj = self.rssm.z_proj(z_proj)       # projected

            kl_losses = []
            rec_losses = []

            for t in range(M + 1):
                # Determine whether we have a ground‑truth observation
                if t < M:
                    obs_in = obs_history_norm[:, t, :]
                else:
                    obs_in = None   # target step, no observation

                act_in = act_history_norm[:, t, :] if t < M else act_seq[:, self.history_len, :]

                # GRU step
                gru_input = torch.cat([z_proj, act_in], dim=-1).unsqueeze(0)   # (1, B, in)
                gru_out, h = self.rssm.gru(gru_input, h)
                h_det = gru_out.squeeze(0)                                     # (B, hidden)

                # Prior
                prior_logits = self.rssm.prior_net(h_det)                      # (B, lat*cats)
                prior_probs = F.softmax(prior_logits.view(B, self.rssm.latent_dim, self.rssm.categories), dim=-1)

                if obs_in is not None:   # history steps: use posterior
                    obs_enc = self.rssm.encoder(obs_in)
                    post_input = torch.cat([h_det, obs_enc], dim=-1)
                    post_logits = self.rssm.posterior_net(post_input)
                    # Sample from posterior (straight‑through)
                    z_onehot, z_proj_cur, post_probs = self.rssm._sample_z(post_logits)
                    # KL loss
                    kl = torch.sum(
                        post_probs * (torch.log(post_probs + 1e-8) - torch.log(prior_probs + 1e-8)), dim=-1
                    ).mean()
                    kl_losses.append(kl)

                    # Decoder reconstruction on the **current** observation (optional but nice)
                    # We only need reconstruction at the final target step; doing it here
                    # helps stabilise learning.
                    dec_input = torch.cat([h_det, z_onehot.view(B, -1)], dim=-1)
                    obs_params_rec = self.rssm.decoder_obs(dec_input)
                    obs_mean_rec, obs_log_std_rec = torch.chunk(obs_params_rec, 2, dim=-1)
                    obs_log_std_rec = torch.clamp(obs_log_std_rec, -20.0, 2.0)
                    dist = Normal(obs_mean_rec, torch.exp(obs_log_std_rec) + self.rssm.std_min)
                    rec_loss_obs = -dist.log_prob(obs_in).sum(dim=-1).mean()
                    # Privileged reconstruction (for this step) – we don't have ground truth priv
                    # for intermediate steps, so skip.
                    rec_losses.append(rec_loss_obs)

                    # Use posterior sample as latent for next step
                    z_proj = z_proj_cur
                else:   # target step: no posterior, use prior
                    # Sample from prior for the decoder prediction
                    z_onehot_target, z_proj_target, _ = self.rssm._sample_z(prior_logits, straight_through=False)
                    dec_input = torch.cat([h_det, z_onehot_target.view(B, -1)], dim=-1)
                    obs_params_pred = self.rssm.decoder_obs(dec_input)
                    obs_mean_pred, obs_log_std_pred = torch.chunk(obs_params_pred, 2, dim=-1)
                    obs_log_std_pred = torch.clamp(obs_log_std_pred, -20.0, 2.0)
                    dist = Normal(obs_mean_pred, torch.exp(obs_log_std_pred) + self.rssm.std_min)
                    rec_loss_target = -dist.log_prob(obs_target_norm).sum(dim=-1).mean()
                    priv_pred = self.rssm.decoder_priv(dec_input)

            # Total loss (mean of KL and reconstruction terms)
            kl_loss = torch.stack(kl_losses).mean() if kl_losses else torch.tensor(0.0, device=device)
            rec_loss_hist = torch.stack(rec_losses).mean() if rec_losses else torch.tensor(0.0, device=device)
            loss = rec_loss_hist + rec_loss_target + kl_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
            optimizer.step()

            if it % log_every == 0:
                print(f"  iter {it:5d}/{max_iterations}  loss: {loss.item():.6f}")

        print("RSSMBaseline training finished.")


# ------------------------------------------------------------------------------
# TransformerBaseline
# ------------------------------------------------------------------------------
class TransformerBaseline(WorldModel):
    """
    Causally‑masked Transformer decoder baseline.

    Architecture (Table S8):
      - Input: sequence of length M (obs+act), projected to dimension 64.
      - Positional encoding: sinusoidal.
      - Transformer decoder: 2 layers, 8 heads, feed‑forward dim 256.
      - Output: from the last token → linear heads for obs and priv.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        priv_dim: int,
        history_len: int = 32,
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        std_min: float = 1e-4,
        **kwargs
    ):
        robot = kwargs.pop("robot", "anymal_d")
        num_binary_priv, num_cont_priv = get_priv_split(robot)

        super().__init__(
            obs_dim=obs_dim,
            act_dim=act_dim,
            priv_dim=priv_dim,
            num_binary_priv=num_binary_priv,
            num_cont_priv=num_cont_priv,
            history_len=history_len,
            forecast_len=1,
            gru_hidden_size=[64],      # dummy
            head_hidden_size=[128],
            activation="relu",
            std_min=std_min,
        )

        self.history_len = history_len

        # Input projection and positional encoding
        self.input_proj = nn.Linear(obs_dim + act_dim, d_model)
        self.register_buffer(
            "pos_encoding",
            self._sinusoidal_pos_encoding(history_len, d_model),
        )

        # Transformer decoder stack
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=False,   # we feed (T, B, E)
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output heads (applied to the last token only)
        self.obs_head = nn.Linear(d_model, obs_dim * 2)
        self.priv_head = nn.Linear(d_model, priv_dim)

    @staticmethod
    def _sinusoidal_pos_encoding(length: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encodings of shape (length, d_model)."""
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(
        self,
        obs_seq: torch.Tensor,       # (B, M, obs_dim)
        act_seq: torch.Tensor,       # (B, M, act_dim)
        h0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, M, _ = obs_seq.shape

        # Concatenate and project
        x = torch.cat([obs_seq, act_seq], dim=-1)          # (B, M, in_dim)
        emb = self.input_proj(x)                           # (B, M, d_model)

        # Add positional encoding (M must equal history_len, which is guaranteed)
        emb = emb + self.pos_encoding[:M, :].unsqueeze(0)  # (B, M, d_model)

        # Permute to (T, B, E) for Transformer
        emb = emb.permute(1, 0, 2)                         # (M, B, d_model)

        # Causal mask (standard square‑subsequent)
        attn_mask = torch.triu(
            torch.ones(M, M, device=emb.device) * float("-inf"), diagonal=1
        )

        # Transformer decoder (no memory, so memory=None)
        out = self.transformer(emb, memory=None, tgt_mask=attn_mask)  # (M, B, d_model)

        # Use the last token output
        last_out = out[-1]                                 # (B, d_model)

        obs_params = self.obs_head(last_out)                # (B, obs_dim*2)
        obs_mean, obs_log_std = torch.chunk(obs_params, 2, dim=-1)
        obs_log_std = torch.clamp(obs_log_std, min=-20.0, max=2.0)
        priv_out = self.priv_head(last_out)

        # Add time dimension (B, 1, ...)
        return (
            obs_mean.unsqueeze(1),
            obs_log_std.unsqueeze(1),
            priv_out.unsqueeze(1),
        )

    def train_teacher_forcing(
        self,
        buffer: "TrajectoryBuffer",
        batch_size: int = 1024,
        max_iterations: int = 2500,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        log_every: int = 100,
        device: str = "cuda",
    ) -> None:
        """Standard teacher‑forcing training loop."""
        self.to(device)
        optimizer = torch.optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.train()
        print(f"TransformerBaseline: starting teacher‑forcing training for {max_iterations} iters")

        for it in range(1, max_iterations + 1):
            batch = buffer.sample_batch(batch_size, self.history_len, 1)
            obs_seq = batch["obs_seq"].to(device)
            act_seq = batch["act_seq"].to(device)
            priv_seq = batch["priv_seq"].to(device)

            obs_history = obs_seq[:, : self.history_len]
            act_history = act_seq[:, : self.history_len]
            obs_target = obs_seq[:, self.history_len]
            priv_target = priv_seq[:, self.history_len]

            # Normalise
            obs_history_norm = self.normalize_obs(obs_history)
            act_history_norm = self.normalize_act(act_history)
            obs_target_norm = self.normalize_obs(obs_target)
            priv_target_norm = priv_target.clone()
            if self.num_cont_priv > 0:
                priv_cont = priv_target_norm[..., self.num_binary_priv:]
                priv_cont_norm = self.normalize_priv_cont(priv_cont)
                priv_target_norm[..., self.num_binary_priv:] = priv_cont_norm

            # Forward
            obs_mean, obs_log_std, priv_params = self.forward(obs_history_norm, act_history_norm)

            # Loss
            loss = self.compute_loss(
                obs_target_norm.unsqueeze(1),
                priv_target_norm.unsqueeze(1),
                obs_mean,
                obs_log_std,
                priv_params,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
            optimizer.step()

            if it % log_every == 0:
                print(f"  iter {it:5d}/{max_iterations}  loss: {loss.item():.6f}")

        print("TransformerBaseline training finished.")
