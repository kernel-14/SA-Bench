"""
model.py

Implements the core GPT and nGPT model as a unified ``GPTModel`` class.

The class assembles:
- Input and output embedding tables (optionally tied).
- A stack of ``TransformerBlock`` layers (self‑attention + MLP).
- For the baseline GPT: a final RMSNorm and standard residual connections.
- For nGPT: a learnable vocabulary‑level logit scaling factor ``s_z`` and
  no final normalisation (hidden states remain on the unit sphere throughout).

All weight matrices are enforced to live on the hypersphere *via* the
``normalize_weights`` method, which is called by the trainer after every
optimizer step.  The scaling parameters (``s_z`` and the eigen learning rates
inside the blocks) use the “init / scale” trick described in Section 2.5.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from layers import TransformerBlock, RMSNorm
from utils import normalize_weights as _normalize_weights_global


# ---------------------------------------------------------------------------
# Helper: Scaled Parameter with init / scale trick
# ---------------------------------------------------------------------------

class ScaledParam(nn.Module):
    """
    A learnable rank‑1 tensor whose effective value is
        ``raw * (init_val / scale_val)``.

    Optionally, the absolute value of ``raw`` is taken before scaling, which is
    used for eigen learning rates to ensure positivity.

    Parameters
    ----------
    shape : tuple
        Shape of the underlying parameter tensor.
    init_val : float
        The desired initial effective value.
    scale_val : float
        The value to which ``raw`` is initialised; the division
        ``init_val / scale_val`` compensates so that the effective value
        starts at ``init_val``.
    positive : bool
        If True, take ``abs(raw)`` before scaling (for eigen learning rates).
    device, dtype
        Passed to the underlying ``torch.Tensor`` creation.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        init_val: float,
        scale_val: float,
        positive: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.shape = shape
        self.init_val = init_val
        self.scale_val = scale_val
        self.positive = positive

        # The raw parameter is initialised to the constant scale_val
        self.raw = nn.Parameter(
            torch.full(shape, scale_val, device=device, dtype=dtype)
        )

    def forward(self) -> torch.Tensor:
        """
        Returns the effective scaling vector.
        """
        x = self.raw
        if self.positive:
            x = torch.abs(x)
        return x * (self.init_val / self.scale_val)


# ---------------------------------------------------------------------------
# Main GPT / nGPT Model
# ---------------------------------------------------------------------------

class GPTModel(nn.Module):
    """
    Decoder‑only Transformer supporting both the baseline GPT and the
    Normalised Transformer (nGPT).

    The behaviour is controlled by ``config.model.use_ngpt``.  All architecture
    dimensions are taken from the ``Config`` object.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        vocab_size = config.model.vocab_size
        d_model = config.model.d_model
        max_seq_len = config.model.max_seq_len

        # -----------------------------------------------------------------
        # Embeddings
        # -----------------------------------------------------------------
        self.emb_in = nn.Embedding(vocab_size, d_model)
        if config.model.tie_embeddings:
            self.emb_out = None
        else:
            self.emb_out = nn.Embedding(vocab_size, d_model)

        # -----------------------------------------------------------------
        # Transformer blocks
        # -----------------------------------------------------------------
        self.blocks = nn.ModuleList(
            [TransformerBlock(config, i) for i in range(config.model.n_layers)]
        )

        # -----------------------------------------------------------------
        # nGPT‑specific components
        # -----------------------------------------------------------------
        if config.model.use_ngpt:
            # Logit scaling factor (vocabulary‑wise)
            self.s_z = ScaledParam(
                shape=(vocab_size,),
                init_val=config.ngpt.s_z_init,
                scale_val=config.ngpt.s_z_scale,
                positive=False,
            )
            self.final_norm = None
        else:
            # Baseline GPT: final layer norm
            self.final_norm = RMSNorm(d_model)
            self.s_z = None

        # -----------------------------------------------------------------
        # Causal mask (registered as buffer so it moves with the model)
        # -----------------------------------------------------------------
        mask = torch.triu(
            torch.ones(max_seq_len, max_seq_len) * float("-inf"), diagonal=1
        )
        self.register_buffer("causal_mask", mask, persistent=False)

        # -----------------------------------------------------------------
        # Weight initialisation
        # -----------------------------------------------------------------
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialize all linear and embedding weights according to the
        configuration.  The actual logic is delegated to ``utils.init_weights``
        but is applied here via a model‑level ``apply`` call.
        """
        from utils import init_weights as _init_weights_global

        def _apply_init(module):
            _init_weights_global(module, self.config)

        self.apply(_apply_init)

    def normalize_weights(self) -> None:
        """
        Project all weight matrices and embedding tables onto the unit
        hypersphere.  This should be called **after** every optimizer step
        when training the nGPT variant.
        """
        _normalize_weights_global(self)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the language model.

        Parameters
        ----------
        idx : torch.Tensor, shape (batch_size, seq_len)
            Token indices.
        targets : torch.Tensor, optional, shape (batch_size, seq_len)
            Target token indices for language modelling loss.  Padded positions
            should be filled with a value ignored by the loss (e.g., ‑100).

        Returns
        -------
        logits : torch.Tensor, shape (batch_size, seq_len, vocab_size)
        loss : Optional[torch.Tensor]
            Scalar cross‑entropy loss if ``targets`` is provided, else ``None``.
        """
        B, T = idx.shape
        config = self.config

        # Input embeddings
        x = self.emb_in(idx)                     # (B, T, d_model)

        # For the baseline GPT we follow the original Transformer by *not*
        # scaling the embeddings – the variance is controlled by the
        # combination of weight initialisation and normalisation.

        # Causal mask (use only the required sequence length)
        mask = self.causal_mask[:T, :T]

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # Output logits
        emb_weight = (
            self.emb_out.weight
            if not config.model.tie_embeddings
            else self.emb_in.weight
        )

        if config.model.use_ngpt:
            # Both x and the embedding rows are unit norm → cosine sim in [‑1,1]
            logits = F.linear(x, emb_weight)     # (B, T, vocab_size)
            logits = logits * self.s_z()         # element‑wise scaling (vocab_size)
        else:
            x = self.final_norm(x)
            logits = F.linear(x, emb_weight)

        # Compute loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, config.model.vocab_size),
                targets.view(-1),
                ignore_index=-100,  # standard padding index
            )

        return logits, loss

    @property
    def dtype(self) -> torch.dtype:
        """Convenience property to obtain the parameter dtype."""
        return next(self.parameters()).dtype
