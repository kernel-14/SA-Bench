```python
## model/moe_layer.py
"""Mixture-of-Experts layer implementation for OLMoE.

Implements the three core components of the MoE layer:
  1. SwiGLUExpert: Individual expert FFN with SwiGLU activation
  2. MoERouter: Learned linear router mapping tokens to expert distributions
  3. MoELayer: Orchestrates dropless token-choice routing and expert dispatch

Architecture details from the paper (Section 2, Table 10, Section 4.1):
  - 64 total experts per layer (fine-grained, Section 4.1.2)
  - 8 activated experts per token (top-k routing, Section 2)
  - Dropless token-choice routing via MegaBlocks or scatter/gather fallback
  - SwiGLU activation: FFN(x) = w2(SiLU(w_gate(x)) ⊙ w1(x))
  - No biases in any linear layers (Table 10: "Biases: -")
  - FFN dimension per expert: 1024 (vs 8192 for equivalent dense model)

MoE module equation (Equation 1 from paper):
  MoE(x) = Σ_{i ∈ Top-k(r(x))} softmax(r(x))_i · E_i(x)

where r is the router linear layer, softmax is applied over all 64 experts,
and Top-k selects the 8 experts with highest routing probability.

Configuration values used (from config.yaml):
  model.hidden_dim: 2048
  model.ffn_dim: 1024
  model.num_experts: 64
  model.top_k: 8
  model.use_bias: false
  model.use_megablocks: true
"""

import logging
import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from config import OLMoEConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional MegaBlocks import for dropless sparse MoE operations.
# MegaBlocks uses block-sparse CUDA kernels for efficient expert dispatch,
# avoiding the Python loop over 64 experts in the fallback implementation.
# Reference: Gale et al. (2022). "MegaBlocks: Efficient Sparse Training with
# Mixture-of-Experts." https://arxiv.org/abs/2211.15841
# ---------------------------------------------------------------------------
try:
    import megablocks  # noqa: F401
    MEGABLOCKS_AVAILABLE: bool = True
    logger.info("MegaBlocks is available. Dropless sparse MoE ops enabled.")
except ImportError:
    MEGABLOCKS_AVAILABLE = False
    logger.info(
        "MegaBlocks not available. Falling back to scatter/gather MoE dispatch. "
        "Install megablocks for better throughput: pip install megablocks"
    )


class SwiGLUExpert(nn.Module):
    """Single expert FFN with SwiGLU activation for OLMoE.

    Each expert is a small feed-forward network using the SwiGLU activation
    function (Shazeer 2020, "GLU Variants Improve Transformer"). With
    hidden_dim=2048 and ffn_dim=1024, each expert has approximately
    3 × 2048 × 1024 = 6.3M parameters.

    SwiGLU formulation:
        output = w2( SiLU(w_gate(x)) ⊙ w1(x) )

    where:
        - w_gate: Linear(hidden_dim, ffn_dim) — gate projection, passed through SiLU
        - w1: Linear(hidden_dim, ffn_dim) — value projection
        - w2: Linear(ffn_dim, hidden_dim) — output projection
        - ⊙: element-wise multiplication

    The input x is a variable-length tensor of tokens assigned to this expert.
    Different experts receive different numbers of tokens per forward pass
    (dropless routing: no capacity constraints, no token dropping).

    All linear layers use bias=False per Table 10 ("Biases: -").

    Attributes:
        hidden_dim: Input/output dimension (2048 for OLMoE-1B-7B).
        ffn_dim: Intermediate FFN dimension (1024 for OLMoE-1B-7B).
        w1: Value projection, Linear(hidden_dim, ffn_dim, bias=False).
        w_gate: Gate projection, Linear(hidden_dim, ffn_dim, bias=False).
        w2: Output projection, Linear(ffn_dim, hidden_dim, bias=False).

    Example:
        >>> expert = SwiGLUExpert(hidden_dim=2048, ffn_dim=1024)
        >>> x = torch.randn(42, 2048)  # 42 tokens assigned to this expert
        >>> out = expert(x)
        >>> out.shape
        torch.Size([42, 2048])
    """

    def __init__(self, hidden_dim: int = 2048, ffn_dim: int = 1024) -> None:
        """Initialize SwiGLUExpert.

        Args:
            hidden_dim: Input and output dimension of the expert.
                        For OLMoE-1B-7B: 2048 (from config.yaml: model.hidden_dim).
            ffn_dim: Intermediate FFN dimension.
                     For OLMoE-1B-7B: 1024 (from config.yaml: model.ffn_dim).
                     This is much smaller than the dense equivalent (8192) because
                     there are 64 experts, each processing only 1/8 of tokens.

        Raises:
            ValueError: If hidden_dim or ffn_dim are not positive integers.
        """
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        if ffn_dim <= 0:
            raise ValueError(f"ffn_dim must be positive, got {ffn_dim}.")

        self.hidden_dim: int = hidden_dim
        self.ffn_dim: int = ffn_dim

        # Value projection: maps hidden_dim -> ffn_dim
        # Produces the "value" signal that is gated by w_gate output.
        self.w1: nn.Linear = nn.Linear(hidden_dim, ffn_dim, bias=False)

        # Gate projection: maps hidden_dim -> ffn_dim
        # Produces the gating signal passed through SiLU activation.
        self.w_gate: nn.Linear = nn.Linear(hidden_dim, ffn_dim, bias=False)

        # Output projection: maps ffn_dim -> hidden_dim
        # Projects the gated intermediate representation back to hidden_dim.
        self.w2: nn.Linear = nn.Linear(ffn_dim, hidden_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Apply SwiGLU FFN to the input tokens.

        Computes: output = w2( SiLU(w_gate(x)) ⊙ w1(x) )

        The input is a variable-length batch of tokens assigned to this expert
        by the router. The number of tokens varies per forward pass depending
        on routing decisions.

        Args:
            x: Input tensor of shape (num_tokens, hidden_dim) where num_tokens
               is the number of tokens routed to this expert in the current batch.
               May be 0 (empty) if no tokens are assigned — callers should guard
               against this case before calling forward().

        Returns:
            Output tensor of shape (num_tokens, hidden_dim), same shape as input.

        Shape example:
            Input:  (42, 2048)  — 42 tokens assigned to this expert
            Output: (42, 2048)
        """
        # Gate projection: (num_tokens, hidden_dim) -> (num_tokens, ffn_dim)
        gate: Tensor = self.w_gate(x)

        # Value projection: (num_tokens, hidden_dim) -> (num_tokens, ffn_dim)
        value: Tensor = self.w1(x)

        # SwiGLU: element-wise product of SiLU(gate) and value
        # SiLU(x) = x * sigmoid(x) — smooth gating function
        # Shape: (num_tokens, ffn_dim)
        activated: Tensor = F.silu(gate) * value

        # Output projection: (num_tokens, ffn_dim) -> (num_tokens, hidden_dim)
        output: Tensor = self.w2(activated)

        return output

    def extra_repr(self) -> str:
        """Return extra representation string for printing the module."""
        return f"hidden_dim={self.hidden_dim}, ffn_dim={self.ffn_dim}"


class MoERouter(nn.Module):
    """Learned linear router for token-to-expert assignment in OLMoE.

    The router maps each token's hidden state to a distribution over experts
    using a single linear layer followed by softmax. The top-k experts with
    highest routing probability are selected for each token.

    Router equation (from paper Section 2):
        logits = linear(x)                    # [num_tokens, num_experts]
        softmax_weights = softmax(logits)     # [num_tokens, num_experts]
        top_k_indices = argtopk(softmax_weights, k)  # [num_tokens, top_k]

    The router returns three tensors:
        1. logits: Pre-softmax router outputs — used by router z-loss
           (AuxiliaryLosses.router_z_loss computes (log Σ exp(logits))²)
        2. softmax_weights: Full softmax probabilities over all experts —
           used by load balancing loss (P_i = mean routing prob for expert i)
        3. top_k_indices: Selected expert indices — used for dispatch and
           load balancing loss (f_i = fraction of tokens routed to expert i)

    No bias in the linear layer per Table 10 ("Biases: -").

    Attributes:
        num_experts: Total number of experts (64 for OLMoE-1B-7B).
        top_k: Number of experts activated per token (8 for OLMoE-1B-7B).
        linear: Router linear layer, Linear(hidden_dim, num_experts, bias=False).

    Example:
        >>> router = MoERouter(hidden_dim=2048, num_experts=64, top_k=8)
        >>> x = torch.randn(4096, 2048)  # 4096 tokens (batch*seq)
        >>> logits, weights, indices = router(x)
        >>> logits.shape, weights.shape, indices.shape
        (torch.Size([4096, 64]), torch.Size([4096, 64]), torch.Size([4096, 8]))
    """

    def __init__(
        self,
        hidden_dim: int = 2048,
        num_experts: int = 64,
        top_k: int = 8,
    ) -> None:
        """Initialize MoERouter.

        Args:
            hidden_dim: Input dimension (token hidden state size).
                        For OLMoE-1B-7B: 2048 (from config.yaml: model.hidden_dim).
            num_experts: Total number of experts to route to.
                         For OLMoE-1B-7B: 64 (from config.yaml: model.num_experts).
            top_k: Number of experts to activate per token.
                   For OLMoE-1B-7B: 8 (from config.yaml: model.top_k).

        Raises:
            ValueError: If hidden_dim, num_experts, or top_k are invalid.
        """
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        if num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {num_experts}.")
        if top_k <= 0 or top_k > num_experts:
            raise ValueError(
                f"top_k must be in [1, num_experts], got top_k={top_k}, "
                f"num_experts={num_experts}."
            )

        self.num_experts: int = num_experts
        self.top_k: int = top_k

        # Router linear layer: maps hidden_dim -> num_experts
        # No bias per Table 10 ("Biases: -")
        # Initialized by OLMoEModel._init_weights() with truncated normal
        self.linear: nn.Linear = nn.Linear(hidden_dim, num_experts, bias=False)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute routing decisions for all tokens.

        Maps each token's hidden state to a distribution over experts and
        selects the top-k experts with highest routing probability.

        The softmax is applied over ALL num_experts=64 experts before top-k
        selection. This means routing weights for selected experts are
        normalized over all 64 experts, not just the top-k. This matches
        Equation 1 in the paper and the original Shazeer et al. [154]
        formulation.

        Args:
            x: Flattened token tensor of shape (num_tokens, hidden_dim)
               where num_tokens = batch_size × seq_len.
               For OLMoE-1B-7B: (B*S, 2048) where B*S can be up to 1024*4096.

        Returns:
            Tuple of three tensors:
                - logits: Pre-softmax router outputs, shape (num_tokens, num_experts).
                  Used by AuxiliaryLosses.router_z_loss() to compute
                  (log Σ_j exp(x_j))² per token.
                - softmax_weights: Full softmax probabilities over all experts,
                  shape (num_tokens, num_experts). Used by
                  AuxiliaryLosses.load_balancing_loss() for P_i computation
                  and by _dispatch_tokens() for scaling expert outputs.
                - top_k_indices: Selected expert indices, shape (num_tokens, top_k),
                  dtype=torch.long. Used by _dispatch_tokens() for routing and
                  by AuxiliaryLosses.load_balancing_loss() for f_i computation.

        Shape example:
            Input:  (4096, 2048)
            Output: logits (4096, 64), weights (4096, 64), indices (4096, 8)
        """
        # Step 1: Compute router logits (pre-softmax)
        # Shape: (num_tokens, num_experts) = (B*S, 64)
        logits: Tensor = self.linear(x)

        # Step 2: Apply softmax over all experts to get routing probabilities
        # Shape: (num_tokens, num_experts) = (B*S, 64)
        # Computed in float32 for numerical stability, then cast back.
        # The softmax is over dim=-1 (the expert dimension).
        softmax_weights: Tensor = torch.softmax(logits.float(), dim=-1).to(x.dtype)

        # Step 3: Select top-k experts per token
        # top_k_indices: (num_tokens, top_k) = (B*S, 8), dtype=torch.long
        # We select based on softmax_weights (post-softmax probabilities),
        # which is equivalent to selecting based on logits since softmax is
        # monotone. Using softmax_weights ensures consistency with the
        # routing weights used for expert output scaling.
        _, top_k_indices = torch.topk(
            softmax_weights,
            k=self.top_k,
            dim=-1,
            largest=True,
            sorted=False,  # Order within top-k doesn't matter for dispatch
        )
        # top_k_indices: (num_tokens, top_k) = (B*S, 8)

        return logits, softmax_weights, top_k_indices

    def extra_repr(self) -> str:
        """Return extra representation string for printing the module."""
        return f"num_experts={self.num_experts}, top_k={self.top_k}"


class MoELayer(nn.Module):
    """Mixture-of-Experts layer with dropless token-choice routing.

    Replaces the FFN in each transformer block with a sparse MoE module.
    Each input token is routed to top_k=8 out of num_experts=64 experts,
    and the outputs are aggregated with routing probability weights.

    MoE module equation (Equation 1 from paper):
        MoE(x) = Σ_{i ∈ Top-k(r(x))} softmax(r(x))_i · E_i(x)

    Key design choices (Table 1, Section 4.1):
        - Token choice routing: each token selects its top-k experts
        - Dropless: no token dropping, no capacity constraints
        - 64 fine-grained experts with FFN dim 1024 (Section 4.1.2)
        - Load balancing loss + router z-loss for training stability

    Two dispatch implementations:
        1. MegaBlocks (preferred): Block-sparse CUDA kernels for efficient
           parallel expert computation. Requires megablocks package.
        2. Scatter/gather fallback: Python loop over experts with boolean
           indexing. Functionally correct but slower.

    Attributes:
        router: MoERouter for token-to-expert assignment.
        experts: nn.ModuleList of num_experts SwiGLUExpert instances.
        num_experts: Total experts per layer (64 for OLMoE-1B-7B).
        top_k: Activated experts per token (8 for OLMoE-1B-7B).
        hidden_dim: Token hidden dimension (2048 for OLMoE-1B-7B).
        ffn_dim: Per-expert FFN dimension (1024 for OLMoE-1B-7B).
        use_megablocks: Whether to attempt MegaBlocks dispatch.

    Example:
        >>> config = OLMoEConfig()
        >>> moe = MoELayer(config)
        >>> x = torch.randn(2, 4096, 2048)  # (batch, seq, hidden)
        >>> output, logits, indices = moe(x)
        >>> output.shape, logits.shape, indices.shape
        (torch.Size([2, 4096, 2048]),
         torch.Size([8192, 64]),
         torch.Size([8192, 8]))
    """

    def __init__(self, config: OLMoEConfig) -> None:
        """Initialize MoELayer.

        Creates the router and all expert FFNs. The number of experts (64)
        and their FFN dimension (1024) are set by the config, which encodes
        the fine-grained expert design choice from Section 4.1.2.

        Args:
            config: OLMoEConfig instance. Key fields used:
                    - hidden_dim (2048): token hidden dimension
                    - ffn_dim (1024): per-expert FFN intermediate dimension
                    - num_experts (64): total experts per layer
                    - top_k (8): activated experts per token
                    - use_bias (False): no biases in linear layers
                    - use_megablocks (True): whether to use MegaBlocks dispatch
        """
        super().__init__()

        self.num_experts: int = config.num_experts
        self.top_k: int = config.top_k
        self.hidden_dim: int = config.hidden_dim
        self.ffn_dim: int = config.ffn_dim
        self.use_megablocks: bool = config.use_megablocks

        # Router: maps each token to a distribution over experts
        self.router: MoERouter = MoERouter(
            hidden_dim=config.hidden_dim,
            num_experts=config.num_experts,
            top_k=config.top_k,
        )

        # Expert FFNs: 64 small SwiGLU networks
        # Each expert has ~6.3M parameters (3 × 2048 × 1024)
        # Total expert parameters per layer: 64 × 6.3M ≈ 402M
        # Total across 16 layers: ≈ 6.4B (bulk of the 6.9B total params)
        self.experts: nn.ModuleList = nn.ModuleList([
            SwiGLUExpert(
                hidden_dim=config.hidden_dim,
                ffn_dim=config.ffn_dim,
            )
            for _ in range(config.num_experts)
        ])

        # Log dispatch strategy
        if self.use_megablocks and MEGABLOCKS_AVAILABLE:
            logger.debug(
                "MoELayer initialized with MegaBlocks dispatch "
                f"({self.num_experts} experts, top_k={self.top_k})."
            )
        else:
            if self.use_megablocks and not MEGABLOCKS_AVAILABLE:
                warnings.warn(
                    "use_megablocks=True but MegaBlocks is not installed. "
                    "Falling back to scatter/gather dispatch. "
                    "Install with: pip install megablocks",
                    RuntimeWarning,
                    stacklevel=2,
                )
            logger.debug(
                "MoELayer initialized with scatter/gather dispatch "
                f"({self.num_experts} experts, top_k={self.top_k})."
            )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute MoE layer output with dropless token-choice routing.

        Routes each token to top_k=8 experts, processes tokens through
        their assigned experts, and aggregates outputs weighted by routing
        probabilities.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim).
               Expected to be pre-normalized (OLMoEBlock applies ffn_norm
               before calling this method).
               For OLMoE-1B-7B: (B, S, 2048) where S <= 4096.

        Returns:
            Tuple of three tensors:
                - output: MoE layer output, shape (batch_size, seq_len, hidden_dim).
                  Added to the residual stream by OLMoEBlock.
                - router_logits: Pre-softmax router logits, shape (B*S, num_experts).
                  Collected by OLMoEModel and stored in OLMoEOutput.router_logits
                  for use in AuxiliaryLosses.router_z_loss().
                - top_k_indices: Expert assignments, shape (B*S, top_k), dtype=long.
                  Collected by OLMoEModel and stored in OLMoEOutput.top_k_indices
                  for use in AuxiliaryLosses.load_balancing_loss() and all
                  analysis modules (router saturation, co-activation, etc.).

        Shape example:
            Input:  (2, 4096, 2048)
            Output: (2, 4096, 2048), (8192, 64), (8192, 8)
        """
        batch_size: int = x.shape[0]
        seq_len: int = x.shape[1]
        num_tokens: int = batch_size * seq_len

        # ------------------------------------------------------------------
        # Step 1: Flatten batch and sequence dimensions for routing
        # (B, S, hidden_dim) -> (B*S, hidden_dim)
        # ------------------------------------------------------------------
        flat_x: Tensor = x.view(num_tokens, self.hidden_dim)

        # ------------------------------------------------------------------
        # Step 2: Compute routing decisions
        # router_logits: (B*S, 64) — pre-softmax, for z-loss
        # softmax_weights: (B*S, 64) — full softmax probs, for LB loss + dispatch
        # top_k_indices: (B*S, 8) — selected expert indices, for dispatch + analysis
        # ------------------------------------------------------------------
        router_logits: Tensor
        softmax_weights: Tensor
        top_k_indices: Tensor
        router_logits, softmax_weights, top_k_indices = self.router(flat_x)

        # ------------------------------------------------------------------
        # Step 3: Dispatch tokens to experts and aggregate outputs
        # ------------------------------------------------------------------
        if self.use_megablocks and MEGABLOCKS_AVAILABLE:
            output_flat: Tensor = self._dispatch_megablocks(
                flat_x, top_k_indices, softmax_weights
            )
        else:
            output_flat = self._dispatch_tokens(
                flat_x, top_k_indices, softmax_weights
            )
        # output_flat: (B*S, hidden_dim)

        # ------------------------------------------------------------------
        # Step 4: Reshape output back to (batch, seq, hidden)
        # ------------------------------------------------------------------
        output: Tensor = output_flat.view(batch_size, seq_len, self.hidden_dim)

        return output, router_logits, top_k_indices

    def _dispatch_tokens(
        self,
        flat_x: Tensor,
        top_k_indices: Tensor,
        softmax_weights: Tensor,
    ) -> Tensor:
        """Dispatch tokens to experts using scatter/gather (fallback implementation).

        Implements dropless token-choice routing: every token is processed by
        exactly top_k=8 experts with no token dropping. Uses a Python loop
        over experts with boolean indexing for correctness.

        Algorithm:
            output = zeros(num_tokens, hidden_dim)
            for expert_id in range(num_experts):
                # Find tokens assigned to this expert
                token_mask = (top_k_indices == expert_id).any(dim=-1)
                if no tokens: continue
                # Gather, process, scale, scatter back
                expert_output = expert(flat_x[token_mask])
                weights = softmax_weights[token_mask, expert_id]
                output[token_mask] += expert_output * weights.unsqueeze(-1)

        The routing weight used is softmax_weights[token, expert_id] — the
        full softmax probability for that expert over all 64 experts. This
        matches Equation 1 in the paper (softmax applied before top-k selection).

        Args:
            flat_x: Flattened token tensor, shape (num_tokens, hidden_dim).
            top_k_indices: Expert assignments, shape (num_tokens, top_k),
                           dtype=torch.long.
            softmax_weights: Full softmax routing probabilities,
                             shape (num_tokens, num_experts).

        Returns:
            Aggregated expert outputs, shape (num_tokens, hidden_dim).
            Each token's output is the weighted sum of its top_k expert outputs.
        """
        num_tokens: int = flat_x.shape[0]

        # Initialize output accumulator with zeros.
        # Each token's output will be accumulated from top_k=8 expert contributions.
        output: Tensor = torch.zeros(
            num_tokens,
            self.hidden_dim,
            dtype=flat_x.dtype,
            device=flat_x.device,
        )

        # Process each expert independently.
        # For 64 experts with top_k=8, on average each expert receives
        # num_tokens * 8 / 64 = num_tokens / 8 tokens.
        for expert_id in range(self.num_experts):
            # ------------------------------------------------------------------
            # Find which tokens are assigned to this expert.
            # top_k_indices has shape (num_tokens, top_k).
            # token_mask[i] = True if expert_id appears in top_k_indices[i].
            # Shape: (num_tokens,), dtype=bool
            # ------------------------------------------------------------------
            token_mask: Tensor = (top_k_indices == expert_id).any(dim=-1)

            # Skip experts that receive no tokens in this batch.
            # This is common early in training before load balancing takes effect,
            # and can happen for any expert in any batch.
            if not token_mask.any():
                continue

            # ------------------------------------------------------------------
            # Gather tokens assigned to this expert.
            # expert_input shape: (num_assigned, hidden_dim)
            # num_assigned varies per expert and per batch.
            # ------------------------------------------------------------------
            expert_input: Tensor = flat_x[token_mask]

            # ------------------------------------------------------------------
            # Process tokens through this expert's SwiGLU FFN.
            # expert_output shape: (num_assigned, hidden_dim)
            # ------------------------------------------------------------------
            expert_output: Tensor = self.experts[expert_id](expert_input)

            # ------------------------------------------------------------------
            # Get routing weights for this expert for the assigned tokens.
            # softmax_weights[token_mask, expert_id] extracts the routing
            # probability that each assigned token gave to this expert.
            # Shape: (num_assigned,)
            # ------------------------------------------------------------------
            weights: Tensor = softmax_weights[token_mask, expert_id]

            # ------------------------------------------------------------------
            # Scale expert output by routing weight and accumulate.
            # weights.unsqueeze(-1): (num_assigned,) -> (num_assigned, 1)
            # Broadcasting: (num_assigned, hidden_dim) * (num_assigned, 1)
            # -> (num_assigned, hidden_dim)
            # ------------------------------------------------------------------
            weighted_output: Tensor = expert_output * weights.unsqueeze(-1)

            # Scatter back to the output tensor using in-place addition.
            # output[token_mask] += weighted_output accumulates contributions
            # from all top_k experts for each token.
            output[token_mask] = output[token_mask] + weighted_output

        return output

    def _dispatch_megablocks(
        self,
        flat_x: Tensor,
        top_k_indices: Tensor,
        softmax_weights: Tensor,
    ) -> Tensor:
        """Dispatch tokens to experts using MegaBlocks sparse operations.

        MegaBlocks uses block-sparse CUDA kernels to process all expert
        computations in a single fused operation, avoiding the Python loop
        over 64 experts. This provides significantly better GPU utilization
        and throughput.

        If MegaBlocks dispatch fails for any reason (API mismatch, CUDA error,
        etc.), falls back to the scatter/gather implementation with a warning.

        Args:
            flat_x: Flattened token tensor, shape (num_tokens, hidden_dim).
            top_k_indices: Expert assignments, shape (num_tokens, top_k),
                           dtype=torch.long.
            softmax_weights: Full softmax routing probabilities,
                             shape (num_tokens, num_experts).

        Returns:
            Aggregated expert outputs, shape (num_tokens, hidden_dim).
        """
        try:
            # Attempt to use MegaBlocks sparse dispatch.
            # The MegaBlocks API for custom expert dispatch uses the
            # ops module for scatter/gather with block-sparse matrices.
            import megablocks.ops as mblocks_ops

            num_tokens: int = flat_x.shape[0]

            # ------------------------------------------------------------------
            # Build the expert assignment metadata required by MegaBlocks.
            # MegaBlocks expects tokens sorted by expert assignment for
            # efficient block-sparse matrix multiplication.
            # ------------------------------------------------------------------

            # Flatten top_k_indices to get all (token, expert) pairs.
            # Each token appears top_k=8 times.
            # expert_indices_flat: (num_tokens * top_k,) = (B*S*8,)
            expert_indices_flat: Tensor = top_k_indices.reshape(-1)

            # Create corresponding token indices.
            # token_indices_flat[i] = which token the i-th (token, expert) pair belongs to
            # Shape: (num_tokens * top_k,)
            token_indices_flat: Tensor = (
                torch.arange(num_tokens, device=flat_x.device)
                .unsqueeze(1)
                .expand(-1, self.top_k)
                .reshape(-1)
            )

            # Sort by expert index to group tokens going to the same expert.
            sort_order: Tensor