# model/moe_layer.py
"""
Mixture-of-Experts layer replacing the FFN in each transformer block.

Implements dropless token-choice routing (token-level top-k selection) with
fine-grained experts (64 small SwiGLU experts, 8 activated per token) exactly
as described in the OLMoE‑1B‑7B paper (Section 2, Table 1, Appendix B).

All hyperparameters are drawn from the configuration dictionary, and the
interface follows the design specification in the project’s data‑structures
document.

Classes:
    MoELayer – one MoE module, to be called as a drop‑in FFN substitution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

# Custom imports – these are expected to be in the same package.
from model.expert import Expert
from model.router import Router

# ---------------------------------------------------------------------------
# Helper function (can be in utils, but kept here for self‑contained file)
def _init_weights_with_trunc_normal(
    module: nn.Module,
    init_std: float,
    init_trunc: int,
) -> None:
    """Apply truncated normal initialisation to all linear layers in `module`."""
    with torch.no_grad():
        for child in module.modules():
            if isinstance(child, nn.Linear):
                nn.init.trunc_normal_(
                    child.weight,
                    mean=0.0,
                    std=init_std,
                    a=-init_trunc * init_std,
                    b=init_trunc * init_std,
                )
                if hasattr(child, "bias") and child.bias is not None:
                    nn.init.zeros_(child.bias)
# ---------------------------------------------------------------------------

class MoELayer(nn.Module):
    """
    Dropless token-choice Mixture-of-Experts layer.

    Replaces the standard dense FFN in a transformer layer.

    Args:
        config:    Full `model` section from the YAML config. Must contain:
                    - hidden_size (int)
                    - moe.num_experts (int)
                    - moe.top_k (int)
                    - moe.expert_ffn_size (int)
                    - init_std (float)
                    - init_truncation (int)
        use_megablocks:  If True, will attempt to use Megablocks' efficient sparse
                        forward.  Default False; custom loop is used otherwise.
    """

    def __init__(
        self,
        config: Dict[str, any],
        use_megablocks: bool = False,
    ) -> None:
        super().__init__()

        # – Read essential MoE parameters -----------------------------------
        hidden_size: int = config["hidden_size"]
        moe_cfg: Dict = config["moe"]
        num_experts: int = moe_cfg["num_experts"]
        top_k: int = moe_cfg["top_k"]
        expert_ffn_size: int = moe_cfg["expert_ffn_size"]
        init_std: float = config.get("init_std", 0.02)
        init_trunc: int = config.get("init_truncation", 3)

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_ffn_size = expert_ffn_size
        self.use_megablocks = use_megablocks

        # – Routing network -------------------------------------------------
        # The Router returns raw logits; we will re‑normalise the top‑k logits
        # ourselves to obtain the final gating probabilities.
        self.router = Router(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            init_std=init_std,
            init_trunc=init_trunc,
            bias=False,
        )

        # – Experts – each is a small SwiGLU MLP --------------------------
        # Create a ModuleList of Expert instances.
        # Their weights are initialised with truncated normal inside Expert.
        self.experts = nn.ModuleList([
            Expert(
                hidden_size=hidden_size,
                ffn_size=expert_ffn_size,
                init_std=init_std,
                init_trunc=init_trunc,
                bias=False,
            )
            for _ in range(num_experts)
        ])

        # Optional: if Megablocks is desired, we would build a fused
        # SparseGLU here using the expert parameters. For simplicity we keep
        # the custom loop path.
        if self.use_megablocks:
            # This line is never reached with the default `use_megablocks=False`,
            # but if a downstream caller enables it they must provide the
            # Megablocks package.
            try:
                import megablocks.layers.glu as megablock_glu
                # Build weight lists required by Megablocks (example)
                # This is a placeholder – full integration would require
                # extracting .weight tensors from each expert.
                self._megablocks_enabled = True
            except ImportError:
                self.use_megablocks = False
                self._megablocks_enabled = False
        else:
            self._megablocks_enabled = False

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states:  Tensor of shape (batch_size, seq_length, hidden_size)
                            The input from the preceding attention block (after RMSNorm).

        Returns:
            output:  Tensor of shape (batch_size, seq_length, hidden_size)
                     The combined expert transform, ready to be added to the residual stream.
            router_logits:  Tensor of shape (batch_size * seq_length, num_experts)
                            Raw logits for all tokens, to be used by auxiliary losses.
        """
        batch_size, seq_length, h = hidden_states.shape
        if h != self.hidden_size:
            raise ValueError(
                f"Expected hidden size {self.hidden_size}, got {h}"
            )

        # Flatten the sequence dimension for token‑by‑token processing
        tokens = hidden_states.view(-1, self.hidden_size)  # (T, H) where T = B*S
        # ------------------------------------------------------------------
        # 1. Obtain router outputs for all tokens
        #    Router.forward returns (topk_indices, topk_probs, raw_logits).
        #    We use the raw logits to re‑compute the *renormalised* top‑k
        #    gating probabilities, which is what the paper describes.
        # ------------------------------------------------------------------
        _, _, router_logits = self.router(tokens)   # (T, N_experts)

        # 2. Select the top‑k experts and obtain renormalised probabilities
        #    a) Take the top‑k logits and their indices.
        #    b) Apply softmax *only over the selected k logits*.
        #    This is the final probability weighting used in Eq. (1) of the paper.
        topk_logits, topk_indices = torch.topk(
            router_logits, self.top_k, dim=-1, sorted=False
        )                               # (T, K)
        topk_probs = F.softmax(topk_logits, dim=-1)   # (T, K) – gating weights

        # 3. Compute expert outputs and accumulate with gating weights
        #    We implement a dropless loop where every selected expert
        #    processes its assigned tokens. This exactly follows the
        #    "dropless token choice" routing scheme.
        output = torch.zeros_like(tokens)  # (T, H)

        for expert_idx in range(self.num_experts):
            # Find which tokens selected expert_idx
            token_mask = (topk_indices == expert_idx).nonzero(as_tuple=False)
            if token_mask.numel() == 0:
                continue   # no tokens routed to this expert

            # token_mask has shape (N_e, 2): rows (token_pos, k_pos)
            token_indices = token_mask[:, 0]   # (N_e,) global token indices
            k_positions = token_mask[:, 1]     # (N_e,) position in top‑k array

            # Extract the corresponding gating probability
            probs_e = topk_probs[token_indices, k_positions]   # (N_e,)

            # Forward the assigned tokens through expert `expert_idx`
            expert_input = tokens[token_indices]                # (N_e, H)
            expert_output = self.experts[expert_idx](expert_input)  # (N_e, H)

            # Scale output by gating probability
            expert_output = expert_output * probs_e.unsqueeze(-1)

            # Scatter‑add back into the full output tensor
            output.index_add_(0, token_indices, expert_output)

        # 4. Reshape output to match input shape
        output = output.view(batch_size, seq_length, self.hidden_size)

        # Return both the final hidden states and the raw router logits
        return output, router_logits

    # ------------------------------------------------------------------
    # Utility to inspect expert usage (can be called by logging code)
    # ------------------------------------------------------------------
    def expert_load(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute the token assignment load per expert for a given input.

        Returns a tensor of shape (num_experts,) containing the fraction
        of tokens routed to each expert.
        """
        tokens = hidden_states.view(-1, self.hidden_size)
        with torch.no_grad():
            _, _, router_logits = self.router(tokens)
            _, topk_indices = torch.topk(router_logits, self.top_k, dim=-1)

        load = torch.zeros(self.num_experts, device=hidden_states.device)
        idx, counts = torch.unique(topk_indices, return_counts=True)
        load[idx] = counts.float() / (tokens.size(0) * self.top_k)
        return load

