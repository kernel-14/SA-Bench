# models/moe_layer.py
"""Mixture-of-Experts layer for the MoE-POT architecture.

Implements the MoELayer class, which orchestrates the interaction between
the router-gating network, shared experts, and routed experts to produce
the sparse MoE output described in Section 4 of the MoE-POT paper.

The core equation implemented here is:
    z^{l+1}(x) = (1/N_s) * Σ_{i=1}^{N_s} E_i^{l(s)}(z_0^l(x))
               + Σ_{k=1}^{K} w_k^l(z_0^l(x)) · E_{i_k}^{l(r)}(z_0^l(x))

where:
  - E_i^{l(s)}: shared experts (always active, N_s=2)
  - E_{i_k}^{l(r)}: routed experts (top-K=4 selected per input from N_r=16)
  - w_k^l: re-normalized routing weights from the router-gating network

Load balancing loss (Section 4, Load Balancing Objective):
    Importance_i = Σ_b w_{i,b}   (sum of routing weights over batch)
    L_balance = w_bal * CV({Importance_i})^2
    where CV = std / mean (coefficient of variation)

References:
  - Paper Section 4: Method
  - Paper Section 3.2: Sparse Pre-training with Mixture-of-Experts
  - config.yaml: architecture.load_balance_weight = 0.1
  - config.yaml: models.tiny.num_routed_experts = 16
  - config.yaml: models.tiny.num_shared_experts = 2
  - config.yaml: models.tiny.top_k = 4
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.expert import ExpertCNN
from models.router import RouterGating


class MoELayer(nn.Module):
    """Sparse Mixture-of-Experts layer for PDE operator learning.

    Combines shared experts (always active) and routed experts (top-K
    selected per input) to achieve efficient capacity scaling without
    proportional inference cost increase.

    Key design properties:
      - N_s=2 shared experts capture universal physical principles
        (conservation laws, symmetry) across all PDE types.
      - N_r=16 routed experts specialize in distinct PDE characteristics;
        only top-K=4 are activated per input (25% of routed experts).
      - Total activated fraction: (N_s + K) / (N_s + N_r) = 6/18 ≈ 33%
        of total expert parameters, matching the paper's efficiency claim.
      - Load balancing loss prevents routing collapse via CV² regularization.
      - Full softmax (all 16 weights) is always computed and returned for
        load balancing and interpretability analysis (Appendix B.4).

    Attributes:
        embed_dim: Feature dimension (attn_dim from config). 512/1024/1024
            for Tiny/Small/Medium models.
        mlp_dim: Hidden dimension inside each expert CNN. 512/1024/2048
            for Tiny/Small/Medium models.
        num_routed_experts: Number of routed experts N_r. Default 16.
        num_shared_experts: Number of shared experts N_s. Default 2.
        top_k: Number of routed experts activated per input K. Default 4.
        load_balance_weight: Scaling factor w_bal for the CV² loss. Default 0.1.
        shared_experts: ModuleList of N_s ExpertCNN instances (always active).
        routed_experts: ModuleList of N_r ExpertCNN instances (top-K active).
        router: RouterGating network producing routing distributions.
    """

    def __init__(
        self,
        embed_dim: int,
        mlp_dim: int,
        num_routed_experts: int = 16,
        num_shared_experts: int = 2,
        top_k: int = 4,
        load_balance_weight: float = 0.1,
    ) -> None:
        """Initializes the MoELayer with shared experts, routed experts, and router.

        Constructs:
          - num_shared_experts ExpertCNN instances (always activated)
          - num_routed_experts ExpertCNN instances (top-K activated per input)
          - One RouterGating network for computing routing distributions

        All experts share the same architecture (ExpertCNN with embed_dim
        input/output channels and mlp_dim hidden channels) but have
        independent parameters, enabling specialization through training.

        Args:
            embed_dim: Number of input/output channels for all experts and
                the router. Corresponds to attn_dim in config.yaml:
                  - Tiny:   512  (config.yaml models.tiny.attn_dim)
                  - Small:  1024 (config.yaml models.small.attn_dim)
                  - Medium: 1024 (config.yaml models.medium.attn_dim)
            mlp_dim: Hidden channel dimension inside each ExpertCNN.
                Corresponds to mlp_dim in config.yaml:
                  - Tiny:   512  (config.yaml models.tiny.mlp_dim)
                  - Small:  1024 (config.yaml models.small.mlp_dim)
                  - Medium: 2048 (config.yaml models.medium.mlp_dim)
            num_routed_experts: Number of routed experts N_r. Default 16
                (config.yaml models.*.num_routed_experts). Ablation studies
                test N_r ∈ {8, 16, 32} (Table 4 in paper).
            num_shared_experts: Number of shared experts N_s. Default 2
                (config.yaml models.*.num_shared_experts). Always activated
                for every input regardless of routing decisions.
            top_k: Number of routed experts to activate per input K.
                Default 4 (config.yaml models.*.top_k). Ablation studies
                test K ∈ {1, 2, 4} (Table 4 in paper).
            load_balance_weight: Scaling factor w_bal for the load balancing
                loss L_balance = w_bal * CV^2. Default 0.1
                (config.yaml architecture.load_balance_weight).

        Raises:
            ValueError: If embed_dim <= 0 or mlp_dim <= 0.
            ValueError: If num_routed_experts <= 0 or num_shared_experts <= 0.
            ValueError: If top_k <= 0 or top_k > num_routed_experts.
            ValueError: If load_balance_weight < 0.
        """
        super().__init__()

        # --- Input validation ---
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}.")
        if mlp_dim <= 0:
            raise ValueError(f"mlp_dim must be positive, got {mlp_dim}.")
        if num_routed_experts <= 0:
            raise ValueError(
                f"num_routed_experts must be positive, got {num_routed_experts}."
            )
        if num_shared_experts <= 0:
            raise ValueError(
                f"num_shared_experts must be positive, got {num_shared_experts}."
            )
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}.")
        if top_k > num_routed_experts:
            raise ValueError(
                f"top_k ({top_k}) cannot exceed num_routed_experts "
                f"({num_routed_experts})."
            )
        if load_balance_weight < 0.0:
            raise ValueError(
                f"load_balance_weight must be non-negative, got {load_balance_weight}."
            )

        # Store configuration attributes for use in forward methods.
        self.embed_dim: int = embed_dim
        self.mlp_dim: int = mlp_dim
        self.num_routed_experts: int = num_routed_experts
        self.num_shared_experts: int = num_shared_experts
        self.top_k: int = top_k
        self.load_balance_weight: float = load_balance_weight

        # --- Shared experts (N_s = 2, always activated) ---
        # These experts are constrained by cross-task learning to capture
        # universal physical principles (conservation laws, symmetry).
        # Each is an independent ExpertCNN with its own parameters.
        self.shared_experts: nn.ModuleList = nn.ModuleList(
            [
                ExpertCNN(in_channels=embed_dim, hidden_channels=mlp_dim)
                for _ in range(num_shared_experts)
            ]
        )

        # --- Routed experts (N_r = 16, top-K = 4 activated per input) ---
        # These experts autonomously develop distinct functional roles to
        # learn unique characteristics of different PDEs. Only top_k are
        # activated per forward pass, controlled by the router.
        self.routed_experts: nn.ModuleList = nn.ModuleList(
            [
                ExpertCNN(in_channels=embed_dim, hidden_channels=mlp_dim)
                for _ in range(num_routed_experts)
            ]
        )

        # --- Router-gating network ---
        # CNN-based network that produces per-sample routing distributions
        # over the num_routed_experts routed experts. Returns both raw
        # logits and full softmax weights.
        self.router: RouterGating = RouterGating(
            embed_dim=embed_dim,
            num_routed_experts=num_routed_experts,
        )

    def _apply_topk_routing(
        self,
        full_softmax: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Selects top-K experts and produces re-normalized routing weights.

        Implements the Top-K selection from Section 4:
            TopK(w^l(z_0^l(x))) = {(i_k, w_k^l(z_0^l(x)))}_{k=1}^{K}

        The selected weights are re-normalized via softmax over the top-K
        subset so they sum to 1.0 per sample. This follows the DeepSeekMoE
        convention (citation [8] in the paper) and ensures the routed
        contribution is properly scaled in the aggregation step.

        The double-softmax design is intentional:
          - First softmax (in RouterGating): produces a distribution over
            all N_r experts for load balancing and interpretability.
          - Second softmax (here): re-normalizes the top-K subset so the
            K routing weights sum to 1 for the weighted aggregation.

        Args:
            full_softmax: Complete softmax distribution over all routed
                experts, shape (B, N_r). Output of RouterGating.forward().
                Values are in (0, 1) and sum to 1.0 along dim=-1.

        Returns:
            A tuple (top_k_indices, top_k_weights) where:
              - top_k_indices: Indices of the top-K selected experts,
                shape (B, K). Values in [0, N_r). Sorted in descending
                order of routing weight per sample.
              - top_k_weights: Re-normalized routing weights for the
                selected experts, shape (B, K). Values are in (0, 1)
                and sum to 1.0 along dim=-1 for each sample.
        """
        # Select top-K experts per sample based on softmax probabilities.
        # torch.topk returns values in descending order by default.
        # top_k_values shape: (B, K)
        # top_k_indices shape: (B, K)
        top_k_values: torch.Tensor
        top_k_indices: torch.Tensor
        top_k_values, top_k_indices = torch.topk(
            full_softmax,
            k=self.top_k,
            dim=-1,
            largest=True,
            sorted=True,
        )

        # Re-normalize the top-K weights via softmax over the selected subset.
        # This ensures the K routing weights sum to 1.0 per sample, which is
        # required for the weighted aggregation in the MoE output formula.
        # Input:  (B, K) — top-K softmax probabilities
        # Output: (B, K) — re-normalized weights summing to 1.0 per sample
        top_k_weights: torch.Tensor = F.softmax(top_k_values, dim=-1)

        return top_k_indices, top_k_weights

    def _compute_load_balance_loss(
        self,
        full_softmax: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the auxiliary load balancing loss to prevent routing collapse.

        Implements the load balancing objective from Section 4:
            Importance_i = Σ_{b=1}^{B} w_{i,b}
            L_balance = w_bal * CV({Importance_i}_{i=1}^{N_r})^2

        where CV = std(Importance) / mean(Importance) is the coefficient
        of variation. A perfectly balanced load gives CV=0 (all experts
        have equal importance), so minimizing CV² drives the router toward
        uniform expert utilization.

        Critical: Uses the FULL softmax (all N_r=16 weights), not just the
        top-K selected ones. Using only top-K would give zero importance to
        non-selected experts, making CV=0 trivially and defeating the purpose
        of the loss. The full distribution captures the router's preference
        for all experts, including those not selected in this batch.

        Gradients flow back to the router through full_softmax, encouraging
        the router to distribute routing weights more uniformly across experts.
        No detach() is applied — this is consistent with Switch Transformer
        and DeepSeekMoE (citations [11, 8] in the paper).

        Args:
            full_softmax: Complete softmax distribution over all routed
                experts for the entire batch, shape (B, N_r). Output of
                RouterGating.forward(). Values are in (0, 1) and sum to
                1.0 along dim=-1 for each sample.

        Returns:
            Scalar tensor representing the load balancing loss for this
            layer. Added to the prediction loss in the training loop:
                L_total = L_pred + Σ_l L_balance^l
        """
        # Compute per-expert importance as the sum of routing weights over
        # the batch dimension. This measures how much total routing weight
        # each expert receives across all B samples.
        # full_softmax shape: (B, N_r)
        # importance shape:   (N_r,)
        importance: torch.Tensor = full_softmax.sum(dim=0)

        # Compute coefficient of variation (CV) across all N_r routed experts.
        # CV = std(Importance) / mean(Importance)
        # A uniform distribution gives CV=0; a collapsed distribution (all
        # weight on one expert) gives CV close to sqrt(N_r - 1).
        mean_importance: torch.Tensor = importance.mean()
        std_importance: torch.Tensor = importance.std()

        # Add epsilon to denominator to avoid division by zero when all
        # experts have exactly equal importance (mean_importance could be
        # very small at the start of training).
        cv: torch.Tensor = std_importance / (mean_importance + 1e-8)

        # Load balancing loss: w_bal * CV^2
        # w_bal = 0.1 (config.yaml architecture.load_balance_weight)
        balance_loss: torch.Tensor = self.load_balance_weight * cv ** 2

        return balance_loss

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the MoE layer output with load balancing loss.

        Implements the full MoE aggregation formula from Section 4:
            z^{l+1}(x) = (1/N_s) * Σ_{i=1}^{N_s} E_i^{l(s)}(z_0^l(x))
                       + Σ_{k=1}^{K} w_k^l(z_0^l(x)) · E_{i_k}^{l(r)}(z_0^l(x))

        Processing pipeline:
          1. Compute shared expert outputs (always active, averaged)
          2. Get routing distribution from router-gating network
          3. Select top-K experts and re-normalize weights
          4. Compute routed expert outputs (only unique selected experts)
          5. Aggregate: shared_out + weighted routed_out
          6. Compute load balancing loss from full routing distribution

        Efficiency: Only experts selected by at least one sample in the
        batch are computed. With top_k=4 and num_routed_experts=16, at
        most 4×B expert calls are needed (vs 16×B naive), and in practice
        many samples share the same top-4 experts, reducing this further.

        Args:
            x: Input feature map of shape (B, embed_dim, H', W') where:
                - B: Batch size (up to 20 for pre-training).
                - embed_dim: Must match self.embed_dim (attn_dim from config).
                - H': Token grid height, typically 16 (= 128 / patch_size=8).
                - W': Token grid width, typically 16.
                This is z_0^l(x) from the paper — the output of the Fourier
                layer passed to the MoE layer.

        Returns:
            A tuple (z_out, balance_loss) where:
              - z_out: Output feature map of shape (B, embed_dim, H', W').
                Same shape as input x, enabling residual connections in
                MoEBlock.
              - balance_loss: Scalar tensor representing the load balancing
                loss for this layer. Accumulated across all N blocks in
                MoEPOT.forward() and added to the prediction loss.
        """
        batch_size: int = x.shape[0]

        # ----------------------------------------------------------------
        # Step 1: Shared Expert Outputs (always active)
        # ----------------------------------------------------------------
        # All N_s=2 shared experts process the full input unconditionally.
        # Their outputs are averaged (not summed) per the paper's equation:
        #   (1/N_s) * Σ_{i=1}^{N_s} E_i^{l(s)}(z_0^l(x))
        # Averaging ensures the shared contribution is properly scaled
        # regardless of N_s, maintaining gradient stability.
        shared_accumulator: torch.Tensor = torch.zeros_like(x)
        for shared_expert in self.shared_experts:
            shared_accumulator = shared_accumulator + shared_expert(x)
        # Average over N_s shared experts.
        shared_out: torch.Tensor = shared_accumulator / self.num_shared_experts

        # ----------------------------------------------------------------
        # Step 2: Get Routing Distribution
        # ----------------------------------------------------------------
        # RouterGating produces both raw logits and full softmax weights.
        # full_softmax shape: (B, N_r) — used for load balancing and
        # interpretability (stored in MoEPOT.get_router_weights()).
        # logits shape: (B, N_r) — not used directly in this method.
        _logits: torch.Tensor
        full_softmax: torch.Tensor
        _logits, full_softmax = self.router(x)

        # ----------------------------------------------------------------
        # Step 3: Apply Top-K Routing
        # ----------------------------------------------------------------
        # Select top-K experts per sample and re-normalize their weights.
        # top_k_indices shape: (B, K) — which experts to activate
        # top_k_weights shape: (B, K) — re-normalized weights summing to 1
        top_k_indices: torch.Tensor
        top_k_weights: torch.Tensor
        top_k_indices, top_k_weights = self._apply_topk_routing(full_softmax)

        # ----------------------------------------------------------------
        # Step 4: Compute Routed Expert Outputs (Efficiently)
        # ----------------------------------------------------------------
        # Only compute experts that are selected by at least one sample.
        # This avoids running all 16 experts when only ~4-6 unique experts
        # are typically selected across a batch of 20 samples.

        # Initialize accumulator for routed expert contributions.
        # Shape: (B, embed_dim, H', W')
        routed_out: torch.Tensor = torch.zeros_like(x)

        # Find unique expert indices selected across the entire batch.
        # top_k_indices shape: (B, K) → flatten to (B*K,) → unique values
        # unique_expert_ids: 1D tensor of unique expert indices in [0, N_r)
        unique_expert_ids: torch.Tensor = top_k_indices.unique()

        # Pre-compute outputs for all unique experts selected in this batch.
        # expert_outputs_cache: dict mapping expert_id (int) → output tensor
        # of shape (B, embed_dim, H', W').
        # We compute on the full batch (not masked subsets) for simplicity
        # and to avoid variable-size tensor operations that complicate
        # gradient flow. The overhead is minimal since we only compute
        # unique experts (typically 4-8 out of 16 per batch).
        expert_outputs_cache: dict = {}
        for expert_id_tensor in unique_expert_ids:
            expert_id: int = int(expert_id_tensor.item())
            # Run this expert on the full batch.
            # Input:  (B, embed_dim, H', W')
            # Output: (B, embed_dim, H', W')
            expert_outputs_cache[expert_id] = self.routed_experts[expert_id](x)

        # Accumulate weighted routed expert contributions.
        # For each of the K routing positions, add the weighted expert output
        # to the accumulator for the samples that selected that expert.
        #
        # top_k_indices[:, k]: shape (B,) — expert index for position k
        # top_k_weights[:, k]: shape (B,) — routing weight for position k
        #
        # We iterate over K positions (not over B samples) to leverage
        # vectorized operations across the batch dimension.
        for k_pos in range(self.top_k):
            # Expert indices selected at position k for each sample.
            # Shape: (B,) — values in [0, N_r)
            k_indices: torch.Tensor = top_k_indices[:, k_pos]

            # Routing weights for position k for each sample.
            # Shape: (B,) — values in (0, 1)
            k_weights: torch.Tensor = top_k_weights[:, k_pos]

            # For each unique expert selected at this position, accumulate
            # its weighted output for the samples that selected it.
            # This inner loop runs at most num_routed_experts times but
            # typically much fewer (only experts selected at position k).
            unique_k_experts: torch.Tensor = k_indices.unique()

            for expert_id_tensor in unique_k_experts:
                expert_id = int(expert_id_tensor.item())

                # Boolean mask: which samples selected this expert at position k.
                # Shape: (B,)
                sample_mask: torch.Tensor = k_indices == expert_id

                if not sample_mask.any():
                    continue

                # Get the pre-computed expert output for this expert.
                # Shape: (B, embed_dim, H', W')
                expert_output: torch.Tensor = expert_outputs_cache[expert_id]

                # Extract routing weights for samples that selected this expert.
                # Shape: (B_masked,) where B_masked = sum(sample_mask)
                masked_weights: torch.Tensor = k_weights[sample_mask]

                # Reshape weights for broadcasting over (embed_dim, H', W').
                # Shape: (B_masked,) → (B_masked, 1, 1, 1)
                masked_weights_broadcast: torch.Tensor = masked_weights.view(
                    -1, 1, 1, 1
                )

                # Accumulate weighted expert output for selected samples.
                # routed_out[sample_mask] shape: (B_masked, embed_dim, H', W')
                # expert_output[sample_mask] shape: (B_masked, embed_dim, H', W')
                routed_out[sample_mask] = (
                    routed_out[sample_mask]
                    + masked_weights_broadcast * expert_output[sample_mask]
                )

        # ----------------------------------------------------------------
        # Step 5: Aggregate Final Output
        # ----------------------------------------------------------------
        # Combine shared and routed contributions:
        #   z^{l+1}(x) = shared_out + routed_out
        # Both terms have shape (B, embed_dim, H', W').
        # The residual connection (adding x) is applied at the MoEBlock
        # level, not here, keeping this layer's responsibility clean.
        z_out: torch.Tensor = shared_out + routed_out

        # ----------------------------------------------------------------
        # Step 6: Compute Load Balance Loss
        # ----------------------------------------------------------------
        # Uses the full softmax distribution (all N_r=16 weights) to
        # compute the CV² regularization loss. Gradients flow back to
        # the router through full_softmax.
        balance_loss: torch.Tensor = self._compute_load_balance_loss(
            full_softmax
        )

        return z_out, balance_loss
