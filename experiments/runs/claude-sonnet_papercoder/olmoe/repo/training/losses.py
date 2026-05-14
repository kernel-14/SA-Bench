## training/losses.py
"""Auxiliary losses for OLMoE pretraining: load balancing and router z-loss.

Implements the two auxiliary losses used during OLMoE pretraining that are
added to the cross-entropy language modeling objective:

  1. Load Balancing Loss (L_LB): Penalizes unequal token distribution across
     experts. Without this, models collapse to using only 1-2 experts per layer
     (Section 4.1.6, Figure 9 and 10).

  2. Router Z-Loss (L_RZ): Penalizes large router logits to prevent numeric
     overflow in BF16 training and improve stability (Section 4.1.7, Figure 11).

Total training loss (Equation 2 from paper):
    L = L_CE + α·L_LB + β·L_RZ
    where α = 0.01 (lb_loss_weight) and β = 0.001 (router_z_loss_weight)

Configuration values used (from config.yaml):
    model.lb_loss_weight: 0.01       # alpha, Section 4.1.6
    model.router_z_loss_weight: 0.001 # beta, Section 4.1.7
    model.num_experts: 64            # N_E in L_LB formula
    model.top_k: 8                   # k activated experts per token
    model.num_layers: 16             # number of MoE layers to average over

CRITICAL: These losses are used ONLY during pretraining.
    SFT config (config.yaml sft.use_lb_loss: false) and
    DPO config (config.yaml dpo.use_lb_loss: false) both disable them.
    See Section 4.3 and Table 7 of the paper.

References:
    - Load balancing loss: Shazeer et al. [154] "Outrageously Large Neural Networks"
    - Router z-loss: Zoph et al. [221] "ST-MoE: Designing Stable and Transferable
      Sparse Expert Models"
    - Paper equations: Section 4.1.6 (Eq. 3) and Section 4.1.7 (Eq. 4)
"""

import logging
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from config import OLMoEConfig

logger = logging.getLogger(__name__)


class AuxiliaryLosses:
    """Computes auxiliary losses for OLMoE MoE training.

    This is a pure computation helper class (not an nn.Module) that computes
    the load balancing loss and router z-loss from routing metadata collected
    during the forward pass. It has no learnable parameters and no torch state.

    The class is designed to be instantiated once and reused across all training
    steps. It reads loss weights and architecture constants from OLMoEConfig at
    construction time.

    Usage during pretraining (training/trainer.py):
        aux_losses = AuxiliaryLosses(config)
        output = model(input_ids, labels=labels)
        total, lb, rz = aux_losses.total_loss(
            output.ce_loss,
            output.router_logits,
            output.top_k_indices,
        )
        total.backward()

    Usage during SFT/DPO (adaptation/sft_trainer.py, adaptation/dpo_trainer.py):
        # Option 1: Skip AuxiliaryLosses entirely (preferred)
        # Option 2: Call with use_lb_loss=False, use_router_z_loss=False
        total, lb, rz = aux_losses.total_loss(
            ce_loss,
            output.router_logits,
            output.top_k_indices,
            use_lb_loss=False,
            use_router_z_loss=False,
        )
        # Returns (ce_loss, tensor(0.0), tensor(0.0))

    Attributes:
        lb_loss_weight: Load balancing loss weight alpha = 0.01.
            (from config.yaml: model.lb_loss_weight)
        router_z_loss_weight: Router z-loss weight beta = 0.001.
            (from config.yaml: model.router_z_loss_weight)
        num_experts: Total experts per layer = 64.
            (from config.yaml: model.num_experts)
        top_k: Activated experts per token = 8.
            (from config.yaml: model.top_k)
        num_layers: Number of MoE layers = 16.
            (from config.yaml: model.num_layers)
    """

    def __init__(self, config: OLMoEConfig) -> None:
        """Initialize AuxiliaryLosses from OLMoEConfig.

        Reads all required constants from the config at construction time.
        No torch state is created — this class is a pure computation helper.

        Args:
            config: OLMoEConfig instance. Key fields used:
                    - lb_loss_weight (0.01): alpha in L = L_CE + α·L_LB + β·L_RZ
                    - router_z_loss_weight (0.001): beta in the same equation
                    - num_experts (64): N_E in L_LB = N_E · Σ f_i · P_i
                    - top_k (8): k activated experts per token
                    - num_layers (16): number of layers to average losses over

        Raises:
            ValueError: If lb_loss_weight or router_z_loss_weight are negative,
                        or if num_experts or top_k are invalid.
        """
        if config.lb_loss_weight < 0.0:
            raise ValueError(
                f"lb_loss_weight must be >= 0, got {config.lb_loss_weight}. "
                f"Set to 0.0 to disable load balancing loss."
            )
        if config.router_z_loss_weight < 0.0:
            raise ValueError(
                f"router_z_loss_weight must be >= 0, got {config.router_z_loss_weight}. "
                f"Set to 0.0 to disable router z-loss."
            )
        if config.num_experts <= 0:
            raise ValueError(
                f"num_experts must be positive, got {config.num_experts}."
            )
        if config.top_k <= 0 or config.top_k > config.num_experts:
            raise ValueError(
                f"top_k must be in [1, num_experts], "
                f"got top_k={config.top_k}, num_experts={config.num_experts}."
            )
        if config.num_layers <= 0:
            raise ValueError(
                f"num_layers must be positive, got {config.num_layers}."
            )

        # Loss weights from config.yaml (model section)
        self.lb_loss_weight: float = config.lb_loss_weight
        """Load balancing loss weight alpha = 0.01 (config.yaml: model.lb_loss_weight)."""

        self.router_z_loss_weight: float = config.router_z_loss_weight
        """Router z-loss weight beta = 0.001 (config.yaml: model.router_z_loss_weight)."""

        # Architecture constants from config.yaml (model section)
        self.num_experts: int = config.num_experts
        """Total experts per layer = 64 (config.yaml: model.num_experts)."""

        self.top_k: int = config.top_k
        """Activated experts per token = 8 (config.yaml: model.top_k)."""

        self.num_layers: int = config.num_layers
        """Number of MoE layers = 16 (config.yaml: model.num_layers)."""

        logger.info(
            f"AuxiliaryLosses initialized: "
            f"lb_loss_weight={self.lb_loss_weight}, "
            f"router_z_loss_weight={self.router_z_loss_weight}, "
            f"num_experts={self.num_experts}, "
            f"top_k={self.top_k}, "
            f"num_layers={self.num_layers}"
        )

    def load_balancing_loss(
        self,
        router_logits: Tensor,
        top_k_indices: Tensor,
    ) -> Tensor:
        """Compute load balancing loss for one MoE layer.

        Implements Equation 3 from the paper (Section 4.1.6):
            L_LB = N_E · Σ_{i=1}^{N_E} f_i · P_i

        where:
            N_E = num_experts = 64
            f_i = fraction of tokens routed to expert i (non-differentiable,
                  computed from hard top-k assignments via top_k_indices)
            P_i = mean routing probability allocated to expert i (differentiable,
                  computed from softmax of router_logits)

        Gradient flow:
            - f_i is treated as a constant (stop-gradient). It is computed from
              the discrete top-k selection which is non-differentiable.
            - P_i is differentiable through the softmax. The gradient pushes
              routing probabilities toward experts already receiving many tokens,
              creating a feedback loop that encourages load balance.
            - This is the standard formulation from Shazeer et al. [154].

        Mathematical properties:
            - sum(f_i) = top_k = 8 (each token activates exactly top_k experts)
            - sum(P_i) = 1.0 (softmax probabilities sum to 1 per token, mean preserves)
            - At perfect balance: f_i = top_k/num_experts = 8/64 = 0.125 for all i
            - At perfect balance: P_i = 1/num_experts = 1/64 ≈ 0.0156 for all i
            - At perfect balance: L_LB = 64 × sum(0.125 × 0.0156) = 64 × 64 × 0.00195 ≈ 0.125

        Args:
            router_logits: Pre-softmax router outputs for one layer,
                           shape (num_tokens, num_experts) = (B*S, 64).
                           These are the raw logits BEFORE softmax, as returned
                           by MoERouter.forward() and stored in OLMoEOutput.router_logits.
                           Used to compute P_i via softmax.
            top_k_indices: Selected expert indices for one layer,
                           shape (num_tokens, top_k) = (B*S, 8), dtype=torch.long.
                           As returned by MoERouter.forward() and stored in
                           OLMoEOutput.top_k_indices. Used to compute f_i.

        Returns:
            Scalar tensor: the load balancing loss for this layer (before
            applying the loss weight alpha=0.01). The trainer scales this
            by lb_loss_weight in total_loss().

        Shape example:
            router_logits: (8192, 64)  [batch=2, seq=4096 -> 8192 tokens]
            top_k_indices: (8192, 8)
            -> scalar tensor
        """
        num_tokens: int = router_logits.shape[0]
        num_experts: int = router_logits.shape[1]

        # Validate shapes
        assert top_k_indices.shape[0] == num_tokens, (
            f"router_logits and top_k_indices must have the same number of tokens. "
            f"Got {num_tokens} and {top_k_indices.shape[0]}."
        )
        assert top_k_indices.shape[1] == self.top_k, (
            f"top_k_indices second dimension must equal top_k={self.top_k}, "
            f"got {top_k_indices.shape[1]}."
        )
        assert num_experts == self.num_experts, (
            f"router_logits num_experts dimension must equal {self.num_experts}, "
            f"got {num_experts}."
        )

        # ------------------------------------------------------------------
        # Step 1: Compute f_i — fraction of tokens routed to each expert.
        #
        # Build a binary expert assignment mask from top_k_indices:
        #   expert_mask[t, i] = 1 if expert i is in the top-k for token t
        #                      = 0 otherwise
        #
        # Method: use scatter_ to set 1.0 at the selected expert positions.
        # expert_mask shape: (num_tokens, num_experts) = (B*S, 64)
        #
        # f_i = mean over tokens = fraction of tokens that selected expert i
        # f_i shape: (num_experts,) = (64,)
        #
        # Note: sum(f_i) = top_k = 8 (each token activates exactly top_k experts)
        # Note: f_i is detached from the computation graph (non-differentiable)
        # ------------------------------------------------------------------
        # Create zero-filled mask on the same device as router_logits
        expert_mask: Tensor = torch.zeros(
            num_tokens,
            num_experts,
            dtype=router_logits.dtype,
            device=router_logits.device,
        )

        # Scatter 1.0 at positions (token_idx, expert_idx) for each selected expert.
        # top_k_indices has shape (num_tokens, top_k); scatter_ fills along dim=1.
        # After scatter_: expert_mask[t, top_k_indices[t, k]] = 1.0 for all k
        expert_mask.scatter_(
            dim=1,
            index=top_k_indices.long(),
            value=1.0,
        )

        # Compute f_i: mean fraction of tokens routed to each expert.
        # Shape: (num_experts,) = (64,)
        # Detach to stop gradients — f_i is treated as a constant.
        f_i: Tensor = expert_mask.float().mean(dim=0).detach()

        # ------------------------------------------------------------------
        # Step 2: Compute P_i — mean routing probability for each expert.
        #
        # Apply softmax over all num_experts=64 experts to get routing probs.
        # Computed in float32 for numerical stability (BF16 has limited precision
        # for softmax over 64 values).
        #
        # routing_probs shape: (num_tokens, num_experts) = (B*S, 64)
        # P_i = mean over tokens = mean routing probability for expert i
        # P_i shape: (num_experts,) = (64,)
        #
        # Note: sum(P_i) = 1.0 (softmax sums to 1 per token, mean preserves this)
        # Note: P_i IS differentiable — gradients flow through softmax to router
        # ------------------------------------------------------------------
        # Compute softmax in float32 for numerical stability, cast back to
        # router_logits dtype for consistent computation
        routing_probs: Tensor = torch.softmax(
            router_logits.float(), dim=-1
        ).to(router_logits.dtype)
        # routing_probs shape: (num_tokens, num_experts) = (B*S, 64)

        # Mean routing probability per expert across all tokens.
        # Shape: (num_experts,) = (64,)
        P_i: Tensor = routing_probs.mean(dim=0)

        # ------------------------------------------------------------------
        # Step 3: Compute L_LB = N_E · Σ_{i=1}^{N_E} f_i · P_i
        #
        # This is the dot product of f_i and P_i, scaled by num_experts.
        # The scaling by N_E ensures the loss magnitude is independent of
        # the number of experts (without scaling, more experts -> smaller loss).
        #
        # At perfect balance:
        #   f_i = top_k/num_experts = 8/64 = 0.125 for all i
        #   P_i = 1/num_experts = 1/64 ≈ 0.01563 for all i
        #   L_LB = 64 × sum(0.125 × 0.01563) = 64 × 64 × 0.001953 ≈ 0.125
        # ------------------------------------------------------------------
        # Element-wise product of f_i and P_i, then sum and scale.
        # (f_i * P_i) shape: (num_experts,) = (64,)
        # sum -> scalar
        lb_loss: Tensor = self.num_experts * (f_i * P_i).sum()

        return lb_loss

    def router_z_loss(self, router_logits: Tensor) -> Tensor:
        """Compute router z-loss for one MoE layer.

        Implements Equation 4 from the paper (Section 4.1.7):
            L_RZ(x) = (1/B) · Σ_{i=1}^{B} (log Σ_{j=1}^{N_E} exp(x_j^(i)))²

        which simplifies to:
            L_RZ = mean((logsumexp(router_logits, dim=-1))²)

        where:
            B = number of tokens in the batch
            x_j^(i) = router logit for expert j, token i, BEFORE softmax
            logsumexp(x) = log(Σ_j exp(x_j)) — numerically stable via max subtraction

        Purpose (Section 4.1.7):
            Large router logits before softmax can cause numeric overflow in the
            large matrix multiplications inside the MoE layer, especially in BF16
            training. This loss penalizes large logits, encouraging the router to
            keep them small and improving training stability (fewer loss spikes).

        Gradient flow:
            Fully differentiable. The gradient of (logsumexp(x))² with respect
            to x_j is: 2 · logsumexp(x) · softmax(x)_j
            This penalizes large logits proportionally to their softmax weight,
            pushing the router toward more uniform logit magnitudes.

        Numerical precision:
            - Computed in float32 to avoid BF16 precision loss in the squaring
              operation. The logsumexp values can be large (e.g., 10-50 for
              typical router logits), and squaring amplifies precision errors.
            - torch.logsumexp uses the max-subtraction trick internally for
              numerical stability: logsumexp(x) = max(x) + log(Σ exp(x - max(x)))

        Args:
            router_logits: Pre-softmax router outputs for one layer,
                           shape (num_tokens, num_experts) = (B*S, 64).
                           These are the raw logits BEFORE softmax, as returned
                           by MoERouter.forward() and stored in OLMoEOutput.router_logits.

        Returns:
            Scalar tensor: the router z-loss for this layer (before applying
            the loss weight beta=0.001). The trainer scales this by
            router_z_loss_weight in total_loss().

        Shape example:
            router_logits: (8192, 64)  [batch=2, seq=4096 -> 8192 tokens]
            -> scalar tensor
        """
        # Validate shape
        assert router_logits.shape[1] == self.num_experts, (
            f"router_logits num_experts dimension must equal {self.num_experts}, "
            f"got {router_logits.shape[1]}."
        )

        # ------------------------------------------------------------------
        # Compute log(Σ_j exp(x_j^(i))) for each token i.
        #
        # torch.logsumexp is numerically stable:
        #   logsumexp(x, dim) = max(x) + log(Σ exp(x - max(x)))
        # This avoids overflow from exp() of large values.
        #
        # Cast to float32 for precision: BF16 has only ~3 decimal digits of
        # precision, which is insufficient for squaring logsumexp values that
        # can range from ~2 to ~50 in typical training.
        #
        # log_sum_exp shape: (num_tokens,) = (B*S,)
        # ------------------------------------------------------------------
        log_sum_exp: Tensor = torch.logsumexp(
            router_logits.float(),  # Cast to float32 for numerical stability
            dim=-1,                 # Sum over expert dimension
        )
        # log_sum_exp shape: (num_tokens,) = (B*S,)

        # ------------------------------------------------------------------
        # Square and average over tokens.
        #
        # (log_sum_exp ** 2) shape: (num_tokens,)
        # .mean() -> scalar
        #
        # This is equivalent to (1/B) · Σ_{i=1}^{B} (log Σ_j exp(x_j^(i)))²
        # from Equation 4 in the paper.
        # ------------------------------------------------------------------
        rz_loss: Tensor = (log_sum_exp ** 2).mean()

        # Cast back to the dtype of router_logits for consistency.
        # The loss will be added to ce_loss which may be in BF16 or float32
        # depending on the training setup.
        rz_loss = rz_loss.to(router_logits.dtype)

        return rz_loss

    def total_loss(
        self,
        ce_loss: Tensor,
        all_router_logits: List[Tensor],
        all_top_k_indices: List[Tensor],
        use_lb_loss: bool = True,
        use_router_z_loss: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute total training loss combining CE loss with auxiliary losses.

        Implements Equation 2 from the paper:
            L = L_CE + α·L_LB + β·L_RZ

        where:
            α = lb_loss_weight = 0.01 (config.yaml: model.lb_loss_weight)
            β = router_z_loss_weight = 0.001 (config.yaml: model.router_z_loss_weight)
            L_LB = mean of load_balancing_loss() across all 16 layers
            L_RZ = mean of router_z_loss() across all 16 layers

        Averaging across layers (not summing):
            The paper specifies loss weights globally (α=0.01, β=0.001).
            Averaging across 16 layers keeps the effective per-layer weight
            at the specified values. Summing would give 16× larger effective
            weights (0.16 and 0.016), which is much larger than intended and
            inconsistent with prior work (Zoph et al. [221], Shen et al. [158]).

        Disabling auxiliary losses (SFT/DPO):
            When use_lb_loss=False and use_router_z_loss=False, returns
            (ce_loss, tensor(0.0), tensor(0.0)) immediately without computing
            any auxiliary losses. This matches the paper's finding (Section 4.3)
            that not using load balancing loss during SFT improves performance.

            Config.yaml confirms:
                sft.use_lb_loss: false
                dpo.use_lb_loss: false

        Args:
            ce_loss: Cross-entropy language modeling loss scalar.
                     Computed by OLMoEModel.forward() when labels are provided.
                     This is the primary training signal.
            all_router_logits: Pre-softmax router logits from all MoE layers.
                               List of num_layers=16 tensors, each of shape
                               (num_tokens, num_experts) = (B*S, 64).
                               From OLMoEOutput.router_logits.
            all_top_k_indices: Selected expert indices from all MoE layers.
                               List of num_layers=16 tensors, each of shape
                               (num_tokens, top_k) = (B*S, 8), dtype=torch.long.
                               From OLMoEOutput.top_k_indices.
            use_lb_loss: Whether to compute and add load balancing loss.
                         True during pretraining (default).
                         False during SFT and DPO (config.yaml: sft.use_lb_loss=false).
            use_router_z_loss: Whether to compute and add router z-loss.
                               True during pretraining (default).
                               False during SFT and DPO (config.yaml: sft.use_router_z_loss=false).

        Returns:
            Tuple of three scalar tensors:
                - total_loss: L = L_CE + α·L_LB + β·L_RZ (or just L_CE if both
                  auxiliary losses are disabled). This is the loss to call
                  .backward() on.
                - mean_lb_loss: Mean load balancing loss across layers (before
                  weighting by alpha). Detached from computation graph — for
                  logging only. Returns tensor(0.0) if use_lb_loss=False.
                - mean_rz_loss: Mean router z-loss across layers (before weighting
                  by beta). Detached from computation graph — for logging only.
                  Returns tensor(0.0) if use_router_z_loss=False.

        Raises:
            ValueError: If all_router_logits and all_top_k_indices have different
                        lengths, or if either list is empty when the corresponding
                        loss is enabled.

        Example (pretraining):
            >>> aux = AuxiliaryLosses(config)
            >>> output = model(input_ids, labels=labels)
            >>> total, lb, rz = aux.total_loss(
            ...     output.ce_loss,
            ...     output.router_logits,
            ...     output.top_k_indices,
            ... )
            >>> total.backward()
            >>> # Log lb.item() and rz.item() to wandb

        Example (SFT — auxiliary losses disabled):
            >>> total, lb, rz = aux.total_loss(
            ...     ce_loss,
            ...     output.router_logits,
            ...     output.top_k_indices,
            ...     use_lb_loss=False,
            ...     use_router_z_loss=False,
            ... )
            >>> # total == ce_loss, lb == 0.0, rz == 0.0
        """
        # ------------------------------------------------------------------
        # Fast path: both auxiliary losses disabled (SFT/DPO case).
        # Return immediately without any auxiliary loss computation.
        # This matches config.yaml: sft.use_lb_loss=false, dpo.use_lb_loss=false
        # ------------------------------------------------------------------
        if not use_lb_loss and not use_router_z_loss:
            zero: Tensor = torch.tensor(
                0.0,
                dtype=ce_loss.dtype,
                device=ce_loss.device,
            )
            return ce_loss, zero, zero

        # ------------------------------------------------------------------
        # Validate inputs when auxiliary losses are enabled.
        # ------------------------------------------------------------------
        if use_lb_loss or use_router_z_loss:
            if len(all_router_logits) == 0:
                raise ValueError(
                    "all_router_logits is empty but auxiliary losses are enabled. "
                    "Ensure the model returns router_logits in OLMoEOutput."
                )
            if use_lb_loss and len(all_top_k_indices) != len(all_router_logits):
                raise ValueError(
                    f"all_router_logits and all_top_k_indices must have the same "
                    f"length. Got {len(all_router_logits)} and "
                    f"{len(all_top_k_indices)}."
                )

        num_layers: int = len(all_router_logits)

        # ------------------------------------------------------------------
        # Compute per-layer auxiliary losses and collect them.
        # ------------------------------------------------------------------
        lb_loss_per_layer: List[Tensor] = []
        rz_loss_per_layer: List[Tensor] = []

        for layer_idx in range(num_layers):
            layer_router_logits: Tensor = all_router_logits[layer_idx]

            # Compute load balancing loss for this layer.
            if use_lb_loss:
                layer_top_k_indices: Tensor = all_top_k_indices[layer_idx]
                layer_lb_loss: Tensor = self.load_balancing_loss(
                    layer_router_logits,
                    layer_top_k_indices,
                )
                lb_loss_per_layer.append(layer_lb_loss)

            # Compute router z-loss for this layer.
            if use_router_z_loss:
                layer_rz_loss: Tensor = self.router_z_loss(layer_router_logits)
                rz_loss_per_layer.append(layer_rz_loss)

        # ------------------------------------------------------------------
        # Average auxiliary losses across all layers.
        #
        # Averaging (not summing) keeps the effective per-layer weight at the
        # paper's specified values (α=0.01, β=0.001). Summing would give
        # 16× larger effective weights, inconsistent with prior work.
        #
        # torch.stack + .mean() is used instead of sum/len to properly handle
        # gradient flow through all layer losses.
        # ------------------------------------------------------------------
        mean_lb_loss: Tensor
        mean_rz_loss: Tensor

        if use_lb_loss and lb_loss_per_layer:
            # Stack per-layer losses: (num_layers,) then mean -> scalar
            mean_lb_loss = torch.stack(lb_loss_per_layer).mean()
        else:
            mean_lb_loss = torch.tensor(
                0.0,
                dtype=ce_loss.dtype,
                device=ce_loss.device,
            )

        if use_router_z_loss and rz_loss_per_layer:
            # Stack per-layer losses: (num_layers,) then mean -> scalar
            mean_rz_loss = torch.stack(rz_loss_per_layer).mean()
        else:
            mean_rz_loss = torch.tensor(
                0.0,
                dtype=ce_loss.dtype,
                device=ce_loss.device,
            )

        # ------------------------------------------------------------------
        # Combine losses: L = L_CE + α·L_LB + β·L_RZ
        #
        # Cast auxiliary losses to match ce_loss dtype for safe addition.
        # ce_loss is typically float32 (computed in full precision even in
        # BF16 training), while auxiliary losses may be in BF16 or float32
        # depending on the router_logits dtype.
        # ------------------------------------------------------------------
        mean_lb_loss_cast: Tensor = mean_lb_loss.to(ce_loss.dtype)
        mean_rz_loss_cast: Tensor = mean_rz_loss.to(ce_loss.dtype)

        total: Tensor = (
            ce_loss
            + self.lb_loss_weight * mean_lb_loss_cast
            + self.router_z_loss_weight * mean_rz_loss_cast
        )

        # ------------------------------------------------------------------
        # Return total loss (for backward()) and detached scalars (for logging).
        #
        # .detach() on the returned lb and rz values is critical:
        #   - These are used for logging only (wandb, console)
        #   - Returning attached tensors could cause accidental double-backward
        #     if the caller tries to use them in further computations
        #   - The gradient already flows through total_loss via the total tensor
        # ------------------------------------------------------------------
        return total, mean_lb_loss.detach(), mean_rz_loss.detach()
