# training/losses.py
"""Loss functions for the MoE-POT training pipeline.

Implements two loss functions used during pre-training and fine-tuning:

1. L2RelativeLoss: The primary evaluation and training metric, computing
   the L2 relative error (L2RE) between predicted and ground-truth PDE
   fields. This is the universal metric reported in all paper tables.

2. LoadBalanceLoss: Auxiliary regularization loss that prevents routing
   collapse in the MoE layer by penalizing non-uniform expert utilization
   via the squared coefficient of variation (CV²) of expert importance scores.

Mathematical formulations from the paper:

    L2RE = ||pred - target||₂ / ||target||₂          (Section B.3)

    Importance_i = Σ_b w_{i,b}                        (Section 4)
    L_balance = w_bal · CV({Importance_i})²            (Section 4)
    where CV = std / mean

From config.yaml:
    architecture.load_balance_weight: 0.1   (w_bal for LoadBalanceLoss)
    evaluation.metric: "L2RE"               (L2RelativeLoss is this metric)
    architecture.max_channels: 4            (input tensors are (B, 4, H, W))
    architecture.target_resolution: 128     (H = W = 128)
"""

import torch
import torch.nn as nn


class L2RelativeLoss(nn.Module):
    """L2 Relative Error loss for PDE field prediction.

    Computes the scale-invariant relative error between predicted and
    ground-truth PDE fields. This is the primary evaluation metric (L2RE)
    reported in all tables of the MoE-POT paper, and is used as the
    prediction loss component during training.

    Mathematical formulation (paper Section B.3):
        Rel-ℓ₂ = ||pred - target||₂ / ||target||₂

    The norm is computed over all spatial locations and channels jointly
    for each sample (full Euclidean norm over the flattened field), then
    averaged over the batch dimension.

    This class returns the **non-squared** relative error, consistent with
    the L2RE metric used in all evaluation tables. The Trainer squares this
    value when computing the training loss L_pred = Σ_t L2RE(pred_t, target_t)²
    as specified in the paper's loss function (Section 4).

    Design notes:
      - Flattens (B, C, H, W) → (B, C*H*W) before computing norms, so the
        L2 norm spans all channels and spatial locations jointly per sample.
      - Adds epsilon=1e-8 to the denominator to prevent NaN gradients when
        the target norm is near zero (e.g., padded channels, zero-initialized
        outputs at the start of training).
      - No learnable parameters — this is a pure functional loss.

    Attributes:
        epsilon: Small constant added to the denominator to prevent division
            by zero. Fixed at 1e-8 and not configurable, as this value is
            universally appropriate for float32 PDE field magnitudes.
    """

    def __init__(self) -> None:
        """Initializes L2RelativeLoss with no learnable parameters.

        The epsilon guard value is fixed at 1e-8 as a class constant.
        No configuration is needed — this loss has no hyperparameters.
        """
        super().__init__()
        # Small constant to prevent division by zero in the relative error
        # denominator. 1e-8 is appropriate for float32 PDE field magnitudes
        # and is consistent with the epsilon used in Preprocessor.normalize().
        self.epsilon: float = 1e-8

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the mean L2 relative error over the batch.

        Implements:
            per_sample_l2re = ||pred_b - target_b||₂ / (||target_b||₂ + ε)
            L2RE = mean_b(per_sample_l2re)

        where the norms are computed over all C*H*W elements jointly for
        each sample b in the batch.

        Args:
            pred: Predicted PDE field tensor of shape (B, C, H, W) where:
                - B: Batch size (up to 20 for pre-training, config.yaml
                  pretraining.batch_size).
                - C: Number of channels = 4 (config.yaml
                  architecture.max_channels, padded in PDEDataset).
                - H: Spatial height = 128 (config.yaml
                  architecture.target_resolution).
                - W: Spatial width = 128.
                This is u_pred from MoEPOT.forward(), the predicted next frame.
            target: Ground-truth PDE field tensor of shape (B, C, H, W).
                Same shape as pred. This is u_target from the DataLoader,
                the actual next frame from the dataset.

        Returns:
            Scalar tensor representing the mean L2 relative error over the
            batch. Values are in [0, ∞) where 0 indicates perfect prediction.
            Typical values during training range from 0.01 to 0.5 depending
            on the PDE dataset and training stage.

        Note:
            This method returns the non-squared L2RE. For the training loss
            L_pred = Σ_t ||G_w(u^{<t} + ε) - u^t||₂² / ||u^t||₂², the
            Trainer should square the output of this method before summing
            over timesteps, or use this directly as the loss (both approaches
            are valid; the paper's notation is slightly ambiguous on this point).
        """
        batch_size: int = pred.shape[0]

        # ----------------------------------------------------------------
        # Step 1: Flatten spatial and channel dimensions
        # ----------------------------------------------------------------
        # Reshape from (B, C, H, W) to (B, C*H*W) so that torch.norm
        # computes the full Euclidean norm over all field elements jointly
        # for each sample. This matches the paper's ||·||₂ notation where
        # the norm is over the entire spatial field, not per-channel.
        # Input:  (B, C, H, W)
        # Output: (B, C*H*W)
        pred_flat: torch.Tensor = pred.reshape(batch_size, -1)
        target_flat: torch.Tensor = target.reshape(batch_size, -1)

        # ----------------------------------------------------------------
        # Step 2: Compute per-sample numerator ||pred - target||₂
        # ----------------------------------------------------------------
        # torch.norm with dim=-1 computes the L2 norm over the last
        # dimension (C*H*W) independently for each sample in the batch.
        # diff_flat shape: (B, C*H*W)
        # numerator shape: (B,)
        diff_flat: torch.Tensor = pred_flat - target_flat
        numerator: torch.Tensor = torch.norm(diff_flat, p=2, dim=-1)
        # Shape: (B,) — per-sample L2 norm of the prediction error

        # ----------------------------------------------------------------
        # Step 3: Compute per-sample denominator ||target||₂ + ε
        # ----------------------------------------------------------------
        # target_norm shape: (B,)
        # Adding epsilon prevents division by zero for:
        #   - Padded channels filled with constant 1.0 (near-zero variation)
        #   - Zero-initialized outputs at the start of training
        #   - PDE fields with very small magnitudes (e.g., near-zero initial
        #     conditions in some SWE or DR configurations)
        target_norm: torch.Tensor = torch.norm(target_flat, p=2, dim=-1)
        # Shape: (B,)
        denominator: torch.Tensor = target_norm + self.epsilon
        # Shape: (B,)

        # ----------------------------------------------------------------
        # Step 4: Per-sample relative error
        # ----------------------------------------------------------------
        # rel_error shape: (B,) — values in [0, ∞)
        # Typical range: 0.001 (excellent) to 1.0+ (poor prediction)
        rel_error: torch.Tensor = numerator / denominator
        # Shape: (B,)

        # ----------------------------------------------------------------
        # Step 5: Batch mean
        # ----------------------------------------------------------------
        # Average over the batch dimension to get a scalar loss value.
        # Using mean (not sum) ensures the loss magnitude is independent
        # of batch size, which is important for consistent hyperparameter
        # settings across different batch size configurations.
        return rel_error.mean()


class LoadBalanceLoss(nn.Module):
    """Auxiliary load balancing loss to prevent MoE routing collapse.

    Penalizes non-uniform expert utilization by computing the squared
    coefficient of variation (CV²) of expert importance scores, scaled
    by a tunable weight factor w_bal.

    Mathematical formulation (paper Section 4):
        Importance_i = Σ_{b=1}^{B} w_{i,b}    (sum of routing weights over batch)
        CV = std({Importance_i}) / mean({Importance_i})
        L_balance = w_bal · CV²

    where w_{i,b} is the FULL softmax routing weight for expert i on sample b
    (not the top-K masked version). Using the full distribution ensures that
    non-selected experts still receive gradient signal to improve their
    routing weights.

    This class is provided as a standalone utility for testing and analysis.
    The actual integration in the training pipeline is inside
    MoELayer._compute_load_balance_loss(), which computes importance directly
    from the full softmax output of RouterGating and applies the same formula.

    Design notes:
      - Receives pre-computed importance scores (shape: (N_r,)) rather than
        raw routing weights, keeping this class simple and testable.
      - Uses population std (unbiased=False) for consistency with the paper's
        CV definition, though with N_r=16 the difference from Bessel-corrected
        std is negligible.
      - Adds epsilon=1e-8 to the mean denominator to handle the edge case
        where all importance scores are exactly zero (e.g., at initialization).
      - CV=0 (perfectly balanced) gives L_balance=0, which is the desired
        minimum. Large CV (routing collapse) gives large L_balance.

    Attributes:
        weight: Scaling factor w_bal for the CV² loss. Default 0.1
            (config.yaml architecture.load_balance_weight). Controls the
            trade-off between prediction accuracy and routing balance.
        epsilon: Small constant added to the mean denominator to prevent
            division by zero. Fixed at 1e-8.
    """

    def __init__(self, weight: float = 0.1) -> None:
        """Initializes LoadBalanceLoss with the given scaling weight.

        Args:
            weight: Scaling factor w_bal for the CV² loss. Default 0.1,
                matching config.yaml architecture.load_balance_weight.
                Higher values enforce stricter load balancing at the cost
                of potentially reducing prediction accuracy. The paper
                uses w_bal=0.1 as the optimal trade-off (Section 4).

        Raises:
            ValueError: If weight is negative, as a negative weight would
                encourage routing collapse rather than preventing it.
        """
        super().__init__()

        if weight < 0.0:
            raise ValueError(
                f"LoadBalanceLoss weight must be non-negative, got {weight}. "
                f"A negative weight would encourage routing collapse rather "
                f"than preventing it."
            )

        # Scaling factor w_bal from config.yaml architecture.load_balance_weight.
        # Default 0.1 as specified in the paper (Section 4).
        self.weight: float = weight

        # Small constant to prevent division by zero in the CV denominator.
        # Applied to mean(importance) to handle the edge case where all
        # importance scores are exactly zero at initialization.
        self.epsilon: float = 1e-8

    def forward(self, importance: torch.Tensor) -> torch.Tensor:
        """Computes the load balancing loss from expert importance scores.

        Implements:
            CV = std(importance) / (mean(importance) + ε)
            L_balance = weight · CV²

        A perfectly balanced routing (all experts have equal importance)
        gives CV=0 and L_balance=0. Routing collapse (all weight on one
        expert) gives large CV and large L_balance.

        Args:
            importance: Per-expert importance scores of shape (N_r,) where
                N_r is the number of routed experts (default 16, config.yaml
                models.*.num_routed_experts). Each entry Importance_i is the
                sum of routing weights for expert i over the batch:
                    Importance_i = Σ_{b=1}^{B} w_{i,b}
                where w_{i,b} is the full softmax weight (not top-K masked).
                This is computed in MoELayer._compute_load_balance_loss() as:
                    importance = full_softmax.sum(dim=0)
                where full_softmax has shape (B, N_r).

        Returns:
            Scalar tensor representing the load balancing loss for one MoE
            layer. Values are in [0, ∞) where 0 indicates perfect balance.
            This is L_balance^l from the paper, summed across all N blocks
            in MoEPOT.forward() to give the total balance loss.

        Note:
            This standalone method is provided for unit testing. In the
            actual training pipeline, the equivalent computation is performed
            inside MoELayer._compute_load_balance_loss() which directly
            receives full_softmax from RouterGating and computes importance
            as full_softmax.sum(dim=0) before applying the CV² formula.
        """
        # ----------------------------------------------------------------
        # Step 1: Compute mean importance across all N_r routed experts
        # ----------------------------------------------------------------
        # mean_importance is a scalar representing the average routing weight
        # received per expert over the batch. For a perfectly balanced router
        # with B samples and N_r experts: mean_importance = B / N_r.
        # importance shape: (N_r,)
        # mean_importance shape: scalar
        mean_importance: torch.Tensor = importance.mean()

        # ----------------------------------------------------------------
        # Step 2: Compute standard deviation of importance scores
        # ----------------------------------------------------------------
        # std_importance measures the spread of routing weights across experts.
        # A uniform distribution gives std=0; a collapsed distribution
        # (all weight on one expert) gives std close to mean * sqrt(N_r - 1).
        #
        # unbiased=True (default in PyTorch): uses Bessel's correction
        # (divides by N_r - 1 instead of N_r). With N_r=16, the difference
        # from population std is small (factor of sqrt(16/15) ≈ 1.03).
        # Using the default is consistent with standard statistical practice.
        #
        # Edge case: if importance has only one element (N_r=1), std() returns
        # NaN with unbiased=True. Guard with unbiased=False for N_r=1.
        if importance.numel() > 1:
            std_importance: torch.Tensor = importance.std(unbiased=True)
        else:
            # N_r=1: CV is undefined (only one expert). Return zero loss.
            std_importance = torch.zeros_like(mean_importance)

        # ----------------------------------------------------------------
        # Step 3: Compute coefficient of variation (CV)
        # ----------------------------------------------------------------
        # CV = std / mean — normalized measure of dispersion.
        # Adding epsilon to the denominator prevents division by zero when:
        #   - All importance scores are exactly zero (e.g., at initialization
        #     before any forward passes have been made)
        #   - The router produces exactly uniform weights (mean is non-zero
        #     but this guard is a safety measure)
        # cv shape: scalar
        cv: torch.Tensor = std_importance / (mean_importance + self.epsilon)

        # ----------------------------------------------------------------
        # Step 4: Compute scaled squared CV
        # ----------------------------------------------------------------
        # L_balance = w_bal · CV²
        # Squaring makes the loss more sensitive to large imbalances and
        # provides smoother gradients near the balanced state (CV=0).
        # The paper explicitly specifies CV² (not CV) in Section 4.
        balance_loss: torch.Tensor = self.weight * cv ** 2

        return balance_loss
