"""
This module provides a `LossCalculator` class responsible for computing
various loss functions used during the pretraining and adaptation phases
of the OLMoE model, including Cross-Entropy, Load Balancing Loss,
Router Z-loss, and Direct Preference Optimization (DPO) loss.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple

class LossCalculator:
    """
    Encapsulates the logic for computing different types of losses required
    during the pretraining and adaptation phases of OLMoE.
    """

    def __init__(self, lbl_weight: float = 0.01, rz_loss_weight: float = 0.001):
        """
        Initializes the LossCalculator with weights for auxiliary losses.

        Args:
            lbl_weight: The weighting factor (alpha) for the Load Balancing Loss.
                        Defaults to 0.01 as per config.yaml.
            rz_loss_weight: The weighting factor (beta) for the Router Z-loss.
                            Defaults to 0.001 as per config.yaml.
        """
        if not isinstance(lbl_weight, (int, float)) or lbl_weight < 0:
            raise ValueError("lbl_weight must be a non-negative float.")
        if not isinstance(rz_loss_weight, (int, float)) or rz_loss_weight < 0:
            raise ValueError("rz_loss_weight must be a non-negative float.")

        self._lbl_weight: float = lbl_weight
        self._rz_loss_weight: float = rz_loss_weight

    def calculate_pretrain_loss(
        self,
        ce_loss: torch.Tensor,
        lbl: torch.Tensor,
        rz_loss: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the total pretraining loss.

        The total pretraining loss is a weighted sum of Cross-Entropy Loss,
        Load Balancing Loss, and Router Z-loss:
        L = L_CE + alpha * L_LB + beta * L_RZ

        Args:
            ce_loss: The Cross-Entropy Loss (L_CE) computed by the OLMoEModel.
                     Expected to be a scalar tensor.
            lbl: The unweighted Load Balancing Loss (L_LB) component, typically
                 accumulated from MoE layers. Expected to be a scalar tensor.
            rz_loss: The unweighted Router Z-loss (L_RZ) component, typically
                     accumulated from MoE layers. Expected to be a scalar tensor.

        Returns:
            A scalar `torch.Tensor` representing the total pretraining loss.
        """
        if not isinstance(ce_loss, torch.Tensor) or ce_loss.dim() != 0:
            raise ValueError("ce_loss must be a scalar torch.Tensor.")
        if not isinstance(lbl, torch.Tensor) or lbl.dim() != 0:
            raise ValueError("lbl must be a scalar torch.Tensor.")
        if not isinstance(rz_loss, torch.Tensor) or rz_loss.dim() != 0:
            raise ValueError("rz_loss must be a scalar torch.Tensor.")

        # Weighted auxiliary losses
        weighted_lbl = self._lbl_weight * lbl
        weighted_rz_loss = self._rz_loss_weight * rz_loss

        # Total pretraining loss
        total_loss = ce_loss + weighted_lbl + weighted_rz_loss
        return total_loss

    def calculate_sft_loss(self, ce_loss: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss for Supervised Fine-Tuning (SFT).

        As per the paper (§4.3), auxiliary losses are NOT used during SFT.
        The SFT loss is simply the Cross-Entropy Loss.

        Args:
            ce_loss: The Cross-Entropy Loss computed by the OLMoEModel.
                     Expected to be a scalar tensor.

        Returns:
            A scalar `torch.Tensor` representing the SFT loss.
        """
        if not isinstance(ce_loss, torch.Tensor) or ce_loss.dim() != 0:
            raise ValueError("ce_loss must be a scalar torch.Tensor.")

        return ce_loss

    def calculate_dpo_loss(
        self,
        logits_chosen: torch.Tensor,
        logits_rejected: torch.Tensor,
        dpo_beta: float = 0.1,
    ) -> torch.Tensor:
        """
        Computes the Direct Preference Optimization (DPO) loss.

        The DPO loss is calculated based on the policy-reference log-likelihood ratios
        for chosen and rejected responses. Auxiliary losses are NOT used during DPO (§4.3).

        Args:
            logits_chosen: A tensor where each element represents the difference between
                           the policy model's log-likelihood and the reference model's
                           log-likelihood for a chosen response in a pair, i.e.,
                           log(p_policy(y_w|x)) - log(p_ref(y_w|x)).
                           Shape: `[batch_size]`.
            logits_rejected: A tensor similar to `logits_chosen`, but for a rejected response,
                             i.e., log(p_policy(y_l|x)) - log(p_ref(y_l|x)).
                             Shape: `[batch_size]`.
            dpo_beta: The beta parameter for DPO, controlling the strength of the preference.
                      Defaults to 0.1 as per config.yaml.

        Returns:
            A scalar `torch.Tensor` representing the averaged DPO loss over the batch.
        """
        if not isinstance(logits_chosen, torch.Tensor) or logits_chosen.dim() not in [0, 1]:
            raise ValueError("logits_chosen must be a scalar or 1D tensor.")
        if not isinstance(logits_rejected, torch.Tensor) or logits_rejected.dim() not in [0, 1]:
            raise ValueError("logits_rejected must be a scalar or 1D tensor.")
        if not isinstance(dpo_beta, (int, float)) or dpo_beta < 0:
            raise ValueError("dpo_beta must be a non-negative float.")
        if logits_chosen.shape != logits_rejected.shape:
            raise ValueError("logits_chosen and logits_rejected must have the same shape.")

        # The core DPO loss term is `beta * (log_ratio_chosen - log_ratio_rejected)`
        # Here, `logits_chosen` already represents `log_ratio_chosen` and `logits_rejected`
        # represents `log_ratio_rejected`.
        dpo_term = dpo_beta * (logits_chosen - logits_rejected)

        # Apply negative log-sigmoid
        # The loss for each sample is -log(sigmoid(dpo_term))
        # F.logsigmoid(x) = log(1 / (1 + exp(-x)))
        loss_per_sample = -F.logsigmoid(dpo_term)

        # Average over the batch
        return torch.mean(loss_per_sample)

