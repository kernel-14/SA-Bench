## training/moe_losses.py
"""Auxiliary loss functions for Mixture-of-Experts (MoE) training.

This module implements the two auxiliary losses required for stable MoE training
as described in the paper "Gated Attention for Large Language Models: Non-linearity,
Sparsity, and Attention-Sink-Free" (Sec 3.1):

    1. Z-loss (Zoph et al., 2022): Penalizes large router logits to prevent
       the softmax from becoming too peaky, improving training stability.

    2. Load Balance Loss (LBL, Qiu et al., 2025): Encourages uniform expert
       utilization across the global batch, preventing expert collapse.

Both losses are applied to the MoE-15A2B model (128 experts, top-8 routing)
during the Table 1/3/4/6 ablation experiments. They are differentiable and
can be directly added to the main cross-entropy loss in the training loop.

Config values used (from config.yaml, moe_15a2b_400b section):
    moe.num_experts: 128 (total experts)
    moe.top_k: 8 (experts selected per token)
    moe.z_loss_coeff: 1.0e-4 (Z-loss scaling coefficient)
    moe.lbl_loss_coeff: 1.0e-2 (load balance loss scaling coefficient)
    hardware.distributed: 'fsdp' (enables global-batch all-reduce for f_i)

Usage in MoELayer.forward():
    z_loss = compute_z_loss(router_logits, coeff=config.moe.z_loss_coeff)
    lbl_loss = compute_load_balance_loss(
        router_logits, top_k=config.moe.top_k, coeff=config.moe.lbl_loss_coeff
    )

Usage in Trainer._compute_loss():
    total_loss = ce_loss + aux_losses['z_loss'] + aux_losses['lbl_loss']
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F


def compute_z_loss(
    router_logits: torch.Tensor,
    coeff: float = 1e-4,
) -> torch.Tensor:
    """Compute Z-loss to penalize large router logits (Zoph et al., 2022).

    Z-loss encourages the router to keep logit magnitudes moderate by penalizing
    the log-sum-exp of the router logits. When logits are large, the softmax
    becomes near one-hot (peaky), which destabilizes training. This is especially
    important for the MoE-15A2B model trained with max_lr=2e-3 (config).

    Mathematical definition:
        L_z = coeff * mean_over_tokens( (log(sum_i(exp(router_logits_i))))^2 )
             = coeff * mean( logsumexp(router_logits, dim=-1)^2 )

    The log-sum-exp formulation is numerically stable and avoids overflow in
    BF16 training (paper Sec 4.3 mentions BF16 training; config precision: bf16).

    Args:
        router_logits: Raw (pre-softmax) router logits from the MoE router linear
            layer. Shape [batch_size * seq_len, num_experts]. For the MoE-15A2B
            model: num_experts=128 (config moe.num_experts).
            Must have requires_grad=True (automatic when coming from nn.Linear).
        coeff: Scaling coefficient for the Z-loss. Default 1e-4 matches
            config moe.z_loss_coeff: 1.0e-4 (Zoph et al., 2022 standard value).

    Returns:
        Scalar tensor representing the Z-loss value. Retains gradient connection
        to router_logits through the logsumexp operation, enabling router weight
        updates via backpropagation.

    Example:
        >>> router_logits = torch.randn(1024 * 4096, 128)  # batch*seq, experts
        >>> loss = compute_z_loss(router_logits, coeff=1e-4)
        >>> loss.shape
        torch.Size([])  # scalar
        >>> loss.item()  # typically small positive value
        0.0023...

    Note:
        The gradient of logsumexp(x)_i with respect to x_j is softmax(x)_j,
        so this loss effectively penalizes the router probabilities being
        concentrated on a few experts (high softmax values → high LSE → high loss).
    """
    # Step 1: Compute log-sum-exp per token using numerically stable implementation.
    # torch.logsumexp(x, dim=-1) = log(sum_i(exp(x_i))) computed stably.
    # Shape: [batch_size * seq_len]
    # This is equivalent to log(sum(exp(router_logits))) but avoids overflow
    # in BF16 by using the max-subtraction trick internally.
    lse: torch.Tensor = torch.logsumexp(router_logits, dim=-1)

    # Step 2: Square the log-sum-exp values per token.
    # Shape: [batch_size * seq_len]
    # Squaring ensures the loss is always non-negative and penalizes large
    # magnitudes more strongly (quadratic penalty).
    lse_squared: torch.Tensor = lse ** 2

    # Step 3: Average over all tokens in the batch.
    # Shape: scalar
    # Mean reduction ensures the loss magnitude is independent of batch size,
    # making the coefficient coeff consistent across different batch sizes.
    loss: torch.Tensor = lse_squared.mean()

    # Step 4: Scale by the coefficient.
    # Default coeff=1e-4 matches config moe.z_loss_coeff: 1.0e-4.
    # The coefficient controls the relative weight of Z-loss vs. cross-entropy.
    return coeff * loss


def compute_load_balance_loss(
    router_logits: torch.Tensor,
    top_k: int = 8,
    coeff: float = 1e-2,
) -> torch.Tensor:
    """Compute global-batch load balance loss (LBL) for MoE training (Qiu et al., 2025).

    Load balance loss encourages uniform expert utilization across the global batch,
    preventing expert collapse (where a small subset of experts handles all tokens).
    The "global-batch" variant computes the expert load fraction f_i across all
    GPUs in the distributed training setup, matching the paper's description:
    "global-batch LBL (Qiu et al., 2025)" (Sec 3.1).

    Mathematical definition:
        L_lbl = coeff * num_experts * sum_i( f_i * p_i )

    Where:
        f_i = fraction of tokens routed to expert i (discrete, non-differentiable)
              computed via top-k selection and averaged over the global batch
        p_i = mean routing probability for expert i (continuous, differentiable)
              computed as mean(softmax(router_logits), dim=0)

    The product f_i * p_i is large when expert i is both heavily used (high f_i)
    and has high routing probability (high p_i). Minimizing this product encourages
    the router to distribute tokens more uniformly across experts.

    Gradient flow:
        - f_i is non-differentiable (based on discrete top-k selection).
          This is intentional — f_i acts as a fixed coefficient in the loss.
        - p_i is differentiable through the softmax operation.
          Gradients flow: L_lbl → p_i → softmax → router_logits → router weights.

    Distributed training (global-batch):
        When torch.distributed is initialized (config hardware.distributed: 'fsdp'),
        the dispatch counts are all-reduced across all ranks before computing f_i.
        This ensures f_i reflects the true global expert utilization, not just the
        local micro-batch. Falls back to local computation for single-GPU debugging.

    Args:
        router_logits: Raw (pre-softmax) router logits from the MoE router linear
            layer. Shape [batch_size * seq_len, num_experts]. For the MoE-15A2B
            model: num_experts=128 (config moe.num_experts).
            Must have requires_grad=True for gradient flow through p_i.
        top_k: Number of experts selected per token. Default 8 matches
            config moe.top_k: 8 (paper Sec 3.1: "top-8 softmax gating").
        coeff: Scaling coefficient for the LBL loss. Default 1e-2 matches
            config moe.lbl_loss_coeff: 1.0e-2 (standard load balance coefficient).

    Returns:
        Scalar tensor representing the load balance loss value. Retains gradient
        connection to router_logits through the p_i (softmax) computation.

    Example:
        >>> router_logits = torch.randn(1024 * 4096, 128)  # batch*seq, experts
        >>> loss = compute_load_balance_loss(router_logits, top_k=8, coeff=1e-2)
        >>> loss.shape
        torch.Size([])  # scalar

    Note:
        The num_experts multiplier in the formula ensures the loss magnitude is
        independent of the number of experts. Without it, adding more experts
        would reduce the loss magnitude, making the coefficient non-transferable.

    References:
        Qiu et al. (2025): "Demons in the Detail: On Implementing Load Balancing
        Loss for Training Specialized Mixture-of-Expert Models"
        https://arxiv.org/abs/2501.11873
    """
    num_tokens: int = router_logits.shape[0]
    num_experts: int = router_logits.shape[1]

    # -------------------------------------------------------------------------
    # Step 1: Compute routing probabilities p_i (differentiable).
    # Apply softmax over the expert dimension to get per-token routing probs.
    # Shape: [num_tokens, num_experts]
    # Using float32 for softmax stability even in BF16 training contexts.
    # -------------------------------------------------------------------------
    routing_probs: torch.Tensor = F.softmax(
        router_logits.float(), dim=-1
    ).to(router_logits.dtype)

    # Mean routing probability per expert across the local batch.
    # Shape: [num_experts]
    # This is differentiable — gradients flow back through softmax to router_logits.
    p_i: torch.Tensor = routing_probs.mean(dim=0)

    # -------------------------------------------------------------------------
    # Step 2: Compute expert load fractions f_i (non-differentiable).
    # Determine which experts each token is routed to via top-k selection.
    # f_i represents the fraction of tokens assigned to expert i.
    # -------------------------------------------------------------------------
    # Get top-k expert indices for each token.
    # Shape: [num_tokens, top_k]
    # We use router_logits (not routing_probs) for top-k selection, consistent
    # with standard MoE implementations where routing is based on raw logits.
    top_k_indices: torch.Tensor = torch.topk(
        router_logits.detach(),  # detach: f_i is non-differentiable
        k=top_k,
        dim=-1,
    ).indices

    # Create a one-hot dispatch tensor indicating which experts each token uses.
    # Shape: [num_tokens, num_experts]
    # Each row has exactly top_k ones (one per selected expert).
    dispatch: torch.Tensor = torch.zeros(
        num_tokens,
        num_experts,
        device=router_logits.device,
        dtype=torch.float32,
    )
    dispatch.scatter_(
        dim=1,
        index=top_k_indices,
        value=1.0,
    )

    # -------------------------------------------------------------------------
    # Step 3: Global-batch all-reduce for f_i (distributed training).
    # Paper Sec 3.1: "global-batch LBL (Qiu et al., 2025)".
    # Config hardware.distributed: 'fsdp' — training uses FSDP across 8 GPUs.
    #
    # Sum dispatch counts across all ranks to get global expert utilization.
    # Then divide by total global tokens to get the global fraction f_i.
    # Falls back to local computation if distributed is not initialized
    # (e.g., single-GPU debugging or unit tests).
    # -------------------------------------------------------------------------
    # Sum dispatch counts per expert across the local batch.
    # Shape: [num_experts] — total tokens routed to each expert locally.
    dispatch_sum: torch.Tensor = dispatch.sum(dim=0)

    # Total tokens in the local batch (for normalization).
    local_total_tokens: torch.Tensor = torch.tensor(
        float(num_tokens),
        device=router_logits.device,
        dtype=torch.float32,
    )

    if dist.is_available() and dist.is_initialized():
        # All-reduce dispatch counts across all ranks (sum).
        # After all-reduce, dispatch_sum contains the global count per expert.
        dist.all_reduce(dispatch_sum, op=dist.ReduceOp.SUM)

        # All-reduce total token count to get global batch size.
        dist.all_reduce(local_total_tokens, op=dist.ReduceOp.SUM)

    # Compute global expert load fraction f_i.
    # Shape: [num_experts]
    # f_i = (global tokens routed to expert i) / (global total tokens)
    # Note: each token contributes top_k to the total dispatch count,
    # so sum(f_i) = top_k (not 1.0). This is consistent with the LBL formula.
    global_total_tokens: float = local_total_tokens.item()
    f_i: torch.Tensor = dispatch_sum / global_total_tokens

    # -------------------------------------------------------------------------
    # Step 4: Compute the load balance loss.
    # L_lbl = coeff * num_experts * sum_i(f_i * p_i)
    #
    # The num_experts multiplier ensures the loss magnitude is independent of
    # the number of experts (128 for MoE-15A2B, config moe.num_experts).
    # -------------------------------------------------------------------------
    # Element-wise product of load fraction and routing probability.
    # Shape: [num_experts]
    # f_i is detached (non-differentiable); gradient flows only through p_i.
    fi_pi: torch.Tensor = f_i.detach() * p_i

    # Sum over all experts and scale.
    # Shape: scalar
    loss: torch.Tensor = coeff * float(num_experts) * fi_pi.sum()

    return loss
