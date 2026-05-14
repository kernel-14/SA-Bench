"""
Sparse Upcycling for OLMoE

Implements sparse upcycling from Komatsuzaki et al. (2022):
"Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints"

The process:
1. Clone the dense MLP for each desired expert to constitute MoE layers
2. Add a newly initialized router in front of each MoE layer
3. Continue pretraining with the new model

From Section 4.1.5 of the paper:
- OLMo-1B (0724) is upcycled at 2T tokens into an MoE with 8 total experts, 2 activated
- After 500B tokens, an MoE trained from scratch catches up with the upcycled model
- At ~600B tokens, the MoE from scratch starts outperforming the upcycled MoE
- This is much faster than the 120% compute budget reported in Komatsuzaki et al.
- Likely because the dense model was already significantly overtrained

Key limitation: The upcycled MoE is constrained by hyperparameters of the dense model
(e.g., no QK-Norm, normal initialization instead of truncated normal).
"""

import torch
import torch.nn as nn
from typing import Optional
import copy
import logging

logger = logging.getLogger(__name__)


def upcycle_dense_to_moe(
    dense_model,
    moe_config,
    noise_std: Optional[float] = None,
) -> "OLMoEForCausalLM":
    """
    Convert a dense language model to a Mixture-of-Experts model.

    Steps:
    1. For each transformer layer, clone the dense FFN for each expert
    2. Initialize a new router for each MoE layer
    3. Keep all other parameters (attention, norms, embeddings) from the dense model

    Args:
        dense_model: Dense language model (e.g., OLMo-1B)
        moe_config: Configuration for the MoE model
        noise_std: If provided, add Gaussian noise to 50% of each MLP's weights
                   (from Qwen2 upcycling approach, but paper finds it doesn't help)

    Returns:
        OLMoE model initialized from the dense model
    """
    from src.model import OLMoEForCausalLM, OLMoEConfig

    logger.info(f"Upcycling dense model to MoE with {moe_config.num_experts} experts")
    logger.info(f"Activated experts per token: {moe_config.num_experts_per_tok}")

    # Create new MoE model
    moe_model = OLMoEForCausalLM(moe_config)

    # Copy embeddings
    moe_model.model.embed_tokens.weight.data.copy_(
        dense_model.model.embed_tokens.weight.data
    )

    # Copy final norm
    if hasattr(dense_model.model, 'norm'):
        moe_model.model.norm.weight.data.copy_(
            dense_model.model.norm.weight.data
        )

    # Copy LM head
    if not moe_config.tie_word_embeddings:
        moe_model.lm_head.weight.data.copy_(
            dense_model.lm_head.weight.data
        )

    # Copy each transformer layer
    for layer_idx, (moe_layer, dense_layer) in enumerate(
        zip(moe_model.model.layers, dense_model.model.layers)
    ):
        # Copy attention weights
        moe_layer.self_attn.q_proj.weight.data.copy_(
            dense_layer.self_attn.q_proj.weight.data
        )
        moe_layer.self_attn.k_proj.weight.data.copy_(
            dense_layer.self_attn.k_proj.weight.data
        )
        moe_layer.self_attn.v_proj.weight.data.copy_(
            dense_layer.self_attn.v_proj.weight.data
        )
        moe_layer.self_attn.o_proj.weight.data.copy_(
            dense_layer.self_attn.o_proj.weight.data
        )

        # Copy layer norms
        moe_layer.input_layernorm.weight.data.copy_(
            dense_layer.input_layernorm.weight.data
        )
        moe_layer.post_attention_layernorm.weight.data.copy_(
            dense_layer.post_attention_layernorm.weight.data
        )

        # Clone dense FFN for each expert
        dense_ffn = dense_layer.mlp if hasattr(dense_layer, 'mlp') else dense_layer.ffn

        for expert_idx in range(moe_config.num_experts):
            expert = moe_layer.moe.experts[expert_idx]

            # Copy gate, up, down projections from dense FFN
            # Note: dense FFN may have different dimensions, so we need to handle this
            _copy_ffn_to_expert(dense_ffn, expert, noise_std=noise_std)

        # Router is newly initialized (already done in __init__)
        logger.debug(f"Layer {layer_idx}: cloned FFN to {moe_config.num_experts} experts")

    logger.info("Upcycling complete!")
    return moe_model


def _copy_ffn_to_expert(dense_ffn, expert, noise_std: Optional[float] = None):
    """
    Copy dense FFN weights to an expert, handling dimension differences.

    If the dense FFN has a larger dimension than the expert (which is typical
    since experts are fine-grained), we take a slice of the dense FFN weights.

    Args:
        dense_ffn: Dense FFN module
        expert: Expert module to copy weights into
        noise_std: If provided, add Gaussian noise to 50% of weights
    """
    expert_ffn_dim = expert.gate_proj.out_features

    # Get dense FFN weights
    if hasattr(dense_ffn, 'gate_proj'):
        dense_gate = dense_ffn.gate_proj.weight.data
        dense_up = dense_ffn.up_proj.weight.data
        dense_down = dense_ffn.down_proj.weight.data
    else:
        # Handle different FFN naming conventions
        raise ValueError(f"Unknown FFN structure: {type(dense_ffn)}")

    dense_ffn_dim = dense_gate.shape[0]

    if dense_ffn_dim >= expert_ffn_dim:
        # Take a slice of the dense FFN
        # For fine-grained experts, each expert gets a portion of the dense FFN
        expert.gate_proj.weight.data.copy_(dense_gate[:expert_ffn_dim, :])
        expert.up_proj.weight.data.copy_(dense_up[:expert_ffn_dim, :])
        expert.down_proj.weight.data.copy_(dense_down[:, :expert_ffn_dim])
    else:
        # Dense FFN is smaller than expert (unusual case)
        # Repeat and truncate
        repeats = (expert_ffn_dim + dense_ffn_dim - 1) // dense_ffn_dim
        expert.gate_proj.weight.data.copy_(
            dense_gate.repeat(repeats, 1)[:expert_ffn_dim, :]
        )
        expert.up_proj.weight.data.copy_(
            dense_up.repeat(repeats, 1)[:expert_ffn_dim, :]
        )
        expert.down_proj.weight.data.copy_(
            dense_down.repeat(1, repeats)[:, :expert_ffn_dim]
        )

    # Optionally add noise (from Qwen2 approach, but paper finds it doesn't help)
    if noise_std is not None:
        _add_noise_to_expert(expert, noise_std)


def _add_noise_to_expert(expert, noise_std: float):
    """
    Add Gaussian noise to 50% of each MLP's weights.

    From the paper's Appendix F (noise upcycling experiment):
    "randomly replacing 50% of each MLP with numbers drawn from a normal
    distribution with a standard deviation of 0.02"

    The paper finds this doesn't lead to better performance.
    """
    for param in expert.parameters():
        # Create a mask for 50% of the weights
        mask = torch.rand_like(param.data) < 0.5
        noise = torch.randn_like(param.data) * noise_std
        param.data[mask] = noise[mask]


def create_upcycled_olmoe(
    dense_checkpoint_path: str,
    num_experts: int = 8,
    num_experts_per_tok: int = 2,
    noise_std: Optional[float] = None,
) -> "OLMoEForCausalLM":
    """
    Create an OLMoE model by upcycling a dense OLMo checkpoint.

    This replicates the experiment in Section 4.1.5:
    - Upcycles OLMo-1B (0724) at 2T tokens
    - Creates MoE with 8 total experts, 2 activated
    - Trains for additional 610B tokens

    Args:
        dense_checkpoint_path: Path to dense model checkpoint
        num_experts: Number of experts in the MoE (default 8 for the paper's experiment)
        num_experts_per_tok: Number of activated experts per token (default 2)
        noise_std: If provided, add noise to expert weights

    Returns:
        Upcycled OLMoE model
    """
    from src.model import OLMoEConfig

    logger.info(f"Loading dense checkpoint from {dense_checkpoint_path}")
    dense_state = torch.load(dense_checkpoint_path, map_location="cpu")

    # Create dense model (OLMo-1B configuration)
    # Note: OLMo-1B uses non-parametric LayerNorm and normal init
    # These constraints carry over to the upcycled model
    from src.model import OLMoEForCausalLM

    # Dense model config (OLMo-1B)
    dense_config = OLMoEConfig(
        hidden_size=2048,
        num_hidden_layers=16,
        num_attention_heads=16,
        vocab_size=50304,
        num_experts=1,  # Dense model has 1 "expert" (the FFN)
        num_experts_per_tok=1,
        expert_ffn_dim=8192,  # Dense FFN dimension
        use_qk_norm=False,  # OLMo-1B doesn't use QK-Norm
        use_load_balancing_loss=False,
        use_router_z_loss=False,
    )

    # MoE config for upcycled model
    # Expert FFN dim = dense_ffn_dim / num_experts (to maintain same compute)
    expert_ffn_dim = dense_config.expert_ffn_dim // num_experts
    moe_config = OLMoEConfig(
        hidden_size=2048,
        num_hidden_layers=16,
        num_attention_heads=16,
        vocab_size=50304,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        expert_ffn_dim=expert_ffn_dim,
        use_qk_norm=False,  # Constrained by dense model
        use_load_balancing_loss=True,
        use_router_z_loss=True,
    )

    # Load dense model
    dense_model = OLMoEForCausalLM(dense_config)
    dense_model.load_state_dict(dense_state["model_state_dict"])

    # Upcycle to MoE
    moe_model = upcycle_dense_to_moe(dense_model, moe_config, noise_std=noise_std)

    return moe_model
