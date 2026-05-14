```python
## models/moe_model.py
"""Mixture-of-Experts (MoE) transformer model for 15A2B parameter experiments.

This module implements MoELayer and MoEModel, used for the primary ablation
experiments in the paper "Gated Attention for Large Language Models: Non-linearity,
Sparsity, and Attention-Sink-Free" (Tables 1, 3, 4, 6).

The MoE-15A2B configuration:
    - 15B total parameters, ~2.54B activated per forward pass
    - 128 total experts with top-8 softmax gating (fine-grained, DeepSeekMoE style)
    - Global-batch load balance loss (Qiu et al., 2025)
    - Z-loss for router stability (Zoph et al., 2022)
    - GQA: q=32 query heads, k=4 KV heads, d_k=128

Architecture:
    MoEModel mirrors DenseModel but replaces each TransformerBlock's FeedForward
    sublayer with a MoELayer. The MoELayer routes each token to top-K experts,
    computes weighted expert outputs, and stores router logits for auxiliary loss
    computation. MoEModel accumulates lbl_loss and z_loss across all layers and
    returns them alongside the language model logits.

Config values used (from config.yaml, moe_15a2b_400b section):
    model.type: Must be 'moe'
    model.num_layers: 24 (assumed)
    model.d_model: 4096 (assumed)
    model.num_heads: 32
    model.num_kv_heads: 4
    model.d_k: 128
    model.ffn_dim: 4096 (per-expert FFN intermediate dimension)
    model.vocab_size: 32000
    model.max_seq_len: 4096
    model.sandwich_norm: false
    moe.num_experts: 128
    moe.top_k: 8
    moe.use_lbl_loss: true
    moe.use_z_loss: true
    moe.z_loss_coeff: 1.0e-4
    moe.lbl_loss_coeff: 1.0e-2
    gate.position: 'G1' (or other variants for ablations)
    gate.granularity: 'elementwise'
    gate.head_specific: true
    gate.gate_type: 'multiplicative'
    gate.activation: 'sigmoid'
    rope.base: 10000.0
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gate_module import GateModule
from models.transformer_block import FeedForward, TransformerBlock
from training.moe_losses import compute_load_balance_loss, compute_z_loss


class MoELayer(nn.Module):
    """Mixture-of-Experts feed-forward layer replacing the dense FFN sublayer.

    Implements fine-grained expert MoE (DeepSeekMoE style, Dai et al. 2024).
    Each token is routed to the top-K experts via softmax gating. The output
    is a weighted sum of the selected experts' outputs, with weights renormalized
    among the top-K selected experts.

    The MoE router uses softmax gating (not sigmoid) — this is distinct from
    the attention gate which uses sigmoid. Paper Sec 3.1: "top-8 softmax gating".

    Router logits are stored as self.last_router_logits after each forward pass,
    enabling MoEModel to collect them for auxiliary loss computation without
    requiring explicit hooks or modifying the TransformerBlock interface.

    Attributes:
        num_experts: Total number of expert networks (128 for MoE-15A2B).
        top_k: Number of experts selected per token (8 for MoE-15A2B).
        d_model: Hidden dimension of the model (4096 for MoE-15A2B).
        d_ffn_per_expert: FFN intermediate dimension per expert (4096 assumed).
        use_lbl_loss: Whether to compute load balance loss.
        use_z_loss: Whether to compute Z-loss.
        z_loss_coeff: Scaling coefficient for Z-loss (1e-4).
        lbl_loss_coeff: Scaling coefficient for load balance loss (1e-2).
        experts: ModuleList of FeedForward expert networks.
        router: Linear projection from d_model to num_experts (no bias).
        last_router_logits: Stored router logits from the most recent forward
            pass. Shape [batch*seq, num_experts]. Used by MoEModel to compute
            auxiliary losses after each layer's forward pass.
    """

    def __init__(
        self,
        num_experts: int = 128,
        top_k: int = 8,
        d_model: int = 4096,
        d_ffn_per_expert: int = 4096,
        use_lbl_loss: bool = True,
        use_z_loss: bool = True,
        z_loss_coeff: float = 1e-4,
        lbl_loss_coeff: float = 1e-2,
    ) -> None:
        """Initialize MoELayer with expert networks and router.

        Args:
            num_experts: Total number of expert networks. From moe.num_experts.
                Default 128 (paper Sec 3.1: "128 total experts").
            top_k: Number of experts selected per token. From moe.top_k.
                Default 8 (paper Sec 3.1: "top-8 softmax gating").
            d_model: Hidden dimension of the model. From model.d_model.
                Default 4096 (assumed for MoE-15A2B).
            d_ffn_per_expert: FFN intermediate dimension per expert.
                From model.ffn_dim. Default 4096 (assumed fine-grained expert width).
                In DeepSeekMoE style, this is smaller than a standard FFN width
                but more experts are used to compensate.
            use_lbl_loss: Whether to compute load balance loss. From moe.use_lbl_loss.
                Default True (paper Sec 3.1: "global-batch LBL").
            use_z_loss: Whether to compute Z-loss. From moe.use_z_loss.
                Default True (paper Sec 3.1: "Z-loss").
            z_loss_coeff: Scaling coefficient for Z-loss. From moe.z_loss_coeff.
                Default 1e-4 (Zoph et al., 2022 standard value).
            lbl_loss_coeff: Scaling coefficient for load balance loss.
                From moe.lbl_loss_coeff. Default 1e-2 (standard LBL coefficient).

        Raises:
            ValueError: If top_k > num_experts.
        """
        super().__init__()

        if top_k > num_experts:
            raise ValueError(
                f"top_k ({top_k}) cannot exceed num_experts ({num_experts})."
            )

        # Store configuration as instance attributes
        self.num_experts: int = num_experts
        self.top_k: int = top_k
        self.d_model: int = d_model
        self.d_ffn_per_expert: int = d_ffn_per_expert
        self.use_lbl_loss: bool = use_lbl_loss
        self.use_z_loss: bool = use_z_loss
        self.z_loss_coeff: float = z_loss_coeff
        self.lbl_loss_coeff: float = lbl_loss_coeff

        # -----------------------------------------------------------------------
        # Expert networks: 128 SwiGLU FeedForward modules
        # Each expert is a full FeedForward(d_model, d_ffn_per_expert) instance.
        # Paper Sec 3.1: "fine-grained experts (Dai et al., 2024)".
        # In DeepSeekMoE, each expert has a smaller FFN width than a standard
        # dense FFN, but more experts are used. Here d_ffn_per_expert=4096.
        # -----------------------------------------------------------------------
        self.experts: nn.ModuleList = nn.ModuleList(
            [
                FeedForward(d_model=d_model, d_ffn=d_ffn_per_expert)
                for _ in range(num_experts)
            ]
        )

        # -----------------------------------------------------------------------
        # Router: linear projection from d_model to num_experts
        # No bias — standard for MoE routers.
        # Paper Sec 3.1: "top-8 softmax gating" — router outputs are passed
        # through softmax (not sigmoid) to get routing probabilities.
        # -----------------------------------------------------------------------
        self.router: nn.Linear = nn.Linear(d_model, num_experts, bias=False)

        # -----------------------------------------------------------------------
        # Storage for router logits (populated during forward pass)
        # MoEModel reads this after each layer to compute auxiliary losses.
        # Shape: [batch*seq, num_experts] after forward pass.
        # -----------------------------------------------------------------------
        self.last_router_logits: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route tokens to top-K experts and compute weighted output.

        Implements the MoE forward pass:
            1. Flatten input to [T, d_model] where T = batch * seq
            2. Compute router logits via linear projection
            3. Apply softmax to get routing probabilities
            4. Select top-K experts per token
            5. Renormalize top-K weights to sum to 1
            6. Compute weighted sum of selected expert outputs
            7. Reshape output back to [batch, seq, d_model]
            8. Store router logits for auxiliary loss computation

        The routing uses softmax gating (not sigmoid), consistent with the
        paper's description: "top-8 softmax gating" (Sec 3.1). This is
        distinct from the attention gate which uses sigmoid.

        Args:
            x: Input tensor from the pre-FFN RMSNorm in TransformerBlock.
                Shape [batch, seq, d_model].

        Returns:
            Output tensor of shape [batch, seq, d_model], representing the
            weighted sum of the top-K selected experts' outputs for each token.

        Note:
            Router logits are stored in self.last_router_logits (detached from
            the computation graph for f_i computation, but the original logits
            with gradients are also stored for p_i gradient flow in LBL loss).
            MoEModel reads self.last_router_logits after each layer's forward
            pass to accumulate auxiliary losses.
        """
        batch_size: int = x.shape[0]
        seq_len: int = x.shape[1]
        total_tokens: int = batch_size * seq_len

        # -----------------------------------------------------------------------
        # Step 1: Flatten to [T, d_model] for routing
        # -----------------------------------------------------------------------
        x_flat: torch.Tensor = x.view(total_tokens, self.d_model)

        # -----------------------------------------------------------------------
        # Step 2: Compute router logits
        # Shape: [T, num_experts]
        # -----------------------------------------------------------------------
        router_logits: torch.Tensor = self.router(x_flat)

        # -----------------------------------------------------------------------
        # Step 3: Store router logits for auxiliary loss computation
        # MoEModel reads this after each layer's forward pass.
        # We store the full logits (with gradient) so that p_i in LBL loss
        # can receive gradients through the softmax operation.
        # -----------------------------------------------------------------------
        self.last_router_logits = router_logits

        # -----------------------------------------------------------------------
        # Step 4: Compute routing probabilities via softmax
        # Paper Sec 3.1: "top-8 softmax gating" — softmax over expert dimension.
        # Using float32 for numerical stability in BF16 training contexts.
        # Shape: [T, num_experts]
        # -----------------------------------------------------------------------
        routing_weights: torch.Tensor = F.softmax(
            router_logits.float(), dim=-1
        ).to(x.dtype)

        # -----------------------------------------------------------------------
        # Step 5: Select top-K experts per token
        # top_k_weights: [T, top_k] — routing weights for selected experts
        # top_k_indices: [T, top_k] — expert indices for selected experts
        # -----------------------------------------------------------------------
        top_k_weights: torch.Tensor
        top_k_indices: torch.Tensor
        top_k_weights, top_k_indices = torch.topk(
            routing_weights, k=self.top_k, dim=-1
        )

        # -----------------------------------------------------------------------
        # Step 6: Renormalize top-K weights to sum to 1
        # Standard for fine-grained expert MoE (DeepSeekMoE style).
        # Ensures the output is a proper convex combination of expert outputs.
        # Shape: [T, top_k]
        # -----------------------------------------------------------------------
        top_k_weights = top_k_weights / (
            top_k_weights.sum(dim=-1, keepdim=True) + 1e-9
        )

        # -----------------------------------------------------------------------
        # Step 7: Compute weighted sum of expert outputs
        # Strategy: iterate over top_k positions, group tokens by expert,
        # run each expert on its assigned tokens, accumulate weighted outputs.
        #
        # This approach avoids running all 128 experts for every token.
        # For each of the top_k positions, we identify which expert is selected
        # and process all tokens assigned to that expert in a single batch.
        #
        # Complexity: O(top_k * num_experts) iterations in the worst case,
        # but in practice each expert handles T/num_experts tokens on average.
        # -----------------------------------------------------------------------
        # Initialize output buffer with zeros
        output: torch.Tensor = torch.zeros(
            total_tokens,
            self.d_model,
            device=x.device,
            dtype=x.dtype,
        )

        # Iterate over each of the top_k expert slots
        for k_idx in range(self.top_k):
            # Expert index selected at position k_idx for each token
            # Shape: [T]
            expert_indices_k: torch.Tensor = top_k_indices[:, k_idx]

            # Routing weight for position k_idx for each token
            # Shape: [T, 1] for broadcasting with expert output [T_expert, d_model]
            weights_k: torch.Tensor = top_k_weights[:, k_idx]

            # Group tokens by expert to enable batched expert computation
            for expert_id in range(self.num_experts):
                # Boolean mask: which tokens are routed to this expert at position k_idx
                # Shape: [T]
                token_mask: torch.Tensor = expert_indices_k == expert_id

                # Skip if no tokens are routed to this expert at this position
                if not token_mask.any():
                    continue

                # Gather tokens assigned to this expert
                # Shape: [num_assigned_tokens, d_model]
                expert_input: torch.Tensor = x_flat[token_mask]

                # Run the expert network on assigned tokens
                # Shape: [num_assigned_tokens, d_model]
                expert_output: torch.Tensor = self.experts[expert_id](expert_input)

                # Scale by routing weight and accumulate into output buffer
                # weights_k[token_mask] shape: [num_assigned_tokens]
                # Unsqueeze for broadcasting: [num_assigned_tokens, 1]
                output[token_mask] += (
                    weights_k[token_mask].unsqueeze(-1) * expert_output
                )

        # -----------------------------------------------------------------------
        # Step 8: Reshape output back to [batch, seq, d_model]
        # -----------------------------------------------------------------------
        output = output.view(batch_size, seq_len, self.d_model)

        return output

    def compute_load_balance_loss(
        self, router_logits: torch.Tensor
    ) -> torch.Tensor:
        """Compute global-batch load balance loss for this MoE layer.

        Delegates to training.moe_losses.compute_load_balance_loss with the
        layer's configured top_k and lbl_loss_coeff.

        Paper Sec 3.1: "global-batch LBL (Qiu et al., 2025)".
        Config moe.lbl_loss_coeff: 1.0e-2.

        Args:
            router_logits: Raw router logits from this layer's forward pass.
                Shape [batch*seq, num_experts]. Should be self.last_router_logits.

        Returns:
            Scalar tensor representing the load balance loss for this layer.
            Retains gradient connection through the softmax (p_i) computation.
        """
        return compute_load_balance_loss(
            router_logits=router_logits,
            top_k=self.top_k,
            coeff=self.lbl_loss_coeff,
        )

    def compute_z_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """Compute Z-loss to penalize large router logits for this MoE layer.

        Delegates to training.moe_losses.compute_z_loss with the layer's
        configured z_loss_coeff.

        Paper Sec 3.1: "Z-loss (Zoph et al., 2022)".
        Config moe.z_loss_coeff: 1.0e-4.

        Args:
            router_logits: Raw router logits from this layer's forward pass.
                Shape [batch*seq, num_experts]. Should be self.last_router_logits.

        Returns:
            Scalar tensor representing the Z-loss for this layer.
            Retains gradient connection through the logsumexp operation.
        """
        return compute_z_loss(
            router_logits=router_logits,
            coeff=self.z_loss_coeff,
        )

    def extra_repr(self) -> str:
        """Return human-readable module description for print(model).

        Returns:
            String summarizing the MoE layer configuration.
        """
        return (
            f"num_experts={self.num_experts}, "
            f"top_k={self.top_k}, "
            f"d_model={self.d_model}, "
            f"d_ffn_per_expert={self.d_ffn_per_expert}, "
            f"use_lbl_loss={self.use_lbl_loss}, "
            f"use_z_loss={self.use_z_loss}, "
            f"z_loss_coeff={self.z_loss_coeff}, "
            f"lbl_loss_coeff={self.lbl_loss_coeff}"
        )


class MoEModel(nn.Module):
    """MoE transformer language model for 15A2B parameter ablation experiments.

    Implements the MoE-15A2B model used for Tables 1, 3, 4, and 6 in the paper.
    Structurally mirrors DenseModel but replaces each TransformerBlock's FeedForward
    sublayer with a MoELayer (128 experts, top-8 routing).

    The model accumulates load balance loss and Z-loss across all MoE layers
    during the forward pass and returns them alongside the language model logits.
    The Trainer adds these auxiliary losses to the cross-entropy loss.

    Architecture:
        embedding → [TransformerBlock with MoELayer] × num_layers → RMSNorm → lm_head

    Each TransformerBlock contains:
        - GatedMultiHeadAttention (with configurable gate from gate.* config)
        - MoELayer (replacing the standard FeedForward)
        - Pre-norm RMSNorm layers
        - Optional sandwich normalization

    Attributes:
        config: The original configuration object (not mutated).
        num_layers: Number of transformer layers (24 assumed for MoE-15A2B).
        d_model: Hidden dimension (4096 assumed for MoE-15A2B).
        vocab_size: Vocabulary size (32000).
        num_experts: Total number of experts per MoE layer (128).
        top_k: Experts selected per token (8).
        use_lbl_loss: Whether to compute load balance loss.
        use_z_loss: Whether to compute Z-loss.
        embedding: Token embedding table, shape [vocab_size, d_model].
        layers: ModuleList of TransformerBlock instances with MoELayer FFN.
        norm: Final RMSNorm before lm_head.
        lm_head: Linear projection to vocabulary logits, weight-tied to embedding.
    """

    def __init__(self, config: object) -> None:
        """Initialize MoEModel from configuration.

        Constructs the full MoE model by:
            1. Building TransformerBlock instances (with GatedMultiHeadAttention)
            2. Replacing each block's FeedForward with a MoELayer
            3. Setting up embedding, final norm, and lm_head with weight tying

        Args:
            config: Configuration object (OmegaConf DictConfig or compatible).
                Must expose model.*, gate.*, rope.*, and moe.* fields as
                described in the module docstring.

        Raises:
            ValueError: If model.type is not 'moe'.
        """
        super().__init__()

        # Validate model type
        model_type: str = str(getattr(config.model, "type", "moe"))
        if model_type != "moe":
            raise ValueError(
                f"MoEModel requires config.model.type='moe', got '{model_type}'"
            )

        # Store original config (never mutated)
        self.config = config

        # -----------------------------------------------------------------------
        # Extract model dimensions from config
        # -----------------------------------------------------------------------
        self.num_layers: int = int(config.model.num_layers)
        self.d_model: int = int(config.model.d_model)
        self.vocab_size: int = int(config.model.vocab_size)
        ffn_dim: int = int(config.model.ffn_dim)  # per-expert FFN intermediate dim

        # -----------------------------------------------------------------------
        # Extract MoE configuration from config.moe.*
        # -----------------------------------------------------------------------
        moe_config = getattr(config, "moe", None)

        self.num_experts: int = int(getattr(moe_config, "num_experts", 128))
        self.top_k: int = int(getattr(moe_config, "top_k", 8))
        self.use_lbl_loss: bool = bool(getattr(moe_config, "use_lbl_loss", True))
        self.use_z_loss: bool = bool(getattr(moe_config, "use_z_loss", True))
        z_loss_coeff: float = float(getattr(moe_config, "z_loss_coeff", 1e-4))
        lbl_loss_coeff: float = float(getattr(moe_config, "lbl_loss_coeff", 1e-2))

        # -----------------------------------------------------------------------
        # Token embedding
        # Shape: [vocab_size, d_model]
        # Weight-tied to lm_head below.
        # -----------------------------------------------------------------------
        self.embedding: nn.Embedding = nn.Embedding(self.vocab_size, self.d_model)

        # -----------------------------------------------------------------------
        # Transformer layers with MoE FFN
        # Strategy: build TransformerBlock (which creates GatedMultiHeadAttention
        # + FeedForward), then replace block.ffn with MoELayer.
        # This keeps TransformerBlock unchanged and cleanly injects MoE.
        # -----------------------------------------------------------------------
        layers: List[TransformerBlock] = []
        for i in range(self.num_layers):
            # Build standard TransformerBlock (creates attention + dense FFN)
            block: TransformerBlock = TransformerBlock(config, layer_idx=i)

            # Replace the dense FeedForward with a MoELayer
            # Paper Sec 3.1: "128 total experts with top-8 softmax gating,
            # fine-grained experts (Dai et al., 2024)"
            block.ffn = MoELayer(
                num_experts=self.num_experts,
                top_k=self.top_k,
                d_model=self.d_model,
                d_ffn_per_expert=ffn_dim,
                use_lbl_loss=self.use_lbl_loss,
                use_z_loss=self.use_z_loss,
                z_loss_coeff=z_loss_coeff,
                lbl_loss_coeff=lbl_loss_coeff,
            )
            layers.append(block)

        self.layers: nn.ModuleList = nn.ModuleList(layers)

        # -----------------------------------------------------------------------
        # Final normalization and language model head
        # -----------------------------------------------------------------------
        self.norm: nn.RMSNorm = nn.RMSNorm(self.d_model)
        self.lm_head: nn.Linear = nn.Linear(self.d_model, self.vocab_size, bias=False)

        # -----------------------------------------------------------------------
        # Weight tying: lm_head.weight = embedding.weight
        # Standard for modern LLMs. Reduces parameter count by vocab_size * d_model.
        # -----------------------------------------------------------------------
        self.lm_head.weight = self.embedding.weight

        # -----------------------------------------------------------------------
        # Weight initialization
        # -----------------------------------------------------------------------
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights following standard LLM conventions.

        Applies:
            - Embedding: normal(0, 0.02)
            - Linear layers: normal(0, 0.02)
            - Output projections (W_O, down_proj): scaled by 1/sqrt(2 * num_layers)
              to prevent residual stream growth (GPT-2 style initialization)
            - Router: normal(0, 0.02) (same as other linear layers)
            - RMSNorm weights: 1.0 (default, not overridden)
        """
        std: float = 0.02
        output_proj_std: float = std / math.sqrt(2.0 * self.num_layers)

        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Scale output projections to prevent residual stream growth
        for layer in self.layers:
            # Attention output projection W_O
            nn.init.normal_(
                layer.attention.W_O.weight,
                mean=0.0,
                std=output_proj_std,
            )
            # Expert down projections (each expert's down_proj)
            if isinstance(layer.ffn, MoELayer):
                for expert in layer.ffn.experts:
                    if hasattr(expert, "down_proj"):
                        nn.init.normal_(
                            expert.down_proj.weight,
                            mean=0.0,
                            std=output_proj_std,
                        )

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass through the MoE transformer model.

        Runs the full autoregressive forward pass, accumulating auxiliary losses
        (load balance loss and Z-loss) from each MoE layer. The auxiliary losses
        are returned alongside the language model logits for the Trainer to add
        to the cross-entropy loss.

        The forward pass manually handles the MoE layer interaction to collect
        router logits from each layer's MoELayer after the TransformerBlock
        forward pass. Since TransformerBlock.forward() calls block.ffn() and
        MoELayer.forward() stores router_logits in self.last_router_logits,
        we can collect them after each block's forward pass.

        Args:
            input_ids: Integer token indices, shape [batch, seq_len].
            mask: Optional additive causal attention mask.
                Shape [1, 1, seq_len, seq_len] or [batch, 1, seq_len, seq_len].
                Values: 0.0 for allowed positions, -inf for masked positions.
                If None, each attention layer builds the causal mask internally.
            position_ids: Token position indices for RoPE.
                Shape [batch, seq_len] or [seq_len].
                If None, defaults to sequential [0, 1, ..., seq_len-1].

        Returns:
            Tuple of:
                - logits: Vocabulary logits, shape [batch, seq_len, vocab_size].
                  Used for cross-entropy loss computation in Trainer._compute_loss.
                - aux_losses: Dict with keys:
                    - 'lbl_loss': Summed load balance loss across all MoE layers.
                      Scalar tensor. Zero if use_lbl_loss=False.
                    - 'z_loss': Summed Z-loss across all MoE layers.
                      Scalar tensor. Zero if use_z_loss=False.
                  The Trainer adds these to the main cross-entropy loss:
                  total_loss = ce_loss + aux_losses['lbl_loss'] + aux_losses['z_loss']
        """
        batch_size: int = input_ids.shape[0]
        seq_len: int = input_ids.shape[1]

        # Generate default position_ids if not provided
        if position_ids is None:
            position_ids = torch.arange(
                seq_len,
                device=input_ids.device,
                dtype=torch.long,
            ).unsqueeze(0).expand(batch_size, -1)

        # Step 1: Token embedding
        # [batch, seq_len] → [batch, seq_len, d_model]
        X: torch.Tensor = self.embedding(input_ids)

        # Initialize auxiliary loss accumulators
        # Using zero tensors on the correct device for proper gradient flow
        device: torch.device = input_ids.device
        total_lbl_loss: torch.Tensor = torch.zeros(1, device=device, dtype=X.dtype)
        total_z_loss: torch.Tensor = torch.zeros(1, device=device, dtype=X.dtype)

        # Step 2: Pass through all transformer layers
        # TransformerBlock.forward() calls block.ffn (which is MoELayer),
        # and MoELayer.forward() stores router_logits in self.last_router_logits.
        # We collect auxiliary losses after each block's forward pass.
        layer: TransformerBlock
        for layer in self.layers:
            # Run the full transformer block (attention + MoE FFN)
            X = layer(X, mask=mask, position_ids=position_ids)

            # Collect auxiliary losses from the MoE layer
            # MoELayer.last_router_logits is set during layer.ffn() call inside
            # TransformerBlock.forward() → self.ffn(ffn_input)
            if isinstance(layer.ffn, MoELayer):
                moe_layer: MoELayer = layer.ffn
                router_logits: Optional[torch.Tensor] = moe_layer.last_router_logits

                if router_logits is not None:
                    # Accumulate load balance loss
                    if self.use_lbl_loss:
                        total_lbl_loss = total_lbl_loss + moe_layer.compute_load_balance_loss(
                            router_logits
                        )

                    # Accumulate Z-loss
                    if self.use_z_loss:
                        total_z_loss = total_z_loss + moe_layer.compute_z_loss(
                            router_logits
                        )

        # Step 3: Final RMS