import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any, Tuple, Union

# Import Config and TransformerBlock
try:
    from config import Config
    from model.transformer_block import TransformerBlock
except ImportError:
    # Fallback for testing or if imports are structured differently
    print("Warning: Could not import Config or TransformerBlock. Using dummy classes/functions.")

    class Config:  # Dummy Config for isolated testing
        def __init__(self):
            self.model = self  # Self-reference for model config
            self.d_model = 2048
            self.vocab_size = 32000
            self.num_layers = 2
            self.type = "dense" # Default for dummy
            self.max_seq_len = 4096
            self.moe = self
            self.load_balancing_loss_coeff = 0.01

            self.evaluation = type('EvaluationConfig', (object,), {
                'attention_sink_analysis_enabled': True,
                'gating_score_analysis_enabled': True,
                'massive_activation_analysis_enabled': True,
            })()
            self.gating_enabled = True # For checking gating metrics

    class TransformerBlock(nn.Module): # Dummy TransformerBlock
        def __init__(self, config: Config):
            super().__init__()
            self.config = config
            self.proj = nn.Linear(config.model.d_model, config.model.d_model)
            self._last_attn_output_pre_residual = None
            self._last_ffn_output_pre_residual = None
            # Dummy GatedAttention to mock metric attributes
            class DummyGatedAttention:
                _last_gating_scores = None
                _last_attention_weights = None
            self.attn = DummyGatedAttention()

        def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            # Simplified dummy logic
            output = self.proj(hidden_states)
            moe_loss = None
            if self.config.model.type == "moe":
                moe_loss = torch.tensor(0.0)
            
            # Simulate massive activations collection for analysis
            if self.config.evaluation.massive_activation_analysis_enabled:
                self._last_attn_output_pre_residual = output.clone()
                self._last_ffn_output_pre_residual = output.clone() # Dummy
            
            return output, moe_loss


class GatedTransformer(nn.Module):
    """
    The main GatedTransformer model. This is the full LLM architecture,
    comprising word embeddings, a sequence of TransformerBlocks, a final layer normalization,
    and a language modeling head.
    """

    def __init__(self, config: Config):
        """
        Initializes the GatedTransformer model.

        Args:
            config: Configuration object containing model hyperparameters.
        """
        super().__init__()
        self.config = config

        self.d_model: int = config.model.d_model
        self.vocab_size: int = config.model.vocab_size
        self.num_layers: int = config.model.num_layers

        # Word embeddings layer
        self.word_embeddings: nn.Embedding = nn.Embedding(self.vocab_size, self.d_model)

        # Stack of Transformer blocks
        self.blocks: nn.ModuleList[TransformerBlock] = nn.ModuleList(
            [TransformerBlock(config) for _ in range(self.num_layers)]
        )

        # Final layer normalization
        self.norm_final: nn.LayerNorm = nn.LayerNorm(self.d_model, eps=1e-5)

        # Language modeling head
        self.lm_head: nn.Linear = nn.Linear(self.d_model, self.vocab_size, bias=False)

        # Weight tying: share weights between input embeddings and output projection
        self.lm_head.weight = self.word_embeddings.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Performs the forward pass of the GatedTransformer model.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len).
            attention_mask: Optional mask for attention (batch_size, 1, seq_len, seq_len).
                            Typically a causal mask, with large negative values for masked positions.
            labels: Optional target token IDs for language modeling loss (batch_size, seq_len).

        Returns:
            A tuple containing:
                - logits: Logits for token prediction (batch_size, seq_len, vocab_size).
                - loss: The computed language modeling loss (scalar), or None if labels are not provided.
                - total_moe_loss: The accumulated MoE regularization loss (scalar), or None if not an MoE model.
        """
        # Get word embeddings
        hidden_states = self.word_embeddings(input_ids)

        total_moe_loss: Optional[torch.Tensor] = None
        if self.config.model.type == "moe":
            total_moe_loss = torch.tensor(0.0, device=hidden_states.device)

        # Pass through Transformer blocks
        for block_idx, block in enumerate(self.blocks):
            block_output, moe_loss_component = block(hidden_states, attention_mask=attention_mask)
            hidden_states = block_output

            if self.config.model.type == "moe" and moe_loss_component is not None:
                if total_moe_loss is None: # Initialize if this is the first MoE block encountered
                    total_moe_loss = torch.tensor(0.0, device=hidden_states.device)
                total_moe_loss += moe_loss_component

        # Apply final layer normalization
        hidden_states = self.norm_final(hidden_states)

        # Language modeling head to get logits
        logits = self.lm_head(hidden_states)

        loss: Optional[torch.Tensor] = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            # Logits are (B, S, V), labels are (B, S)
            # We want to predict token i+1 from token i's output
            shifted_logits = logits[..., :-1, :].contiguous()
            shifted_labels = labels[..., 1:].contiguous()

            # Flatten for CrossEntropyLoss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shifted_logits.view(-1, self.vocab_size), shifted_labels.view(-1))

            if self.config.model.type == "moe" and total_moe_loss is not None:
                # Combine LM loss with MoE regularization loss
                # The moe.load_balancing_loss_coeff from config.yaml also implies scaling the total MoE loss.
                loss = loss + self.config.moe.load_balancing_loss_coeff * total_moe_loss

        return logits, loss, total_moe_loss

    def get_gating_metrics(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Performs a forward pass to collect detailed internal metrics for analysis.
        This method will put the model in evaluation mode and disable gradient computation.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len).
            attention_mask: Optional mask for attention (batch_size, 1, seq_len, seq_len).

        Returns:
            A dictionary containing lists of collected metrics (gating scores, attention weights,
            massive activations from FFN and attention sub-layers). All returned tensors are moved to CPU.
        """
        # Store original training state and set to eval mode
        original_training_state = self.training
        self.eval()

        metrics: Dict[str, List[torch.Tensor]] = {
            "gating_scores": [],
            "attention_weights": [],
            "massive_activations_ffn": [],
            "massive_activations_attn": [],
        }

        with torch.no_grad():
            hidden_states = self.word_embeddings(input_ids)

            for block_idx, block in enumerate(self.blocks):
                # Call the standard forward pass of the block.
                # The block and its attention sub-module are expected to store
                # relevant metrics as attributes if evaluation flags are enabled.
                block_output, _ = block(hidden_states, attention_mask=attention_mask)
                hidden_states = block_output

                # Collect metrics from the block and its sub-modules if enabled in config
                if self.config.gating_enabled and self.config.evaluation.gating_score_analysis_enabled:
                    if hasattr(block.attn, '_last_gating_scores') and block.attn._last_gating_scores is not None:
                        metrics["gating_scores"].append(block.attn._last_gating_scores)

                if self.config.evaluation.attention_sink_analysis_enabled:
                    if hasattr(block.attn, '_last_attention_weights') and block.attn._last_attention_weights is not None:
                        metrics["attention_weights"].append(block.attn._last_attention_weights)

                if self.config.evaluation.massive_activation_analysis_enabled:
                    if hasattr(block, '_last_ffn_output_pre_residual') and block._last_ffn_output_pre_residual is not None:
                        metrics["massive_activations_ffn"].append(block._last_ffn_output_pre_residual.detach().cpu())
                    if hasattr(block, '_last_attn_output_pre_residual') and block._last_attn_output_pre_residual is not None:
                        metrics["massive_activations_attn"].append(block._last_attn_output_pre_residual.detach().cpu())

        # Restore original training state
        self.train(original_training_state)

        return metrics

