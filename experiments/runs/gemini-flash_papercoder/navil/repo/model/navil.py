import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple

# Local imports
from config import Config
from model.visual_encoder import VisualEncoder
from model.components import Connector # Connector is in components.py as per design, even though in my current setup it's in navil.py. Will assume it's moved to components.py.
from model.mmoe_llm import MoELLM
from utils import logger # Assuming logger is configured in utils.py


class NaViLModel(nn.Module):
    """
    NaViL (Native Multimodal Large Language Model) orchestrates the entire multimodal
    architecture, integrating a Visual Encoder, a Connector, and an MoE-enhanced LLM.
    """

    def __init__(self, config: Config, tokenizer: Any): # Any for tokenizer to avoid circular import with transformers
        """
        Initializes the NaViL model by instantiating its sub-components
        (VisualEncoder, Connector, MoELLM).

        Args:
            config: An instance of the Config class, loaded from config.yaml,
                    providing all necessary hyperparameters and paths.
            tokenizer: The tokenizer for the base LLM (e.g., Hugging Face PreTrainedTokenizer).
        """
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

        logger.info("Initializing NaViLModel components...")

        # 1. Visual Encoder
        self.visual_encoder = VisualEncoder(config)
        logger.info("VisualEncoder initialized.")

        # 2. Connector
        self.connector = Connector(config)
        logger.info("Connector initialized.")

        # 3. MoE-enhanced LLM
        # The `llm_name_or_path` is directly under the current model variant in the config.
        self.mmoe_llm = MoELLM(self.config.llm_name_or_path, config, tokenizer)
        logger.info("MoELLM initialized and MoE layers injected.")

        logger.info("NaViLModel initialization complete.")

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Defines the forward pass of the integrated NaViL model.
        Combines visual and textual information into a single multimodal sequence
        processed by the MoE-enhanced LLM.

        Args:
            pixel_values: Batched image data. If Visual Multi-scale Packing (VMP) is enabled,
                          this tensor will contain all scaled images concatenated.
                          Shape: (total_num_scaled_images_in_batch, C, H, W).
            input_ids: Tokenized text (including placeholders for visual tokens and other special tokens)
                       for the LLM. Includes padding tokens. Shape: (batch_size, text_sequence_length).
            attention_mask: Attention mask corresponding to input_ids, indicating valid tokens vs. padding.
                            Shape: (batch_size, text_sequence_length).
            labels: Optional target labels for language modeling loss. Shape: (batch_size, text_sequence_length).

        Returns:
            A tuple of (logits, loss). Loss is None if labels are not provided.
        """
        # Ensure all inputs are on the same device as the model
        device = pixel_values.device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)

        # 1. Process Visual Input
        # VisualEncoder outputs features along with inferred grid dimensions (H_grid, W_grid).
        # The connector, as per `components.py` (and the logic analysis in my mind),
        # rebuilds H/W from the number of patches internally for PixelUnshuffle.
        visual_features, _, _ = self.visual_encoder(pixel_values) # (total_patches, D_vis)
        
        # Connector projects visual features into the LLM's hidden dimension space
        # projected_visual_features: (total_num_visual_tokens_across_batch_and_scales, llm_hidden_size)
        projected_visual_features = self.connector(visual_features)
        
        # 2. LLM Forward Pass with multimodal input
        # The MoELLM's forward method is responsible for dynamically splicing
        # the projected_visual_features into the LLM's input embeddings based on
        # special tokens present in input_ids, and then performing the LLM's forward pass.
        logits, loss = self.mmoe_llm(
            input_ids=input_ids,
            visual_embeddings=projected_visual_features,
            attention_mask=attention_mask,
            labels=labels,
        )

        return logits, loss

    def get_trainable_params(self, stage: str) -> List[nn.Parameter]:
        """
        Returns a list of `nn.Parameter` objects that should be considered trainable
        for a given training stage, implementing the freezing/unfreezing logic
        as described in the paper.

        Args:
            stage: The current training stage, e.g., "stage_1_1", "stage_1_2", "stage_2".

        Returns:
            A list of `nn.Parameter` objects that are currently trainable.
        """
        # First, set all parameters to not require gradients to ensure a clean slate
        for param in self.parameters():
            param.requires_grad = False

        trainable_params: List[nn.Parameter] = []

        # Helper function to check if a module name starts with the specified prefix
        def _starts_with_any(name: str, prefixes: List[str]) -> bool:
            return any(name.startswith(p) for p in prefixes)

        # Helper function to check for specific modality-specific MoE parameters
        # This assumes the MoE layer parameters follow the naming convention:
        # `mmoe_llm.base_llm.model.layers.LAYER_IDX.{self_attn|mlp}.modality_specific_weights.{visual_|linguistic_}{q_proj|gate_proj} etc.`
        def _is_moe_expert_param(name: str, modality: str, component_type: Optional[str] = None) -> bool:
            base_moe_prefix = 'mmoe_llm.base_llm.model.layers'
            if not name.startswith(base_moe_prefix):
                return False
            
            # Check for modality-specific part
            if f'{modality}_' not in name: # e.g., 'visual_' or 'linguistic_'
                return False
            
            # Check for component type (attention or FFN) if specified
            if component_type == 'attention' and 'self_attn.' not in name:
                return False
            if component_type == 'ffn' and 'mlp.' not in name:
                return False
            
            # Only count the actual weights within `modality_specific_weights`
            if 'modality_specific_weights' not in name:
                return False
            
            return True


        if stage == "stage_1_1":
            # Stage 1.1: Visual Encoder, Connector, MoE visual experts are trainable.
            for name, param in self.named_parameters():
                if _starts_with_any(name, ['visual_encoder.', 'connector.']) or \
                   _is_moe_expert_param(name, 'visual'):
                    param.requires_grad = True
                    trainable_params.append(param)
            logger.info(f"Stage {stage}: Visual encoder, connector, and MoE visual experts are trainable.")

        elif stage == "stage_1_2":
            # Stage 1.2: All from S1.1, PLUS linguistic attention experts.
            for name, param in self.named_parameters():
                if _starts_with_any(name, ['visual_encoder.', 'connector.']) or \
                   _is_moe_expert_param(name, 'visual') or \
                   _is_moe_expert_param(name, 'linguistic', component_type='attention'):
                    param.requires_grad = True
                    trainable_params.append(param)
            logger.info(f"Stage {stage}: Visual encoder, connector, MoE visual experts, and MoE linguistic attention experts are trainable.")

        elif stage == "stage_2":
            # Stage 2: All parameters are trainable.
            for name, param in self.named_parameters():
                param.requires_grad = True
                trainable_params.append(param)
            logger.info(f"Stage {stage}: All parameters are trainable.")
        else:
            raise ValueError(f"Unknown training stage: {stage}")
        
        logger.info(f"Number of trainable parameters for stage {stage}: {len(trainable_params)}")
        return trainable_params

