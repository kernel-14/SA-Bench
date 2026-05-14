```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Iterator, Optional, Dict, Any, List, Tuple
import collections
import math
import copy

# To avoid circular imports, BaseDatasetLoader is not directly imported.
# It's an abstract concept.

# For ViT specific layers from Hugging Face transformers
# We assume the backbone passed will be an instance of transformers.models.vit.modeling_vit.ViTModel
# or transformers.models.clip.modeling_clip.CLIPVisionModel
# We need to access its internal structure, e.g., `model.encoder.layer`
# We'll use type hints for nn.Module where specific HF types aren't easily imported without HF as a hard dependency.


class BasePEFTModule(ABC):
    """
    Abstract Base Class for all Parameter-Efficient Fine-Tuning (PEFT) methods.
    Defines the common interface for initializing, applying modifications to the backbone,
    exposing trainable parameters, and handling Weight-space ensembles (WiSE) scaling.
    """

    def __init__(self, backbone: nn.Module, config: Dict[str, Any]) -> None:
        """
        Initializes the BasePEFTModule.

        Args:
            backbone (nn.Module): The pre-trained Vision Transformer backbone.
            config (Dict[str, Any]): A dictionary containing method-specific and
                                     general PEFT configuration settings.
        """
        self.backbone: nn.Module = backbone
        self.config: Dict[str, Any] = config
        self.peft_config: Dict[str, Any] = config.get('peft_hyperparameters', {})
        self.base_peft_config: Dict[str, Any] = config.get('peft', {})

        # Extract feature_dim and num_layers from the backbone
        # Assuming backbone is a HuggingFace ViTModel or CLIPVisionModel
        if hasattr(backbone, 'config') and hasattr(backbone.config, 'hidden_size'):
            self.feature_dim: int = backbone.config.hidden_size
        else:
            raise AttributeError("Backbone does not have a 'config.hidden_size' attribute.")
        
        if hasattr(backbone, 'config') and hasattr(backbone.config, 'num_hidden_layers'):
            self.num_layers: int = backbone.config.num_hidden_layers
        else:
            raise AttributeError("Backbone does not have a 'config.num_hidden_layers' attribute.")

        self.trainable_parameters: nn.ParameterList = nn.ParameterList()
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.wrapped_modules: Dict[str, Any] = {} # To store original modules/forward methods if replaced
        self._wise_alpha: float = 1.0 # Default alpha for WiSE, scales additive components

    @abstractmethod
    def apply_to_backbone(self) -> None:
        """
        Integrates the PEFT logic into the backbone. This typically involves:
        - Freezing all original backbone parameters.
        - Creating new PEFT-specific modules.
        - Replacing existing modules, wrapping forward methods, or registering hooks.
        - Making specific backbone parameters trainable for selective tuning methods.
        """
        pass

    @abstractmethod
    def get_trainable_parameters(self) -> Iterator[nn.Parameter]:
        """
        Returns an iterator over the nn.Parameter objects that belong specifically
        to this PEFT method and should be optimized during training.
        """
        pass

    @abstractmethod
    def configure_wise_scaling(self, alpha: float) -> None:
        """
        Updates the internal scaling factor for WiSE. This method is called by
        PEFTModelWrapper for methods that internally scale an additive component
        (e.g., Adapter, LoRA, FacT).

        For other methods (e.g., VPT, BitFit) where parameters themselves are
        interpolated, this method should be a no-op or raise an error, as
        PEFTModelWrapper handles direct parameter interpolation.

        Args:
            alpha (float): The current alpha value for WiSE interpolation.
        """
        pass

    def _freeze_all_backbone_params(self) -> None:
        """
        Helper method to iterate through all parameters of self.backbone and
        set p.requires_grad = False. This ensures only PEFT-specific or
        explicitly unfrozen parameters are trained.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _get_transformer_layers(self) -> nn.ModuleList:
        """
        Helper method to abstract access to the backbone's Transformer layers.
        Assumes standard HuggingFace ViT-like structure: backbone.encoder.layer.
        """
        if hasattr(self.backbone, 'encoder') and hasattr(self.backbone.encoder, 'layer'):
            return self.backbone.encoder.layer
        else:
            raise AttributeError("Backbone does not have a 'encoder.layer' attribute "
                                 "or it's not a ModuleList as expected for ViT-like models.")

    def _clean_hooks(self) -> None:
        """Removes all registered forward/backward hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def _restore_wrapped_modules(self) -> None:
        """Restores original modules or forward methods that were replaced."""
        for key, original_func in self.wrapped_modules.items():
            # Assuming key encodes the path to the module/attribute
            parts = key.split('.')
            target = self.backbone
            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    setattr(target, part, original_func)
                else:
                    target = getattr(target, part)
        self.wrapped_modules.clear()

    def __del__(self) -> None:
        """Ensure hooks are removed and wrapped modules are restored when object is deleted."""
        self._clean_hooks()
        self._restore_wrapped_modules()


# --- Helper classes for specific PEFT methods ---

class VPTDeepTransformerLayer(nn.Module):
    """
    A wrapper module for a single ViT Transformer layer to inject VPT-Deep prompts.
    """
    def __init__(self, original_layer: nn.Module, prompt_embedding: nn.Parameter) -> None:
        super().__init__()
        self.original_layer = original_layer
        self.prompt_embedding = prompt_embedding

    def forward(self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor] = None,
                output_attentions: bool = False, output_hidden_states: bool = False,
                return_dict: bool = True) -> Any:
        """
        Prepends the prompt embedding to hidden_states, passes through the original layer,
        and then removes the prompt tokens from the output.
        """
        batch_size = hidden_states.shape[0]
        # Expand prompt_embedding to match batch size
        expanded_prompt = self.prompt_embedding.expand(batch_size, -1, -1) # Shape: (B, num_prompts, D)

        # Concatenate prompts with input tokens
        # hidden_states: (B, 1+N, D) -> [CLS, patch_tokens]
        # Concat -> (B, num_prompts + 1 + N, D)
        combined_hidden_states = torch.cat([expanded_prompt, hidden_states], dim=1)

        # Pass through original Transformer layer
        layer_output = self.original_layer(
            combined_hidden_states,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        # Extract output. For HF ViTLayer, output is usually a tuple or BaseModelOutput
        # We need the hidden_states part.
        if return_dict:
            output_hidden_states_from_layer = layer_output.hidden_states
        else:
            output_hidden_states_from_layer = layer_output[0]

        # Remove the prompt tokens from the output
        # Output shape should be: (B, 1+N, D)
        final_hidden_states = output_hidden_states_from_layer[:, expanded_prompt.shape[1]:, :]

        if return_dict:
            layer_output.hidden_states = final_hidden_states
            return layer_output
        else:
            return (final_hidden_states,) + layer_output[1:]


class AdapterBlock(nn.Module):
    """
    Implements the generic bottleneck MLP structure for adapter-based methods.
    Adapter(h) = s · W_up σ(W_down h) + h
    """
    def __init__(self, input_dim: int, bottleneck_dim: int, scale_factor: float, activation: nn.Module = nn.GELU()) -> None:
        super().__init__()
        self.down_proj = nn.Linear(input_dim, bottleneck_dim)
        self.activation = activation
        self.up_proj = nn.Linear(bottleneck_dim, input_dim)
        
        # Initialize up_proj weights to zero to ensure identity mapping at initialization
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

        self.scale = nn.Parameter(torch.tensor(scale_factor))
        self._wise_alpha_internal: float = 1.0 # For WiSE scaling of the additive component

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Adapter block.
        The paper shows `Adapter(h) = s * W_up σ(W_down h) + h` (residual connection)
        However, the implementation details in the paper describe it as adding to previous values.
        The `Adapter(h)` in the paper often refers to the output of `s * W_up σ(W_down h)`,
        which is then added to the main path.
        For Houlsby, Pfeiffer, AdaptFormer, the structure is usually: `original_output + Adapter(original_output)` or `original_output + Adapter(input_to_mlp)`.
        So, this `forward` will return `s * W_up σ(W_down x)`. The residual addition will happen outside.
        """
        output = self.down_proj(x)
        output = self.activation(output)
        output = self.up_proj(output)
        return output * self.scale * self._wise_alpha_internal

    def set_wise_alpha(self, alpha: float) -> None:
        """Sets the internal WiSE alpha for this adapter block."""
        self._wise_alpha_internal = alpha


class AdaptFormerMLPBlock(nn.Module):
    """
    Wrapper for ViT MLP block to integrate AdaptFormer in parallel.
    `h9 = h9 + Adapter(h7)` where `h9` is `original_mlp(h7)`.
    So, `MLP_out = original_mlp(input_features) + adapter(input_features)`
    """
    def __init__(self, original_mlp: nn.Module, adapter: AdapterBlock) -> None:
        super().__init__()
        self.original_mlp = original_mlp
        self.adapter = adapter

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Combines output from original MLP and AdaptFormer adapter.
        """
        mlp_output = self.original_mlp(hidden_states)
        adapter_output = self.adapter(hidden_states)
        return mlp_output + adapter_output


class ConvPassBlock(nn.Module):
    """
    Implements the convolutional bottleneck module for ConvPass.
    Requires reshaping tokens to image-like structures for 2D convolutions.
    """
    def __init__(self, input_dim: int, bottleneck_dim: int, scale_factor: float, 
                 kernel_size: int = 3, xavier_init: bool = True) -> None:
        super().__init__()
        # 1x1 conv for down-projection (D -> r)
        self.conv_down = nn.Conv2d(input_dim, bottleneck_dim, kernel_size=1, bias=False)
        # 3x3 conv (r -> r)
        self.conv_mid = nn.Conv2d(bottleneck_dim, bottleneck_dim, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        # 1x1 conv for up-projection (r -> D)
        self.conv_up = nn.Conv2d(bottleneck_dim, input_dim, kernel_size=1, bias=False)

        self.activation = nn.GELU() # Assuming GELU as a standard non-linearity

        self.scale = nn.Parameter(torch.tensor(scale_factor))
        self._wise_alpha_internal: float = 1.0

        if xavier_init:
            nn.init.xavier_uniform_(self.conv_down.weight)
            nn.init.xavier_uniform_(self.conv_mid.weight)
            nn.init.xavier_uniform_(self.conv_up.weight)
        else: # Initialize to zero for identity behavior at start
            nn.init.zeros_(self.conv_down.weight)
            nn.init.zeros_(self.conv_mid.weight)
            nn.init.zeros_(self.conv_up.weight)

    def forward(self, x: torch.Tensor, patch_resolution: int) -> torch.Tensor:
        """
        Forward pass for ConvPass block.
        Args:
            x (torch.Tensor): Input token sequence (B, SeqLen, D).
            patch_resolution (int): The spatial resolution of the patches (e.g., 14 for 224x224/16x16).
                                    Used to reshape tokens into a 2D grid.
        Returns:
            torch.Tensor: Output of the ConvPass block (B, SeqLen, D).
        """
        batch_size, seq_len, dim = x.shape

        # Separate CLS token if present. Assuming CLS token is at index 0.
        # ConvPass specifically mentions operating over 'patch tokens'.
        if seq_len == patch_resolution * patch_resolution + 1:
            cls_token = x[:, :1, :]
            patch_tokens = x[:, 1:, :]
        else: # Assume all tokens are patch tokens
            cls_token = None
            patch_tokens = x

        # Reshape patch tokens for 2D convolution
        # (B, N_patches, D) -> (B, D, N_patches_sqrt, N_patches_sqrt)
        h = patch_tokens.transpose(1, 2).reshape(batch_size, dim, patch_resolution, patch_resolution)

        h = self.conv_down(h)
        h = self.activation(h)
        h = self.conv_mid(h)
        h = self.activation(h)
        h = self.conv_up(h)

        # Reshape back to (B, N_patches, D)
        h = h.flatten(2).transpose(1, 2)

        # Re-attach CLS token if it was separated
        if cls_token is not None:
            h = torch.cat([cls_token, h], dim=1)
        
        return h * self.scale * self._wise_alpha_internal

    def set_wise_alpha(self, alpha: float) -> None:
        """Sets the internal WiSE alpha for this ConvPass block."""
        self._wise_alpha_internal = alpha


class RepAdapterBlock(nn.Module):
    """
    Implements the linear Adapter with group-wise transformation for RepAdapter.
    Can be re-parameterized for zero inference overhead.
    """
    def __init__(self, input_dim: int, bottleneck_dim: int, num_groups: int, scale_factor: float) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.num_groups = num_groups
        
        if bottleneck_dim % num_groups != 0 or input_dim % num_groups != 0:
            raise ValueError(f"bottleneck_dim ({bottleneck_dim}) and input_dim ({input_dim}) "
                             f"must be divisible by num_groups ({num_groups}) for RepAdapter.")

        # phi_down: W_down h (D -> r)
        self.W_down = nn.Linear(input_dim, bottleneck_dim, bias=False)
        
        # phi_up: group-wise linear (r -> D)
        # Each group gets (r/G) input and (D/G) output
        self.W_ups = nn.ModuleList([
            nn.Linear(bottleneck_dim // num_groups, input_dim // num_groups, bias=False)
            for _ in range(num_groups)
        ])
        
        # Initialize W_ups weights to zero for identity mapping at initialization
        for W_up in self.W_ups:
            nn.init.zeros_(W_up.weight)
        nn.init.zeros_(self.W_down.weight) # Initialize W_down as well

        self.scale = nn.Parameter(torch.tensor(scale_factor))
        self._wise_alpha_internal: float = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RepAdapter block.
        x: (B, SeqLen, D)
        """
        # phi_down
        h_tilde = self.W_down(x) # (B, SeqLen, r)

        # phi_up (group-wise)
        # Split h_tilde into G groups: (B, SeqLen, r/G) * G
        h_tilde_groups = torch.split(h_tilde, self.bottleneck_dim // self.num_groups, dim=-1)
        
        # Apply W_up to each group
        up_outputs = []
        for i in range(self.num_groups):
            up_outputs.append(self.W_ups[i](h_tilde_groups[i])) # (B, SeqLen, D/G)
        
        # Concatenate outputs: (B, SeqLen, D)
        h_up = torch.cat(up_outputs, dim=-1)

        return h_up * self.scale * self._wise_alpha_internal

    def set_wise_alpha(self, alpha: float) -> None:
        """Sets the internal WiSE alpha for this RepAdapter block."""
        self._wise_alpha_internal = alpha

    def reparameterize(self) -> torch.Tensor:
        """
        Computes the effective weight matrix for reparameterization after training.
        This allows merging the adapter into the original linear layer.
        Returns:
            torch.Tensor: The effective weight matrix (D_out, D_in) that represents
                          the adapter's contribution when added to a linear layer.
        """
        # (r, D_in)
        W_down_matrix = self.W_down.weight
        
        # W_ups are (D_out/G, r/G). Need to form a full (D_out, r) matrix.
        W_up_matrix_blocks = [W_up.weight for W_up in self.W_ups]
        # Concatenate along output dimension to get (D_out, r)
        W_up_matrix = torch.block_diag(*W_up_matrix_blocks) # This assumes block diagonal if groups are independent.
                                                           # If groups are interleaved, it's more complex.
                                                           # Standard RepAdapter paper suggests concatenating.
        # This part requires careful check of exact W_up_matrix construction
        # Simple concat:
        W_up_matrix = torch.cat([W.transpose(0,1) for W in W_up_matrix_blocks], dim=1).transpose(0,1) # (D, r)
        
        delta_W = (W_up_matrix @ W_down_matrix) * self.scale.item()
        return delta_W


class LoRALayer(nn.Module):
    """
    Wraps an original nn.Linear layer and adds LoRA components.
    The effective weight becomes W_orig + (W_up @ W_down) * scale.
    """
    def __init__(self, original_linear_layer: nn.Linear, rank: int, lora_alpha: float = 1.0, lora_dropout: float = 0.0) -> None:
        super().__init__()
        self.original_linear_layer = original_linear_layer
        self.rank = rank
        self.lora_alpha = lora_alpha
        
        # Clone and detach original weight for direct access
        self.original_weight = original_linear_layer.weight.data.clone().detach().requires_grad_(False)
        self.original_bias = original_linear_layer.bias.data.clone().detach().requires_grad_(False) if original_linear_layer.bias is not None else None

        self.lora_down = nn.Linear(original_linear_layer.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, original_linear_layer.out_features, bias=False)
        
        # LoRA initialization: A (down) is usually random normal, B (up) is zero.
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        # Scale factor based on lora_alpha and rank.
        self.scaling = self.lora_alpha / self.rank

        # Optional dropout
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        self._wise_alpha_internal: float = 1.0 # For WiSE scaling of the additive component

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LoRA layer.
        """
        # Original part
        result = F.linear(x, self.original_weight, self.original_bias)

        # LoRA additive part: x @ (W_down @ W_up) * scaling * wise_alpha
        lora_result = self.lora_up(self.lora_down(self.lora_dropout(x))) * self.scaling * self._wise_alpha_internal
        
        return result + lora_result

    def set_wise_alpha(self, alpha: float) -> None:
        """Sets the internal WiSE alpha for this LoRA layer."""
        self._wise_alpha_internal = alpha


class SSFModule(nn.Module):
    """
    Applies Scale and Shift factors to features.
    h_out = w * h_in + b
    """
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.ones(feature_dim))
        self.b = nn.Parameter(torch.zeros(feature_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies element-wise scale and shift to the input tensor.
        x: (B, SeqLen, D)
        w, b: (D)
        """
        return x * self.w + self.b


# --- Implementations for PEFT methods (inheriting from BasePEFTModule) ---

class VPTShallow(BasePEFTModule):
    """
    Visual Prompt Tuning (VPT) - Shallow variant.
    Prepends learnable prompt embeddings to the input tokens of the first Transformer layer.
    """
    def __init__(self, backbone: nn.Module, config: Dict[str, Any]) -> None:
        super().__init__(backbone, config)
        self.prompt_number: int = self.peft_config.get('prompt_number', 100) # Default from paper's Table 3

        # Create learnable prompt embeddings
        self.prompt_embedding = nn.Parameter(torch.empty(1, self.prompt_number, self.feature_dim))
        nn.init.trunc_normal_(self.prompt_embedding, std=0.02) # Common initialization for embeddings

        self.trainable_parameters.append(self.prompt_embedding)

    def apply_to_backbone(self) -> None:
        """
        Freezes backbone parameters and replaces the original embedding forward
        method to inject shallow prompts.
        """
        self._freeze_all_backbone_params()

        # Store original forward method to restore later if needed
        # Assuming `self.backbone.embeddings` is the module responsible for initial token embeddings
        if hasattr(self.backbone, 'embeddings') and hasattr(self.backbone.embeddings, 'forward'):
            self.wrapped_modules['backbone.embeddings.forward'] = self.backbone.embeddings.forward
            self.backbone.embeddings.forward = self._vpt_shallow_embedding_forward
        else:
            raise AttributeError("Backbone does not have a 'embeddings.forward' method for VPTShallow integration.")

    def _vpt_shallow_embedding_forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Custom forward method for backbone.embeddings to prepend prompts.
        """
        # Call original embedding forward to get initial embeddings for pixel values
        original_embeddings = self.wrapped_modules['backbone.embeddings.forward'](pixel_values)

        batch_size = original_embeddings.shape[0]
        # Expand prompt_embedding to match batch size
        expanded_prompt = self.prompt_embedding.expand(batch_size, -1, -1) # Shape: (B, num_prompts, D)

        # Concatenate prompts with original embeddings
        # original_embeddings: (B, 1+N, D) -> [CLS, patch_tokens]
        # Concat -> (B, num_prompts + 1 + N, D)
        combined_embeddings = torch.cat([expanded_prompt, original_embeddings], dim=1)
        return combined_embeddings

    def get_trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Returns an iterator over the prompt embeddings."""
        yield self.prompt_embedding

    def configure_wise_scaling(self, alpha: float) -> None:
        """
        No-op for VPT-Shallow. PEFTModelWrapper handles direct interpolation
        of prompt_embedding if WiSE is applied.
        """
        pass


class VPTDeep(BasePEFTModule):
    """
    Visual Prompt Tuning (VPT) - Deep variant.
    Inserts learnable prompt embeddings to the input of each Transformer layer.
    """
    def __init__(self, backbone: nn.Module, config: Dict[str, Any]) -> None:
        super().__init__(backbone, config)
        self.prompt_number: int = self.peft_config.get('prompt_number', 100) # Default from paper's Table 3

        # Create learnable prompt embeddings for each Transformer layer
        self.prompt_embeddings = nn.ParameterList([
            nn.Parameter(torch.empty(1, self.prompt_number, self.feature_dim))
            for _ in range(self.num_layers)
        ])
        for prompt_emb in self.prompt_embeddings:
            nn.init.trunc_normal_(prompt_emb, std=0.02) # Common initialization

        self.trainable_parameters.extend(self.prompt_embeddings)

    def apply_to_backbone(self) -> None:
        """
        Freezes backbone parameters and replaces each original Transformer layer
        with a wrapper that injects and removes deep prompts.
        """
        self._freeze_all_backbone_params()

        transformer_layers = self._get_transformer_layers()
        for i, layer in enumerate(transformer_layers):
            # Replace original layer with VPTDeepTransformerLayer
            vpt_layer = VPTDeepTransformerLayer(layer, self.prompt_embeddings[i])
            self.wrapped_modules[f'backbone.encoder.layer.{i}'] = layer # Store original for potential restore
            transformer_layers[i] = vpt_layer

        # Also need to modify the initial embedding forward to prepend prompts for the first layer,
        # and then remove them *before* the first layer passes its output to the wrapper.
        # This implies that the 'VPTDeepTransformerLayer' wrapper already handles the
        # prompts being present in the input from the previous layer, and strips them.
        # The paper says: `[P_m, Z_m] = L_m([P_{m-1}, Z_{m-1}])`, and `outputs are discarded at the end of the layer`.
        # This means the prompt from previous layer passes *through* the layer and is then discarded by the next layer's input.
        # The `VPTDeepTransformerLayer` assumes prompts are already present, so the initial embedding
        # part of the original ViT is NOT modified, the prompts are *inserted* per layer.
        # Let's adjust VPTDeepTransformerLayer to *receive* hidden_states without prompts, *add* prompts,
        # pass through original layer, then *strip* prompts before returning.
        # The current VPTDeepTransformerLayer does this.
        pass # No need to modify backbone.embeddings for VPTDeep directly, layers handle it.

    def get_trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Returns an iterator over all deep prompt embeddings."""
        for param in self.prompt_embeddings:
            yield param

    def configure_wise_scaling(self, alpha: float) -> None:
        """
        No-op for VPT-Deep. PEFTModelWrapper handles direct interpolation
        of prompt_embeddings if WiSE is applied.
        """
        pass


class HoulsbyAdapter(BasePEFTModule):
    """
    Houlsby Adapter (Adapter-based method).
    Inserts two AdapterBlocks per Transformer layer: one after MSA block, one after MLP block.
    """
    def __init__(self, backbone: nn.Module, config: Dict[str, Any]) -> None:
        super().__init__(backbone, config)
        self.adapter_bottleneck: int = self.peft_config.get('adapter_bottleneck', 8)
        self.adapter_scale_factor: float = self.peft_config.get('adapter_scale_factor', 1.0)
        self.activation = nn.GELU() # Assuming GELU for non-linear activation

        self.adapters_msa = nn.ModuleList([
            AdapterBlock(self.feature_dim, self.adapter_bottleneck, self.adapter_scale_factor, self.activation)
            for _ in range(self.num_layers)
        ])
        self.adapters_mlp = nn.ModuleList([
            AdapterBlock(self.feature_dim, self.adapter_bottleneck, self.adapter_scale_factor, self.activation)
            for _ in range(self.num_layers)
        ])
        self.trainable_parameters.extend(list(self.adapters_msa.parameters()))
        self.trainable_parameters.extend(list(self.adapters_mlp.parameters()))

    def apply_to_backbone(self) -> None:
        """
        Freezes backbone parameters and injects Houlsby adapters after MSA and MLP outputs.
        """
        self._freeze_all_backbone_params()

        transformer_layers = self._get_transformer_layers()
        for i, layer in enumerate(transformer_layers):
            # Store original forward methods
            self.wrapped_modules[f'backbone.encoder.layer.{i}.attention.output.forward'] = layer.attention.output.forward
            self.wrapped_modules[f'backbone.encoder.layer.{i}.output.forward'] = layer.output.forward

            # Wrap attention.output.forward (after MSA, before residual add)
            def create_msa_adapter_forward(original_forward, adapter_module):
                def msa_adapter_forward(hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
                    original_output = original_forward(hidden_states, input_tensor)
                    adapter_output = adapter_module(original_output) # Adapter(h5) -> paper uses h5 as output from MSA
                    return original_output + adapter_output
                return msa_adapter_forward
            
            # Wrap output.forward (after MLP, before residual add)
            def create_mlp_adapter_forward(original_forward, adapter_module):
                def mlp_adapter_forward(hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
                    original_output = original_forward(hidden_states, input_tensor)
                    adapter_output = adapter_module(original_output) # Adapter(h9) -> paper uses h9 as output from MLP
                    return original_output + adapter_output
                return mlp_adapter_forward

            layer.attention.output.forward = create_msa_adapter_forward(
                layer.attention.output.forward, self.adapters_msa[i]
            )
            layer.output.forward = create_mlp_adapter_forward(
                layer.output.forward, self.adapters_mlp[i]
            )

    def get_trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Returns an iterator over all adapter parameters."""
        for adapter_module in self.adapters_msa:
            for param in adapter_module.parameters():
                yield param
        for adapter_module in self.adapters_mlp:
            for param in adapter_module.parameters():
                yield param

    def configure_wise_scaling(self, alpha: float) -> None:
        """Configures WiSE scaling for all adapter blocks."""
        for adapter_module in self.adapters_msa:
            adapter_module.set_wise_alpha(alpha)
        for adapter_module in self.adapters_mlp:
            adapter_module.set_wise_alpha(alpha)


class PfeifferAdapter(BasePEFTModule):
    """
    Pfeiffer Adapter (Adapter-based method).
    Inserts one AdapterBlock per Transformer layer, solely after the MLP block.
    """
    def __init__(self, backbone: nn.Module, config: Dict[str, Any]) -> None:
        super().__init__(backbone, config)
        self.adapter_bottleneck: int = self.peft_config.get('adapter_bottleneck', 8)
        self.adapter_scale_factor: float = self.peft_config.get('adapter_scale_factor', 1.0)
        self.activation = nn.GELU()

        self.adapters_mlp = nn.ModuleList([
            AdapterBlock(self.feature_dim, self.adapter_bottleneck, self.adapter_scale_factor, self.activation)
            for _ in range(self.num_layers)
        ])
        self.trainable_parameters.extend(list(self.adapters_mlp.parameters()))

    def apply_to_backbone(self) -> None:
        """
        Freezes backbone parameters and injects Pfeiffer adapters after MLP outputs.
        """
        self._freeze_all_backbone_params()

        transformer_layers = self._get_transformer_layers()
        for i, layer in enumerate(transformer_layers):
            # Store original forward method
            self.wrapped_modules[f'backbone.encoder.layer.{i}.output.forward'] = layer.output.forward

            # Wrap output.forward (after MLP, before residual add)
            def create_mlp_adapter_forward(original_forward, adapter_module):
                def mlp_adapter_forward(hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
                    original_output = original_forward(hidden_states, input_tensor)
                    adapter_output = adapter_module(original_output)
                    return original_output + adapter_output
                return mlp_adapter_forward

            layer.output.forward = create_mlp_adapter_forward(
                layer.output.forward, self.adapters_mlp[i]
            )

    def get_trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Returns an iterator over all MLP adapter parameters."""
        for adapter_module in self.adapters_mlp:
            for param in adapter_module.parameters():
                yield param

    def configure_wise_scaling(self, alpha: float) -> None:
        """Configures WiSE scaling for all MLP adapter blocks."""
        for adapter_module in self.adapters_mlp:
            adapter_module.set_wise_alpha(alpha)


class AdaptFormer(BasePEFTModule):
    """
    AdaptFormer (Adapter-based method).
    Inserts an AdapterBlock in parallel with the MLP block.
    """
    def __init__(self, backbone: nn.Module, config: Dict[str, Any]) -> None:
        super().__init__(backbone, config)
        self.adapter_bottleneck: int = self.peft_config.get('adapter_bottleneck', 16)
        self.adapter_scale_factor: float = self.peft_config.get('adapter_scale_factor', 0.1)
        self.activation = nn.GELU()

        self.adapters_parallel_mlp = nn.ModuleList([
            AdapterBlock(self.feature_dim, self.adapter_bottleneck, self.adapter_scale_factor, self.activation)
            for _ in range(self.num_layers)
        ])
        self.trainable_parameters.extend(list(self.adapters_parallel_mlp.parameters()))

    def apply_to_backbone(self) -> None:
        """
        Freezes backbone parameters and replaces the original MLP block
        with AdaptFormerMLPBlock that runs adapter in parallel.
        """
        self._freeze_all_backbone_params()

        transformer_layers = self._get_transformer_layers()
        for i, layer in enumerate(transformer_layers):
            # Store original MLP module
            self.wrapped_modules[f'backbone.encoder.layer.{i}.mlp'] = layer.mlp
            # Replace with AdaptFormerMLPBlock
            layer.mlp = AdaptFormerMLPBlock(layer.mlp, self.adapters_parallel_mlp[i])

    def get_trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Returns an iterator over all parallel MLP adapter parameters."""
        for adapter_module in self.adapters_parallel_mlp:
            for param in adapter_module.parameters():
                yield param

    def configure_wise_scaling(self, alpha: float) -> None:
        """Configures WiSE scaling for all parallel MLP adapter blocks."""
        for adapter_module in self.adapters_parallel_mlp:
            adapter_module.set_wise_alpha(alpha)


class ConvPass(BasePEFTModule):
    """
    ConvPass (Adapter-based method).
    Inserts a convolutional bottleneck module parallel to the MSA and MLP blocks.
    """
    def __init__(self, backbone: nn.Module, config: Dict[str, Any]) -> None:
        super().__init__(backbone, config)
        self.convpass_bottleneck: int = self.peft_config.get('convpass_bottleneck', 16)
        self.convpass_scale_factor: float = self.peft_config.get('convpass_scale_factor', 1.0)
        self.convpass_xavier_init: bool = self.peft_config.get('convpass_xavier_init', True)

        # Calculate patch_resolution from backbone config
        # Assuming image_size and patch_size are available
        if hasattr(self.backbone.config, 'image_size') and hasattr(self.backbone.config, 'patch_size'):
            self.patch_resolution: int = self.backbone.config.image_size // self.backbone.config.patch_size
        else:
            raise AttributeError("Backbone config must have 'image_size' and 'patch_size' for ConvPass.")

        self.convpass_msa = nn.ModuleList([
            ConvPassBlock(self.feature_dim, self.convpass_bottleneck, self.convpass_scale_factor, 
                          xavier_init=self.convpass_xavier_init)
            for _ in range(self.num_layers)
        ])
        self.convpass_mlp = nn.ModuleList([
            ConvPassBlock(self.feature_dim, self.convpass_bottleneck, self.convpass_scale_factor, 
                          xavier_init=self.convpass_xavier_init)
            for _ in range(self.num_layers)
        ])
        self.trainable_parameters.extend(list(self.convpass_msa.parameters()))
        self.trainable_parameters.extend(list(self.convpass_mlp.parameters()))

    def apply_to_backbone(self) -> None:
        """
        Freezes backbone parameters and injects ConvPass modules using forward hooks.
        The paper states: h5 = Convpass1(h2) + h5 and h9 = Convpass2(h7) + h9.
        Mapping to HF ViT:
        h2: output of layernorm_before_attention
        h5: output of layernorm_before_mlp
        h7: input to mlp.dense (after layernorm_before_mlp)
        h9: output of output.dense (after mlp.dense)
        
        The equations imply:
        For MSA: `output_after_msa_residual = MSA_output + Convpass_MSA(LN_before_MSA_output)`.
                 This would mean modifying the `attention.output.forward` to add Convpass.
        For MLP: `output_after_mlp_residual = MLP_output + Convpass_MLP(LN_before_MLP_output)`.
                 This would mean modifying the `output.forward` to add Convpass.
        """
        self._freeze_all_backbone_params()

        transformer_layers = self._get_transformer_layers()
        for i, layer in enumerate(transformer_layers):
            # For h5 = Convpass1(h2) + h5
            # This is complex because h2 is LN_before_MSA_output, h5 is LN_before_MLP_output
            # The paper's notation often reflects a conceptual data flow, not exact insertion points.
            # A common interpretation for parallel adapters is:
            #   original_output + adapter(original_input)
            # Given the phrasing "parallel to the MSA and MLP blocks", let's follow a simpler
            # interpretation where Convpass operates on the same input as MSA/MLP and adds to its output.

            # We'll apply Convpass_MSA to the output of `attention.output` (h3 in paper) and add it.
            # And Convpass_MLP to the input of `mlp.dense` (h7 in paper) and add it.
            # This is similar to how AdaptFormer is applied.

            # Original MSA path: hidden_states -> LN -> self_attention -> output_projection (h3) -> residual + hidden_states (h5 conceptually)
            # Original MLP path: hidden_states -> LN -> intermediate.dense (h7) -> output.dense (h9) -> residual + hidden_states

            # Let's wrap the entire `attention` and `mlp` modules to inject parallel Convpass
            # This is a more robust way to handle the internal forward flow.

            # For MSA: original `attention` module
            self.wrapped_modules[f'backbone.encoder.layer.{i}.attention.forward'] = layer.attention.forward
            def create_msa_convpass_forward(original_attention_forward, convpass_module, patch_res):
                def msa_convpass_forward(hidden_states: torch.Tensor, *args, **kwargs) -> Any:
                    # original_attention_forward takes `hidden_states` (after LN before attention)
                    original_output_tuple = original_attention_forward(hidden_states, *args, **kwargs)
                    original_output = original_output_tuple[0] # The actual output tensor
                    
                    convpass_output = convpass_module(hidden_states, patch_res)
                    # The paper implies `MSA_output + Convpass_MSA(LN_before_MSA_output)`
                    # If we wrap `attention.forward`, hidden_states IS LN_before_MSA_output.
                    
                    # The `h5 = Convpass1(h2) + h5` in paper means output from Convpass 
                    # is added to output of MSA.
                    # ViT self_attention returns: (attn_output, attn_weights, past_key_value)
                    # We modify attn_output (first element)
                    return (original_output + convpass_output,) + original_output_tuple[1:]
                return msa_convpass_forward
            
            layer.attention.forward = create_msa_convpass_forward(
                layer.attention.forward, self.convpass_msa[i], self.patch_resolution
            )

            # For MLP: original `mlp` module
            self.wrapped_modules[f'backbone.encoder.layer.{i}.mlp.forward'] = layer.mlp.forward
            def create_mlp_convpass_forward(original_mlp_forward, convpass_module, patch_res):
                def mlp_convpass_forward(hidden_states: torch.Tensor) -> torch.Tensor:
                    # original_mlp_forward takes `hidden_states` (after LN before MLP)
                    original_output = original_mlp_forward(hidden_states)
                    convpass_output = convpass_module(hidden_states, patch_res)
                    return original_output + convpass_output
                return mlp_convpass_forward
            
            layer.mlp.forward = create_mlp_convpass_forward(
                layer.mlp.forward, self.convpass_mlp[i], self.patch_resolution
            )

    def get_trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Returns an iterator over all ConvPass parameters."""
        for convpass_module in self.convpass_msa:
            for param in convpass_module.parameters():
                yield param
        for convpass_module in self.convpass_mlp:
            for param in convpass_module.parameters():
                yield param

    def configure_wise_scaling(self, alpha: float) -> None:
        """Configures WiSE scaling for all ConvPass blocks."""
        for convpass_module in self.convpass_msa:
            convpass_module.set_wise_alpha(alpha)
        for convpass_module in self.convpass_mlp:
            convpass_module.set_wise_alpha(alpha)


class RepAdapter(BasePEFTModule):
    """
    RepAdapter (Adapter-based method).
    Linear Adapters with group-wise transformation, placed sequentially.
    """
    def __init__(self, backbone: nn.Module, config: Dict[str, Any]) -> None:
        super().__init__(backbone, config)
        self.repadapter_bottleneck: int = self.peft_config.get('repadapter_bottleneck', 16)
        self.repadapter_scale_factor: float = self.peft_config.get('repadapter_scale_factor', 1.0)
        self.num_groups: int = self.peft_config.get('num_groups', 4) # Default num_groups for RepAdapter
        
        self.rep_adapters_msa = nn.ModuleList([
            RepAdapterBlock(self.feature_dim, self.repadapter_bottleneck, self.num_groups, self.repadapter_scale_factor)
            for _ in range(self.num_layers)
        ])
        self.rep_adapters_mlp = nn.ModuleList([
            RepAdapterBlock(self.feature_dim, self.repadapter_bottleneck, self.num_groups, self.repadapter_scale_factor)
            for _ in range(self.num_layers)
        ])
        self.trainable_parameters.extend(list(self.rep_adapters_msa.parameters()))
        self.trainable_parameters.extend(list(self.rep_adapters_mlp.parameters()))

    def apply_to_backbone(self) -> None:
        """
        Freezes backbone parameters and injects RepAdapters sequentially.
        Paper: `h2 = RepAdapter1(h2)` and `h7 = RepAdapter2(h7)`.
        `h2`: output of `layernorm_before_attention`.
        `h7`: input to `mlp.dense` (output of `layernorm_before_mlp`).
        """
        self._freeze_all_backbone_params()

        transformer_layers = self._get_transformer_layers()
        for i, layer in enumerate(transformer_layers):
            # For h2 = RepAdapter1(h2)
            # Hook the output of layernorm_before_attention
            def create_msa_pre_hook(rep_adapter_module):
                def msa_pre_hook(module, input):
                    # `input` is a tuple (hidden_states,)
                    return (input[0] + rep_adapter_module(input[0]),)
                return msa_pre_hook
            
            # This is an input pre-hook to `layer.attention` module
            # So, the input `hidden_states` to `layer.attention` is modified
            self.hooks.append(
                layer.attention.register_forward_pre_hook(create_msa_pre_hook(self.rep_adapters_msa[i]))
            )

            # For h7 = RepAdapter2(h7)
            # Hook the output of layernorm_before_mlp
            def create_mlp_pre_hook(rep_adapter_module):
                def mlp_pre_hook(module, input):
                    # `input` is a tuple (hidden_states,)
                    return (input[0] + rep_adapter_module(input[0]),)
                return mlp_pre_hook

            # This is an input pre-hook to `layer.mlp` module
            # So, the input `hidden_states` to `layer.mlp` is modified
            self.hooks.append(
                layer.mlp.register_forward_pre_hook(create_mlp_pre_hook(self.rep_adapters_mlp[i]))
            )
            
    def get_trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Returns an iterator over all RepAdapter parameters."""
        for adapter_module in self.rep_adapters_msa:
            for param in adapter_module.parameters():
                yield param
        for adapter_module in self.rep_adapters_mlp:
            for param in adapter_module.parameters():
                yield param

    def configure_wise_scaling(self, alpha: float) -> None:
        """Configures WiSE scaling for all RepAdapter blocks."""
        for adapter_module in self.rep_adapters_msa:
            adapter_module.set_wise_alpha(alpha)
        for adapter_module in self.rep_adapters_mlp:
            adapter_module.set_wise_alpha(alpha)


class BitFit(BasePEFTModule):
    """
    BitFit (Direct Selective Tuning).
    Only tunes the bias terms of the pre-trained model.
    """
    def __init__(self, backbone: nn.Module, config: Dict[str, Any]) -> None:
        super().__init__(backbone, config)

    def apply_to_backbone(self) -> None:
        """
        Iterates through backbone parameters, setting `requires_grad=True` for biases
        and `requires_grad=False` for all other parameters.
        """
        # First freeze everything
        self._freeze_all_backbone_params()

        # Then selectively unfreeze bias terms
        for name, param in self.backbone.named_parameters():
            if 'bias' in name:
                param.requires_grad = True

    def get_trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Returns an iterator over the trainable bias parameters from the backbone."""
        for param in self.backbone.parameters():
            if param.requires_grad:
                yield param

    def configure_wise_scaling(self, alpha: float) -> None:
        """
        No-op for BitFit. PEFTModelWrapper handles direct interpolation
        of bias parameters if WiSE is applied.
        """
        pass


class LayerNormTune(BasePEFTModule):
    """
    LayerNorm (Direct Selective Tuning).
    Only tunes the parameters of the Layer Normalization (LN) blocks.
    """
    def __init__(self, backbone: nn.Module, config: Dict[str, Any]) -> None:
        super().__init__(backbone, config)

    def apply_to_backbone(self) -> None:
        """
        Iterates through backbone modules, setting `requires_grad=True` for parameters
        within `nn.LayerNorm` layers.
        """
        # First freeze everything
        self._freeze_all_backbone_params()

        # Then selectively unfreeze LayerNorm parameters
        for module in self.backbone.modules():
            if isinstance(module, nn.LayerNorm):
                for param in module.parameters():
                    param.requires_grad = True

    def get_trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Returns an iterator over the trainable LayerNorm parameters from the backbone."""
        for param in self.backbone.parameters():
            if param.requires_grad:
                yield param

    def configure_wise_scaling(self, alpha: float) -> None:
        """
        No-op for LayerNormTune. PEFTModelWrapper handles direct interpolation
        of LayerNorm parameters if WiSE is applied.
        """
        pass


class DiffFit(BasePEFTModule):
    """
    DiffFit (Direct Selective Tuning).
    Combines BitFit and LayerNorm tuning, and adds learnable scale factors (gamma)
    after MSA and MLP outputs.
    """
    def __init__(self, backbone: nn.Module, config: Dict[str, Any]) -> None:
        super().__init__(backbone, config)
        
        # Create learnable scale factors for MSA and MLP outputs per layer
        self.gamma_msa = nn.ParameterList([
            nn.Parameter(torch.ones(self.feature_dim)) for _ in range(self.num_layers)
        ])
        self.gamma_mlp = nn.ParameterList([
            nn.Parameter(torch.ones(self.feature_dim)) for _ in range(self.num_layers)
        ])
        self.trainable_parameters.extend(self.gamma_msa)
        self.trainable_parameters.extend(self.gamma_mlp)

    def apply_to_backbone(self) -> None:
        """
        Applies BitFit and LayerNormTune logic, and registers forward hooks
        to apply learnable scale factors (gamma) after MSA and MLP outputs.
        """
        # Apply BitFit and LayerNormTune logic (combined)
        self._freeze_all_backbone_params()
        for name, param in self.backbone.named_parameters():
            if 'bias' in name:
                param.requires_grad = True
        for module in self.backbone.modules():
            if isinstance(module, nn.LayerNorm):
                for param in module.parameters():
                    param.requires_grad = True

        transformer_layers = self._get_transformer_layers()
        for i, layer in enumerate(transformer_layers):
            # For h5 = gamma1 * h5
            # h5 is the output of LN before MLP (layer.layernorm_before_mlp)
            def create_msa_gamma_hook(gamma_param):
                def msa_gamma_hook(module, input, output):
                    return output * gamma_param # Apply scale to output of LN
                return msa_gamma_hook
            
            # Hook the output of the LayerNorm BEFORE the MLP block
            self.hooks.append(
                layer.layernorm_before_mlp.register_forward_hook(create_msa_gamma_hook(self.gamma_msa[i]))
            )

            # For h9 = gamma2 * h9
            # h9 is the output of the final dense layer in MLP (layer.output.dense)
            def create_mlp_gamma_hook(gamma_param):
                def mlp_gamma_hook(module, input, output):
                    return output * gamma_param # Apply scale to output of MLP's final dense
                return mlp_gamma_hook
            
            # Hook the output of the final dense layer in the MLP block
            self.hooks.append(
                layer.mlp.output.dense.register_forward_hook(create_mlp_gamma_hook(self.gamma_mlp[i]))
            )


    def get_trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Returns an iterator over all trainable parameters (backbone biases, LN, gammas)."""
        for param in self.backbone.parameters():
            if param.requires_grad:
                yield param
        for param in self.gamma_msa:
            yield param
        for param in self.gamma_mlp:
            yield param

    def configure_wise_scaling(self, alpha: float) -> None:
        """
        No-op for DiffFit. PEFTModelWrapper handles direct interpolation
        of backbone and gamma parameters if WiSE is applied.
        """
        pass


class SSF(BasePEFTModule):
    """
    SSF (Scale & Shift deep Features).
    Applies linear transformations (scale 'w' and shift 'b') to specific
    intermediate features within Transformer layers.
    """
    def __init__(self, backbone: nn.Module, config: Dict[str, Any]) -> None:
        super().__init__(backbone, config)
        
        # Store SSF modules for each layer and feature point
        self.ssf_modules: nn.ModuleDict = nn.ModuleDict()
        
        for i in range(self.num_layers):
            self.ssf_modules[f'layer_{i}_h2'] = SSFModule(self.feature_dim)
            self.ssf_modules[f'layer_{i}_h3'] = SSFModule(self.feature_dim)
            self.ssf_modules[f'layer_{i}_h5'] = SSFModule(self.feature_dim)
            self.ssf_modules[f'layer_{i}_h7'] = SSFModule(self.feature_dim)
            self.ssf_modules[f'layer_{i}_h8'] = SSFModule(self.feature_dim) # h8 is after MLP in a specific config
            self.ssf_modules[f'layer_{i}_h9'] = SSFModule(self.feature_dim) # h9 is another point
        
        self.trainable_parameters.extend(list(self.ssf_modules.parameters()))

    def apply_to_backbone(self) -> None:
        """
        Freezes backbone parameters and registers forward hooks to apply SSF modules.
        Mapping to HF ViT (simplified from paper's h-notation):
        - h2: Output of `layer.layernorm_before_attention`
        - h3: Output of `layer.attention` (specifically, its first element: hidden_states after attention)
        - h5: Output of `layer.layernorm_before_mlp`
        - h7: Not directly a module output in HF, but effectively input to MLP block after LN
        - h8: Output of `layer.mlp.intermediate.dense` (first linear layer in MLP)
        - h9: Output of `layer.mlp.output.dense` (second linear layer in MLP)
        """
        self._freeze_all_backbone_params()

        transformer_layers = self._get_transformer_layers()
        for i, layer in enumerate(transformer_layers):
            # h2: output of layernorm_before_attention
            self.hooks.append(
                layer.layernorm_before_attention.register_forward_hook(
                    self._create_ssf_hook(self.ssf_modules[f'layer_{i}_h2'])
                )
            )

            # h3: output of layer.attention (the main output tensor)
            def create_attention_ssf_hook(ssf_module):
                def ssf_hook(module, input, output):
                    # output is a tuple (hidden_states, attention_probs)
                    return (ssf_module(output[0]),) + output[