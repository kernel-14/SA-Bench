import math
from typing import Any, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming Config class is available from config.py and logger from utils.py
# To avoid direct import and potential circular dependencies during setup,
# the Config object is passed explicitly, and logger is assumed to be available
# or will be passed/imported in main.
from config import Config # Import actual Config
from utils import logger # Import logger


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm) layer.
    As described in the paper, applied before attention and FFN operations.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initializes the RMSNorm layer.

        Args:
            dim: The feature dimension over which normalization is applied.
            eps: A small epsilon value to prevent division by zero.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the RMS for the input tensor.
        """
        # x.norm(2, dim=-1, keepdim=True) computes the L2 norm for the last dimension
        # x.shape is (..., dim)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RMS Normalization to the input tensor.

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            Normalized tensor of the same shape.
        """
        return self._norm(x) * self.weight


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding module for the visual encoder.
    Transforms raw input images into a sequence of embedded patches.
    """
    def __init__(self, config: Config):
        """
        Initializes the PatchEmbed layer.

        Args:
            config: The global configuration object.
        """
        super().__init__()
        visual_encoder_config = config.model_architecture.visual_encoder
        
        self.in_chans = 3  # For RGB images
        self.patch_embedding_stride = visual_encoder_config.patch_embedding_stride
        self.embed_dim = visual_encoder_config.width # Visual encoder's hidden dimension
        
        # As per the logic analysis, assume patch_size == patch_embedding_stride
        self.patch_size = self.patch_embedding_stride 

        # Convolutional layer for patch embedding
        # padding=0 implies no padding in the conv layer itself,
        # padding will be handled explicitly in forward pass
        self.proj = nn.Conv2d(
            in_channels=self.in_chans,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_embedding_stride,
            padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transforms input images into patch embeddings.

        Args:
            x: Input image tensor of shape (B, C, H, W).

        Returns:
            Patch embeddings tensor of shape (B, N_patches, embed_dim).
        """
        _, _, H, W = x.shape

        # Calculate padding to ensure H and W are multiples of patch_embedding_stride
        pad_h = (self.patch_embedding_stride - H % self.patch_embedding_stride) % self.patch_embedding_stride
        pad_w = (self.patch_embedding_stride - W % self.patch_embedding_stride) % self.patch_embedding_stride
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
            logger.debug(f"Padded image from ({H},{W}) to ({H + pad_h},{W + pad_w})")

        # Apply convolutional projection
        # Output shape: (B, embed_dim, H_out, W_out)
        x = self.proj(x)
        
        # Flatten spatial dimensions and transpose to (B, N_patches, embed_dim)
        # N_patches = H_out * W_out
        x = x.flatten(2).transpose(1, 2)
        return x


class MoELayer(nn.Module):
    """
    Modality-specific Mixture-of-Experts layer for either attention or FFN.
    It applies RMSNorm and then routes computation to modality-specific experts.
    """
    def __init__(self, config: Config, is_attention_expert: bool):
        """
        Initializes the MoE layer.

        Args:
            config: The global configuration object.
            is_attention_expert: If True, this layer acts as an MHA-MMoE (attention expert).
                                 If False, it acts as an FFN-MMoE (feed-forward expert).
        """
        super().__init__()
        llm_moe_config = config.model_architecture.llm_moe
        
        self.hidden_size = llm_moe_config.width
        self.num_attention_heads = llm_moe_config.num_attention_heads
        self.mlp_width = llm_moe_config.mlp_width
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.is_attention_expert = is_attention_expert

        self.rms_norm = RMSNorm(self.hidden_size)
        self.modality_specific_weights = nn.ModuleDict()

        modalities = ["visual", "linguistic"]

        if self.is_attention_expert:
            # MHA-MMoE: Modality-specific Q, K, V, O projections
            for m in modalities:
                self.modality_specific_weights[f"{m}_q_proj"] = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
                self.modality_specific_weights[f"{m}_k_proj"] = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
                self.modality_specific_weights[f"{m}_v_proj"] = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
                self.modality_specific_weights[f"{m}_o_proj"] = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.softmax = nn.Softmax(dim=-1)
        else:
            # FFN-MMoE: Modality-specific gate, up, down projections
            self.silu = nn.SiLU()
            for m in modalities:
                self.modality_specific_weights[f"{m}_gate_proj"] = nn.Linear(self.hidden_size, self.mlp_width, bias=False)
                self.modality_specific_weights[f"{m}_up_proj"] = nn.Linear(self.hidden_size, self.mlp_width, bias=False)
                self.modality_specific_weights[f"{m}_down_proj"] = nn.Linear(self.mlp_width, self.hidden_size, bias=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Splits a tensor into multiple heads for multi-head attention.
        Input: (batch_size, seq_len, hidden_size)
        Output: (batch_size, num_heads, seq_len, head_dim)
        """
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merges multiple heads back into a single hidden dimension.
        Input: (batch_size, num_heads, seq_len, head_dim)
        Output: (batch_size, seq_len, hidden_size)
        """
        batch_size, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        modality_id: int, # 0 for visual, 1 for linguistic
        attention_mask: Optional[torch.Tensor] = None, # For MHA-MMoE
        rope_func: Optional[Any] = None, # For MHA-MMoE if RoPE is applied inside layer
        seq_len: Optional[int] = None, # For MHA-MMoE 1D RoPE
        img_h: Optional[int] = None, # For MHA-MMoE 2D RoPE
        img_w: Optional[int] = None, # For MHA-MMoE 2D RoPE
    ) -> torch.Tensor:
        """
        Performs the forward pass for the MoE layer.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size).
            modality_id: Integer indicating the modality (0 for visual, 1 for linguistic).
            attention_mask: Optional mask for attention scores (for MHA-MMoE).
            rope_func: Optional RoPE application function (e.g., from utils.py).
            seq_len: Sequence length for 1D RoPE.
            img_h: Image height for 2D RoPE.
            img_w: Image width for 2D RoPE.

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size).
        """
        normed_hidden_states = self.rms_norm(hidden_states)
        expert_prefix = "visual" if modality_id == 0 else "linguistic"

        if self.is_attention_expert:
            # MHA-MMoE logic
            q_proj = self.modality_specific_weights[f"{expert_prefix}_q_proj"]
            k_proj = self.modality_specific_weights[f"{expert_prefix}_k_proj"]
            v_proj = self.modality_specific_weights[f"{expert_prefix}_v_proj"]
            o_proj = self.modality_specific_weights[f"{expert_prefix}_o_proj"]

            query = self._split_heads(q_proj(normed_hidden_states))
            key = self._split_heads(k_proj(normed_hidden_states))
            value = self._split_heads(v_proj(normed_hidden_states))

            # Apply RoPE if provided. The specific RoPE will be handled by the caller,
            # but this MoELayer provides a hook. The paper implies RoPE is applied at LLM level
            # for 1D and Visual Encoder for 2D, but also says `Qwen2VL-2B [63] ... 1D-RoPE`.
            # For simplicity and adhering to typical LLM structures, RoPE is often applied *before*
            # the attention computation in the LLM's attention block.
            # If `rope_func` is None, it means the LLM's base attention or a wrapper handles it.
            # For NaViL, it is stated that 1D-RoPE is for LLM and 2D-RoPE for visual encoder.
            # The LLM's base attention layer (e.g., from InternLM2 or Qwen3) already handles its RoPE.
            # So, for the MoELayer, we will NOT apply RoPE here. The `rope_func` parameters
            # are kept for future flexibility or if the base LLM does not include RoPE.
            # For this implementation, we assume the LLM's own attention mechanism
            # (which this MoELayer replaces/wraps) would apply its 1D-RoPE.
            # For now, `rope_func` is ignored, and RoPE is not applied directly in `MoELayer`'s attention.

            # Scaled dot-product attention
            attention_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)

            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask

            attention_weights = self.softmax(attention_scores)
            attention_output = torch.matmul(attention_weights, value)

            attention_output = self._merge_heads(attention_output)
            output = o_proj(attention_output)
            return output
        else:
            # FFN-MMoE logic
            gate_proj = self.modality_specific_weights[f"{expert_prefix}_gate_proj"]
            up_proj = self.modality_specific_weights[f"{expert_prefix}_up_proj"]
            down_proj = self.modality_specific_weights[f"{expert_prefix}_down_proj"]

            gate_output = gate_proj(normed_hidden_states)
            up_output = up_proj(normed_hidden_states)

            # SiLU(gate_output) * up_output (element-wise product)
            intermediate_output = self.silu(gate_output) * up_output
            output = down_proj(intermediate_output)
            return output


class Connector(nn.Module):
    """
    Connects the visual encoder's output to the LLM's feature space.
    It downsamples the visual token sequence using PixelUnshuffle and projects
    its feature dimension using an MLP.
    """
    def __init__(self, config: Config):
        """
        Initializes the Connector.

        Args:
            config: The global configuration object.
        """
        super().__init__()
        visual_encoder_config = config.model_architecture.visual_encoder
        llm_moe_config = config.model_architecture.llm_moe

        self.vis_embed_dim = visual_encoder_config.width
        self.llm_embed_dim = llm_moe_config.width

        # The paper says "downsamples ... through pixel shuffle [15]".
        # `nn.PixelShuffle` is an upsampling operation.
        # `nn.PixelUnshuffle` is the inverse, performing downsampling by
        # increasing channel dimension and decreasing spatial dimensions.
        # This aligns with the need to reduce the number of visual tokens (sequence length)
        # while preparing for projection to the LLM's feature space.
        # A downscale_factor of 2 is a common default, as not specified in the paper.
        self.downscale_factor = 2 # Default value, can be made configurable if needed.

        # Ensure that the visual encoder's embedding dimension is divisible by downscale_factor^2
        # if PixelUnshuffle were to operate on the feature dimension as channels.
        # However, PixelUnshuffle operates on spatial dimensions (H, W) and pushes them to channels.
        # So, the input must be (B, C, H, W). C here is self.vis_embed_dim.
        # The output channels will be C * (downscale_factor^2).
        
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=self.downscale_factor)
        
        # After PixelUnshuffle, the channels will be vis_embed_dim * (downscale_factor ** 2)
        mlp_input_dim = self.vis_embed_dim * (self.downscale_factor ** 2)
        
        self.mlp_projector = nn.Linear(mlp_input_dim, self.llm_embed_dim)

    def forward(self, visual_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Connects visual embeddings from the encoder to the LLM's input space.

        Args:
            visual_embeddings: Tensor from VisualEncoder, shape (B, N_patches, D_ve).

        Returns:
            Projected visual embeddings, shape (B, N_new_patches, D_llm).
        """
        batch_size, num_patches, vis_embed_dim = visual_embeddings.shape

        if num_patches == 0:
            logger.warning("Received empty visual embeddings for connector.")
            return torch.empty((batch_size, 0, self.llm_embed_dim), device=visual_embeddings.device, dtype=visual_embeddings.dtype)

        # Infer grid dimensions. This assumes a square grid from the visual encoder.
        # A more robust solution would pass these dimensions from the visual encoder.
        h_grid = w_grid = int(math.sqrt(num_patches))
        if h_grid * w_grid != num_patches:
            # Fallback for non-square patch grids (e.g., from rectangular images)
            # This is a heuristic and might not be optimal.
            logger.warning(
                f"Number of visual patches ({num_patches}) is not a perfect square. "
                "Assuming height is the larger dimension for reshaping. "
                "This might lead to incorrect spatial arrangement for PixelUnshuffle."
            )
            # A common approach is to try to derive H, W from original image H, W and patch_size
            # For now, let's assume we can derive it. This is a potential area for improvement if specific aspect ratios are common.
            # Example: if original H, W are H_img, W_img, and patch_size=16, then H_grid=H_img/16, W_grid=W_img/16.
            # Without original image dims, `sqrt` is the simplest heuristic.
            # If num_patches is not perfect square, the output of pixel unshuffle might be weird.
            # A direct linear projection without spatial reshaping might be an alternative if
            # spatial reasoning is fully handled by LLM after projection.
            # For now, let's proceed with an approximate square root, and make sure to handle integer division carefully.
            # Alternatively, if we know the input image dimensions H_orig, W_orig and patch_stride,
            # H_grid = (H_orig + pad_h) // patch_stride
            # W_grid = (W_orig + pad_w) // patch_stride
            # But PatchEmbed has already performed padding.
            
            # Let's try to infer H_grid, W_grid assuming it was initially a rectangular grid
            # and it should ideally be derived from the visual encoder's output shape
            # e.g., if (B, D_ve, H_out, W_out) -> (B, H_out*W_out, D_ve)
            # We would need H_out, W_out from visual encoder.
            # For now, let's assume num_patches is always a perfect square for simplicity of `sqrt` based `h_grid, w_grid`.
            # If not a perfect square, pixel_unshuffle would still require 4D.
            # The current design assumes (H, W) are from a square region for simplicity of the paper's vague description.
            # A simple workaround: if it's not a perfect square, infer width as the closest integer
            # and height from that, or vice-versa.
            w_grid = int(round(math.sqrt(num_patches)))
            h_grid = num_patches // w_grid
            if h_grid * w_grid != num_patches: # This check handles non-integer ratios after rounding.
                raise ValueError(
                    f"Connector received {num_patches} patches which cannot form a simple 2D grid "
                    f"for PixelUnshuffle (h_grid={h_grid}, w_grid={w_grid}). "
                    "This usually means the visual encoder output is not structured for PixelUnshuffle."
                )

        # Reshape to (B, C, H, W) for PixelUnshuffle
        # Visual embeddings are (B, N_patches, D_ve)
        # Transpose to (B, D_ve, N_patches) then reshape to (B, D_ve, H_grid, W_grid)
        reshaped_embeddings = visual_embeddings.transpose(1, 2).reshape(batch_size, vis_embed_dim, h_grid, w_grid)

        # Apply PixelUnshuffle for downsampling
        # Output shape: (B, D_ve * (downscale_factor^2), H_grid/k, W_grid/k)
        downsampled_embeddings = self.pixel_unshuffle(reshaped_embeddings)

        # Flatten new spatial dimensions and transpose to (B, N_new_patches, D_ve_new)
        # N_new_patches = (H_grid/k) * (W_grid/k)
        # D_ve_new = D_ve * (downscale_factor^2)
        new_batch_size, new_channels, new_h, new_w = downsampled_embeddings.shape
        flattened_embeddings = downsampled_embeddings.flatten(2).transpose(1, 2)
        
        # Apply MLP projector to match LLM's feature space
        projected_embeddings = self.mlp_projector(flattened_embeddings)

        return projected_embeddings

