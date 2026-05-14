## model.py
import torch
import torch.nn as nn
from transformers import AutoModel
import utils

class VisualEncoder(nn.Module):
    """
    Implementation of the Visual Encoder based on transformer layers and bidirectional attention mechanisms.
    
    Attributes:
        depth (int): Number of transformer layers.
        width (int): Hidden size of transformer layers.
        patch_size (int): Size of image patches.
        mlp_width (int): Width of feed-forward layers.
        attention_heads (int): Number of attention heads in transformers.
        pixel_shuffle_connector (torch.nn.Linear): Downsampling module to align visual and textual embeddings.
    """
    def __init__(self, depth: int = 24, width: int = 1472, patch_size: int = 16, mlp_width: int = 5888, attention_heads: int = 23):
        super(VisualEncoder, self).__init__()
        
        self.depth = depth
        self.width = width
        self.patch_size = patch_size
        self.mlp_width = mlp_width
        self.attention_heads = attention_heads

        # Patch Embedding Layer
        self.patch_embedding = nn.Conv2d(in_channels=3, out_channels=self.width, kernel_size=self.patch_size, stride=self.patch_size)

        # Transformer Layers
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.width,
            nhead=self.attention_heads,
            dim_feedforward=self.mlp_width,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=self.depth)

        # Pixel Shuffle Connector
        self.pixel_shuffle_connector = nn.Linear(self.width, self.width)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Processes input image tensor to generate visual embeddings using transformer layers.

        Args:
            image (torch.Tensor): Input image tensor (shape: [batch_size, 3, H, W]).

        Returns:
            torch.Tensor: Visual embeddings (shape: [batch_size, sequence_length, width]).
        """
        # Apply patch embedding
        patches = self.patch_embedding(image)  # (batch_size, width, H/patch_size, W/patch_size)
        patches = patches.flatten(2).permute(0, 2, 1)  # (batch_size, sequence_length, width)

        # Transformer encoding
        encoded_patches = self.transformer_encoder(patches)

        # Pixel shuffle and alignment
        aligned_embeddings = self.pixel_shuffle_connector(encoded_patches)
        return aligned_embeddings

    def apply_multiscale_packing(self, image: torch.Tensor, scales: list, tau: float, area_threshold: int) -> torch.Tensor:
        """
        Performs multiscale packing to process input image across multiple resolutions.

        Args:
            image (torch.Tensor): Input image tensor (shape: [batch_size, 3, H, W]).
            scales (list): List of scale factors for multisampling.
            tau (float): Downsampling factor.
            area_threshold (int): Minimum area size threshold.

        Returns:
            torch.Tensor: Multiscale visual embeddings packed together.
        """
        return utils.process_visual_multiscale(image=image, scales=scales, tau=tau, area_threshold=area_threshold)


class LLM(nn.Module):
    """
    Large Language Model initialized from a pre-trained model with modality-specific Mixture-of-Experts (MoE).
    
    Attributes:
        model_name (str): Pre-trained model name.
        use_moe (bool): Flag to enable or disable Mixture-of-Experts architecture.
        num_experts (int): Number of experts for modality-specific operations.
        depth (int): Number of transformer layers.
        width (int): Hidden dimensions of transformer layers.
        mlp_width (int): Width of feed-forward layers.
        attention_heads (int): Number of attention heads for attention mechanism.
    """
    def __init__(self, model_name: str = "InternLM2-Base", use_moe: bool = True, num_experts: int = 2, depth: int = 24,
                 width: int = 2048, mlp_width: int = 8192, attention_heads: int = 16):
        super(LLM, self).__init__()

        self.use_moe = use_moe
        self.num_experts = num_experts
        self.depth = depth
        self.width = width
        self.mlp_width = mlp_width
        self.attention_heads = attention_heads

        # Load base pre-trained model
        self.llm = AutoModel.from_pretrained(model_name)

        # Modality-Specific Mixture-of-Experts: Attention Experts
        if self.use_moe:
            self.attention_moe = nn.ModuleList([
                nn.MultiheadAttention(embed_dim=self.width, num_heads=self.attention_heads) for _ in range(self.num_experts)
            ])

            # Modality-Specific Mixture-of-Experts: Feed-Forward Experts
            self.ffn_moe = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.width, self.mlp_width),
                    nn.ReLU(),
                    nn.Linear(self.mlp_width, self.width)
                ) for _ in range(self.num_experts)
            ])

    def forward(self, text: torch.Tensor, multimodal_inputs: torch.Tensor = None) -> torch.Tensor:
        """
        Processes textual inputs and optional multimodal embeddings using the LLM.

        Args:
            text (torch.Tensor): Input textual token embeddings (shape: [batch_size, sequence_length]).
            multimodal_inputs (torch.Tensor): Optional visual embeddings.

        Returns:
            torch.Tensor: Processed outputs combining visual and textual embeddings.
        """
        # Process text through the pre-trained LLM
        text_outputs = self.llm(input_ids=text)

        # Process multimodal inputs using modality-specific MoEs if provided
        if multimodal_inputs is not None and self.use_moe:
            for idx in range(len(self.attention_moe)):
                multimodal_inputs = self.attention_moe[idx](multimodal_inputs, multimodal_inputs, multimodal_inputs)[0]
                multimodal_inputs = self.ffn_moe[idx](multimodal_inputs)

            return multimodal_inputs + text_outputs.last_hidden_state
        else:
            return text_outputs.last_hidden_state


class Model(nn.Module):
    """
    Integrated multimodal model combining VisualEncoder and LLM components.
    
    Attributes:
        visual_encoder (VisualEncoder): Visual encoder for processing images.
        llm (LLM): Large Language Model for processing text and multimodal embeddings.
    """
    def __init__(self, visual_encoder: VisualEncoder, llm: LLM):
        super(Model, self).__init__()
        self.visual_encoder = visual_encoder
        self.llm = llm

        # Projection layer to align visual tokens with LLM embedding dimensions
        self.visual_to_llm_projection = nn.Linear(visual_encoder.width, llm.width)

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        """
        Combines visual and textual inputs into a multimodal embedding.

        Args:
            image (torch.Tensor): Input image tensor (shape: [batch_size, 3, H, W]).
            text (torch.Tensor): Input textual tokens (shape: [batch_size, sequence_length]).

        Returns:
            torch.Tensor: Processed multimodal embeddings.
        """
        # Process image through the visual encoder
        visual_embeddings = self.visual_encoder(image)
        visual_embeddings_aligned = self.visual_to_llm_projection(visual_embeddings)

        # Apply special tokens
        visual_embeddings_with_tokens = utils.apply_special_tokens(visual_embeddings_aligned, token_type="add")

        # Process combined multimodal embeddings through the LLM
        multimodal_inputs = torch.cat([visual_embeddings_with_tokens, text], dim=1)
        multimodal_outputs = self.llm(text=None, multimodal_inputs=multimodal_inputs)

        return multimodal_outputs

    def apply_special_tokens(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Applies special positional tokens to multimodal inputs.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Processed tensor with special tokens.
        """
        return utils.apply_special_tokens(inputs, token_type="add")


if __name__ == "__main__":
    from config import Config
    config_data = Config("config.yaml").get_config()

    # Initialize components based on configurations
    visual_encoder_config = config_data["model"]["visual_encoder"]
    llm_config = config_data["model"]["llm"]

    visual_encoder = VisualEncoder(
        depth=visual_encoder_config["depth"],
        width=visual_encoder_config["width"],
        patch_size=visual_encoder_config["patch_size"],
        mlp_width=visual_encoder_config["mlp_width"],
        attention_heads=visual_encoder_config["attention_heads"]
    )

    llm = LLM(
        model_name=llm_config["model_name"],
        use_moe=llm_config["use_moe"],
        num_experts=llm_config["num_experts"],
        depth=llm_config["depth"],
        width=llm_config["width"],
        mlp_width=llm_config["mlp_width"],
        attention_heads=llm_config["attention_heads"]
    )

    model = Model(visual_encoder, llm)
    print("Model initialized successfully.")
