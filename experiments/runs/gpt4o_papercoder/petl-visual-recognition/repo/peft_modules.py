## peft_modules.py

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict


class AdaptationLayer(nn.Module):
    """Base class for Adapters that contains common logic for up/down projections."""
    def __init__(self, input_dim: int, bottleneck_dim: int, scale: float = 1.0):
        super(AdaptationLayer, self).__init__()
        self.down = nn.Linear(input_dim, bottleneck_dim, bias=False)
        self.up = nn.Linear(bottleneck_dim, input_dim, bias=False)
        self.scale = scale

    def forward(self, x: Tensor) -> Tensor:
        return self.scale * self.up(torch.relu(self.down(x))) + x


class ConvPassLayer(nn.Module):
    """Convolution-Based PEFT Layer for ConvPass adapters."""
    def __init__(self, input_dim: int, bottleneck_dim: int, kernel_size: int = 3, scale: float = 1.0):
        super(ConvPassLayer, self).__init__()
        self.down = nn.Conv2d(input_dim, bottleneck_dim, kernel_size=1, stride=1, bias=False)
        self.mid = nn.Conv2d(bottleneck_dim, bottleneck_dim, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.up = nn.Conv2d(bottleneck_dim, input_dim, kernel_size=1, stride=1, bias=False)
        self.scale = scale

    def forward(self, x: Tensor) -> Tensor:
        # Expect x to be [batch, channels, height, width] for ConvPass
        identity = x
        x = self.down(x)
        x = torch.relu(self.mid(x))
        x = self.up(x)
        return self.scale * x + identity


class PEFTModules:
    """
    Implements Parameter-Efficient Fine-Tuning (PEFT) methods for Vision Transformers
    based on the configurations from 'config.yaml' and paper specifications.
    """

    def __init__(self):
        pass

    # Prompt-Based Methods
    def apply_vpt_shallow(self, params: Dict) -> nn.Module:
        """
        Applies VPT-Shallow by adding learnable prompts to the input of the first Transformer layer.
        """
        num_prompts = params.get("num_prompts", 5)
        embedding_dim = params.get("embedding_dim", 768)  # ViT-B/16 default
        return nn.Parameter(torch.randn(num_prompts, embedding_dim))

    def apply_vpt_deep(self, params: Dict) -> nn.Module:
        """
        Applies VPT-Deep by adding learnable prompts to all Transformer layers.
        """
        layers = params.get("num_layers", 12)  # Default for ViT-B/16
        num_prompts = params.get("num_prompts", 5)
        embedding_dim = params.get("embedding_dim", 768)
        return nn.ParameterList([torch.randn(num_prompts, embedding_dim) for _ in range(layers)])

    # Adapter-Based Methods
    def apply_houl_adapter(self, params: Dict) -> nn.Module:
        """
        Houl. Adapter: Adds lightweight adapters after MSA & MLP blocks.
        """
        bottleneck_dim = params.get("bottleneck_dim", 16)
        embedding_dim = params.get("embedding_dim", 768)
        scale = params.get("scale", 1.0)
        return AdaptationLayer(embedding_dim, bottleneck_dim, scale)

    def apply_pfeif_adapter(self, params: Dict) -> nn.Module:
        """
        Pfeif. Adapter: Adds a lightweight adapter after the MLP block only.
        """
        bottleneck_dim = params.get("bottleneck_dim", 16)
        embedding_dim = params.get("embedding_dim", 768)
        scale = params.get("scale", 1.0)
        return AdaptationLayer(embedding_dim, bottleneck_dim, scale)

    def apply_adaptformer(self, params: Dict) -> nn.Module:
        """
        AdaptFormer: Adds adapters in parallel with MLP blocks.
        """
        bottleneck_dim = params.get("bottleneck_dim", 16)
        embedding_dim = params.get("embedding_dim", 768)
        scale = params.get("scale", 1.0)
        return AdaptationLayer(embedding_dim, bottleneck_dim, scale)

    def apply_convpass(self, params: Dict) -> nn.Module:
        """
        ConvPass Adapter: Combines convolutional bottlenecks with adapters.
        """
        bottleneck_dim = params.get("bottleneck_dim", 16)
        embedding_dim = params.get("embedding_dim", 768)
        kernel_size = params.get("kernel_size", 3)
        scale = params.get("scale", 1.0)
        return ConvPassLayer(embedding_dim, bottleneck_dim, kernel_size, scale)

    def apply_repadapter(self, params: Dict) -> nn.Module:
        """
        RepAdapter: Sequential linear adapters with re-parameterization logic.
        """
        bottleneck_dim = params.get("bottleneck_dim", 16)
        embedding_dim = params.get("embedding_dim", 768)
        scale = params.get("scale", 1.0)
        return AdaptationLayer(embedding_dim, bottleneck_dim, scale)

    # Direct Selective Tuning Methods
    def apply_bitfit(self, model: nn.Module) -> None:
        """
        BitFit: Updates bias terms in the model. Does not return a new module.
        """
        for name, param in model.named_parameters():
            if "bias" in name:
                param.requires_grad = True

    def apply_layernorm(self, model: nn.Module) -> None:
        """
        LayerNorm Tuning: Fine-tunes LayerNorm parameters only.
        """
        for name, param in model.named_parameters():
            if "LayerNorm" in name or "norm" in name or "ln" in name:
                param.requires_grad = True

    def apply_difffit(self, model: nn.Module, params: Dict) -> None:
        """
        DiffFit: Combines BitFit and LayerNorm tuning, with additional scale factors.
        """
        self.apply_bitfit(model)
        self.apply_layernorm(model)

        scale_factors = params.get("scales", [1.0, 1.0])  # Default scale factors for MSA and FFN
        gamma_msa, gamma_ffn = scale_factors
        model.gamma_msa = nn.Parameter(torch.tensor(gamma_msa))
        model.gamma_ffn = nn.Parameter(torch.tensor(gamma_ffn))

    # Efficient Selective Tuning Methods
    def apply_lora(self, params: Dict) -> nn.Module:
        """
        LoRA: Injects low-rank decomposition matrices into QKV weights.
        """
        rank = params.get("rank", 4)
        embedding_dim = params.get("embedding_dim", 768)
        return AdaptationLayer(embedding_dim, rank)

    def apply_fact_tt(self, params: Dict) -> nn.Module:
        """
        FacT-TT (Tensor-Train): Tensor-based low-rank weight updates.
        """
        rank = params.get("rank", 4)
        embedding_dim = params.get("embedding_dim", 768)
        return AdaptationLayer(embedding_dim, rank)

    def apply_fact_tk(self, params: Dict) -> nn.Module:
        """
        FacT-TK (Tucker): Tensor-based low-rank weight updates.
        """
        rank = params.get("rank", 4)
        embedding_dim = params.get("embedding_dim", 768)
        return AdaptationLayer(embedding_dim, rank)

    def apply_ssf(self, params: Dict) -> nn.Module:
        """
        SSF: Scales and shifts intermediate features of the ViT backbone.
        """
        embedding_dim = params.get("embedding_dim", 768)
        scale = params.get("scale", 1.0)
        shift = params.get("shift", 0.0)
        return nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim, bias=True),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Parameter(torch.full((embedding_dim,), scale)),
            nn.Parameter(torch.full((embedding_dim,), shift))
        )
