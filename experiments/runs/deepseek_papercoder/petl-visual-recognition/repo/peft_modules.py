## peft_modules.py

"""
Implementations of 14 PEFT methods as subclasses of PEFTModule.

Each module encapsulates the logic required to modify a ViT-B/16 backbone
during fine-tuning. Features are referenced by the h_i notation defined in
the paper appendix: h1–h9. The model builder injects these modules into
individual Transformer layers and calls their forward methods (or uses
their internal adapters/linear wrappers) to construct the adapted model.
"""

import math
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


# ------------------------------------------------------------------------ #
#  Constants (defaults for ViT-B/16)
# ------------------------------------------------------------------------ #
_D_DEFAULT = 768
_N_PATCHES_DEFAULT = 196
_PATCH_SIZE_DEFAULT = 16


# ------------------------------------------------------------------------ #
#  Abstract Base Class
# ------------------------------------------------------------------------ #
class PEFTModule(ABC, nn.Module):
    """
    Base class for all parameter-efficient fine‑tuning modules.

    Subclasses must override `forward(features: Dict[str, Tensor])`.
    """

    def __init__(self, layer_idx: int, config: Dict):
        """
        Args:
            layer_idx: index (0‑based) of the Transformer layer this module
                       is attached to.  For global modules (e.g., FacT) a
                       sentinel value of -1 is used.
            config:    dictionary with method‑specific hyperparameters and
                       optionally backbone constants (D, N, patch_size).
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.D = config.get("D", _D_DEFAULT)
        self.N = config.get("N", _N_PATCHES_DEFAULT)
        self.patch_size = config.get("patch_size", _PATCH_SIZE_DEFAULT)

    @abstractmethod
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ...

    def trim_output(self, features: Dict[str, torch.Tensor], num_tokens: int) -> Dict[str, torch.Tensor]:
        """
        Remove a given number of leading tokens from every feature tensor.
        Used by VPT‑Deep (and optionally others) to discard prompt tokens.
        """
        if num_tokens <= 0:
            return features
        for k in features:
            if features[k] is not None:
                # features[k].shape: (B, seq_len, D)
                features[k] = features[k][:, num_tokens:, :]
        return features

    def num_trainable_params(self) -> int:
        """Return the total number of trainable elements in this module."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ------------------------------------------------------------------------ #
#  Helper building blocks (not exposed as public PEFTModules)
# ------------------------------------------------------------------------ #
class AdapterBlock(nn.Module):
    """
    Standard bottleneck adapter with a residual connection.

    Parameters:
        down_proj: Linear(D, r)
        up_proj:   Linear(r, D)
        scale:     float multiplier
        with_skip: if True, output = x + scale * up(act(down(x)))
                   if False, output = scale * up(act(down(x)))
    """

    def __init__(self, D: int, r: int, scale: float = 1.0, with_skip: bool = True):
        super().__init__()
        self.D = D
        self.r = r
        self.scale = scale
        self.with_skip = with_skip
        self.down_proj = nn.Linear(D, r, bias=False)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(r, D, bias=False)
        self._init_weights()

    def _init_weights(self):
        # Down projection: Xavier uniform, Up projection: zero
        init.xavier_uniform_(self.down_proj.weight)
        nn.init.zeros_(self.up_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.down_proj(x))
        h = self.up_proj(h)
        h = h * self.scale
        if self.with_skip:
            return x + h
        return h


class ConvpassBlock(nn.Module):
    """
    Convolutional bottleneck block that processes only patch tokens.
    The class token is left unchanged.  The output is a delta that can be
    added to a feature map (e.g., h5 or h9) without affecting the class token.
    """

    def __init__(self, D: int, r: int, scale: float, patch_size: int = 16,
                 xavier_init: bool = False):
        super().__init__()
        self.D = D
        self.r = r
        self.scale = scale
        self.num_patches = (224 // patch_size) ** 2   # 14^2 = 196 for ViT-B

        # Convs: 1x1 -> dw 3x3 -> 1x1
        self.down_conv = nn.Conv2d(D, r, kernel_size=1, bias=False)
        self.mid_conv = nn.Conv2d(r, r, kernel_size=3, padding=1, groups=r, bias=False)   # depthwise
        self.up_conv = nn.Conv2d(r, D, kernel_size=1, bias=False)
        self.act = nn.GELU()

        if xavier_init:
            self._apply_xavier()
        else:
            self._default_init()

    def _default_init(self):
        # Default: Kaiming uniform for down/mid, zero for up
        init.kaiming_uniform_(self.down_conv.weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.mid_conv.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_conv.weight)

    def _apply_xavier(self):
        init.xavier_uniform_(self.down_conv.weight)
        init.xavier_uniform_(self.mid_conv.weight)
        init.xavier_uniform_(self.up_conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (B, 1 + N, D), where first token is class token.
        Returns: delta of same shape, with zeros for class token
                 and `scale * conv_output` for patches.
        """
        B, S, D = x.shape
        x_cls = x[:, 0:1, :]               # (B, 1, D)
        x_patches = x[:, 1:, :]             # (B, N, D)

        # Reshape patches to 2D
        H = W = int(self.num_patches ** 0.5)
        x_patches = x_patches.reshape(B, H, W, D).permute(0, 3, 1, 2)  # (B, D, H, W)

        # Apply convolutions
        h = self.act(self.down_conv(x_patches))
        h = self.act(self.mid_conv(h))
        h = self.up_conv(h)

        # Flatten back to tokens
        h = h.permute(0, 2, 3, 1).reshape(B, self.num_patches, D)      # (B, N, D)

        # Create delta: zeros for class token
        delta = torch.cat([torch.zeros_like(x_cls), h * self.scale], dim=1)
        return delta


class RepAdapterBlock(nn.Module):
    """
    Linear adapter with group‑wise up‑projection, no activation.
    Used by RepAdapter.
    """

    def __init__(self, D: int, r: int, scale: float, groups: int = 4):
        super().__init__()
        assert r % groups == 0, "Bottleneck dimension must be divisible by groups"
        self.D = D
        self.r = r
        self.scale = scale
        self.groups = groups

        self.down_proj = nn.Linear(D, r, bias=False)
        # group‑wise up‑project layers
        self.up_projs = nn.ModuleList()
        for g in range(groups):
            self.up_projs.append(
                nn.Linear(r // groups, D // groups, bias=False)
            )
        self._init_weights()

    def _init_weights(self):
        init.xavier_uniform_(self.down_proj.weight)
        for up in self.up_projs:
            init.xavier_uniform_(up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, seq_len, D)
        Returns delta to be added externally.
        """
        B, S, _ = x.shape
        down = self.down_proj(x)                    # (B, S, r)
        # Reshape to split into groups
        down_g = down.reshape(B, S, self.groups, self.r // self.groups)
        up_parts = []
        for g in range(self.groups):
            part = down_g[:, :, g, :]                # (B, S, r_g)
            up = self.up_projs[g](part)               # (B, S, D_g)
            up_parts.append(up)
        up = torch.cat(up_parts, dim=-1)             # (B, S, D)
        return up * self.scale


# ------------------------------------------------------------------------ #
#  PEFT Method Implementations
# ------------------------------------------------------------------------ #

# ---- Prompt‑Based ----
class VPTShallow(PEFTModule):
    """Visual Prompt Tuning – Shallow version. Only first layer is injected."""

    def __init__(self, layer_idx: int, config: Dict):
        super().__init__(layer_idx, config)
        num_prompts = config["num_prompts"]
        self.num_prompts = num_prompts
        if self.layer_idx == 0:
            self.prompts = nn.Parameter(torch.randn(num_prompts, self.D) * 0.02)
        else:
            self.prompts = None   # placeholder

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.layer_idx == 0 and self.prompts is not None:
            B = features["h1"].shape[0]
            prompts = self.prompts.unsqueeze(0).expand(B, -1, -1)   # (B, l, D)
            features["h1"] = torch.cat([prompts, features["h1"]], dim=1)
        return features


class VPTDeep(PEFTModule):
    """Visual Prompt Tuning – Deep version. Each layer has its own prompts."""

    def __init__(self, layer_idx: int, config: Dict):
        super().__init__(layer_idx, config)
        num_prompts = config["num_prompts"]
        self.num_prompts = num_prompts
        self.prompts = nn.Parameter(torch.randn(num_prompts, self.D) * 0.02)

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        B = features["h1"].shape[0]
        prompts = self.prompts.unsqueeze(0).expand(B, -1, -1)
        features["h1"] = torch.cat([prompts, features["h1"]], dim=1)
        return features

    def trim_output(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Remove the leading prompt tokens from all features."""
        return super().trim_output(features, self.num_prompts)


# ---- Adapter‑Based ----
class HoulAdapter(PEFTModule):
    """
    Houlsby Adapter: two adapters, one after MSA (h5) and one after MLP (h9).
    """

    def __init__(self, layer_idx: int, config: Dict):
        super().__init__(layer_idx, config)
        r = config["bottleneck"]
        s = config["scale"]
        self.adapter1 = AdapterBlock(self.D, r, s, with_skip=True)
        self.adapter2 = AdapterBlock(self.D, r, s, with_skip=True)

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        features["h5"] = self.adapter1(features["h5"])
        features["h9"] = self.adapter2(features["h9"])
        return features


class PfeifAdapter(PEFTModule):
    """Pfeiffer Adapter: single adapter after MLP (h9)."""

    def __init__(self, layer_idx: int, config: Dict):
        super().__init__(layer_idx, config)
        r = config["bottleneck"]
        s = config["scale"]
        self.adapter = AdapterBlock(self.D, r, s, with_skip=True)

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        features["h9"] = self.adapter(features["h9"])
        return features


class AdaptFormer(PEFTModule):
    """
    AdaptFormer: parallel adapter added to the MLP output.
    Adapter takes h7 (LN before MLP) and produces a delta added to h9.
    """

    def __init__(self, layer_idx: int, config: Dict):
        super().__init__(layer_idx, config)
        r = config["bottleneck"]
        s = config["scale"]
        self.adapter = AdapterBlock(self.D, r, s, with_skip=False)

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        features["h9"] = features["h9"] + self.adapter(features["h7"])
        return features


class Convpass(PEFTModule):
    """
    Convpass: convolutional adapters in parallel to MSA and MLP.
    """

    def __init__(self, layer_idx: int, config: Dict):
        super().__init__(layer_idx, config)
        r = config["bottleneck"]
        s = config["scale"]
        xavier = config.get("xavier_init", False)
        self.convpass1 = ConvpassBlock(self.D, r, s, self.patch_size, xavier)
        self.convpass2 = ConvpassBlock(self.D, r, s, self.patch_size, xavier)

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # convpass1 takes h2 and adds delta to h5
        features["h5"] = features["h5"] + self.convpass1(features["h2"])
        # convpass2 takes h7 and adds delta to h9
        features["h9"] = features["h9"] + self.convpass2(features["h7"])
        return features


class RepAdapter(PEFTModule):
    """
    RepAdapter: two linear group‑wise adapters without activation.
    """

    def __init__(self, layer_idx: int, config: Dict):
        super().__init__(layer_idx, config)
        r = config["bottleneck"]
        s = config["scale"]
        groups = config.get("groups", 4)
        self.repadapter1 = RepAdapterBlock(self.D, r, s, groups)
        self.repadapter2 = RepAdapterBlock(self.D, r, s, groups)

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        features["h5"] = features["h5"] + self.repadapter1(features["h2"])
        features["h9"] = features["h9"] + self.repadapter2(features["h6"])
        return features


# ---- Direct Selective Tuning ----
class BitFit(PEFTModule):
    """
    BitFit: no added modules; signals model builder to unfreeze all biases.
    Forward is identity.
    """

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return features


class LayerNormTune(PEFTModule):
    """
    LayerNorm Tuning: no added modules; signals model builder to unfreeze
    only LayerNorm weights/biases. Forward is identity.
    """

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return features


class DiffFit(PEFTModule):
    """
    DiffFit: combines BitFit + LayerNorm tuning and inserts learnable scaling
    factors after MSA (h5) and MLP (h9).
    """

    def __init__(self, layer_idx: int, config: Dict):
        super().__init__(layer_idx, config)
        self.gamma1 = nn.Parameter(torch.ones(self.D))
        self.gamma2 = nn.Parameter(torch.ones(self.D))

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        features["h5"] = features["h5"] * self.gamma1.view(1, 1, -1)
        features["h9"] = features["h9"] * self.gamma2.view(1, 1, -1)
        return features


class SSF(PEFTModule):
    """
    Scale & Shift deep Features: applies learned scale/w and shift/b to
    six key intermediate features (h2, h3, h5, h7, h8, h9).
    """

    def __init__(self, layer_idx: int, config: Dict):
        super().__init__(layer_idx, config)
        self.scales = nn.ParameterDict()
        self.shifts = nn.ParameterDict()
        for key in ["h2", "h3", "h5", "h7", "h8", "h9"]:
            self.scales[key] = nn.Parameter(torch.ones(self.D))
            self.shifts[key] = nn.Parameter(torch.zeros(self.D))

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for key in self.scales.keys():
            features[key] = features[key] * self.scales[key].view(1, 1, -1) + self.shifts[key].view(1, 1, -1)
        return features


# ---- Efficient Selective Tuning (weight‑modifying) ----
class LoRALinear(nn.Module):
    """
    Wrapper around a frozen Linear layer that adds a low‑rank residual.
    Used by LoRA PEFTModule to inject trainable delta into Q and V matrices.
    """

    def __init__(self, original: nn.Linear, down: nn.Linear, up: nn.Linear):
        super().__init__()
        self.original = original      # frozen, parameters stored here
        self.down = down          # trainable
        self.up = up              # trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # original forward with frozen weights
        # We must ensure original's weights are not trained.
        with torch.no_grad():
            base = self.original(x)
        # low‑rank update
        delta = self.up(self.down(x))
        return base + delta


class LoRA(PEFTModule):
    """
    LoRA: Low‑Rank Adaptation applied to Q and V projection weights.
    The PEFTModule itself does not modify features; it provides the
    low‑rank adapters to be wrapped around the original linear layers
    by the model builder.
    """

    def __init__(self, layer_idx: int, config: Dict):
        super().__init__(layer_idx, config)
        r = config["bottleneck"]      # rank
        self.r = r
        # Q
        self.down_Q = nn.Linear(self.D, r, bias=False)
        self.up_Q = nn.Linear(r, self.D, bias=False)
        # V
        self.down_V = nn.Linear(self.D, r, bias=False)
        self.up_V = nn.Linear(self.D, r, bias=False)
        self._init_weights()

    def _init_weights(self):
        # Down matrices: Kaiming uniform; Up matrices: zero
        init.kaiming_uniform_(self.down_Q.weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.down_V.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_Q.weight)
        nn.init.zeros_(self.up_V.weight)

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return features   # No direct feature manipulation

    def get_lora_linear(self, original: nn.Linear, key: str) -> LoRALinear:
        """Return a LoRALinear wrapper for the specified projection."""
        if key == "Q":
            return LoRALinear(original, self.down_Q, self.up_Q)
        elif key == "V":
            return LoRALinear(original, self.down_V, self.up_V)
        else:
            raise ValueError(f"LoRA only supports Q and V, got {key}")


# ------------------------------------------------------------------------
#  FacT Family (global modules, shared across layers)
# ------------------------------------------------------------------------
class FacTLinear(nn.Module):
    """
    Linear wrapper that adds a delta fetched from a FacT PEFTModule.
    """

    def __init__(self, original: nn.Linear, fact_module: "FacTBase",
                 weight_index: int):
        super().__init__()
        self.original = original      # frozen
        self.fact_module = fact_module
        self.weight_index = weight_index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.fact_module.get_delta(self.weight_index)   # (D_out, D_in)
        # delta should be added to weight: output = x @ W^T + x @ delta^T
        # original.forward already computes x @ W^T + bias.
        base = self.original(x)
        extra = F.linear(x, delta)
        return base + extra


class FacTBase(PEFTModule):
    """Common base for FacT_TT and FacT_TK (global modules)."""

    def __init__(self, config: Dict):
        # layer_idx is irrelevant, set to -1
        super().__init__(-1, config)
        self.s = config["scale"]
        self.r = config["bottleneck"]
        self.D = config.get("D", _D_DEFAULT)
        # Determine the number of weight matrices to be factorised.
        # ViT-B has 12 layers; each layer contributes 6 matrices:
        # Q, K, V, O (MSA) + W1, W2 (MLP).  Total = 72.
        num_layers = config.get("num_layers", 12)
        self.num_weights = num_layers * 6
        # Mapping from (layer_idx, type) -> index.
        self._type_to_idx = {"Q": 0, "K": 1, "V": 2, "O": 3, "W1": 4, "W2": 5}

    def weight_index(self, layer_idx: int, weight_type: str) -> int:
        """Return the 0‑based index into the decomposition for a given layer and weight."""
        if weight_type not in self._type_to_idx:
            raise ValueError(f"Unknown weight type: {weight_type}")
        return layer_idx * 6 + self._type_to_idx[weight_type]

    @abstractmethod
    def get_delta(self, weight_idx: int) -> torch.Tensor:
        """Return the additive delta for the specified weight index (shape (D, D))."""
        ...

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return features   # no direct feature modification


class FacT_TT(FacTBase):
    """
    FacT with Tensor‑Train decomposition.
    Parameters: U (D,r), V (D,r), Sigma (num_weights, r, r)
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.U = nn.Parameter(torch.empty(self.D, self.r))
        self.V = nn.Parameter(torch.empty(self.D, self.r))
        self.Sigma = nn.Parameter(torch.empty(self.num_weights, self.r, self.r))
        self._init_weights()

    def _init_weights(self):
        init.xavier_uniform_(self.U)
        init.xavier_uniform_(self.V)
        # Sigma initialized small to keep initial delta near zero
        nn.init.normal_(self.Sigma, std=1e-3)

    def get_delta(self, weight_idx: int) -> torch.Tensor:
        slice_sigma = self.Sigma[weight_idx]            # (r, r)
        delta = self.s * (self.U @ slice_sigma @ self.V.t())   # (D, D)
        return delta


class FacT_TK(FacTBase):
    """
    FacT with Tucker decomposition.
    Parameters: U (D,r), V (D,r), B (num_weights, r), A (r, r, r)
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.U = nn.Parameter(torch.empty(self.D, self.r))
        self.V = nn.Parameter(torch.empty(self.D, self.r))
        self.B = nn.Parameter(torch.empty(self.num_weights, self.r))
        self.A = nn.Parameter(torch.empty(self.r, self.r, self.r))
        self._init_weights()

    def _init_weights(self):
        init.xavier_uniform_(self.U)
        init.xavier_uniform_(self.V)
        nn.init.normal_(self.B, std=1e-3)
        nn.init.normal_(self.A, std=1e-3)

    def get_delta(self, weight_idx: int) -> torch.Tensor:
        # Compute C = A x1 B^T  -> shape (num_weights, r, r)
        C = torch.einsum('abc,ia->ibc', self.A, self.B)[weight_idx]   # (r, r)
        delta = self.s * (self.U @ C @ self.V.t())
        return delta


# ------------------------------------------------------------------------ #
#  Factory function
# ------------------------------------------------------------------------ #
def create_peft_module(method: str,
                       layer_idx: int,
                       hyperparams: Dict,
                       backbone_cfg: Optional[Dict] = None) -> PEFTModule:
    """
    Instantiate the correct PEFTModule subclass.

    Args:
        method:       string key matching config.yaml peft_methods keys.
        layer_idx:    which transformer layer this module belongs to.
        hyperparams:  dictionary of method‑specific hyperparameters.
        backbone_cfg: optional dict with backbone constants (D, N, patch_size).
    Returns:
        An instance of the appropriate PEFTModule subclass.
    """
    cfg = hyperparams.copy()
    if backbone_cfg:
        cfg.update(backbone_cfg)

    if method == "vpt_shallow":
        return VPTShallow(layer_idx, cfg)
    elif method == "vpt_deep":
        return VPTDeep(layer_idx, cfg)
    elif method == "bitfit":
        return BitFit(layer_idx, cfg)
    elif method == "difffit":
        return DiffFit(layer_idx, cfg)
    elif method == "layernorm":
        return LayerNormTune(layer_idx, cfg)
    elif method == "ssf":
        return SSF(layer_idx, cfg)
    elif method == "pfeif_adapter":
        return PfeifAdapter(layer_idx, cfg)
    elif method == "houl_adapter":
        return HoulAdapter(layer_idx, cfg)
    elif method == "adaptformer":
        return AdaptFormer(layer_idx, cfg)
    elif method == "repadapter":
        return RepAdapter(layer_idx, cfg)
    elif method == "convpass":
        return Convpass(layer_idx, cfg)
    elif method == "lora":
        return LoRA(layer_idx, cfg)
    elif method == "fact_tt":
        # FacT is global; layer_idx is ignored, but factory still receives one.
        return FacT_TT(cfg)
    elif method == "fact_tk":
        return FacT_TK(cfg)
    else:
        raise ValueError(f"Unknown PEFT method: '{method}'")
