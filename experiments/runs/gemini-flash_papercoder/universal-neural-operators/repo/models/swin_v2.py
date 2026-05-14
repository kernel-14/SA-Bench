import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from typing import Optional, Union, List, Tuple

from models.base_operator import CoreOperator
from utils import get_activation_fn # Assuming get_activation_fn is defined in utils.py

# Helper function to ensure input is a 2-tuple
def to_2tuple(val: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """
    Converts a single integer or a tuple of two integers into a 2-tuple.
    """
    if isinstance(val, int):
        return (val, val)
    if isinstance(val, tuple) and len(val) == 2 and all(isinstance(v, int) for v in val):
        return val
    raise ValueError(f"Input must be an int or a 2-tuple of ints, but got {val}")


# Helper functions for Swin Windowing
def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Partitions feature map into windows.
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """
    Reverses window partition to feature map.
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of feature map
        W (int): Width of feature map
    Returns:
        x: (B, H, W, C)
    """
    B_num_windows, win_H, win_W, C = windows.shape
    B = int(B_num_windows / (H * W / win_H / win_W))
    x = windows.view(B, H // win_H, W // win_W, win_H, win_W, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)
    return x


class Mlp(nn.Module):
    """
    Multi-Layer Perceptron (MLP) block for Swin Transformer.
    """
    def __init__(self, in_features: int, hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None, act_layer: nn.Module = nn.GELU, drop: float = 0.):
        """
        Initializes the MLP block.

        Args:
            in_features (int): Number of input features.
            hidden_features (Optional[int]): Number of hidden features. If None, defaults to `in_features`.
            out_features (Optional[int]): Number of output features. If None, defaults to `in_features`.
            act_layer (nn.Module): Activation layer to use. Defaults to `nn.GELU`.
            drop (float): Dropout rate. Defaults to 0.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        if not isinstance(in_features, int) or in_features <= 0:
            raise ValueError(f"in_features must be a positive int, got {in_features}")
        if not isinstance(hidden_features, int) or hidden_features <= 0:
            raise ValueError(f"hidden_features must be a positive int, got {hidden_features}")
        if not isinstance(out_features, int) or out_features <= 0:
            raise ValueError(f"out_features must be a positive int, got {out_features}")
        if not isinstance(drop, (int, float)) or not (0.0 <= drop <= 1.0):
            raise ValueError(f"drop must be a float between 0 and 1, got {drop}")

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in blocks).
    """
    def __init__(self, drop_prob: float = 0.):
        """
        Initializes the DropPath module.
        Args:
            drop_prob (float): Probability of dropping a path. Defaults to 0.
        """
        super().__init__()
        if not isinstance(drop_prob, (int, float)) or not (0.0 <= drop_prob <= 1.0):
            raise ValueError(f"drop_prob must be a float between 0 and 1, got {drop_prob}")
        self.drop_prob = drop_prob

    def drop_path(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies drop path to the input tensor.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor with drop path applied.
        """
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # work with diff dim tensors, not just 2D ConvNets
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for DropPath.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor after potential drop path application.
        """
        return self.drop_path(x)


class WindowAttention(nn.Module):
    """
    Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both shifted and non-shifted window attention.
    """

    def __init__(self, dim: int, window_size: Tuple[int, int], num_heads: int,
                 qkv_bias: bool = True, attn_drop: float = 0., proj_drop: float = 0.,
                 pretrained_window_size: Tuple[int, int] = (0, 0)):
        """
        Initializes the WindowAttention module.

        Args:
            dim (int): Number of input channels.
            window_size (Tuple[int, int]): The height and width of the window.
            num_heads (int): Number of attention heads.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            attn_drop (float): Dropout ratio of attention weight.
            proj_drop (float): Dropout ratio of output.
            pretrained_window_size (Tuple[int, int]): Pretrained window resolution, not used in this reproduction
                                                       but included for API completeness.
        """
        super().__init__()
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"dim must be a positive int, got {dim}")
        if not isinstance(window_size, tuple) or len(window_size) != 2 or \
           not all(isinstance(ws, int) and ws > 0 for ws in window_size):
            raise ValueError(f"window_size must be a 2-tuple of positive ints, got {window_size}")
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError(f"num_heads must be a positive int, got {num_heads}")
        if not isinstance(attn_drop, (int, float)) or not (0.0 <= attn_drop <= 1.0):
            raise ValueError(f"attn_drop must be a float between 0 and 1, got {attn_drop}")
        if not isinstance(proj_drop, (int, float)) or not (0.0 <= proj_drop <= 1.0):
            raise ValueError(f"proj_drop must be a float between 0 and 1, got {proj_drop}")
        if not isinstance(pretrained_window_size, tuple) or len(pretrained_window_size) != 2:
            raise ValueError(f"pretrained_window_size must be a 2-tuple, got {pretrained_window_size}")

        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for WindowAttention.
        Args:
            x (torch.Tensor): Input features with shape of (num_windows*B, N, C).
            mask (Optional[torch.Tensor], optional): Mask for shifted window attention.
                                                     Defaults to None.
        Returns:
            torch.Tensor: Output features.
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make type annotations happy

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block.
    """

    def __init__(self, dim: int, input_resolution: Tuple[int, int], num_heads: int, window_size: int = 7,
                 shift_size: int = 0, mlp_ratio: float = 4., qkv_bias: bool = True, drop: float = 0.,
                 attn_drop: float = 0., drop_path: float = 0., act_layer: nn.Module = nn.GELU,
                 norm_layer: type = nn.LayerNorm):
        """
        Initializes the SwinTransformerBlock.

        Args:
            dim (int): Number of input channels.
            input_resolution (Tuple[int, int]): Input resolution (height, width).
            num_heads (int): Number of attention heads.
            window_size (int): Window size. Defaults to 7.
            shift_size (int): Shift size for shifted window attention. Defaults to 0.
            mlp_ratio (float): Ratio of MLP hidden dimension to embedding dimension. Defaults to 4.
            qkv_bias (bool): If True, add a learnable bias to query, key, value. Defaults to True.
            drop (float): Dropout rate. Defaults to 0.
            attn_drop (float): Attention dropout rate. Defaults to 0.
            drop_path (float): Stochastic depth rate. Defaults to 0.
            act_layer (nn.Module): Activation function for MLP. Defaults to nn.GELU.
            norm_layer (type): Normalization layer. Defaults to nn.LayerNorm.
        """
        super().__init__()
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"dim must be a positive int, got {dim}")
        if not isinstance(input_resolution, tuple) or len(input_resolution) != 2 or \
           not all(isinstance(res, int) and res > 0 for res in input_resolution):
            raise ValueError(f"input_resolution must be a 2-tuple of positive ints, got {input_resolution}")
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError(f"num_heads must be a positive int, got {num_heads}")
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError(f"window_size must be a positive int, got {window_size}")
        if not isinstance(shift_size, int) or not (0 <= shift_size < window_size):
            raise ValueError(f"shift_size must be an int between 0 and window_size-1, got {shift_size}")
        if not isinstance(mlp_ratio, (int, float)) or mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be a positive number, got {mlp_ratio}")
        
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            # If window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            H, W = self.input_resolution
            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for SwinTransformerBlock.
        Args:
            x (torch.Tensor): Input features with shape (B, H * W, C).
        Returns:
            torch.Tensor: Output features.
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, N, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, N, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H W C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """
    def __init__(self, img_size: Tuple[int, int] = (224, 224), patch_size: int = 4,
                 in_chans: int = 3, embed_dim: int = 96, norm_layer: Optional[type] = None):
        """
        Initializes the PatchEmbed module.
        Args:
            img_size (Tuple[int, int]): Input image resolution. Defaults to (224, 224).
            patch_size (int): Patch token size. Defaults to 4.
            in_chans (int): Number of input image channels. Defaults to 3.
            embed_dim (int): Number of linear projection output channels. Defaults to 96.
            norm_layer (Optional[type]): Normalization layer. Defaults to None.
        """
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        
        if not isinstance(in_chans, int) or in_chans <= 0:
            raise ValueError(f"in_chans must be a positive int, got {in_chans}")
        if not isinstance(embed_dim, int) or embed_dim <= 0:
            raise ValueError(f"embed_dim must be a positive int, got {embed_dim}")

        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for PatchEmbed.
        Args:
            x (torch.Tensor): Input tensor with shape (B, C, H, W).
        Returns:
            torch.Tensor: Patch embeddings with shape (B, num_patches, embed_dim).
        """
        B, C, H, W = x.shape
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        if C != self.in_chans:
            raise ValueError(f"Input channels ({C}) do not match expected in_chans ({self.in_chans}) for PatchEmbed.")

        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging(nn.Module):
    """
    Patch Merging Layer.
    Downsamples feature map size by 2x.
    """

    def __init__(self, input_resolution: Tuple[int, int], dim: int, norm_layer: type = nn.LayerNorm):
        """
        Initializes the PatchMerging module.
        Args:
            input_resolution (Tuple[int, int]): Resolution of input feature.
            dim (int): Number of input channels.
            norm_layer (type): Normalization layer. Defaults to nn.LayerNorm.
        """
        super().__init__()
        if not isinstance(input_resolution, tuple) or len(input_resolution) != 2:
            raise ValueError(f"input_resolution must be a 2-tuple, got {input_resolution}")
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"dim must be a positive int, got {dim}")

        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for PatchMerging.
        Args:
            x (torch.Tensor): Input features with shape of (B, H * W, C).
        Returns:
            torch.Tensor: Output features after patch merging.
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"
        # assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
    """
    A basic Swin Transformer layer for one stage.
    Consists of `depth` Swin Transformer Blocks and an optional PatchMerging layer.
    """

    def __init__(self, dim: int, input_resolution: Tuple[int, int], depth: int, num_heads: int,
                 window_size: int, mlp_ratio: float = 4., qkv_bias: bool = True, drop: float = 0.,
                 attn_drop: float = 0., drop_path: Union[List[float], float] = 0.,
                 norm_layer: type = nn.LayerNorm, downsample: Optional[nn.Module] = None):
        """
        Initializes the BasicLayer.
        Args:
            dim (int): Number of input channels.
            input_resolution (Tuple[int, int]): Input resolution.
            depth (int): Number of blocks in this layer.
            num_heads (int): Number of attention heads.
            window_size (int): Local window size.
            mlp_ratio (float): Ratio of MLP hidden dimension to embedding dimension.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop (float): Dropout rate.
            attn_drop (float): Attention dropout rate.
            drop_path (Union[List[float], float]): Stochastic depth rate. Can be a single float or list.
            norm_layer (type): Normalization layer.
            downsample (Optional[nn.Module]): Patch merging layer at the end of the layer.
        """
        super().__init__()
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"dim must be a positive int, got {dim}")
        if not isinstance(input_resolution, tuple) or len(input_resolution) != 2:
            raise ValueError(f"input_resolution must be a 2-tuple, got {input_resolution}")
        if not isinstance(depth, int) or depth <= 0:
            raise ValueError(f"depth must be a positive int, got {depth}")
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError(f"num_heads must be a positive int, got {num_heads}")
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError(f"window_size must be a positive int, got {window_size}")

        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])

        # patch merging layer
        self.downsample = downsample
        if self.downsample is not None:
            if not isinstance(self.downsample, PatchMerging):
                raise TypeError(f"downsample must be an instance of PatchMerging, got {type(self.downsample)}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Forward pass for BasicLayer.
        Args:
            x (torch.Tensor): Input features with shape (B, H * W, C).
        Returns:
            Tuple[torch.Tensor, Tuple[int, int]]: Output features and output resolution.
        """
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
            new_resolution = (self.input_resolution[0] // 2, self.input_resolution[1] // 2)
            return x, new_resolution
        return x, self.input_resolution


class SwinV2(CoreOperator):
    """
    Swin Transformer V2 model, adapted as a CoreOperator.
    """

    def __init__(self, input_dim: int, output_dim: int, img_size: Tuple[int, int], patch_size: int,
                 embed_dim: int, depths: List[int], num_heads: List[int], window_size: int = 7,
                 mlp_ratio: float = 4., qkv_bias: bool = True, drop_rate: float = 0.,
                 attn_drop_rate: float = 0., drop_path_rate: Union[List[float], float] = 0.1,
                 norm_layer: type = nn.LayerNorm, **kwargs):
        """
        Initializes the SwinV2 model.

        Args:
            input_dim (int): Number of input channels from the LiftingAdapter output.
            output_dim (int): Number of output channels for the ProjectionAdapter input.
                              This should typically be the same as `input_dim` (i.e., `hidden_dim`).
            img_size (Tuple[int, int]): Input image resolution (e.g., (64, 64)).
                                       Corresponds to the reshaped spatial dimensions of the LiftingAdapter's output.
            patch_size (int): Patch size.
            embed_dim (int): Patch embedding dimension.
            depths (List[int]): Number of blocks in each stage.
            num_heads (List[int]): Number of attention heads in each stage.
            window_size (int): Window size. Defaults to 7.
            mlp_ratio (float): Ratio of MLP hidden dimension to embedding dimension. Defaults to 4.
            qkv_bias (bool): If True, add a learnable bias to query, key, value. Defaults to True.
            drop_rate (float): Dropout rate. Defaults to 0.
            attn_drop_rate (float): Attention dropout rate. Defaults to 0.
            drop_path_rate (Union[List[float], float]): Stochastic depth rate. Defaults to 0.1.
            norm_layer (type): Normalization layer. Defaults to nn.LayerNorm.
        """
        super().__init__()
        if not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError(f"input_dim must be a positive int, got {input_dim}")
        if not isinstance(output_dim, int) or output_dim <= 0:
            raise ValueError(f"output_dim must be a positive int, got {output_dim}")
        if not isinstance(img_size, tuple) or len(img_size) != 2:
            raise ValueError(f"img_size must be a 2-tuple, got {img_size}")
        if not isinstance(patch_size, int) or patch_size <= 0:
            raise ValueError(f"patch_size must be a positive int, got {patch_size}")
        if not isinstance(embed_dim, int) or embed_dim <= 0:
            raise ValueError(f"embed_dim must be a positive int, got {embed_dim}")
        if not isinstance(depths, list) or not all(isinstance(d, int) and d > 0 for d in depths):
            raise ValueError(f"depths must be a list of positive ints, got {depths}")
        if not isinstance(num_heads, list) or not all(isinstance(n, int) and n > 0 for n in num_heads):
            raise ValueError(f"num_heads must be a list of positive ints, got {num_heads}")
        if len(depths) != len(num_heads):
            raise ValueError("depths and num_heads must have the same length.")
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError(f"window_size must be a positive int, got {window_size}")
        
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.output_dim = output_dim # Final output channels for ProjectionAdapter

        # Store initial input image size for reshaping
        self.input_img_size = img_size 
        self.input_flat_res = img_size[0] * img_size[1] # H*W from LiftingAdapter

        # patch embedding layer
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=input_dim,  # input_dim from LiftingAdapter
            embed_dim=embed_dim,
            norm_layer=norm_layer
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution

        # stochastic depth (drop path) for blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        current_dim = embed_dim
        current_resolution = patches_resolution

        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=current_dim,
                input_resolution=current_resolution,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging(current_resolution, current_dim, norm_layer)
                if (i_layer < self.num_layers - 1) else None # Downsample all but the last layer
            )
            self.layers.append(layer)
            if (i_layer < self.num_layers - 1): # Update dim and resolution if downsampled
                current_dim *= 2
                current_resolution = (current_resolution[0] // 2, current_resolution[1] // 2)

        self.norm = norm_layer(current_dim)
        # Final head to project to output_dim
        self.head = nn.Linear(current_dim, output_dim) if current_dim != output_dim else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for SwinV2.

        Args:
            x (torch.Tensor): Input tensor from the LiftingAdapter.
                              Expected shape: (batch_size, num_points, lifting_hidden_dim).
                              `num_points` should be H*W of the `img_size`.

        Returns:
            torch.Tensor: Output tensor for the ProjectionAdapter.
                          Expected shape: (batch_size, final_num_patches, output_dim).
        """
        B, N_flat, C_in = x.shape
        H, W = self.input_img_size

        if N_flat != H * W:
            raise ValueError(
                f"Input tensor's spatial dimension ({N_flat}) must match "
                f"img_size ({H}x{W}={H*W}) after flattening. Got {N_flat} and {H*W}."
            )
        if C_in != self.patch_embed.in_chans:
             raise ValueError(
                f"Input tensor's channel dimension ({C_in}) must match "
                f"PatchEmbed's input_chans ({self.patch_embed.in_chans})."
            )

        # Reshape for 2D processing by PatchEmbed: (B, H*W, C_in) -> (B, C_in, H, W)
        x = x.permute(0, 2, 1).view(B, C_in, H, W)

        # Patch Embedding
        x = self.patch_embed(x) # (B, num_patches, embed_dim)
        x_resolution = self.patch_embed.patches_resolution

        # Swin Transformer Layers
        for layer in self.layers:
            x, x_resolution = layer(x) # x_resolution updates after each downsample

        # Final LayerNorm and Head
        x = self.norm(x)  # (B, final_num_patches, current_dim)
        x = self.head(x) # (B, final_num_patches, output_dim)

        return x

