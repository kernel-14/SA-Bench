import torch
import torch.nn as nn
from einops import rearrange
from typing import Dict, Any, Union, List, Tuple

from models.base_operator import CoreOperator
from models.fno import FNO # Assuming FNO is correctly imported from its module

# Mamba-SSM library import. This is a third-party package.
# Ensure it's installed via pip install mamba-ssm
try:
    from mamba_ssm import Mamba
except ImportError:
    raise ImportError(
        "The 'mamba-ssm' library is required for MambaFNO. "
        "Please install it using 'pip install mamba-ssm'."
    )


class MambaSSMBlock(nn.Module):
    """
    A wrapper for the Mamba-SSM block, designed to integrate into the neural operator
    architecture. It expects input in (batch, sequence_length, dim) format.
    """

    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        """
        Initializes the MambaSSMBlock.

        Args:
            dim (int): The input and output feature dimension of the Mamba block.
                       This typically corresponds to the hidden_dim of the neural operator.
            d_state (int): The dimension of the state `h` in the Mamba-SSM. Default: 16.
            d_conv (int): The kernel size for the 1D convolution within the Mamba block. Default: 4.
            expand (int): The expansion factor for the intermediate hidden dimensions
                          within the Mamba block. Default: 2.
        """
        super().__init__()
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"dim must be a positive integer, got {dim}")
        if not isinstance(d_state, int) or d_state <= 0:
            raise ValueError(f"d_state must be a positive integer, got {d_state}")
        if not isinstance(d_conv, int) or d_conv <= 0:
            raise ValueError(f"d_conv must be a positive integer, got {d_conv}")
        if not isinstance(expand, int) or expand <= 0:
            raise ValueError(f"expand must be a positive integer, got {expand}")

        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass through the Mamba block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, dim).
        """
        if x.dim() != 3:
            raise ValueError(
                f"Expected input to MambaSSMBlock to be 3D (batch, seq_len, dim), "
                f"but got {x.shape}"
            )
        return self.mamba(x)


class MambaFNO(CoreOperator):
    """
    Implements the Mamba FNO model, which integrates a Mamba-SSM module
    after the lifting layer and before the FNO blocks.

    The Mamba module processes the lifted features as a sequence (flattened spatial dimensions),
    then the output is reshaped back for the FNO layers.
    """

    def __init__(self, hidden_dim: int, fno_config: Dict[str, Any], mamba_config: Dict[str, Any]):
        """
        Initializes the MambaFNO model.

        Args:
            hidden_dim (int): The dimensionality of the hidden feature representation
                              throughout the model. This is used for both Mamba and FNO.
            fno_config (Dict[str, Any]): Configuration parameters for the FNO component.
                                         Expected keys: 'num_fourier_modes', 'num_layers', 'mlp_width', 'activation'.
            mamba_config (Dict[str, Any]): Configuration parameters for the Mamba component.
                                          Expected keys: 'd_state', 'd_conv', 'expand'.
        """
        super().__init__()
        if not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be a positive integer, got {hidden_dim}")
        if not isinstance(fno_config, dict):
            raise TypeError(f"fno_config must be a dictionary, got {type(fno_config)}")
        if not isinstance(mamba_config, dict):
            raise TypeError(f"mamba_config must be a dictionary, got {type(mamba_config)}")

        self.hidden_dim = hidden_dim

        # Initialize MambaSSMBlock
        # The 'dim' parameter for MambaSSMBlock is the hidden_dim of the overall operator.
        self.mamba_block = MambaSSMBlock(dim=self.hidden_dim, **mamba_config)

        # Initialize FNO component
        # Ensure FNO uses the same hidden_dim
        # It's better to explicitly pass the common hidden_dim to FNO,
        # even if fno_config might contain it.
        self.fno = FNO(hidden_dim=self.hidden_dim, **fno_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass through the MambaFNO model.

        Input `x` is the output from the LiftingAdapter, typically
        (batch_size, H, W, hidden_dim).

        The process is:
        1. Reshape `x` from (B, H, W, C) to (B, H*W, C) for Mamba.
        2. Pass through MambaSSMBlock.
        3. Reshape output from Mamba back to (B, H, W, C) for FNO.
        4. Pass through FNO.

        Args:
            x (torch.Tensor): Input tensor from the LiftingAdapter.
                              Expected shape: (batch_size, H, W, hidden_dim).

        Returns:
            torch.Tensor: Output tensor from the FNO.
                          Expected shape: (batch_size, H, W, hidden_dim).
        """
        if x.dim() != 4 or x.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Expected input to MambaFNO to have shape (batch_size, H, W, hidden_dim={self.hidden_dim}), "
                f"but got {x.shape}"
            )

        # Store spatial dimensions (H, W)
        batch_size, H, W, channels = x.shape

        # 1. Reshape for Mamba: (B, H, W, C) -> (B, H*W, C)
        x_reshaped_for_mamba = rearrange(x, 'b h w c -> b (h w) c')

        # 2. Pass through MambaSSMBlock
        mamba_output = self.mamba_block(x_reshaped_for_mamba)

        # 3. Reshape output from Mamba back to (B, H, W, C) for FNO
        fno_input = rearrange(mamba_output, 'b (h w) c -> b h w c', h=H, w=W)

        # 4. Pass through FNO
        fno_output = self.fno(fno_input)

        return fno_output

