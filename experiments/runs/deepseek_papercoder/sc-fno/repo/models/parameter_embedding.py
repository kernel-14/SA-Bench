# models/parameter_embedding.py
# ============================================================================
# Purpose: Implements ParameterEmbedding, a module that transforms raw scalar
#          physical parameters (a flat vector) into a constant spatio‑temporal
#          feature map. The resulting feature map is concatenated with the lifted
#          input of the Fourier Neural Operator (FNO), enabling conditioning on
#          physics parameters. This module is intended only for equations where
#          parameters are spatially constant (ODEs and standard PDEs). For the
#          zoned Burgers’ equation (pde2_zoned), the parameters are spatially
#          varying and must be handled differently – this class is **not** used
#          in that case.
# ============================================================================

from typing import Optional, Tuple

import torch
import torch.nn as nn

from config import Config


class ParameterEmbedding(nn.Module):
    """MLP‑based parameter embedding for scalar‑constant physical parameters.

    The module consists of a small multi‑layer perceptron that maps a parameter
    vector of length `num_params` to a vector of length `embed_dim`. This vector
    is then broadcast across the full spatio‑temporal domain (determined from
    the configuration) to produce a feature map of shape
        (batch_size, embed_dim, S_x, N_time)   for 1D + time PDEs,
        (batch_size, embed_dim, S_x, S_y)      for 2D spatial PDEs (Navier‑Stokes),
        (batch_size, embed_dim, N_time)        for ODEs.

    The spatial shape is inferred once from the configuration object and is fixed
    for the lifetime of the module. The `grid` argument in the forward pass is
    accepted only for interface compatibility and is completely ignored.

    Attributes:
        mlp: Sequential MLP mapping (num_params → hidden_dim → embed_dim).
        spatial_shape: Tuple of ints describing the domain grid.
        embed_dim: Number of output channels (from config.model.param_embedding_hidden).
    """

    def __init__(self, config: Config) -> None:
        """Initialise the ParameterEmbedding module.

        Args:
            config: Global, frozen Config object. Relevant sections:
                    - config.equation : current problem key.
                    - config.sol_params[eq] : equation‑specific settings
                      (S_x, S_y, N_time, param_names, spatial_dims).
                    - config.model.param_embedding_hidden : int, output channels.

        Raises:
            ValueError: If the equation is not supported for this embedding
                        (currently only pde2_zoned is unsupported, and the caller
                        is expected to avoid instantiation for that case).
        """
        super().__init__()

        # ---- Retrieve configuration values ----
        eq = config.equation
        eq_params = config.sol_params
        model_params = config.model_params
        embed_dim = model_params.get("param_embedding_hidden", 32)

        # ---- Determine the number of scalar parameters ----
        if eq == "pde2_zoned":
            # This embedding is not designed for zoned parameters (spatially varying).
            # The FNO should handle that case separately. We raise an error to
            # prevent accidental misuse.
            raise ValueError(
                "ParameterEmbedding is not intended for the zoned Burgers' equation "
                "(pde2_zoned). The FNO implementation should bypass this module for "
                "that equation."
            )

        param_names: list = eq_params["param_names"]
        num_params = len(param_names)

        # ---- Build the MLP ----
        # Hidden dimension heuristically chosen as max(32, embed_dim) to ensure
        # enough capacity for simple transformations.
        hidden_dim = max(32, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(num_params, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # ---- Determine the spatial shape for feature map expansion ----
        spatial_dims: int = eq_params.get("spatial_dims", 0)
        if eq == "pde3":
            # Navier‑Stokes: purely spatial (no time dimension in output)
            Sx = eq_params["S_x"]
            Sy = eq_params["S_y"]
            self.spatial_shape: Tuple[int, ...] = (Sx, Sy)
        elif spatial_dims == 1:
            Sx = eq_params["S_x"]
            Nt = eq_params["N_time"]
            self.spatial_shape = (Sx, Nt)
        elif spatial_dims == 0:
            Nt = eq_params["N_time"]
            self.spatial_shape = (Nt,)
        else:
            raise ValueError(f"Unsupported spatial_dims={spatial_dims} for equation '{eq}'.")

        # Store for later reference
        self.embed_dim = embed_dim

    def forward(self, p: torch.Tensor, grid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Produce the embedded feature map from a batch of parameter vectors.

        Args:
            p:    Tensor of shape (batch_size, num_params) containing the raw
                  physical parameters for each sample in the batch.
            grid: Ignored – kept only for compatibility with the FNO interface.

        Returns:
            Tensor of shape (batch_size, embed_dim, *spatial_shape) containing
            the parameter embedding broadcast across the full domain.
        """
        # (B, num_params) -> (B, embed_dim)
        x: torch.Tensor = self.mlp(p)

        # Expand to spatial dimensions. For each spatial dimension, add a
        # trailing dimension of size 1 and then expand to the target size.
        for _ in range(len(self.spatial_shape)):
            x = x.unsqueeze(-1)                     # add a 1‑dimension at the end
        # Now x.shape == (B, embed_dim, 1, ..., 1) with as many trailing 1s as spatial dims.
        x = x.expand(-1, -1, *self.spatial_shape)   # broadcast to full shape

        return x

