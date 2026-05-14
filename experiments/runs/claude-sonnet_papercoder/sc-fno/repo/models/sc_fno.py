## models/sc_fno.py
"""SC-FNO model factory and FNO building blocks.

This module serves two distinct responsibilities:

1. **FNO building blocks** — ``FNOBlock1d`` and ``FNOBlock2d``, the per-layer
   components assembled inside ``FNO`` (defined in ``models/fno.py``).  Each
   block implements one Fourier layer:

       output = activation(spectral_conv(x) + linear_skip(x))

2. **Model factory** — ``build_model(cfg)``, which constructs an ``FNO``
   instance and stamps it with a ``variant`` string tag.  The ``Trainer``
   reads this tag to decide which loss terms to activate, keeping the
   architecture code completely free of loss logic.

Design principle
----------------
FNO, SC-FNO, FNO-PINN, and SC-FNO-PINN are **identical neural networks**.
The only difference is the loss configuration.  This module enforces that
principle by making the variant a metadata attribute rather than a subclass.

Valid variant tags (``VALID_VARIANTS``)
---------------------------------------
+------------------+-------------------+------------+------------------+
| Tag              | use_sensitivity   | use_pinn   | Loss terms       |
+==================+===================+============+==================+
| ``"fno"``        | False             | False      | L_u only         |
+------------------+-------------------+------------+------------------+
| ``"sc_fno"``     | True              | False      | L_u + L_s        |
+------------------+-------------------+------------+------------------+
| ``"fno_pinn"``   | False             | True       | L_u + L_Eq       |
+------------------+-------------------+------------+------------------+
| ``"sc_fno_pinn"``| True              | True       | L_u + L_s + L_Eq |
+------------------+-------------------+------------+------------------+

Hyperparameters from Table C.7 of the SC-FNO paper
----------------------------------------------------
- width = 20 (number of channels in hidden layers)
- n_fourier_layers = 4
- modes = 8 for all dimensions
- activation = GELU

References
----------
- Li et al. (2021): "Fourier Neural Operator for Parametric Partial
  Differential Equations" (https://arxiv.org/abs/2010.08895)
- SC-FNO paper Table C.7: Hyperparameters for FNOs
- SC-FNO paper Section 2.4: Implementation Details
- config.yaml: model.width=20, model.n_fourier_layers=4, model.activation='gelu'
"""

from typing import Dict, List, Optional, Type

import torch
import torch.nn as nn

from models.spectral_conv import SpectralConv1d, SpectralConv2d


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: All valid variant tags.  Used for validation in ``build_model`` and as
#: CLI argument choices in ``main.py``.
VALID_VARIANTS: List[str] = ["fno", "sc_fno", "fno_pinn", "sc_fno_pinn"]

#: Mapping from activation name (config.yaml ``model.activation``) to the
#: corresponding ``nn.Module`` class.  Extend this dict to support additional
#: activations without changing any other code.
_ACTIVATION_MAP: Dict[str, Type[nn.Module]] = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
    "elu":  nn.ELU,
}


def _build_activation(name: str = "gelu") -> nn.Module:
    """Instantiates an activation module by name.

    Args:
        name: Lowercase activation name.  Must be a key in ``_ACTIVATION_MAP``.
              Sourced from ``config.yaml`` key ``model.activation`` (default
              ``'gelu'``).

    Returns:
        An instantiated ``nn.Module`` activation.

    Raises:
        ValueError: If ``name`` is not in ``_ACTIVATION_MAP``.

    Example:
        >>> act = _build_activation("gelu")
        >>> isinstance(act, nn.GELU)
        True
    """
    name_lower: str = name.strip().lower()
    if name_lower not in _ACTIVATION_MAP:
        raise ValueError(
            f"Unknown activation '{name}'. "
            f"Valid options: {sorted(_ACTIVATION_MAP.keys())}. "
            f"Check config.yaml key 'model.activation'."
        )
    return _ACTIVATION_MAP[name_lower]()


# ---------------------------------------------------------------------------
# FNO building blocks
# ---------------------------------------------------------------------------

class FNOBlock1d(nn.Module):
    """Single Fourier layer for 1D sequences (ODEs and 1D PDEs).

    Implements one FNO layer for 1D inputs:

        output = activation(spectral_conv(x) + linear_skip(x))

    where ``spectral_conv`` is a ``SpectralConv1d`` (the Fourier integral
    operator) and ``linear_skip`` is a pointwise ``nn.Conv1d`` with
    ``kernel_size=1`` (the local residual/skip connection).

    Used by ``FNO`` in ``models/fno.py`` for:
      - ODE1, ODE2: 1D time sequences, ``modes=8`` (Table C.7)
      - PDE1, PDE2, PDE4: when the 1D FNO variant is selected

    Attributes:
        spectral_conv: ``SpectralConv1d`` applying the Fourier integral operator.
        linear: Pointwise ``Conv1d`` (``kernel_size=1``) as the skip connection.
        activation: Activation module (default ``nn.GELU``).

    Example:
        >>> block = FNOBlock1d(width=20, modes=8)
        >>> x = torch.randn(4, 20, 100)   # [B, C, L]
        >>> out = block(x)
        >>> out.shape
        torch.Size([4, 20, 100])
    """

    def __init__(
        self,
        width: int = 20,
        modes: int = 8,
        activation: str = "gelu",
    ) -> None:
        """Initialises ``FNOBlock1d``.

        Args:
            width: Number of feature channels (FNO hidden dimension).
                   Sourced from ``config.yaml`` ``model.width`` (default 20,
                   Table C.7).
            modes: Number of Fourier modes for the spectral convolution.
                   Sourced from ``config.yaml`` ``model.modes_t`` or
                   ``model.modes_x`` (default 8, Table C.7).
                   Must satisfy ``modes <= L // 2 + 1`` at runtime.
            activation: Activation function name.  Sourced from
                        ``config.yaml`` ``model.activation`` (default
                        ``'gelu'``).
        """
        super().__init__()

        self.spectral_conv: SpectralConv1d = SpectralConv1d(
            in_channels=width,
            out_channels=width,
            modes=modes,
        )

        # Pointwise linear skip connection.
        # ``kernel_size=1`` makes this equivalent to a per-position linear
        # layer — it provides a local linear transformation that complements
        # the global Fourier convolution.  The two paths are *summed* (not
        # concatenated) before the activation.
        self.linear: nn.Conv1d = nn.Conv1d(
            in_channels=width,
            out_channels=width,
            kernel_size=1,
        )

        self.activation: nn.Module = _build_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies one 1D FNO layer.

        Computes ``activation(spectral_conv(x) + linear(x))``.

        Args:
            x: Input tensor of shape ``[B, width, L]`` where:
               - ``B`` is the batch size
               - ``width`` is the number of feature channels
               - ``L`` is the sequence length (time steps or spatial points)
               Must be real-valued (float32).

        Returns:
            Output tensor of shape ``[B, width, L]``.  Same spatial/temporal
            resolution as the input.

        Example:
            >>> block = FNOBlock1d(width=20, modes=8)
            >>> x = torch.randn(4, 20, 100)
            >>> block(x).shape
            torch.Size([4, 20, 100])
        """
        # Spectral path: F⁻¹(R_φ · F(v_t))
        x_spectral: torch.Tensor = self.spectral_conv(x)   # [B, width, L]

        # Skip connection: pointwise linear W·v_t
        x_linear: torch.Tensor = self.linear(x)            # [B, width, L]

        # Sum and apply activation.
        return self.activation(x_spectral + x_linear)      # [B, width, L]


class FNOBlock2d(nn.Module):
    """Single Fourier layer for 2D grids (1D PDEs and PDE3).

    Implements one FNO layer for 2D inputs:

        output = activation(spectral_conv(x) + linear_skip(x))

    where ``spectral_conv`` is a ``SpectralConv2d`` and ``linear_skip`` is a
    pointwise ``nn.Conv2d`` with ``kernel_size=1``.

    Used by ``FNO`` in ``models/fno.py`` for:
      - PDE1, PDE2, PDE4: 2D ``(x, t)`` inputs, ``modes1=modes2=8`` (Table C.7)
      - PDE3 (Navier-Stokes): 2D spatial ``(x, y)`` inputs, ``modes1=modes2=8``

    Dimension assignment by equation:

    +----------+-------+-------+------------------+------------------+
    | Equation | H     | W     | modes1 source    | modes2 source    |
    +==========+=======+=======+==================+==================+
    | PDE1     | Sx=20 | T=30  | ``modes_x``      | ``modes_t``      |
    +----------+-------+-------+------------------+------------------+
    | PDE2     | Sx=40 | T=30  | ``modes_x``      | ``modes_t``      |
    +----------+-------+-------+------------------+------------------+
    | PDE3     | Sx=64 | Sy=64 | ``modes_x``      | ``modes_y``      |
    +----------+-------+-------+------------------+------------------+
    | PDE4     | Sx=40 | T=30  | ``modes_x``      | ``modes_t``      |
    +----------+-------+-------+------------------+------------------+

    All mode values are 8 in every case (Table C.7), so ``modes1 == modes2 == 8``
    always.  The code reads them separately from config to remain general.

    Attributes:
        spectral_conv: ``SpectralConv2d`` applying the 2D Fourier integral operator.
        linear: Pointwise ``Conv2d`` (``kernel_size=1``) as the skip connection.
        activation: Activation module (default ``nn.GELU``).

    Example:
        >>> block = FNOBlock2d(width=20, modes1=8, modes2=8)
        >>> x = torch.randn(4, 20, 64, 64)   # [B, C, H, W]
        >>> out = block(x)
        >>> out.shape
        torch.Size([4, 20, 64, 64])
    """

    def __init__(
        self,
        width: int = 20,
        modes1: int = 8,
        modes2: int = 8,
        activation: str = "gelu",
    ) -> None:
        """Initialises ``FNOBlock2d``.

        Args:
            width: Number of feature channels (FNO hidden dimension).
                   Sourced from ``config.yaml`` ``model.width`` (default 20,
                   Table C.7).
            modes1: Number of Fourier modes in the first spatial dimension.
                    Sourced from ``config.yaml`` ``model.modes_x`` (default 8,
                    Table C.7).  Must satisfy ``modes1 <= H // 2 + 1`` at
                    runtime.
            modes2: Number of Fourier modes in the second spatial dimension.
                    Sourced from ``config.yaml`` ``model.modes_t`` or
                    ``model.modes_y`` (default 8, Table C.7).  Must satisfy
                    ``modes2 <= W // 2 + 1`` at runtime.
            activation: Activation function name.  Sourced from
                        ``config.yaml`` ``model.activation`` (default
                        ``'gelu'``).
        """
        super().__init__()

        self.spectral_conv: SpectralConv2d = SpectralConv2d(
            in_channels=width,
            out_channels=width,
            modes1=modes1,
            modes2=modes2,
        )

        # Pointwise linear skip connection.
        # ``kernel_size=1`` makes this a per-position linear transformation
        # applied independently at each (H, W) location.
        self.linear: nn.Conv2d = nn.Conv2d(
            in_channels=width,
            out_channels=width,
            kernel_size=1,
        )

        self.activation: nn.Module = _build_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies one 2D FNO layer.

        Computes ``activation(spectral_conv(x) + linear(x))``.

        Args:
            x: Input tensor of shape ``[B, width, H, W]`` where:
               - ``B`` is the batch size
               - ``width`` is the number of feature channels
               - ``H`` is the first spatial dimension (e.g., Sx or T)
               - ``W`` is the second spatial dimension (e.g., Sy or Sx)
               Must be real-valued (float32).

        Returns:
            Output tensor of shape ``[B, width, H, W]``.  Same spatial
            resolution as the input.

        Example:
            >>> block = FNOBlock2d(width=20, modes1=8, modes2=8)
            >>> x = torch.randn(4, 20, 64, 64)
            >>> block(x).shape
            torch.Size([4, 20, 64, 64])
        """
        # Spectral path: F⁻¹(R_φ · F(v_t))
        x_spectral: torch.Tensor = self.spectral_conv(x)   # [B, width, H, W]

        # Skip connection: pointwise linear W·v_t
        x_linear: torch.Tensor = self.linear(x)            # [B, width, H, W]

        # Sum and apply activation.
        return self.activation(x_spectral + x_linear)      # [B, width, H, W]


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(cfg: dict) -> "FNO":  # type: ignore[name-defined]
    """Constructs an FNO instance and stamps it with a variant tag.

    This is the single entry point for model construction throughout the
    SC-FNO codebase.  It reads the variant from ``cfg['variant']``, builds
    an ``FNO`` with the equation-specific hyperparameters, and attaches the
    variant string as ``model.variant``.

    The ``Trainer`` reads ``model.variant`` to decide which loss terms to
    activate:

    .. code-block:: python

        use_sensitivity = model.variant in ('sc_fno', 'sc_fno_pinn')
        use_pinn        = model.variant in ('fno_pinn', 'sc_fno_pinn')

    The ``FNO`` class itself is completely agnostic to the variant — it only
    implements the forward pass.

    Config key resolution
    ---------------------
    ``cfg`` should be a *merged* equation-specific dict where all required
    keys are present at the top level.  ``main.py`` is responsible for
    extracting the equation sub-config (e.g., ``cfg['pde1']``) and merging
    it with global defaults before calling this function.

    Required ``cfg`` keys
    ---------------------
    - ``'equation'``: str, e.g. ``'ode1'``, ``'pde1'``, ``'pde3'``
    - ``'n_params'``: int, number of physical parameters
    - ``'variant'``: str, one of ``VALID_VARIANTS``
    - ``'model'``: dict with sub-keys:
        - ``'dim'``: int (1 for ODEs, 2 for PDEs)
        - ``'width'``: int (default 20, Table C.7)
        - ``'n_fourier_layers'``: int (default 4, Table C.7)
        - ``'modes_t'``: int (default 8, Table C.7)
        - ``'modes_x'``: int (default 8, Table C.7)
        - ``'modes_y'``: int (default 8, Table C.7, PDE3 only)
        - ``'activation'``: str (default ``'gelu'``)
        - ``'normalize_params'``: bool (default True)
    - ``'discretization'``: dict with sub-keys:
        - ``'M'``: int, input time steps
        - ``'N'``: int, total time steps
    - ``'params'``: dict mapping param_name → [a, b] (for normalisation)

    Args:
        cfg: Merged equation-specific configuration dictionary.  See above
             for required keys.  Sourced from ``config.yaml`` after
             equation-specific extraction and global-default merging in
             ``main.py``.

    Returns:
        An ``FNO`` instance with ``model.variant`` set to ``cfg['variant']``.
        The model is on CPU; move to device in the training loop.

    Raises:
        ValueError: If ``cfg['variant']`` is not in ``VALID_VARIANTS``.
        KeyError: If a required config key is missing.

    Example:
        >>> from utils.config_loader import ConfigLoader
        >>> cfg_loader = ConfigLoader('config.yaml')
        >>> master_cfg = cfg_loader.cfg
        >>> # Merge PDE1 sub-config with global defaults
        >>> pde1_cfg = master_cfg['pde1'].copy()
        >>> pde1_cfg['variant'] = 'sc_fno'
        >>> pde1_cfg['model'] = {**master_cfg['model'], **pde1_cfg.get('model', {})}
        >>> model = build_model(pde1_cfg)
        >>> model.variant
        'sc_fno'
        >>> model.count_parameters()  # ~107,897 for PDE1 (Table C.7)
    """
    # ------------------------------------------------------------------
    # Step 1: Validate the variant tag.
    # ------------------------------------------------------------------
    variant: str = str(cfg.get("variant", "fno")).lower()
    if variant not in VALID_VARIANTS:
        raise ValueError(
            f"Invalid model variant '{variant}'. "
            f"Must be one of {VALID_VARIANTS}. "
            f"Check config.yaml key 'variants[*].name' or the --variant CLI flag."
        )

    # ------------------------------------------------------------------
    # Step 2: Validate that required top-level keys are present.
    # ------------------------------------------------------------------
    required_keys: List[str] = ["equation", "n_params"]
    for key in required_keys:
        if key not in cfg:
            raise KeyError(
                f"build_model: required config key '{key}' is missing. "
                f"Available keys: {sorted(cfg.keys())}. "
                f"Ensure the equation sub-config is merged with global defaults "
                f"before calling build_model()."
            )

    # ------------------------------------------------------------------
    # Step 3: Validate mode counts against grid sizes.
    # This is a best-effort check using config values; the FNO's forward
    # pass will raise a more specific RuntimeError if modes exceed the
    # actual tensor dimensions at runtime.
    # ------------------------------------------------------------------
    model_cfg: dict = cfg.get("model", {})
    disc_cfg: dict = cfg.get("discretization", {})
    equation: str = str(cfg.get("equation", "pde1")).lower()

    modes_t: int = int(model_cfg.get("modes_t", 8))
    modes_x: int = int(model_cfg.get("modes_x", 8))
    modes_y: int = int(model_cfg.get("modes_y", 8))

    N: int = int(disc_cfg.get("N", 30))
    Sx: int = int(disc_cfg.get("Sx", 20))
    Sy: int = int(disc_cfg.get("Sy", 64))

    # For 1D sequences (ODEs), check modes_t against N.
    if equation in ("ode1", "ode2"):
        if modes_t > N // 2 + 1:
            raise ValueError(
                f"build_model: modes_t={modes_t} exceeds the number of "
                f"available rfft frequencies for N={N} time steps "
                f"(max={N // 2 + 1}). Reduce modes_t in config.yaml."
            )

    # For 2D PDEs, check modes against spatial and temporal dimensions.
    if equation in ("pde1", "pde2", "pde4"):
        if modes_x > Sx // 2 + 1:
            raise ValueError(
                f"build_model: modes_x={modes_x} exceeds the number of "
                f"available rfft frequencies for Sx={Sx} spatial points "
                f"(max={Sx // 2 + 1}). Reduce modes_x in config.yaml."
            )
        if modes_t > N // 2 + 1:
            raise ValueError(
                f"build_model: modes_t={modes_t} exceeds the number of "
                f"available rfft frequencies for N={N} time steps "
                f"(max={N // 2 + 1}). Reduce modes_t in config.yaml."
            )

    if equation == "pde3":
        if modes_x > Sx // 2 + 1:
            raise ValueError(
                f"build_model: modes_x={modes_x} exceeds the number of "
                f"available rfft frequencies for Sx={Sx} (max={Sx // 2 + 1}). "
                f"Reduce modes_x in config.yaml."
            )
        if modes_y > Sy // 2 + 1:
            raise ValueError(
                f"build_model: modes_y={modes_y} exceeds the number of "
                f"available rfft frequencies for Sy={Sy} (max={Sy // 2 + 1}). "
                f"Reduce modes_y in config.yaml."
            )

    # ------------------------------------------------------------------
    # Step 4: Import FNO here (inside the function body) to avoid the
    # circular import that would arise if fno.py imports FNOBlock1d/2d
    # from sc_fno.py at module level while sc_fno.py imports FNO from
    # fno.py at module level.
    # ------------------------------------------------------------------
    from models.fno import FNO  # pylint: disable=import-outside-toplevel

    # ------------------------------------------------------------------
    # Step 5: Construct the FNO instance.
    # ------------------------------------------------------------------
    model: FNO = FNO(cfg)

    # ------------------------------------------------------------------
    # Step 6: Stamp the variant tag onto the model instance.
    # This is the *only* place the variant is attached — FNO.__init__
    # sets self.variant = 'fno' as a default; we override it here.
    # ------------------------------------------------------------------
    model.variant = variant

    # ------------------------------------------------------------------
    # Step 7: Log construction summary.
    # ------------------------------------------------------------------
    n_params_learnable: int = model.count_parameters()
    print(
        f"[build_model] Built model: equation='{equation}' | "
        f"variant='{variant}' | "
        f"learnable_params={n_params_learnable:,} | "
        f"use_sensitivity={variant in ('sc_fno', 'sc_fno_pinn')} | "
        f"use_pinn={variant in ('fno_pinn', 'sc_fno_pinn')}"
    )

    return model


# ---------------------------------------------------------------------------
# Convenience helpers used by main.py and evaluator.py
# ---------------------------------------------------------------------------

def get_variant_flags(variant: str) -> Dict[str, bool]:
    """Returns a dict of boolean flags for the given variant tag.

    Provides a single source of truth for which loss terms are active for
    each variant.  Used by ``Trainer._compute_total_loss`` and
    ``Evaluator.evaluate_sensitivity``.

    Args:
        variant: One of ``VALID_VARIANTS``.

    Returns:
        Dict with keys:
          - ``'use_sensitivity'``: True if the sensitivity loss ``L_s`` is active.
          - ``'use_pinn'``: True if the PINN equation loss ``L_Eq`` is active.

    Raises:
        ValueError: If ``variant`` is not in ``VALID_VARIANTS``.

    Example:
        >>> get_variant_flags('sc_fno')
        {'use_sensitivity': True, 'use_pinn': False}
        >>> get_variant_flags('sc_fno_pinn')
        {'use_sensitivity': True, 'use_pinn': True}
        >>> get_variant_flags('fno')
        {'use_sensitivity': False, 'use_pinn': False}
    """
    variant_lower: str = str(variant).lower()
    if variant_lower not in VALID_VARIANTS:
        raise ValueError(
            f"Unknown variant '{variant}'. Must be one of {VALID_VARIANTS}."
        )
    return {
        "use_sensitivity": variant_lower in ("sc_fno", "sc_fno_pinn"),
        "use_pinn": variant_lower in ("fno_pinn", "sc_fno_pinn"),
    }


def list_variants() -> List[str]:
    """Returns the list of all valid variant tag strings.

    Convenience wrapper around ``VALID_VARIANTS`` for use in CLI argument
    parsers (``argparse.add_argument('--variant', choices=list_variants())``).

    Returns:
        A copy of ``VALID_VARIANTS`` as a new list.

    Example:
        >>> list_variants()
        ['fno', 'sc_fno', 'fno_pinn', 'sc_fno_pinn']
    """
    return list(VALID_VARIANTS)
