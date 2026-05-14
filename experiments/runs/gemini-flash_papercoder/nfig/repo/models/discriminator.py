import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

# Assuming Config class is available from config.py
# To avoid circular import or ensure standalone testability,
# we might define a dummy Config or explicitly import.
# For the purpose of this single file generation, we'll assume Config can be imported.
from config import Config


class Discriminator(nn.Module):
    """
    Discriminator network for the FR-VAE, implemented as a PatchGAN.
    It takes an image as input and outputs a feature map of scores indicating
    realness/fakeness of image patches.
    """

    def __init__(self, config: Config):
        """
        Initializes the Discriminator model.

        Args:
            config: Configuration object containing discriminator hyperparameters
                    under `config.fr_vae.discriminator`.
                    Expected keys:
                        - ndf: Number of filters in the first convolutional layer (default: 64).
                        - n_layers: Number of convolutional blocks (default: 4).
                        - use_spectral_norm: Boolean to apply spectral normalization (default: True).
        """
        super().__init__()

        # --- Hyperparameter extraction with default values ---
        discriminator_cfg = config.fr_vae.get("discriminator", {})
        # Input to discriminator is an RGB image, so in_channels is 3.
        in_channels: int = 3
        ndf: int = discriminator_cfg.get("ndf", 64)
        n_layers: int = discriminator_cfg.get("n_layers", 4)
        use_spectral_norm: bool = discriminator_cfg.get("use_spectral_norm", True)

        model_layers = []

        # --- First layer (no BatchNorm, LeakyReLU only) ---
        # Input: (B, in_channels, H, W) -> Output: (B, ndf, H/2, W/2)
        conv_layer = nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1)
        if use_spectral_norm:
            model_layers.append(spectral_norm(conv_layer))
        else:
            model_layers.append(conv_layer)
        model_layers.append(nn.LeakyReLU(0.2, inplace=True))

        # --- Intermediate layers ---
        # Each layer downsamples spatially by 2
        current_ndf = ndf
        for i in range(1, n_layers - 1):  # Loop for n_layers-2 intermediate blocks
            in_features = current_ndf
            out_features = current_ndf * 2
            conv_layer = nn.Conv2d(in_features, out_features, kernel_size=4, stride=2, padding=1, bias=False)
            if use_spectral_norm:
                model_layers.append(spectral_norm(conv_layer))
            else:
                model_layers.append(conv_layer)
            model_layers.append(nn.LeakyReLU(0.2, inplace=True))
            current_ndf = out_features

        # --- Last intermediate layer before final output (stride 1, no downsampling) ---
        # Output: (B, current_ndf*2, H/2^(n_layers-1), W/2^(n_layers-1))
        # This layer does not downsample spatially
        in_features = current_ndf
        out_features = current_ndf * 2
        conv_layer = nn.Conv2d(in_features, out_features, kernel_size=4, stride=1, padding=1, bias=False)
        if use_spectral_norm:
            model_layers.append(spectral_norm(conv_layer))
        else:
            model_layers.append(conv_layer)
        model_layers.append(nn.LeakyReLU(0.2, inplace=True))
        current_ndf = out_features

        # --- Output layer ---
        # Output: (B, 1, H_patch, W_patch) - a 2D map of scores
        # No activation applied here; loss function typically handles logits.
        conv_layer = nn.Conv2d(current_ndf, 1, kernel_size=4, stride=1, padding=1)
        if use_spectral_norm:
            model_layers.append(spectral_norm(conv_layer))
        else:
            model_layers.append(conv_layer)

        self.model = nn.Sequential(*model_layers)

        # Apply custom weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """
        Custom weight initialization for convolutional layers.
        Follows a common practice for GAN discriminators.
        """
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass through the discriminator.

        Args:
            x: A batch of image tensors (B, 3, H, W), typically normalized to [-1, 1].

        Returns:
            A torch.Tensor of raw logits from the discriminator (B, 1, H_patch, W_patch).
        """
        if x.ndim != 4:
            raise ValueError(f"Input to Discriminator must be 4D (B, C, H, W), but got {x.ndim}D.")
        return self.model(x)

