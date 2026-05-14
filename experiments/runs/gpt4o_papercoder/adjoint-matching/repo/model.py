## model.py

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict


class BaseModel(nn.Module):
    """
    BaseModel class that serves as the foundational implementation for Flow Matching and Diffusion Models.
    It includes functionality for forward sampling, noise prediction, velocity field modeling, 
    and checkpoint management.

    Attributes:
        architecture (nn.Module): Core network architecture (e.g., U-Net-based or custom).
        model_type (str): Specifies whether this is a 'FlowMatching' or 'Diffusion' model.
        device (str): Sets the device ('cuda' or 'cpu') for model computation.
    """

    def __init__(self, params: Dict):
        """
        Initialize the BaseModel with parameters from the configuration file.

        Args:
            params (Dict): Configuration dictionary loaded from `config.yaml`.
        """
        super(BaseModel, self).__init__()
        self.params = params

        # Determine model type and device
        self.model_type = params.get("model", {}).get("type", "FlowMatching")  # Default to FlowMatching
        self.device = params.get("general", {}).get("device", "cpu")

        # Initialize network architecture dynamically based on model type
        if self.model_type == "FlowMatching":
            self.architecture = self._initialize_flow_matching(params)
        elif self.model_type == "Diffusion":
            self.architecture = self._initialize_diffusion_model(params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        # Move model to device
        self.to(self.device)
        print(f"[INFO] BaseModel initialized as {self.model_type} on device {self.device}")

    def forward(self, x: Tensor, t: float) -> Tensor:
        """
        Perform a forward pass through the model.

        For FlowMatching: Returns velocity predictions v(x, t).
        For Diffusion: Returns noise predictions ε(x_k, k).

        Args:
            x (Tensor): Input tensor (current state or sample) of shape [B, C, H, W].
            t (float): Timestep index. For Diffusion models, this may be discrete.

        Returns:
            Tensor: Predicted velocity fields (FlowMatching) or noise (Diffusion).
        """
        if self.model_type == "FlowMatching":
            # Continuous SDE velocity modeling
            return self.architecture(x, t)
        elif self.model_type == "Diffusion":
            # Discrete timestep noise prediction
            return self.architecture(x, int(t))
        else:
            raise ValueError(f"Unsupported model type in forward: {self.model_type}")

    def save_checkpoint(self, path: str) -> None:
        """
        Save the model's state to a checkpoint file.

        Args:
            path (str): Path to save the model checkpoint.
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_type': self.model_type,
        }
        torch.save(checkpoint, path)
        print(f"[INFO] Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """
        Load the model's state from a checkpoint file.

        Args:
            path (str): Path to the checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.model_type = checkpoint['model_type']
        print(f"[INFO] Checkpoint loaded from {path}")

    def _initialize_flow_matching(self, params: Dict) -> nn.Module:
        """
        Initialize the Flow Matching model architecture.

        Args:
            params (Dict): Configuration dictionary.

        Returns:
            nn.Module: Initialized Flow Matching network.
        """
        input_channels = 3  # Default input for 3-channel images
        layers = params.get("FlowMatching", {}).get("layers", 4)  # e.g., number of U-Net layers
        out_channels = params.get("FlowMatching", {}).get("output_channels", 3)

        print("[INFO] Initializing Flow Matching architecture...")
        return UNet(input_channels=input_channels, layers=layers, out_channels=out_channels)

    def _initialize_diffusion_model(self, params: Dict) -> nn.Module:
        """
        Initialize the Diffusion model (DDPM/DDIM) architecture.

        Args:
            params (Dict): Configuration dictionary.

        Returns:
            nn.Module: Initialized Diffusion network.
        """
        input_channels = 3  # Default input for images
        time_embedding_dim = params.get("Diffusion", {}).get("time_embedding_dim", 256)  # e.g., dim for t embeddings
        hidden_dim = params.get("Diffusion", {}).get("hidden_dim", 256)  # Hidden feature size

        print("[INFO] Initializing Diffusion architecture...")
        return UNetWithTime(input_channels=input_channels, time_embedding_dim=time_embedding_dim, hidden_dim=hidden_dim)


class UNet(nn.Module):
    """
    A simplified U-Net implementation for Flow Matching.

    Args:
        input_channels (int): Number of input channels (e.g., 3 for RGB).
        layers (int): Number of downsampling/upsampling layers.
        out_channels (int): Number of output channels (e.g., 3 corresponding to velocity field dimensions).
    """

    def __init__(self, input_channels: int = 3, layers: int = 4, out_channels: int = 3):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            *[nn.Conv2d(input_channels if i == 0 else 64, 64, kernel_size=3, stride=1, padding=1) for i in range(layers)]
        )
        self.decoder = nn.Sequential(
            *[nn.ConvTranspose2d(64, 64 if i + 1 < layers else out_channels, kernel_size=3, stride=1, padding=1) for i in range(layers)]
        )
        self.activate = nn.LeakyReLU(inplace=True)
    
    def forward(self, x: Tensor, t: float) -> Tensor:
        # Simple U-Net forward pass
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return self.activate(dec)


class UNetWithTime(nn.Module):
    """
    U-Net with time embeddings for DDPM/DDIM.

    Args:
        input_channels (int): Number of input channels (e.g., 3 for RGB).
        time_embedding_dim (int): Dimension of the time embeddings.
        hidden_dim (int): Intermediate feature size.
    """

    def __init__(self, input_channels: int, time_embedding_dim: int, hidden_dim: int):
        super(UNetWithTime, self).__init__()
        self.time_embedding = nn.Sequential(
            nn.Linear(1, time_embedding_dim),
            nn.ReLU(),
            nn.Linear(time_embedding_dim, hidden_dim)
        )
        self.base_model = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, input_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x: Tensor, t: int) -> Tensor:
        # Compute time embedding
        t_emb = self.time_embedding(t.view(-1, 1).float())
        t_emb = t_emb.view(t_emb.size(0), -1, 1, 1)  # Reshape for broadcasting

        # Forward pass with time-dependent modifications
        x = x + t_emb
        return self.base_model(x)
