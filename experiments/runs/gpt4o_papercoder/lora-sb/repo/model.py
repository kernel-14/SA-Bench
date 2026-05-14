## model.py
import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Tuple


class LoRA_SB_Model(nn.Module):
    """
    LoRA_SB_Model: Implements the LoRA-SB methodology for low-rank adaptation using update approximation.
    Supports initialization via Truncated SVD and efficient forward propagation augmented with low-rank updates.
    """
    def __init__(self, base_model_name: str = "mistral-7B", rank: int = 32, scaling_factor: float = 1.0) -> None:
        """
        Initialize the LoRA_SB_Model.

        Args:
            base_model_name (str): Name of the pre-trained backbone model (e.g., "mistral-7B", "roberta-large").
            rank (int): Desired rank for low-rank matrices \(r\) (\(r \ll \min(m, n)\)).
            scaling_factor (float): Scaling factor \(s\) to stabilize updates (\(s = 1.0\) for scaling independence).
        """
        super(LoRA_SB_Model, self).__init__()
        
        # Load pre-trained backbone model using HuggingFace Transformers
        self.base_model = AutoModel.from_pretrained(base_model_name)
        self.base_model.eval()  # Freeze base model parameters by default
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Extract embedding dimensions from model configuration
        self.hidden_size = self.base_model.config.hidden_size
        self.rank = rank
        self.scaling_factor = scaling_factor

        # Initialize fixed low-rank matrices (B, A) and trainable matrix (R)
        self.B = nn.Parameter(torch.randn(self.hidden_size, self.rank), requires_grad=False)
        self.A = nn.Parameter(torch.randn(self.rank, self.hidden_size), requires_grad=False)
        self.R = nn.Parameter(torch.eye(self.rank) / scaling_factor, requires_grad=True)

        # Normalize B and A matrices to be orthonormal
        self._normalize_matrices()

    def _normalize_matrices(self) -> None:
        """
        Normalize B and A matrices to ensure orthonormality.

        Ensures:
            B^T B = I
            A A^T = I
        """
        self.B.data = self._orthonormalize(self.B.data)
        self.A.data = self._orthonormalize(self.A.data)

    @staticmethod
    def _orthonormalize(matrix: torch.Tensor) -> torch.Tensor:
        """
        Perform orthonormalization of a matrix using QR decomposition.

        Args:
            matrix (torch.Tensor): Input matrix to orthonormalize.

        Returns:
            torch.Tensor: Orthonormal matrix.
        """
        q, _ = torch.linalg.qr(matrix)
        return q

    def initialize_with_svd(self, gradients: torch.Tensor, num_samples: int) -> None:
        """
        Initialize B, A, and R matrices using SVD decomposition of averaged gradients.

        Args:
            gradients (torch.Tensor): Gradient matrix \(\Delta W_{\text{avg}}\), size \(m \times n\).
            num_samples (int): Number of samples used for gradient averaging.
        """
        if not isinstance(gradients, torch.Tensor):
            raise TypeError("Gradients must be a torch.Tensor.")
        if gradients.ndim != 2:
            raise ValueError("Gradients must be a 2D tensor.")

        # Perform truncated SVD on the gradient matrix
        U, S, Vt = torch.linalg.svd(gradients, full_matrices=False)

        # Resize and extract the top-rank components
        self.B.data = U[:, :self.rank]  # Top `rank` columns of U
        self.A.data = Vt[:self.rank, :].T  # Top `rank` rows of Vt transposed
        self.R.data = torch.diag(S[:self.rank]) / self.scaling_factor  # Scale singular values into R

        # Ensure orthonormality for B and A post-initialization
        self._normalize_matrices()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for LoRA-SB, augmenting the base model's predictions with low-rank updates.

        Args:
            x (torch.Tensor): Input batch tensor from tokenized sequences.

        Returns:
            torch.Tensor: Augmented predictions with low-rank updates.
        """
        # Base model forward pass
        base_output = self.base_model(**x)

        # Compute low-rank update: s * B * R * A
        low_rank_update = self.scaling_factor * torch.mm(torch.mm(self.B, self.R), self.A)

        # Augment base model outputs with the low-rank updates
        augmented_output = base_output.last_hidden_state + low_rank_update

        return augmented_output
