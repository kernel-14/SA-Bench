import torch
from normalization import normalize

class Embeddings(torch.nn.Module):
    def __init__(self, vocab_size: int, d_model: int, s_z_init: float = 1.0, s_z_scale: float = None):
        super().__init__()
        self.d_model = d_model
        self.E_input = torch.nn.Parameter(torch.rand(vocab_size, d_model))
        self.E_output = torch.nn.Parameter(torch.rand(vocab_size, d_model))
        
        if s_z_scale is None:
            s_z_scale = 1.0 / (d_model**0.5) # Default from paper Section 2.6, point 6

        # s_z is a trainable scaling parameter for logits (Section 2.1, 2.6.6)
        self.s_z_unscaled = torch.nn.Parameter(torch.full((vocab_size,), s_z_init)) # Initialize with s_z_init
        self.s_z_scale_factor = s_z_scale

        self.s_z = self.s_z_unscaled * (s_z_init / s_z_scale) # Effective s_z as per Section 2.5

        # Positional encoding (RoPE) would be applied here, but its specific implementation
        # is abstracted for this reproduction as it's not a core nGPT modification.

    def forward_input(self, tokens: torch.Tensor) -> torch.Tensor:
        # Normalize E_input after each training step (or during forward pass) - Section 2.6, point 2
        self.E_input.data = normalize(self.E_input.data, dim=-1)
        return torch.nn.functional.embedding(tokens, self.E_input)

    def get_logits(self, h: torch.Tensor) -> torch.Tensor:
        # Normalize E_output after each training step (or during forward pass) - Section 2.6, point 2
        self.E_output.data = normalize(self.E_output.data, dim=-1)
        
        # Compute logits (Section 2.1, Equation 1)
        logits = torch.matmul(h, self.E_output.transpose(0, 1))
        
        # Apply trainable scaling parameter s_z (Section 2.1, Equation 3)
        # s_z is applied element-wise to the logits
        return logits * self.s_z


