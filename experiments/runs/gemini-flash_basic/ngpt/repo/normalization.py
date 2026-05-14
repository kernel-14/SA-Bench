import torch

def normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Normalizes a tensor to have unit norm along the specified dimension.
    """
    return x / x.norm(p=2, dim=dim, keepdim=True)


def get_norm_factors(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Returns the norm factors of a tensor along the specified dimension.
    """
    return x.norm(p=2, dim=dim, keepdim=True)


