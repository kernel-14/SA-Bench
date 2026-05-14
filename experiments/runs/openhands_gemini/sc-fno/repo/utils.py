
import torch

def relative_l2_error(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """
    Calculates the relative L2 error between predicted and true values.
    """
    return torch.norm(pred - true) / torch.norm(true)

def r2_score(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """
    Calculates the R^2 score.
    """
    ss_res = torch.sum((true - pred)**2)
    ss_tot = torch.sum((true - torch.mean(true))**2)
    return 1 - ss_res / ss_tot
