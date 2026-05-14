import torch
import torch.nn.functional as F

def init_scaling_parameter(shape, scale_factor=1.0):
    """ Initialize a scaling parameter. """
    param = torch.ones(shape) * scale_factor
    return torch.nn.Parameter(param)


def apply_scaling(param, scale_init, scale_factor):
    """Adjust and apply scaling to the parameter during usage."""
    return param * (scale_init / scale_factor)


def normalize_matrix(matrix):
    """Normalize a given matrix along its embedding dimensions."""
    return F.normalize(matrix, p=2, dim=0)


def normalize_vector(vector):
    """Normalize a vector to have unit L2 norm."""
    return F.normalize(vector, p=2, dim=-1)
