import torch
def memoryless_noise_schedule(t, beta_fn):
    """
    Implements the memoryless noise schedule: 
    """
    # Compute the noise coefficient using theoretical definitions
    eta_t = beta_fn(t) * (beta_fn(t) - torch.autograd.grad(beta_fn(t), t, retain_graph=True)[0])
    return torch.sqrt(2 * eta_t)

