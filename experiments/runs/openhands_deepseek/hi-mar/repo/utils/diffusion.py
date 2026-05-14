import math
import torch


class NoiseScheduler:
    """Linear noise schedule for diffusion process in continuous token space."""

    def __init__(self, num_train_timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.num_train_timesteps = num_train_timesteps
        betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

    def register_buffer(self, name: str, tensor: torch.Tensor):
        # Mimics nn.Module buffer registration for standalone use
        setattr(self, name, tensor)

    def add_noise(self, x_start: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Forward diffusion: q(x_t | x_0)."""
        sqrt_alpha = self.sqrt_alphas_cumprod.to(x_start.device)[timesteps]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod.to(x_start.device)[timesteps]
        while sqrt_alpha.dim() < x_start.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

    def get_timesteps(self, num_inference_steps: int, device: torch.device) -> torch.Tensor:
        """Return timesteps for inference sampling (descending)."""
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = (torch.arange(0, num_inference_steps, device=device) * step_ratio).long()
        timesteps = torch.cat([timesteps, torch.tensor([self.num_train_timesteps - 1], device=device)])
        return timesteps.flip(0)


def cosine_mask_schedule(num_tokens: int, masked_ratio: float, device: torch.device) -> torch.Tensor:
    """Cosine schedule for selecting which tokens to mask (MaskGIT-style)."""
    rand = torch.rand(num_tokens, device=device)
    threshold = torch.cos(math.pi / 2 * masked_ratio)
    mask = rand > threshold
    return mask


def generate_mask_by_ratio(N: int, ratio: float, B: int, device: torch.device) -> torch.Tensor:
    """Randomly mask ceil(ratio * N) tokens per batch item. Returns [B, N] boolean mask where True = masked."""
    num_mask = math.ceil(ratio * N)
    mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    for b in range(B):
        indices = torch.randperm(N, device=device)[:num_mask]
        mask[b, indices] = True
    return mask


def iterative_decoding_mask(
    num_tokens: int,
    step: int,
    total_steps: int,
    B: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Iterative decoding schedule: at each step, predict some tokens and keep others masked.
    Uses cosine schedule to determine how many tokens to unmask.
    Returns boolean mask where True = still masked.
    """
    progress = step / total_steps
    mask_ratio = math.cos(math.pi / 2 * progress)
    num_masked = int(mask_ratio * num_tokens + 0.5)
    mask = torch.zeros(B, num_tokens, dtype=torch.bool, device=device)
    if num_masked > 0:
        for b in range(B):
            indices = torch.randperm(num_tokens, device=device)[:num_masked]
            mask[b, indices] = True
    return mask


def betas_for_alpha_bar(alpha_bar: torch.Tensor, max_beta: float = 0.999) -> torch.Tensor:
    """Compute beta schedule from pre-computed alpha_bar."""
    betas = []
    for i in range(len(alpha_bar)):
        t1 = i
        t2 = i + 1
        beta = 1 - alpha_bar[t2] / alpha_bar[t1] if i > 0 else 1 - alpha_bar[0]
        betas.append(min(beta, max_beta))
    return torch.tensor(betas)
