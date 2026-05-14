"""Pyramidal Flow Matching Algorithm.

Core implementation of the paper's pyramidal flow matching:
- Spatial pyramid: piecewise flow across K resolution stages
- Temporal pyramid: compressed history conditions for autoregressive generation
- Unified flow matching objective (Eq. 11)
- Coupled noise sampling (Eqs. 9-10)
- Inference with corrective renoising (Algorithm 1, Eq. 15)
"""
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import math
import einops


def down_sample(x: torch.Tensor, factor: int) -> torch.Tensor:
    """Bilinear downsampling by factor (2^k)."""
    if factor == 1:
        return x
    return F.interpolate(x, scale_factor=1.0 / factor, mode="bilinear", align_corners=False)


def up_sample(x: torch.Tensor, factor: int) -> torch.Tensor:
    """Nearest-neighbor upsampling by factor (2^k)."""
    if factor == 1:
        return x
    return F.interpolate(x, scale_factor=factor, mode="nearest")


def down_sample_nearest(x: torch.Tensor, factor: int) -> torch.Tensor:
    """Nearest-neighbor downsampling (block averaging for noise)."""
    if factor == 1:
        return x
    return F.interpolate(x, scale_factor=1.0 / factor, mode="nearest")


def get_stage_boundaries(
    num_stages: int = 3,
    uniform: bool = True,
) -> List[Tuple[int, float, float]]:
    """Get (resolution_level, s_k, e_k) for each pyramid stage.

    Paper convention: stage k = resolution level, where factor = 2^k.
    - k = 0: full resolution (factor 1), latest timesteps
    - k = K-1: most compressed (factor 2^{K-1}), earliest timesteps

    Algorithm 1: for k = K-1 to 0 do
    The most compressed stages handle early (noisy) timesteps,
    the full-resolution stage handles the final timesteps.

    For K=3 uniform:
      resolution_level=2 (factor 4): [0, 1/3]
      resolution_level=1 (factor 2): [1/3, 2/3]
      resolution_level=0 (factor 1): [2/3, 1]

    Returns list sorted by resolution_level (ascending: 0=full, K-1=compressed),
    which is reverse timestep order — the inference loop should iterate reversed.
    """
    boundaries = []
    for resolution_level in range(num_stages):
        # Higher resolution_level → earlier timestep window
        # resolution_level = 0 (full res) → last window
        # resolution_level = K-1 (most compressed) → first window
        window_idx = num_stages - 1 - resolution_level
        s_k = window_idx / num_stages
        e_k = (window_idx + 1) / num_stages
        boundaries.append((resolution_level, s_k, e_k))
    return boundaries


def compute_endpoints_coupled(
    x1: torch.Tensor,
    stage_k: int,
    s_k: float,
    e_k: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute coupled endpoints for pyramidal flow (Eqs. 7-10).

    Coupled sampling ensures the noise has the same direction for start and end,
    improving flow trajectory straightness.

    Stage k in the paper corresponds to resolution 2^k (k=0 is full res).
    Here stage_k=0 means full resolution (factor=1=2^0).

    Args:
        x1: Clean data latent at full resolution (B, C, H, W).
        stage_k: Current stage index (0 = full resolution).
        s_k: Start timestep for this stage.
        e_k: End timestep for this stage.

    Returns:
        x_hat_end: Noisy endpoint at resolution 2^k.
        x_hat_start: Noisy start point at resolution 2^(k+1) upsampled to 2^k.
        noise: The shared noise vector at resolution 2^k.
    """
    factor_end = 2 ** stage_k  # Resolution factor for endpoint
    factor_start = 2 ** (stage_k + 1)  # Resolution factor for start

    # Downsample clean latent to endpoint resolution
    x1_down_end = down_sample(x1, factor_end)

    # Downsample clean latent further, then upsample back
    x1_down_start = down_sample(x1, factor_start)
    x1_up_start = up_sample(x1_down_start, 2)  # Upsample by factor 2

    # Sample shared noise at endpoint resolution
    noise = torch.randn_like(x1_down_end)

    # Eq. 9: Endpoint
    x_hat_end = e_k * x1_down_end + (1 - e_k) * noise

    # Eq. 10: Start point (upsampled from lower resolution)
    x_hat_start = s_k * x1_up_start + (1 - s_k) * noise

    return x_hat_end, x_hat_start, noise


def compute_endpoints_uncoupled(
    x1: torch.Tensor,
    stage_k: int,
    s_k: float,
    e_k: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute uncoupled endpoints (Eqs. 7-8, without coupling)."""
    factor_end = 2 ** stage_k
    factor_start = 2 ** (stage_k + 1)

    x1_down_end = down_sample(x1, factor_end)
    x1_down_start = down_sample(x1, factor_start)
    x1_up_start = up_sample(x1_down_start, 2)

    noise_end = torch.randn_like(x1_down_end)
    noise_start = torch.randn_like(x1_up_start)

    x_hat_end = e_k * x1_down_end + (1 - e_k) * noise_end
    x_hat_start = s_k * x1_up_start + (1 - s_k) * noise_start

    return x_hat_end, x_hat_start


def interpolate_flow(
    x_hat_start: torch.Tensor,
    x_hat_end: torch.Tensor,
    t_prime: float,
) -> torch.Tensor:
    """Linear interpolation within a pyramid stage (Eq. 6).

    x_hat_t = t_prime * x_hat_end + (1 - t_prime) * x_hat_start
    """
    return t_prime * x_hat_end + (1 - t_prime) * x_hat_start


def pyramidal_flow_matching_loss(
    model: torch.nn.Module,
    x1: torch.Tensor,
    stage_boundaries: List[Tuple[int, float, float]],
    context: torch.Tensor,
    pooled_text: torch.Tensor,
    clip_context: Optional[torch.Tensor] = None,
    temporal_pos_ids: Optional[torch.Tensor] = None,
    frame_boundaries: Optional[List[int]] = None,
    coupled_sampling: bool = True,
) -> torch.Tensor:
    """Unified flow matching loss for pyramidal flow (Eq. 11).

    Loss = E_{k, t, (x_hat_e_k, x_hat_s_k)} || v_t(x_hat_t) - (x_hat_end - x_hat_start) ||^2

    Args:
        model: MM-DiT model that predicts velocity.
        x1: Clean latent (B, C, H, W) at full resolution.
        stage_boundaries: List of (k, s_k, e_k) tuples.
        context: T5 text embeddings.
        pooled_text: Pooled T5 embedding.
        clip_context: CLIP text embeddings.
        temporal_pos_ids: Frame indices for RoPE.
        frame_boundaries: Token boundaries for causal masking.
        coupled_sampling: Whether to use coupled noise sampling.

    Returns:
        Scalar loss value.
    """
    B = x1.shape[0]
    H, W = x1.shape[2], x1.shape[3]

    # Select a random stage
    stage_idx = torch.randint(0, len(stage_boundaries), (1,)).item()
    k, s_k, e_k = stage_boundaries[stage_idx]

    # Sample random t within the stage window
    t = s_k + torch.rand(B, device=x1.device) * (e_k - s_k)
    t_prime = (t - s_k) / (e_k - s_k)

    if coupled_sampling:
        x_hat_end, x_hat_start, _ = compute_endpoints_coupled(x1, k, s_k, e_k)
    else:
        x_hat_end, x_hat_start = compute_endpoints_uncoupled(x1, k, s_k, e_k)

    # Interpolate to get x_hat_t (Eq. 6)
    x_hat_t = interpolate_flow(x_hat_start, x_hat_end, t_prime.view(-1, 1, 1, 1))

    # Target velocity: delta between endpoint and start point
    target_velocity = x_hat_end - x_hat_start

    # Compute spatial dimensions at resolution 2^k
    factor = 2 ** k
    current_h, current_w = H // factor, W // factor

    # Predict velocity using the model
    # For low-resolution stages, we need to handle the different resolution
    predicted_velocity = model(
        latent=x_hat_t,
        timestep=t,
        context=context,
        pooled_text=pooled_text,
        clip_context=clip_context,
        spatial_h=current_h,
        spatial_w=current_w,
        temporal_pos_ids=temporal_pos_ids,
        frame_boundaries=frame_boundaries,
        resolution_level=k,
    )

    loss = F.mse_loss(predicted_velocity, target_velocity, reduction="mean")
    return loss


def renoise_jump_point(
    x_up: torch.Tensor,
    s_k: float,
    e_k_plus_1: float,
    corrective_gamma: float = -1.0 / 3.0,
) -> torch.Tensor:
    """Apply corrective renoising at jump points between pyramid stages.

    Implements Eq. 15 from the paper (derived in Appendix A).
    For nearest-neighbor upsampling with gamma = -1/3:

    x_hat_{s_k} = (1 + s_k) / 2 * Up(x_hat_{e_{k+1}}) + sqrt(3) * (1 - s_k) / 2 * n'

    The timestep relationship: e_{k+1} = 2 * s_k / (1 + s_k)

    Args:
        x_up: Upsampled latent from previous stage endpoint.
        s_k: Start timestep of current stage.
        e_k_plus_1: End timestep of previous (more compressed) stage.
        corrective_gamma: Gamma parameter (-1/3 for minimum noise).

    Returns:
        Renoised latent for the start of the current stage.
    """
    # Eq. 25: alpha = (1 - s_k) / sqrt(1 - gamma)
    alpha = (1 - s_k) / math.sqrt(1 - corrective_gamma)

    # Eq. 25: e_{k+1} = s_k * sqrt(1-gamma) / ((1-s_k)*sqrt(-gamma) + s_k*sqrt(1-gamma))
    # For gamma = -1/3: e_{k+1} = 2*s_k / (1 + s_k)
    # And rescaling coeff: s_k / e_{k+1} = (1 + s_k) / 2
    rescale_coeff = s_k / e_k_plus_1

    # Generate corrective noise
    n_prime = torch.randn_like(x_up)

    # Generate corrective noise with blockwise correlation structure
    if corrective_gamma < 0:
        # Apply decorrelation correction
        # For nearest-neighbor blocks (2x2 spatial), correlate with gamma
        B, C, H, W = n_prime.shape
        n_prime_reshaped = n_prime.view(B, C, H // 2, 2, W // 2, 2)
        # Create correlated noise within 2x2 blocks
        n_corr = torch.zeros_like(n_prime_reshaped)
        block_mean = n_prime_reshaped.mean(dim=(3, 5), keepdim=True)
        # Apply -gamma correlation within blocks
        n_corr = n_prime_reshaped + corrective_gamma * (block_mean - n_prime_reshaped)
        n_prime = n_corr.view(B, C, H, W)

    x_renoised = rescale_coeff * x_up + alpha * n_prime
    return x_renoised


def renoise_jump_point_nearest(
    x_up: torch.Tensor,
    s_k: float,
) -> torch.Tensor:
    """Specialized renoising for nearest-neighbor upsampling with gamma=-1/3.

    Eq. 15:  x_hat_{s_k} = (1+s_k)/2 * Up(x_hat_{e_{k+1}}) + sqrt(3)*(1-s_k)/2 * n'
    """
    coef_rescale = (1.0 + s_k) / 2.0
    coef_noise = math.sqrt(3.0) * (1.0 - s_k) / 2.0

    n_prime = torch.randn_like(x_up)

    B, C, H, W = n_prime.shape
    if H >= 2 and W >= 2:
        n_prime_reshaped = n_prime.view(B, C, H // 2, 2, W // 2, 2)

        block_mean = n_prime_reshaped.mean(dim=(3, 5), keepdim=True)
        # gamma = -1/3 for maximum decorrelation
        n_corr = n_prime_reshaped - (1.0 / 3.0) * (n_prime_reshaped - block_mean)
        n_prime = n_corr.view(B, C, H, W)

    x_renoised = coef_rescale * x_up + coef_noise * n_prime
    return x_renoised


def generate_pyramidal_flow(
    model: torch.nn.Module,
    stage_boundaries: List[Tuple[int, float, float]],
    context: torch.Tensor,
    pooled_text: torch.Tensor,
    num_steps_per_stage: int = 20,
    spatial_h: int = 96,
    spatial_w: int = 96,
    clip_context: Optional[torch.Tensor] = None,
    temporal_pos_ids: Optional[torch.Tensor] = None,
    frame_boundaries: Optional[List[int]] = None,
    corrective_gamma: float = -1.0 / 3.0,
    cfg_scale: float = 7.0,
    vae: Optional[torch.nn.Module] = None,
) -> torch.Tensor:
    """Generate samples using pyramidal flow matching (Algorithm 1).

    Inference proceeds from the lowest resolution to full resolution:
    1. Start from pure Gaussian noise at the lowest resolution
    2. For each stage k = K-1 down to 0:
       - Solve ODE from s_k to e_k using the flow model
       - At jump point: upsample + renoise for next stage
    3. Output full-resolution latent

    Args:
        model: Trained MM-DiT model.
        stage_boundaries: (k, s_k, e_k) for each stage, sorted by decreasing k.
        context: T5 text embeddings (B, L, D).
        pooled_text: Pooled T5 embedding (B, D).
        num_steps_per_stage: Number of ODE steps per stage.
        spatial_h, spatial_w: Full-resolution latent spatial dims.
        clip_context: CLIP text embeddings (optional).
        temporal_pos_ids: Frame indices for RoPE.
        frame_boundaries: Token boundaries for causal masking.
        corrective_gamma: Gamma for corrective noise.
        cfg_scale: Classifier-free guidance scale.
        vae: Optional VAE for decoding to pixel space.

    Returns:
        Generated latent at full resolution.
    """
    device = model.parameters().__next__().device
    B = context.shape[0]
    K = len(stage_boundaries)

    # Start with pure noise at the lowest resolution (stage K-1)
    _, s_km1, _ = stage_boundaries[-1]  # Most compressed stage
    init_factor = 2 ** (K - 1)
    init_h, init_w = spatial_h // init_factor, spatial_w // init_factor
    x_hat_s_k = torch.randn(B, model.in_channels, init_h, init_w, device=device)

    # Iterate from lowest resolution (K-1) to full resolution (0)
    for stage_idx in range(K - 1, -1, -1):
        k, s_k, e_k = stage_boundaries[stage_idx]
        factor = 2 ** k
        stage_h, stage_w = spatial_h // factor, spatial_w // factor

        # Solve ODE from s_k to e_k
        dt = (e_k - s_k) / num_steps_per_stage
        x_t = x_hat_s_k.clone()
        current_t = s_k

        for step in range(num_steps_per_stage):
            t_vec = torch.full((B,), current_t, device=device)

            # Classifier-free guidance
            if cfg_scale > 1.0:
                # Unconditional (null text)
                null_context = torch.zeros_like(context)
                null_pooled = torch.zeros_like(pooled_text)
                null_clip = torch.zeros_like(clip_context) if clip_context is not None else None

                v_cond = model(
                    latent=x_t,
                    timestep=t_vec,
                    context=context,
                    pooled_text=pooled_text,
                    clip_context=clip_context,
                    spatial_h=stage_h,
                    spatial_w=stage_w,
                    temporal_pos_ids=temporal_pos_ids,
                    frame_boundaries=frame_boundaries,
                    resolution_level=k,
                )
                v_uncond = model(
                    latent=x_t,
                    timestep=t_vec,
                    context=null_context,
                    pooled_text=null_pooled,
                    clip_context=null_clip,
                    spatial_h=stage_h,
                    spatial_w=stage_w,
                    temporal_pos_ids=temporal_pos_ids,
                    frame_boundaries=frame_boundaries,
                    resolution_level=k,
                )
                v_t = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v_t = model(
                    latent=x_t,
                    timestep=t_vec,
                    context=context,
                    pooled_text=pooled_text,
                    clip_context=clip_context,
                    spatial_h=stage_h,
                    spatial_w=stage_w,
                    temporal_pos_ids=temporal_pos_ids,
                    frame_boundaries=frame_boundaries,
                    resolution_level=k,
                )

            # Euler step: x_{t+dt} = x_t + v_t * dt
            x_t = x_t + v_t * dt
            current_t += dt

        x_hat_e_k = x_t

        # Jump point: renoise for next stage (if not final stage)
        if k > 0:
            # Upsample to next (higher) resolution
            x_up = up_sample(x_hat_e_k, 2)

            # Get next stage's s_k
            _, s_next, _ = stage_boundaries[stage_idx - 1]

            # Compute e_{k} for renoising
            # From the paper: e_{k+1} = 2 * s_k / (1 + s_k)
            e_current = 2 * s_next / (1 + s_next)

            # Renoised jump point (Eq. 15)
            x_hat_s_k = renoise_jump_point(
                x_up, s_next, e_current, corrective_gamma
            )

    return x_hat_e_k


def temporal_pyramid_condition(
    history_latents: List[torch.Tensor],
    stage_k: int,
    noise_strength: float = 0.0,
) -> torch.Tensor:
    """Build temporal pyramid condition from history latents.

    Eq. 16 (training): ..., Down(x_{t'}^{i-2}, 2^{k+1}), Down(x_{t'}^{i-1}, 2^k) -> x_hat_t^i
    Eq. 17 (inference): ..., Down(x_1^{i-2}, 2^{k+1}), Down(x_1^{i-1}, 2^k) -> x_hat_t^i

    Later frames in history are at higher resolution, earlier frames at lower resolution.

    Args:
        history_latents: List of previous frame latents [x^{i-T}, ..., x^{i-1}].
        stage_k: Current pyramid stage (resolution level).
        noise_strength: Strength of corruptive noise to add (only during training).

    Returns:
        Concatenated condition tensor with varying resolutions.
    """
    conditions = []
    T = len(history_latents)
    K_total = T  # Each history frame gets a different resolution

    for t_idx, latent in enumerate(reversed(history_latents)):
        # Earlier frames get more compressed (higher factor)
        # Frame i-1 gets factor 2^k, i-2 gets 2^{k+1}, etc.
        frame_factor = 2 ** (k + t_idx)
        down_latent = down_sample(latent, frame_factor)

        if noise_strength > 0:
            noise_scale = noise_strength * (1 - 0)  # uniform [0, noise_strength]
            noise = torch.randn_like(down_latent) * noise_scale
            down_latent = down_latent + noise

        conditions.append(down_latent)

    return conditions


def count_tokens(
    num_frames: int,
    spatial_h: int,
    spatial_w: int,
    patch_size: int,
    num_stages: int,
    history_frames: int = 3,
) -> Tuple[int, int, int]:
    """Count tokens for pyramidal flow vs full-sequence diffusion.

    Returns:
        pyramidal_tokens: Tokens in pyramidal flow training.
        full_sequence_tokens: Tokens in full-sequence diffusion.
        reduction_ratio: Reduction ratio.
    """
    tokens_per_frame = (spatial_h // patch_size) * (spatial_w // patch_size)

    pyramidal_tokens = 0
    for k in range(num_stages):
        factor = 2 ** k
        frame_tokens = tokens_per_frame // (factor ** 2)
        pyramidal_tokens += frame_tokens

    # History conditions (temporal pyramid)
    history_tokens = 0
    for k in range(history_frames):
        factor = 2 ** (k + 1)
        history_tokens += tokens_per_frame // (factor ** 2)

    pyramidal_tokens += history_tokens

    full_sequence_tokens = num_frames * tokens_per_frame

    reduction = full_sequence_tokens / pyramidal_tokens if pyramidal_tokens > 0 else 0

    return pyramidal_tokens, full_sequence_tokens, reduction
