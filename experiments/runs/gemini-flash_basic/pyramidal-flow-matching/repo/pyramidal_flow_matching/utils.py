import torch
import torch.nn.functional as F

def downsample_video(video: torch.Tensor, scale_factor: int, mode: str = 'nearest') -> torch.Tensor:
    """
    Downsamples a video tensor (B, C, T, H, W) by a given scale factor.
    Args:
        video (torch.Tensor): Input video tensor.
        scale_factor (int): Downsampling factor. Applied to H and W dimensions.
        mode (str): Upsampling mode (e.g., 'nearest', 'bilinear').
    Returns:
        torch.Tensor: Downsampled video tensor.
    """
    if scale_factor == 1:
        return video
    # For video, typically downsample spatial dimensions
    # F.interpolate expects (N, C, H_in, W_in) or (N, C, D_in, H_in, W_in)
    # We want to scale H and W, so we use scale_factor for those dimensions.
    # The time dimension (T) remains the same.
    _, _, T, H, W = video.shape
    new_H = H // scale_factor
    new_W = W // scale_factor
    return F.interpolate(video, size=(T, new_H, new_W), mode=mode, antialias=True)

def upsample_video(video: torch.Tensor, scale_factor: int, mode: str = 'nearest') -> torch.Tensor:
    """
    Upsamples a video tensor (B, C, T, H, W) by a given scale factor.
    Args:
        video (torch.Tensor): Input video tensor.
        scale_factor (int): Upsampling factor. Applied to H and W dimensions.
        mode (str): Upsampling mode (e.g., 'nearest', 'bilinear').
    Returns:
        torch.Tensor: Upsampled video tensor.
    """
    if scale_factor == 1:
        return video
    # For video, typically upsample spatial dimensions
    # F.interpolate expects (N, C, H_in, W_in) or (N, C, D_in, H_in, W_in)
    # We want to scale H and W, so we use scale_factor for those dimensions.
    # The time dimension (T) remains the same.
    _, _, T, H, W = video.shape
    new_H = H * scale_factor
    new_W = W * scale_factor
    return F.interpolate(video, size=(T, new_H, new_W), mode=mode)

def get_video_patch_embedding_size(original_size: tuple[int, int, int], patch_size: tuple[int, int, int]) -> tuple[int, int, int]:
    """
    Calculates the resulting number of patches in T, H, W dimensions.
    Args:
        original_size (tuple): Original dimensions (T, H, W).
        patch_size (tuple): Patch dimensions (t, h, w).
    Returns:
        tuple: Number of patches (num_t, num_h, num_w).
    """
    return (
        original_size[0] // patch_size[0],
        original_size[1] // patch_size[1],
        original_size[2] // patch_size[2]
    )

# Renoising related utility functions (conceptual based on paper's description)

def calculate_corrective_noise(current_xk1_end: torch.Tensor, sk: float, ek1: float, upsample_mode: str = 'nearest') -> torch.Tensor:
    """
    Calculates and applies corrective noise as described in the paper (Eq. 12-14).
    This is a simplified conceptual implementation based on Eq. 139.
    
    Args:
        current_xk1_end (torch.Tensor): The output latent from the previous stage (x_e_{k+1}).
        sk (float): The start time `s_k` for the current stage.
        ek1 (float): The end time `e_{k+1}` for the previous stage (used for the relationship e_{k+1} = 2 * s_k / (1 + s_k)).
        upsample_mode (str): Upsampling mode for U_p.

    Returns:
        torch.Tensor: The starting point for the current stage with corrective noise (x_s_k).
    """
    # Paper states: x_s_k = (1 + s_k) / 2 * Up(x_e_{k+1}) + sqrt(3) * (1 - s_k) / 2 * n'
    # with e_{k+1} = 2 * s_k / (1 + s_k)

    # Up(x_e_{k+1}) needs to upsample the spatial dimensions of the latent.
    # The upsampling factor is implicitly 2x spatial for each pyramid stage when moving to the next stage.
    upsampled_x = upsample_video(current_xk1_end, scale_factor=2, mode=upsample_mode) 
    
    # Coefficient for the upsampled term
    coeff_upsampled = (1 + sk) / 2.0
    
    # Coefficient for the noise term
    coeff_noise = (math.sqrt(3) * (1 - sk)) / 2.0
    
    # Sample corrective noise n_prime from N(0, I). The specific coefficients in the formula
    # are derived assuming a certain Sigma' (covariance) for n', such that gamma = -1/3 for decorrelation.
    # For conceptual implementation, we generate standard normal noise and rely on the coefficients.
    noise_n_prime = torch.randn_like(upsampled_x)
    
    x_s_k = coeff_upsampled * upsampled_x + coeff_noise * noise_n_prime
    
    return x_s_k
