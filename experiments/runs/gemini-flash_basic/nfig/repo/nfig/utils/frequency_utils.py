import torch
import numpy as np

def get_frequency_band_boundaries(H_prime: int, W_prime: int, scaling_factors: list[int]) -> list[float]:
    """
    Calculates the frequency band boundaries (sigma_i) based on the given formula.
    sigma_i = sigma_{i-1} + (h_i * w_i) / (sum(h_j * w_j)) * sigma_max
    The maximum frequency sigma_max is assumed to be related to the Nyquist frequency,
    which for a discrete 2D FFT of size H'xW' could be approximated by sqrt((H'/2)^2 + (W'/2)^2).
    For simplicity and alignment with the formula, we can define sigma_max as 1 (normalized).
    """
    num_bands = len(scaling_factors)
    
    # Calculate h_i * w_i for each band
    h_w_products = []
    for s in scaling_factors:
        h_i = H_prime // s
        w_i = W_prime // s
        h_w_products.append(h_i * w_i)
    
    sum_h_w_products = sum(h_w_products)
    
    # We normalize sigma_max to 1 for calculation, and the output sigma_i will be normalized.
    sigma_max = 1.0 
    
    sigma_boundaries = [0.0] # sigma_0 = 0
    for i in range(num_bands):
        prev_sigma = sigma_boundaries[-1]
        current_sigma = prev_sigma + (h_w_products[i] / sum_h_w_products) * sigma_max
        sigma_boundaries.append(current_sigma)
        
    # The last boundary is sigma_max, as per the paper
    # sigma_n = sigma_max
    sigma_boundaries[-1] = sigma_max # Ensure the last one is exactly sigma_max

    return sigma_boundaries[1:] # Return [sigma_1, sigma_2, ..., sigma_n]

def generate_circular_frequency_masks(H_prime: int, W_prime: int, sigma_boundaries: list[float], device: torch.device) -> list[torch.Tensor]:
    """
    Generates circular frequency masks for 2D FFT.
    The masks are applied to the shifted FFT spectrum, where the zero-frequency component is at the center.
    """
    masks = []
    
    center_h, center_w = H_prime / 2, W_prime / 2
    
    # Create a grid of frequency coordinates
    # The FFT frequencies range from -N/2 to N/2 - 1
    # For shifted FFT, 0 frequency is at the center.
    freq_h = torch.linspace(-center_h, center_h -1, H_prime, device=device) if H_prime % 2 == 0 else torch.linspace(-center_h, center_h, H_prime, device=device)
    freq_w = torch.linspace(-center_w, center_w -1, W_prime, device=device) if W_prime % 2 == 0 else torch.linspace(-center_w, center_w, W_prime, device=device)

    # Create a meshgrid
    grid_h, grid_w = torch.meshgrid(freq_h, freq_w, indexing='ij')
    
    # Calculate radial distance from the center (normalized)
    # The maximum possible frequency in this normalized space is related to sqrt((H'/2)^2 + (W'/2)^2)
    # We need to normalize this distance so that the max radial distance is 0.5 or 1, 
    # to correspond with the `sigma_boundaries` being normalized to 0-1.
    max_radial_dist = torch.sqrt(torch.tensor(center_h**2 + center_w**2, device=device))
    
    # For the definition of sigma_i using h_i*w_i, sigma_max is normalized to 1.
    # So, we should normalize radial_dist to be in [0, 1].
    radial_distances = torch.sqrt(grid_h**2 + grid_w**2) / max_radial_dist
    
    # Add a zero boundary for the lowest frequency band (i.e., [0, sigma_1))
    all_boundaries = [0.0] + sigma_boundaries
    
    for i in range(len(sigma_boundaries)):
        lower_bound = all_boundaries[i]
        upper_bound = all_boundaries[i+1]
        
        # Create a circular mask. Frequencies within [lower_bound, upper_bound) are 1, others are 0.
        mask = ((radial_distances >= lower_bound) & (radial_distances < upper_bound)).float()
        masks.append(mask.unsqueeze(0).unsqueeze(0)) # Add C and B dimensions for broadcasting (1, 1, H, W)
        
    return masks

def generate_frequency_masks(H_prime: int, W_prime: int, scaling_factors: list[int], device: torch.device) -> list[torch.Tensor]:
    """
    Combines the logic to get frequency band boundaries and generate the masks.
    """
    sigma_boundaries = get_frequency_band_boundaries(H_prime, W_prime, scaling_factors)
    masks = generate_circular_frequency_masks(H_prime, W_prime, sigma_boundaries, device)
    return masks
