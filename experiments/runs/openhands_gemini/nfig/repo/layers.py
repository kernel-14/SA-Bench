
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Utility for FFT and IFFT
def fft_2d(x):
    # Move channel to the last dimension for rfft and irfft
    x = x.permute(0, 2, 3, 1) # B, H, W, C
    # Perform real FFT
    # Output is a complex tensor of shape (B, H, W//2 + 1, C, 2) for real and imag parts
    fft_output = torch.fft.rfftn(x, dim=(1, 2))
    return fft_output

def ifft_2d(fft_output, output_size):
    # Perform inverse real FFT
    x_recon = torch.fft.irfftn(fft_output, s=output_size, dim=(1, 2))
    x_recon = x_recon.permute(0, 3, 1, 2) # B, C, H, W
    return x_recon

class FrequencyMask(nn.Module):
    def __init__(self, height, width, channels, band_dims, device='cpu'):
        super().__init__()
        self.height = height
        self.width = width
        self.channels = channels
        self.band_dims = band_dims # List of (h_i, w_i) for each band
        self.num_bands = len(band_dims)
        self.device = device
        self.masks = self._create_frequency_masks().to(device)

    def _create_frequency_masks(self):
        masks = []
        
        # Calculate sigma_max
        # Based on the paper's formula, sigma_max is related to the sum of h_i * w_i
        # and then scaled. Let's assume an arbitrary maximum frequency extent for now.
        # A simple conceptual max could be the highest possible radial frequency index.
        sigma_max = np.sqrt((self.height / 2)**2 + (self.width // 2)**2) 

        # Calculate h_i * w_i for each band and sum for normalization
        h_w_products = [h * w for h, w in self.band_dims]
        sum_h_w_products = sum(h_w_products)

        # Calculate sigma_i boundaries
        sigmas = [0.0] * (self.num_bands + 1)
        for i in range(self.num_bands):
            # sigma_i = sigma_{i-1} + (h_i * w_i) / (sum_h_w_products) * sigma_max
            sigmas[i+1] = sigmas[i] + (h_w_products[i] / sum_h_w_products) * sigma_max

        # Create frequency coordinates grid
        # These represent normalized frequencies from -0.5 to 0.5.
        # To get pixel-like frequency coordinates (e.g., from -H/2 to H/2), multiply by dimension.
        freq_h_coords = torch.fft.fftfreq(self.height, d=1) * self.height 
        freq_w_coords = torch.fft.rfftfreq(self.width, d=1) * self.width

        # Create 2D meshgrid for radial distance calculation
        h_mesh, w_mesh = torch.meshgrid(freq_h_coords, freq_w_coords, indexing='ij')
        radial_distances = torch.sqrt(h_mesh**2 + w_mesh**2) # (H, W//2 + 1)

        for i in range(self.num_bands):
            mask = torch.zeros((self.height, self.width // 2 + 1, self.channels), dtype=torch.float32)
            
            # Band F_i = [sigma_i, sigma_{i+1})
            lower_bound = sigmas[i]
            upper_bound = sigmas[i+1]

            # Create a boolean mask based on radial distance
            # For the first band, it should be [0, sigma_1). For subsequent bands [sigma_i, sigma_{i+1})
            # This logic needs to be exact from the paper: [0, sigma_1), [sigma_1, sigma_2), etc.
            if i == 0:
                band_mask_2d = (radial_distances >= lower_bound) & (radial_distances < upper_bound)
            else:
                band_mask_2d = (radial_distances >= lower_bound) & (radial_distances < upper_bound)
            
            # Ensure the DC component is only in the first band.
            # If `radial_distances[0,0]` corresponds to DC, it should be 0.
            # If lower_bound is 0 for i=0, then this is handled.
            
            # Expand to (H, W//2 + 1, C)
            mask[:, :, :] = band_mask_2d.unsqueeze(-1).float().repeat(1, 1, self.channels)
            masks.append(mask)
        
        # Stack masks to have a shape (num_bands, H, W//2 + 1, C)
        return torch.stack(masks, dim=0)

    def forward(self, fft_features, band_idx):
        # fft_features: B, H, W//2 + 1, C, 2 (complex tensor from rfftn)
        # band_idx: integer indicating which frequency band mask to apply
        if not (0 <= band_idx < self.num_bands):
            raise ValueError(f"band_idx must be between 0 and {self.num_bands - 1}")

        # Select the mask for the current band
        mask = self.masks[band_idx] # H, W//2 + 1, C

        # Apply mask
        # fft_features has B, H, W//2+1, C, 2
        # mask needs to be broadcastable to this.
        # Unsqueezing mask for batch dimension and real/imaginary parts
        masked_fft_features = fft_features * mask.unsqueeze(0).unsqueeze(-1)
        return masked_fft_features

class SpatialResampler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feature_map, target_height, target_width):
        # feature_map: B, C, H, W
        # target_height, target_width: H', W'
        if feature_map.shape[-2:] == (target_height, target_width):
            return feature_map
        return F.interpolate(feature_map, size=(target_height, target_width), mode='bilinear', align_corners=False)

