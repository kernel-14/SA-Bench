import torch
import torch.nn.functional as F

def downsample(tensor, scale):
    """Downsample a tensor by a given scale factor."""
    return F.interpolate(tensor, scale_factor=1/scale, mode="bilinear", align_corners=False)

def upsample(tensor, scale):
    """Upsample a tensor by a given scale factor."""
    return F.interpolate(tensor, scale_factor=scale, mode="bilinear", align_corners=False)

def spatial_pyramidal_flow(latent, resolutions):
    """Perform spatial pyramidal flow generation across multiple resolutions."""
    stages = len(resolutions)
    pyramidal_results = []

    for i, resolution in enumerate(resolutions):
        t_prime = (i + 1) / stages
        downsampled = downsample(latent, resolution)
        upsampled = upsample(downsampled, resolution)
        interpolated = t_prime * latent + (1 - t_prime) * upsampled
        pyramidal_results.append(interpolated)

    return pyramidal_results

if __name__ == "__main__":
    # Example usage
    input_tensor = torch.rand(1, 3, 256, 256)
    resolutions = [2, 4, 8]
    pyramids = spatial_pyramidal_flow(input_tensor, resolutions)
    print(f"Generated {len(pyramids)} pyramid stages.")

