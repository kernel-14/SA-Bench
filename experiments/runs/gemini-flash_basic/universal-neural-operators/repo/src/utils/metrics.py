import torch

def nmae(prediction, target, epsilon=1e-6):
    # prediction, target: Tensors of shape (batch_size, ..., channels)
    # The NMAE formula in the paper is: 
    # NMAE(theta) = (1 / |D_RD^test|) * sum_((a, u) in D_RD^test) (||G_theta(a) - u||_1,G / (max_G u - min_G u + epsilon))
    # where ||.||_1,G denotes the L1 norm over the spatial domain G.
    # For simplicity, we'll calculate the mean of absolute errors normalized by the range of the target.

    # Calculate L1 norm (mean absolute error over the spatial dimensions)
    # Assuming the last dimension is channels, and the preceding dimensions are spatial.
    # We need to sum over all spatial dimensions (from 1 to -2) and divide by their product
    
    # Reshape prediction and target to (batch_size, num_spatial_points, channels) if not already
    if prediction.dim() > 2:
        # Flatten spatial dimensions if necessary
        spatial_dims = tuple(range(1, prediction.dim() - 1)) # All dimensions except batch and channel
        abs_diff = torch.abs(prediction - target)
        l1_norm_per_sample = torch.mean(abs_diff, dim=spatial_dims)
    else:
        # Assume 1D case (batch, spatial_dim, channels) or (batch, channels) - should be spatial for FNO
        l1_norm_per_sample = torch.mean(torch.abs(prediction - target), dim=1) # Average over spatial dim

    # Calculate max and min of target over spatial dimensions
    if target.dim() > 2:
        max_u = torch.max(target, dim=1, keepdim=True).values # Keepdim for correct broadcasting
        for dim in range(2, target.dim() - 1):
            max_u = torch.max(max_u, dim=dim, keepdim=True).values
        
        min_u = torch.min(target, dim=1, keepdim=True).values
        for dim in range(2, target.dim() - 1):
            min_u = torch.min(min_u, dim=dim, keepdim=True).values

        # Squeeze to remove extra dimensions from keepdim=True for final calculation
        max_u = max_u.squeeze()
        min_u = min_u.squeeze()
    else:
        # 1D case
        max_u = torch.max(target, dim=1).values # (batch_size, channels)
        min_u = torch.min(target, dim=1).values # (batch_size, channels)

    # Range for normalization
    data_range = max_u - min_u + epsilon

    # NMAE per channel and then average over channels and batch
    nmae_val = torch.mean(l1_norm_per_sample / data_range)

    return nmae_val
