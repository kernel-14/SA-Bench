import numpy as np
from skimage.transform import rescale

def downsample_data(data, scale_factor):
    """
    Downsample the given data by a scale factor.
    :param data: Input data (numpy array)
    :param scale_factor: Factor by which to downsample (e.g., 0.5 for half-resolution).
    :return: Downsampled data.
    """
    return rescale(data, scale=scale_factor, mode='reflect', multichannel=False, anti_aliasing=True)

def create_multi_resolution_dataset(data, steps):
    """
    Generate multi-resolution dataset pairs.
    :param data: Original high-resolution data (numpy array).
    :param steps: Number of resolution levels to generate.
    :return: List of data pairs [(high-res, low-res), ...]
    """
    dataset = []
    current_data = data
    for _ in range(steps):
        low_res_data = downsample_data(current_data, 0.5)
        dataset.append((current_data, low_res_data))
        current_data = low_res_data
    return dataset
