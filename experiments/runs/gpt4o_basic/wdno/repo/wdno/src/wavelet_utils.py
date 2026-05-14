import pywt
import numpy as np

def wavelet_decompose(data, wavelet='bior2.4', mode='periodization'):
    """
    Perform wavelet decomposition for 1D or 2D data.
    :param data: Input signal or image (numpy array)
    :param wavelet: Wavelet type (default is 'bior2.4')
    :param mode: Signal extension mode (default is 'periodization')
    :return: Decomposed coefficients
    """
    coeffs = pywt.wavedec2(data, wavelet=wavelet, mode=mode)
    return coeffs

def wavelet_reconstruct(coeffs, wavelet='bior2.4', mode='periodization'):
    """
    Reconstruct the original signal/image from wavelet coefficients.
    :param coeffs: Wavelet coefficients
    :param wavelet: Wavelet type (default is 'bior2.4')
    :param mode: Signal extension mode (default is 'periodization')
    :return: Reconstructed signal/image
    """
    data_reconstructed = pywt.waverec2(coeffs, wavelet=wavelet, mode=mode)
    return data_reconstructed
