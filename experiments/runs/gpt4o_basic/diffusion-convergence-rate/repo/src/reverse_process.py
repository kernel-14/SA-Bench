import numpy as np

def reverse_process(Y_0, score_function, alphas):
    """
    Perform the reverse process to transform noise into data.

    Parameters:
    Y_0 : np.ndarray
        Initial Gaussian noise sample.
    score_function : callable
        Function that computes the score (gradient of log-density).
    alphas : np.ndarray
        Sequence of alpha values defining the noise schedule.

    Returns:
    Y_T : np.ndarray
        Data sample approximating the target distribution.
    """
    T = len(alphas)
    Y_t = Y_0.copy()

    for t in range(T - 1, -1, -1):
        alpha_t = alphas[t]
        score_t = score_function(Y_t)
        dt = 1 / (T * np.sqrt(1 - alpha_t))
        dY = (-Y_t / (2 * (1 - alpha_t)) + score_t) * dt
        Y_t += dY

    return Y_t

