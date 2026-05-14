import numpy as np

def forward_process(X_0, alphas):
    """
    Perform the forward process to transform data into noise.

    Parameters:
    X_0 : np.ndarray
        Initial data sample (d-dimensional).
    alphas : np.ndarray
        Sequence of alpha values defining the noise schedule.

    Returns:
    X_t : np.ndarray
        Data transformed at each iteration step.
    """
    T = len(alphas)
    X_t = X_0.copy()

    for t in range(T):
        alpha_t = alphas[t]
        noise_t = np.random.normal(0, 1, X_t.shape)
        X_t = np.sqrt(alpha_t) * X_t + np.sqrt(1 - alpha_t) * noise_t

    return X_t

