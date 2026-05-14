import numpy as np
from forward_process import forward_process
from reverse_process import reverse_process

def randomized_midpoint_sampler(X_0, alphas, score_function, K, N):
    """
    Randomized midpoint sampling algorithm.

    Parameters:
    X_0 : np.ndarray
        Initial data distribution.
    alphas : np.ndarray
        Sequence of alpha values defining the noise schedule.
    score_function : callable
        Function to compute the score (gradient of log-density).
    K : int
        Number of rounds.
    N : int
        Steps per round.

    Returns:
    Y_final : np.ndarray
        Final data sample approximating the target distribution.
    """
    T = len(alphas)
    assert K * N == 2 * T, "K*N should equal 2*T for initialization."

    # Step 1: Start with Gaussian noise
    Y = np.random.normal(0, 1, X_0.shape)

    # Step 2: Iterate over K rounds
    for k in range(K):
        for n in range(N):
            # Compute intermediate values and noise injection
            tau_k_n = 1 - alphas[T - k * N // 2 - n]
            tau_prev = 1 - alphas[T - k * N // 2 - n - 1]
            delta_tau = tau_prev - tau_k_n

            score_val = score_function(Y)
            Y += score_val * delta_tau

        # Inject Noise after each round
        noise = np.random.normal(0, 1, X_0.shape)
        Y = np.sqrt(1 - tau_k_n) * Y + np.sqrt(tau_k_n) * noise

    return Y

