import numpy as np

def sample_dirichlet_1(n: int, num_samples: int) -> np.ndarray:
    """
    Samples from a Dirichlet distribution Dir(1, ..., 1) with n+1 components.

    Args:
        n: The number of observed losses (n in the paper, which means n+1 components).
        num_samples: The number of Dirichlet samples to generate.

    Returns:
        A numpy array of shape (num_samples, n + 1) where each row is a sample
        from Dir(1, ..., 1).
    """
    # Sample from Gamma(1, 1) which is equivalent to Exponential(1)
    gamma_samples = np.random.exponential(scale=1.0, size=(num_samples, n + 1))
    
    # Normalize to get Dirichlet samples
    dirichlet_samples = gamma_samples / np.sum(gamma_samples, axis=1, keepdims=True)
    
    return dirichlet_samples

if __name__ == '__main__':
    # Example usage:
    num_observed_losses = 10
    num_dirichlet_samples = 5

    # Each row sums to 1.0
    dir_samples = sample_dirichlet_1(num_observed_losses, num_dirichlet_samples)
    print(f"Dirichlet samples (first 5 rows):
{dir_samples}")
    print(f"Sum of each row: {np.sum(dir_samples, axis=1)}")
