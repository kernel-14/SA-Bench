import numpy as np
import torch
from torch.utils.data import Dataset

class ODE1Dataset(Dataset):
    def __init__(self, num_samples, t_steps=100):
        super(ODE1Dataset, self).__init__()
        self.num_samples = num_samples
        self.t_steps = t_steps
        self.data = []

        # Parameter Ranges for ODE1
        param_ranges = {
            "alpha": (0.1, 2.0),
            "beta": (0.1, 2.0),
            "gamma": (0.0, 1.0),
        }

        for _ in range(num_samples):
            alpha = np.random.uniform(*param_ranges["alpha"])
            beta = np.random.uniform(*param_ranges["beta"])
            gamma = np.random.uniform(*param_ranges["gamma"])

            t = np.linspace(0, 1, t_steps)
            u = (
                -(1 / np.pi) * np.cos(alpha * np.pi * t)
                + (1 / np.pi) * np.sin(beta * np.pi * t)
                + np.sin(gamma * np.pi)
                + (1 / np.pi)
            )

            du_dalpha = t * np.sin(alpha * np.pi * t)
            du_dbeta = t * np.cos(beta * np.pi * t)
            du_dgamma = np.pi * np.cos(gamma * np.pi)

            params = np.array([alpha, beta, gamma])
            sensitivities = np.stack([du_dalpha, du_dbeta, du_dgamma], axis=1)

            self.data.append((params, u, sensitivities))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        params, u, sensitivities = self.data[idx]
        return (
            torch.tensor(params, dtype=torch.float32),
            torch.tensor(u, dtype=torch.float32),
            torch.tensor(sensitivities, dtype=torch.float32),
        )

# Example Usage
if __name__ == "__main__":
    dataset = ODE1Dataset(num_samples=1000)
    print(f"Dataset size: {len(dataset)}")
    for sample in dataset[:2]:
        print("Sample:", sample)
