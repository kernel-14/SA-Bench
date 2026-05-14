import torch

class SyntheticDataGenerator:
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def generate_arithmetic_data(self):
        data = []
        for _ in range(self.num_samples):
            x = torch.randint(1, 100, (1,))
            y = x % 7  # Example modular arithmetic target
            data.append((x, y))
        return data

    def generate_reasoning_data(self):
        data = []
        for _ in range(self.num_samples):
            x = torch.randn((10,))  # Example reasoning input
            y = torch.sum(x) > 0  # Example reasoning target
            data.append((x, y))
        return data


