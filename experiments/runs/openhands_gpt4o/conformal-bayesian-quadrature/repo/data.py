import numpy as np

def generate_synthetic_binomial_data(n: int, k: int, alpha: float):
    data = np.random.uniform(0, 1, size=(n, k))
    losses = np.mean(data > alpha, axis=1)
    return losses

def generate_synthetic_heteroskedastic_data(n: int):
    x = np.random.uniform(0, 4, size=n)
    y = np.random.normal(0, x**2, size=n)
    return x, y

def load_mscoco_data():
    # Placeholder for MS-COCO data loading
    # Replace with actual data loading logic
    raise NotImplementedError("MS-COCO data loading is not implemented.")