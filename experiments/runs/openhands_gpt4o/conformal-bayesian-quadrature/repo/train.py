import numpy as np
from model import BayesianConformalPredictor
from data import generate_synthetic_binomial_data, generate_synthetic_heteroskedastic_data

def train_on_synthetic_binomial():
    n, k, alpha = 10, 4, 0.4
    losses = generate_synthetic_binomial_data(n, k, alpha)
    model = BayesianConformalPredictor(alpha=alpha, max_loss=1.0)
    model.fit(losses)
    upper_bound = model.compute_upper_bound(beta=0.95)
    print(f"Upper bound on expected loss: {upper_bound}")

def train_on_synthetic_heteroskedastic():
    n = 200
    x, y = generate_synthetic_heteroskedastic_data(n)
    print("Synthetic heteroskedastic data generated.")

def main():
    print("Training on synthetic binomial data...")
    train_on_synthetic_binomial()

    print("Training on synthetic heteroskedastic data...")
    train_on_synthetic_heteroskedastic()

if __name__ == "__main__":
    main()