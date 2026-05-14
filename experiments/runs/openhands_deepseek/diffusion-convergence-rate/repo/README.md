# Instance-dependent Convergence Theory for Diffusion Models

Reproduction of numerical experiments from:

> Jiao, Y., & Li, G. (2025). *Instance-dependent Convergence Theory for Diffusion Models*.

## Codebase Structure

```
repo/
├── __init__.py              # Package init
├── config.py                # Hyperparameters and configuration
├── score_function.py        # Exact score functions for Gaussian targets
├── sampler.py               # Randomized midpoint sampler (sequential)
├── parallel_sampler.py      # Parallel implementation (Appendix E.1)
├── experiment.py            # Main experiment script (Appendix A / Figure 2)
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Key Components

### Sampler (`sampler.py`)
Implements the randomized midpoint sampler from Section 2.2:
- Forward process: DDPM-style with a randomized learning rate schedule
- Reverse process: Discretized probability flow ODE integration
- Contains both sequential and parallel implementations

### Score Function (`score_function.py`)
Exact score function for Gaussian target distributions as described in Example 1 (Appendix C.1).

### Experiments (`experiment.py`)
Reproduces Figure 2 from Appendix A:
- Gaussian target with diagonal covariance
- First k components ~ Uniform[0, 10]
- K=10 rounds, varying T
- KL divergence measured via Monte Carlo estimation

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run experiments
python experiment.py \
  --d_values 10 100 500 \
  --k_values 10 10 100 \
  --T_values 500 1000 2000 4000 \
  --K 10 \
  --num_mc_samples 10000 \
  --output_dir results
```

## Theoretical Results

The paper establishes an iteration complexity of:
```
min{d, d^{2/3} L^{1/3}, d^{1/3} L} * ε^{-2/3} * polylog(T)
```
where L is a relaxed Lipschitz constant for the normalized score functions.
