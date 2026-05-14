# Instance-dependent Convergence Theory for Diffusion Models

Code reproduction of:
> **Instance-dependent Convergence Theory for Diffusion Models**  
> Yuchen Jiao, Gen Li (2025)

## Overview

This repository implements the randomized midpoint sampler for score-based diffusion models and reproduces the convergence experiments from the paper. The main contribution is an instance-dependent convergence bound:

**Theorem 1**: TV(q_K, p_{Y_K}) ≤ C · min{d^{3/2}, d·L^{1/2}, d^{1/2}·L^{3/2}} · log⁴(T) / T^{3/2} + C · ε_score · log^{1/2}(T)

This yields iteration complexity:

**T ≥ min{d, d^{2/3}·L^{1/3}, d^{1/3}·L} · ε^{-2/3} · log^{8/3}(T)**

where L is the non-uniform Lipschitz constant (Definition 2), which scales only logarithmically with the number of GMM components.

## File Structure

```
repo/
├── config.py              # All hyperparameters and experiment configurations
├── score_functions.py     # Exact score functions (Gaussian, GMM)
├── forward_process.py     # Forward diffusion process and learning rate schedule
├── sampler.py             # Randomized midpoint sampler (Algorithm, Section 2.2)
├── parallel_sampler.py    # Parallel sampler (Section 3.3, Appendix E)
├── convergence_metrics.py # KL divergence, TV distance, score estimation error
├── theory.py              # Theoretical complexity bounds for all methods
├── baselines.py           # Baseline samplers (DDPM, ODE, Langevin)
├── data.py                # Data distributions (Gaussian, GMM)
├── experiments.py         # Experiment runners for all figures
├── visualization.py       # Plotting functions for Figures 1, 2, 3
├── train.py               # Main entry point
└── requirements.txt       # Dependencies
```

## Key Algorithms

### Randomized Midpoint Sampler (Section 2.2)

The sampler operates over K rounds, each with N = 2T/K steps:

1. **Initialize**: Y₀ ~ N(0, I_d)
2. **For k = 0, ..., K-1**:
   - Sample τ_{k,n} ~ Unif(τ̂_{k,n}, τ̂_{k,n-1}) (randomized schedule, Eq. 9)
   - Compute Y_{k,n} via ODE discretization (Eq. 10)
   - Inject noise: Y_{k+1} = √((1-τ_{k+1,0})/(1-τ_{k,N})) Y_{k,N} + √((τ_{k+1,0}-τ_{k,N})/(1-τ_{k,N})) Z_k

The randomized schedule enables unbiased estimation of the ODE integral (key for the improved convergence rate).

### Non-uniform Lipschitz Condition (Definition 2)

Instead of requiring the uniform Lipschitz condition used in prior work, we use:

P_{x~X_t}{ (1-ᾱ_t)||s_t*(x') - s_t*(x)||₂ ≤ L||x'-x||₂, ∀||x'-x||₂ ≤ C√(d(1-ᾱ_t)logT)/L } ≥ 1 - c/(T+d)⁴

For GMMs, L = O(log(H(T+d))) — logarithmic in components, dimension, and iterations.

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run all experiments
python train.py --experiment all --output_dir results

# Run specific experiments
python train.py --experiment figure2 --d 10 --K 10 --n_samples 3000
python train.py --experiment figure1 --d 100
python train.py --experiment figure3 --d 100
python train.py --experiment lipschitz --d 100
python train.py --experiment theorem1 --d 50
python train.py --experiment parallel --d 50
python train.py --experiment summary --d 100
```

## Experiments

### Figure 2 (Appendix A): KL Divergence Convergence

Verifies the theoretical rate O(log⁴(T)/T³) for KL divergence with Gaussian target:
- (a) d=10, k=10
- (b) d=100, k=10  
- (c) d=500, k=100

### Figure 1: Iteration Complexity Comparison

- **Left**: Complexity vs L for ε = O(1) — our result achieves best across full range of L
- **Right**: TV distance vs T for L = ∞ — improvement when T ≲ d²

### Figure 3 (Appendix B): TV Distance vs L

Comparison for fixed T = O(d), O(d^{3/2}), O(d²).

### Example 2: Lipschitz Constant Comparison

Demonstrates that for GMMs, the non-uniform L = O(log(H(T+d))) while the uniform Lipschitz constant can be O(d/σ⁴) — exponentially larger.

## Comparison with Prior Works

| Method | Iteration Complexity |
|--------|---------------------|
| **Ours (Theorem 1)** | **min{d, d^{2/3}L^{1/3}, d^{1/3}L} · ε^{-2/3}** |
| Benton et al. (2023) | d · ε^{-2} |
| Li & Yan (2024a) | d · ε^{-1} |
| Li & Cai (2024) | d^{5/4} · ε^{-1/2} |
| Li & Jiao (2024) | d^{1/3} · L · ε^{-2/3} |
| Gupta et al. (2024) | d^{2/3} · L^{1/3} · ε^{-2/3} |

Our result improves over Li & Jiao (2024) by a factor of max{d^{-2/3}L, d^{-1/3}L^{2/3}, 1} when L ≳ √d.

## Parallel Sampler (Theorem 2)

The parallel sampler achieves ε-accuracy with:
- N ≳ (min{d^{2/3}L^{-2/3}, d^{1/3}} + 1) · log^{5/3}(T) / ε^{2/3} processors
- MK ≳ min{d·log(T), L} · log²(T) parallel rounds

## References

- Jiao, Y. and Li, G. (2025). Instance-dependent Convergence Theory for Diffusion Models.
- Ho, J., Jain, A., and Abbeel, P. (2020). Denoising diffusion probabilistic models. NeurIPS.
- Song, J., Meng, C., and Ermon, S. (2021). Denoising diffusion implicit models. ICLR.
- Li, G. and Jiao, Y. (2024). Improved convergence rate for diffusion probabilistic models.
- Li, G. and Yan, Y. (2024a). O(d/T) convergence theory for diffusion probabilistic models.
- Li, G. and Cai, C. (2024). Provable acceleration for diffusion models under minimal assumptions.
- Gupta, S., Cai, L., and Chen, S. (2024). Faster diffusion-based sampling with randomized midpoints.
- Benton, J. et al. (2023). Nearly d-linear convergence bounds for diffusion models.
