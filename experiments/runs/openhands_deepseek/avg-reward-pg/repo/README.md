# Global Convergence of Policy Gradient in Average Reward MDPs

Reproduction of the paper "Global Convergence of Policy Gradient in Average Reward MDPs"
by Navdeep Kumar, Yashaswini Murthy, Itai Shufaro, Kfir Y. Levy, R. Srikant, Shie Mannor.

## Codebase Structure

```
repo/
├── mdp.py              # TabularMDP class: transition, reward, value functions, stationary distribution
├── policy_gradient.py  # Projected Policy Gradient algorithm and policy projection
├── constants.py        # MDP complexity constants: Cm, Cp, Cr, kappa_r, L1^Pi, L2^Pi, C_PL
├── experiments.py      # Three experiments from the paper (Figures 1a, 1b, 2)
├── config.py           # Hyperparameters and configuration
├── utils.py            # Utility functions (soft policies, theoretical bounds)
├── train.py            # Main entry point to run all experiments
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Algorithm

The paper implements **Projected Policy Gradient (PPG)** for average reward MDPs:

```
pi_{k+1} = Proj_Pi[ pi_k + eta * grad(rho^{pi_k}) ]
```

where `grad(rho^pi)(s,a) = d^pi(s) * Q^pi(s,a)`.

## Experiments

1. **Varying State/Action Sizes** (Figure 1a): MDPs with (S,A) in {(3,3), (9,9), (81,81)}
2. **Varying Reward Variance** (Figure 1b): Fixed (16,16) MDP with different reward distributions
3. **Varying Transition Kernels** (Figure 2): Fixed (16,16) MDP with uniform, non-uniform, and deterministic transitions

## Usage

```bash
cd repo
python train.py
```

Results will be saved as PNG figures in the `figures/` directory.

## Key Theoretical Results (Theorem 1)

- For all ergodic MDPs: rho* - rho^{pi_k} <= 1 / (1/(rho*-rho^{pi_0}) + nu*k)
- For simple MDPs (L2^Pi << 1): exponential convergence
- MDP complexity captured by constants: Cm (mixing), Cp (transition kernel diameter), Cr (reward diameter), kappa_r (reward variance)
