# Conformal Prediction as Bayesian Quadrature — Reproduction

This repository reproduces the core contributions of:

> **Conformal Prediction as Bayesian Quadrature**
> Jake C. Snell & Thomas L. Griffiths

The paper reinterprets conformal prediction through Bayesian probability, using Bayesian quadrature to produce a posterior distribution over the expected loss rather than just marginal guarantees.

## Repository Structure

```
.
├── src/
│   ├── __init__.py                    # Package init, exports
│   ├── bayesian_quadrature.py         # Core implementation
│   └── theoretical_verification.py    # Verification of key theorems
├── experiments/
│   ├── __init__.py
│   ├── synthetic_binomial.py          # Exp 5.1 (Table 1)
│   ├── synthetic_heteroskedastic.py   # Exp 5.2 (Table 2)
│   ├── mscoco.py                      # Exp 5.3 (Table 3)
│   └── visualize_L_plus.py            # Figure 4 reproduction
├── run_all.py                         # Run all experiments
└── README.md
```

## Core Contributions Reproduced

### 1. Decision-theoretic Formulation (Section 3)

The paper shows that both split conformal prediction and conformal risk control can be formulated as instances of a general decision problem. We implement:
- `compute_split_conformal_lambda()` — Proposition 3.1
- `compute_crc_decision_rule()` — Proposition 3.2

### 2. Bayesian Quadrature Framework (Section 4)

The key methodological contribution: using Bayesian quadrature over quantile functions of the loss distribution to bound the posterior expected loss.

Implemented theoretical results:
- **Theorem 4.1**: Upper bound on posterior expected loss using quantile spacings
- **Lemma 4.2**: Distribution of quantile spacings is Dirichlet(1,...,1)
- **Theorem 4.3**: Random variable $L^+ = \sum_i U_i \ell_{(i)}$ stochastically dominates posterior risk
- **Corollary 4.4**: Upper confidence bounds via quantiles of $L^+$

### 3. HPD Decision Rule (Section 4.5, Eq. 31)

The main algorithm: $\lambda_{hpd}^\beta = \inf\{\lambda : \Pr(L^+ \leq \alpha \mid \ell_{1:n}) \geq \beta\}$

Implemented via Monte Carlo simulation of Dirichlet random variates (the approach used in the paper's experiments).

### 4. Recovery of Existing Methods (Section 4.6)

We verify that:
- CRC corresponds to taking $\mathbb{E}[L^+] \leq \alpha$
- SCP is recovered when $k \geq (n+1)(1-\alpha)$

### 5. Experiments

**Experiment 5.1 — Synthetic Binomial Data (Table 1):**
- $n=10$, $K=4$, $\alpha=0.4$, scaled binomial loss
- Compares CRC, RCPS (Hoeffding), and Ours ($\beta=0.95$)
- Code: `experiments/synthetic_binomial.py`

**Experiment 5.2 — Synthetic Heteroskedastic Data (Table 2):**
- $X \sim U[0,4]$, $Y|X \sim \mathcal{N}(0, X^2)$
- $n=200$, $\alpha=0.1$, miscoverage loss
- Compares SCP/CRC, RCPS, and Ours ($\beta=0.95$)
- Code: `experiments/synthetic_heteroskedastic.py`

**Experiment 5.3 — MS-COCO False Negative Rate (Table 3):**
- Multilabel classification, 1000 cal / 3952 test examples
- FNR loss, $\alpha=0.1$
- Compares CRC, RCPS, and Ours ($\beta=0.95$)
- Code: `experiments/mscoco.py`
- Note: Uses synthetic data to demonstrate methodology; real MS-COCO requires dataset download and pre-trained model.

**Figure 4 — L^+ Distribution:**
- Code: `experiments/visualize_L_plus.py`
- Visualizes the posterior distribution of $L^+$ for $\lambda \in \{0.7, 0.8, 0.9\}$

## Usage

```bash
# Install dependencies
pip install numpy scipy matplotlib

# Run all experiments (full 10K trials)
python run_all.py

# Quick run (200 trials for faster testing)
python run_all.py --quick

# Skip MS-COCO
python run_all.py --skip-mscoco

# Individual experiments
python experiments/synthetic_binomial.py
python experiments/synthetic_heteroskedastic.py
python experiments/mscoco.py

# Theoretical verification
python src/theoretical_verification.py

# Generate Figure 4
python experiments/visualize_L_plus.py
```

## Key Implementation Details

### L^+ Random Variable (Theorem 4.3)
$L^+$ is the core random variable. Given sorted losses $\ell_{(1)}, \ldots, \ell_{(n)}$ and maximum loss $B$, we define $\ell_{(n+1)} = B$ and:

$$L^+ = \sum_{i=1}^{n+1} U_i \ell_{(i)}, \quad (U_1, \ldots, U_{n+1}) \sim \text{Dir}(1, \ldots, 1)$$

### HPD Decision Rule
We search over a grid of $\lambda$ values. For each $\lambda$:
1. Compute individual losses $\ell_i(\lambda)$
2. Generate Dirichlet samples and compute $L^+$ samples
3. Estimate $\Pr(L^+ \leq \alpha)$
4. Select smallest $\lambda$ where this probability $\geq \beta$

### Assumptions
- Data at deployment time are i.i.d. with calibration data
- Losses have a known upper bound $B$
- Loss functions are monotonically non-increasing in $\lambda$

## Unresolved Details / Assumptions

1. **MS-COCO data**: We provide the framework but use synthetic data. Real experiments require downloading MS-COCO and a pre-trained model as in Angelopoulos & Bates (2023).

2. **Number of Dirichlet samples**: The paper states "1000 samples" for the experiments and "100,000 Dirichlet samples" for Figure 4. These values are used.

3. **Lambda grid resolution**: The paper does not specify the grid resolution. We use 200 grid points.

4. **RCPS Hoeffding delta**: We set $\delta = 1 - \beta$ as the natural correspondence between confidence levels.

5. **The proof of Theorem 4.3** in the appendix (B.5) appears to have corrupted/malformed equations. Our implementation follows the theorem statement and the described construction of $L^+$.
