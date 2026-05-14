-e # Reproduction of Conformal Prediction as Bayesian Quadrature

This repository aims to reproduce the core contributions of the paper Conformal Prediction as Bayesian Quadrature by Jake C. Snell and Thomas L. Griffiths.

## Paper Summary

The paper introduces a novel Bayesian framework for conformal prediction, reinterpreting existing distribution-free uncertainty quantification techniques (Split Conformal Prediction and Conformal Risk Control) from a Bayesian perspective. The key insight is to formulate the problem in terms of Bayesian quadrature, where the uncertainty in quantile values is explicitly modeled.

Instead of relying on a prior distribution over functions, the paper derives an upper bound on the posterior expected loss. This is achieved by leveraging properties of distribution-free tolerance regions, specifically that quantile spacings follow a Dirichlet distribution. The main contribution is the introduction of a random variable ^+$, which stochastically dominates the posterior risk. By analyzing the distribution of ^+$, the authors propose a decision rule that provides conditional guarantees on the expected loss.

The paper demonstrates that existing conformal methods can be recovered as special cases (by taking the expectation of ^+$), and that the proposed Bayesian approach offers a richer characterization of uncertainty, leading to more robust guarantees in practice.

## Core Contributions to be Reproduced

1.  **Bayesian Quadrature for Expected Loss (^+$ and $\lambda_{hpd}^{eta}$)**:
    *   Implementation of functions to sample from a Dirichlet distribution.
    *   Calculation of ^+$ (Theorem 4.3, Equation 27) given observed losses and a maximum possible loss $.
    *   Determination of the critical value ^*_{eta}$ (Corollary 4.4, Equation 29) through Monte Carlo simulation of ^+$.
    *   Implementation of the decision rule $\lambda_{hpd}^{eta}$ (Equation 31), which involves finding the $\lambda$ that satisfies the HPD criterion. This will require searching for $\lambda$ by iteratively calculating ^+$ for different $\lambda$ values.

2.  **Recovery of Conformal Methods (Baselines)**:
    *   Implementation of the **Conformal Risk Control (CRC)** decision rule (Proposition 3.2, Equation 15). The paper shows this is equivalent to (L^+)$.
    *   Implementation of the **Split Conformal Prediction (SCP)** decision rule (Proposition 3.1, Equation 12), also shown to be recoverable from (L^+)$.

## Experiments to be Reproduced

1.  **Synthetic Binomial Data (Section 5.1)**:
    *   Implement the binomial loss function (Equation 34).
    *   Simulate data and run the CRC and proposed HPD methods.
    *   Evaluate the relative frequency of exceeding the target risk $lpha$.
    *   Generate histograms of chosen $\lambda$ values.

2.  **Synthetic Heteroskedastic Data (Section 5.2)**:
    *   Implement the data generation process for heteroskedastic data (|X \sim \mathcal{N}(0, X^2)$) and the miscoverage loss.
    *   Run simulations for CRC and proposed HPD methods.
    *   Evaluate relative frequency of exceeding target risk and mean prediction interval length.

## Assumptions and Limitations

*   **No Prior Specification**: The core method explicitly avoids the need for a specific prior over quantile functions, relying on a conservative bound.
*   **i.i.d. Data**: Assumes calibration data and future deployment data are independent and identically distributed.
*   **Bounded Losses**: Assumes an upper bound $ on losses.
*   **Computational Resources**: Monte Carlo simulations will be used to approximate distributions and critical values. The number of samples will be chosen based on the paper's suggestions (e.g., 1000 samples for ^*_{eta}$, 100,000 for PDF plots if needed).
*   **RCPS Baseline**: Implementation of the RCPS baseline is outside the scope of this initial reproduction attempt due to time constraints and lack of detailed implementation description in the main paper.
*   **MS-COCO Experiment**: The MS-COCO experiment will not be reproduced due to its complexity and the lack of readily available simplified data/model setup instructions within the paper. The focus will be on the synthetic experiments that demonstrate the core methodological contributions.

## Code Structure

The code will be organized into  and will contain:
*   : Functions for sampling from Dirichlet distributions.
*   : Core implementation of ^+$, ^*_{eta}$, and $\lambda_{hpd}^{eta}$.
*   : Implementations of Split Conformal Prediction and Conformal Risk Control.
*   : Code for running the synthetic experiments and collecting results.
*   : Any utility functions like loss definitions, sorting, etc.

This  will be updated as the reproduction progresses.
