# LUNO: Linearization Turns Neural Operators into Function-Valued Gaussian Processes

This repository aims to reproduce the core contributions of the paper "Linearization Turns Neural Operators into Function-Valued Gaussian Processes (LUNO)".

## Core Contributions and Reproduction Plan

The paper proposes LUNO, a framework for approximate Bayesian uncertainty quantification in trained neural operators. The key idea is to leverage model linearization to push Gaussian weight-space uncertainty forward to the neural operator's predictions, interpreting this as a probabilistic version of currying to yield a function-valued Gaussian process belief.

The main steps outlined in the paper are:

1.  **Uncurrying the Neural Operator (Step 1 in paper, Section 3.2):**
    *   Transform the neural operator $\pmb{F}: \mathbb{A} 	imes \mathbb{W} 	o \mathbb{U}$ into a function $\pmb{f}: (\mathbb{A} 	imes \mathbb{D}_\mathbb{U}) 	imes \mathbb{W} 	o \mathbb{R}^{d'_\mathbb{U}}$. This effectively treats the operator as a standard neural network.

2.  **Obtaining a Gaussian Weight-Space Belief (Step 2 in paper, Section 3.2.1):**
    *   This involves approximating the posterior distribution over the neural network's weights $\pmb{w}$ with a Gaussian distribution $\mathcal{N}(\pmb{\mu}, \pmb{\Sigma})$. The paper mentions methods like Laplace approximation, variational inference, or SWAG. For the purpose of reproduction, we will assume such a $\pmb{\mu}$ and $\pmb{\Sigma}$ are given, as the paper focuses on *using* this belief rather than *deriving* it within the core LUNO framework.

3.  **Linearization of the Neural Network (Part of Step 2 in paper, Section 3.2.1):**
    *   Linearize the uncurried neural network $\pmb{f}$ around the mean weight $\pmb{\mu}$:
        $\pmb{f}((\pmb{a}, \pmb{x}), \pmb{w}) \approx \pmb{f}^	ext{lin}_\mu((\pmb{a}, \pmb{x}), \pmb{w}) = \pmb{f}((\pmb{a}, \pmb{x}), \pmb{\mu}) + \mathrm{D}_{\pmb{w}}\pmb{f}((\pmb{a}, \pmb{x}), \pmb{w})|_{\pmb{\mu}}(\pmb{w} - \pmb{\mu})$.
    *   This linearization induces a multi-output Gaussian Process $\pmb{f}^	ext{lin}_\mu((\pmb{a}, \pmb{x}), \pmb{w}) \sim \mathcal{GP}(m, K)$, with:
        *   Mean function: $m(\pmb{a}, \pmb{x}) = \pmb{f}((\pmb{a}, \pmb{x}), \pmb{\mu})$
        *   Covariance function: $K((\pmb{a}_1, \pmb{x}_1), (\pmb{a}_2, \pmb{x}_2)) = \mathrm{D}_{\pmb{w}}\pmb{f}((\pmb{a}_1, \pmb{x}_1), \pmb{w})|_{\pmb{\mu}} \pmb{\Sigma} \mathrm{D}_{\pmb{w}}\pmb{f}((\pmb{a}_2, \pmb{x}_2), \pmb{w})|_{\pmb{\mu}}^	op$.

4.  **Probabilistic Currying (Step 3 in paper, Section 3.2):**
    *   Construct a function-valued Gaussian random operator $\pmb{F}: \mathbb{A} 	imes \Omega 	o \mathbb{U}$ from the multi-output GP $\pmb{f}$. This means for each $\pmb{a} \in \mathbb{A}$, $\pmb{F}(\pmb{a})$ is a Gaussian random function, and specifically $\pmb{F}(\pmb{a})(\pmb{x}) = \pmb{f}((\pmb{a}, \pmb{x}))$ almost surely.
    *   The resulting function-valued GP has:
        *   Mean: $\mathbb{E}[\pmb{F}(\pmb{a})(\pmb{x})] = \pmb{F}(\pmb{a}, \pmb{\mu})(\pmb{x})$ (which is the original neural operator's prediction with mean weights)
        *   Covariance: $\mathrm{Cov}[\pmb{F}(\pmb{a}_1)(\pmb{x}_1), \pmb{F}(\pmb{a}_2)(\pmb{x}_2)] = \mathrm{D}_{\pmb{w}}\pmb{F}(\pmb{a}_1, \pmb{w})(\pmb{x}_1)|_{\pmb{\mu}} \pmb{\Sigma} \mathrm{D}_{\pmb{w}}\pmb{F}(\pmb{a}_2, \pmb{w})(\pmb{x}_2)|_{\pmb{\mu}}^	op$.

## Implementation Strategy

I will aim to implement a modular Python codebase that reflects these steps.

*   Define abstract base classes or interfaces for Neural Operators.
*   Implement the `uncurry` operation.
*   Provide a structure to define the Jacobian of the neural network with respect to its weights.
*   Implement the mean and covariance functions for the induced function-valued Gaussian Process.
*   Focus on the general framework first, and then consider the Fourier Neural Operator (FNO) case study if time permits. The FNO case study in the paper focuses on a "last-layer Laplace approximation," which simplifies the Jacobian calculation.
