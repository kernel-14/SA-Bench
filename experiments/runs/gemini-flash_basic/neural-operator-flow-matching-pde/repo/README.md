# Reproduction of Bridging Neural Operator and Flow Matching for a Generative PDE Foundation Model

This repository contains an attempt to reproduce the core contributions of the paper "Bridging Neural Operator and Flow Matching for a Generative PDE Foundation Model".

## Implemented Components:

### P2VAE (Pretrained Physics Variational Autoencoder)
- **Description:** Compresses static physical field snapshots into a compact latent grid to reduce the cost of generative training and inference.
- **Status:** (To be implemented/described)

### FMT (Flow Marching Transformer)
- **Description:** Implements a flow marching algorithm that bridges deterministic neural operator and stochastic flow matching through a bridge parameter 'k'.
- **Status:** (To be implemented/described)

### Flow Marching Algorithm
- **Description:** Core algorithm for learning a unified velocity field that transports a noisy current state toward its clean successor.
- **Status:** (To be implemented/described)

### Diffusion Forcing Scheme
- **Description:** Adaptively injects small stochastic increments during autoregressive prediction to mitigate error accumulation without sacrificing stability.
- **Status:** (To be implemented/described)

### Latent Temporal Pyramids
- **Description:** Executes coarse-to-fine transport, cutting training and inference cost while improving long-range consistency.
- **Status:** (To be implemented/described)

## Dataset
- **Description:** A curated training corpus of ~2M trajectories across 12 distinct PDE families, combining FNO-v, PDEBench, PDEArena, and The Well.
- **Status:** The dataset itself is not reproduced, but its specifications and usage are documented.

## Reproduction Notes
- This reproduction focuses on the architectural and algorithmic aspects described in the main body of the paper.
- Any experiments or details solely present in the Appendix are considered out of scope.
- This is a static-only reproduction, meaning no code execution or training was performed. The deliverable is the codebase itself.

## Assumptions and Unresolved Details
- Specific hyperparameter tuning for training might require further experimentation.
- The exact implementation details of some modules (e.g., AdaLN-Zero, SiT) are based on their original papers and general understanding, as the paper provides a high-level description.
- The RNN used for diffusion forcing is assumed to be a GRU, as mentioned in the paper, with the same internal dimension as the SiT embedding.
