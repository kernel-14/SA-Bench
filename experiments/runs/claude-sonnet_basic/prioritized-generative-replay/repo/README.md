# Prioritized Generative Replay (PGR) - Reproduction

This repository reproduces the paper:

> **Prioritized Generative Replay**  
> Renhao Wang, Kevin Frans, Pieter Abbeel, Sergey Levine, Alexei A. Efros  
> UC Berkeley

## Overview

PGR is a framework for scalable, guidable generative replay in online reinforcement learning. It uses a conditional diffusion model to generate synthetic transitions that are guided towards more "relevant" parts of the agent's experience, as measured by a relevance function F(tau).

### Key Contributions Reproduced

1. **Conditional Diffusion Model** (`pgr/diffusion.py`): A residual MLP-based DDPM with classifier-free guidance (CFG) for generating RL transitions. Implements Eq. 2 from the paper.

2. **Relevance Functions** (`pgr/relevance.py`): Three main variants:
   - **Curiosity (ICM)**: F(s,a,s',r) = 0.5 * ||g(h(s), a) - h(s')||^2 (Eq. 5, recommended default)
   - **Return**: F(s,a,s',r) = Q(s, pi(s)) (Eq. 3)
   - **TD Error**: F(s,a,s',r) = |r + gamma*Q_target(s', argmax Q(s',a')) - Q(s,a)| (Eq. 4)
   - **Reward**: F(s,a,s',r) = r (naive baseline)

3. **REDQ Agent** (`pgr/redq.py`): Randomized Ensembled Double Q-Learning backbone (Chen et al., 2021) with N=10 Q-networks, M=2 for target computation, UTD=20.

4. **PGR Training Loop** (`pgr/pgr_trainer.py`): Implements Algorithm 1:
   - Outer loop: collect real transitions, update relevance function F
   - Inner loop (every 10K steps): train diffusion model, generate 1M synthetic transitions
   - Policy update: train on mixed real+synthetic data (50/50 ratio)

5. **Prompting Strategy**: Conditions generation on top-k (50%) highest relevance values from D_real.

## Architecture Details

### Diffusion Model
- Residual MLP denoiser with 4 layers, 256 hidden dim
- Sinusoidal time embeddings (128 dim)
- Condition embedding: scalar relevance -> 64 dim MLP
- Null condition embedding (learned) for CFG
- Cosine noise schedule, 100 diffusion steps
- p_uncond = 0.25 (CFG dropout probability)
- Guidance scale omega = 1.2

### REDQ Policy
- 10 Q-networks ensemble, 2 randomly selected for target
- 2-layer MLP, 256 hidden dim
- UTD ratio = 20
- Batch size = 256 (50% real, 50% synthetic)

### Curiosity Module (ICM)
- Feature encoder: 2-layer MLP, 64 feature dim
- Forward dynamics model: (feature, action) -> next_feature
- Inverse dynamics model (optional): (feature, next_feature) -> action
- Updated every 20 policy gradient steps (~5% of updates)

## Experiments

### Table 1: DMC State-Based (100K steps)
```bash
bash scripts/run_dmc_experiments.sh
```

Environments: quadruped-walk, cheetah-run, reacher-hard, finger-turn-hard (300K)

### Table 2: OpenAI Gym (100K steps)
```bash
bash scripts/run_gym_experiments.sh
```

Environments: Walker2d-v2, HalfCheetah-v2, Hopper-v2

### Figure 7: Scaling Experiments
```bash
bash scripts/run_scaling_experiments.sh
```

Tests: (a) larger networks (3 layers, 512 hidden), (b) higher synthetic ratio (r=0.75), (c) UTD=40

### Single Run
```bash
# PGR with curiosity (main method)
python train.py --env quadruped-walk --mode pgr --relevance curiosity --seed 0

# SYNTHER baseline (unconditional generation)
python train.py --env quadruped-walk --mode synther --seed 0

# REDQ baseline (model-free)
python train.py --env quadruped-walk --mode redq --seed 0
```

## Analysis

```bash
# Generate figures from saved results
python analysis.py --results_dir results --output_dir figures

# Evaluate a checkpoint
python evaluate.py --checkpoint results/quadruped-walk_pgr_curiosity_seed0/checkpoint_100000
```

## File Structure

```
pgr/
  __init__.py          - Package exports
  diffusion.py         - Conditional DDPM with CFG
  relevance.py         - Relevance functions (curiosity, return, TD-error, reward)
  replay_buffer.py     - Real and synthetic replay buffers with normalization
  redq.py              - REDQ agent (Q-ensemble + SAC actor)
  pgr_trainer.py       - Main PGR training loop (Algorithm 1)
  dmc_wrapper.py       - DeepMind Control Suite environment wrapper

train.py               - Main training script
evaluate.py            - Evaluation and result aggregation
analysis.py            - Analysis scripts for paper figures
scripts/
  run_dmc_experiments.sh    - DMC benchmark (Table 1)
  run_gym_experiments.sh    - Gym benchmark (Table 2)
  run_scaling_experiments.sh - Scaling experiments (Figure 7)
```

## Assumptions and Missing Details

1. **Diffusion architecture**: The paper references SynthER (Lu et al., 2024) for architecture details. We use a residual MLP denoiser matching the ~7M parameter count reported in Table 3.

2. **Inner loop frequency**: The paper states "every 10K iterations" (confirmed in Appendix D, Figure 9).

3. **Prompting strategy**: The paper references Peebles et al. (2022) for the prompting strategy. We implement this as sampling conditioning values from the top-k (50%) highest relevance transitions.

4. **Curiosity update frequency**: The paper states the curiosity head is "updated for only 5% of all policy gradient steps." With UTD=20, this means updating every 20 policy steps.

5. **Normalization**: Transitions are normalized to zero mean and unit variance before diffusion training, following SynthER.

6. **Pixel-based tasks**: The paper generates in the latent space of the CNN visual encoder (DRQ-V2). This requires a pixel-based wrapper not fully implemented here.

## Results

Expected performance at 100K environment steps (from paper):

| Method | Quadruped-Walk | Cheetah-Run | Reacher-Hard |
|--------|---------------|-------------|--------------|
| REDQ | 496.75 ± 151.00 | 606.86 ± 99.77 | 733.54 ± 79.66 |
| SYNTHER | 727.01 ± 86.66 | 729.35 ± 49.59 | 838.60 ± 131.15 |
| PGR (Curiosity) | **927.98 ± 25.18** | **817.36 ± 35.93** | **915.21 ± 48.24** |
