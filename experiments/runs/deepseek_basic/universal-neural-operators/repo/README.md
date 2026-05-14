# Towards Universal Neural Operators through Multiphysics Pretraining

Reproduction attempt of the paper: "Towards Universal Neural Operators through Multiphysics Pretraining" by Mikhail Masliaev, Dmitry A. Gusarov, Ilya Markov, Alexander Hvatov (ITMO University).

## Overview

This repository implements the core contributions of the paper:

1. **Adapter-based pretraining framework**: Separate lifting and projection layers act as adapters, allowing the same core neural operator to handle different PDE problems with varying input function cardinalities.

2. **MambaFNO**: A neural operator with a post-lifting Mamba SSM module that encodes long-range temporal and spatial dependencies, acting as a latent preconditioner.

3. **Perceiver IO-based Neural Operator**: Cross-attention blocks with FNO-based key/value mappings, operating on learned latent arrays for efficient multi-physics representation.

4. **CoDA-NO (Codomain Attention Neural Operator)**: Uses function-space dot products to compute attention between feature channels rather than spatial samples.

5. **Swin-v2 Transformer Neural Operator**: Hierarchical vision transformer with shifted windows, adapted for PDE operator learning.

6. **Fine-tuning with frozen core**: During fine-tuning, only adapter parameters (lifting + projection) are trained while core operator weights remain frozen, significantly reducing computational cost.

## Architecture

All models follow the lifting-operator-projection paradigm:

```
Input(x) → Lifting(L) → [Mamba/Attention] → FNO/Operator Blocks(F) → Projection(P) → Output(ũ)
               ↑                                                           ↑
          Adapter (trainable in FT)                                  Adapter (trainable in FT)
```

### Pretraining & Fine-tuning (Section 3)

- **Pretraining**: All parameters (adapters + core) are optimized on multiple physics problems simultaneously
- **Fine-tuning**: Core operator parameters θ_F are frozen, only new adapter parameters (θ_P_ft, θ_L_ft) are trained

## Repository Structure

```
.
├── models/                     # Neural operator architectures
│   ├── __init__.py
│   ├── fno.py                  # Baseline Fourier Neural Operator
│   ├── mamba_fno.py            # MambaFNO (post-lifting Mamba SSM)
│   ├── perceiver_fno.py        # Perceiver IO-based NO
│   ├── codomain_attention.py   # CoDA-NO
│   ├── swin_transformer.py     # Swin-v2 Transformer NO
│   └── adapters.py             # Lift/Projection adapters, LocalAttnFNO
├── data/                       # Dataset generation and loading
│   ├── __init__.py
│   ├── pde_dataset.py          # Base dataset class, dataloader utils
│   └── pde_generators.py       # PDE solvers and dataset generators
├── training/                   # Training framework
│   ├── __init__.py
│   ├── pretrain.py             # Multi-physics pretraining
│   ├── finetune.py             # Fine-tuning with frozen core
│   └── metrics.py              # NMAE and MSE metrics
├── utils/                      # Helper utilities
│   ├── __init__.py
│   └── helpers.py              # Parameter counting, grid creation
├── configs/                    # Configuration files
│   └── default.yaml            # Default experiment config
├── experiments.py              # Main experiment runner
└── README.md
```

## Supported PDE Problems

Based on PDEBench [20] and the experiments described in Section 4:

- **Burgers' equation** (viscous, 2D): du/dt + u·∇u = ν∇²u
- **Gray-Scott reaction-diffusion**: Pattern-forming RD system
- **Navier-Stokes** (incompressible, 2D vorticity formulation)
- **Heat equation**: du/dt = α∇²u
- **Advection equation**: du/dt + v·∇u = 0
- **Heat + Convection**: Extended heat equation with advection
- **Reaction-Diffusion + Advection**: Extended RD with advective transport

## Experiment Scenarios (Section 4)

### Scenario 1: Out-of-sample parameter values
Pretraining on one parameter range, fine-tuning on a different range for the same equation type. Evaluated on Burgers', Gray-Scott, and Navier-Stokes.

### Scenario 2: Input function set extension
Pretraining on simpler equations, fine-tuning on extended versions with additional terms (heat → heat+convection, reaction-diffusion → RD+advection).

### Scenario 3: General multi-physics learning
Cross-domain transfer: pretrain on advection and Burgers', fine-tune on reaction-diffusion.

## Usage

### Installation

```bash
pip install torch numpy scipy h5py pyyaml
```

### Running Experiments

```bash
# Scenario 1: Out-of-sample parameters with MambaFNO
python experiments.py --scenario 1 --model mamba_fno --pretrain_epochs 50 --finetune_epochs 100

# Scenario 2: Input function extension with Perceiver
python experiments.py --scenario 2 --model perceiver --pretrain_epochs 50 --finetune_epochs 100

# Scenario 3: Cross-domain transfer with CoDA-NO
python experiments.py --scenario 3 --model codano --pretrain_epochs 50 --finetune_epochs 100

# Baseline FNO from scratch
python experiments.py --scenario 1 --model fno --pretrain_epochs 0 --finetune_epochs 100
```

### Available Models
- `fno` - Baseline Fourier Neural Operator
- `mamba_fno` - MambaFNO with post-lifting SSM
- `perceiver` - Perceiver IO-based NO
- `codano` - Codomain Attention NO
- `swin` - Swin-v2 Transformer NO
- `local_attn_fno` - Post-lifting Local Attention FNO

## Metrics

Following the paper, we use:

- **MSE**: Mean Squared Error between prediction and ground truth
- **NMAE**: Range-Normalized Mean Absolute Error (Equation 3 in paper)
  ```
  NMAE(θ) = (1/|D_test|) Σ ||G_θ(a) - u||_{1,G} / (max_G u - min_G u + ε)
  ```

## Key Design Decisions

1. **Adapter-based design**: Lift and projection layers are treated as problem-specific adapters, analogous to LoRA in language models. This enables:
   - Different input cardinalities per physics problem
   - Small parameter count for adapters
   - Efficient fine-tuning by freezing core weights

2. **Mamba SSM placement**: The Mamba module is placed post-lifting (before FNO blocks), as described in the paper. It acts as a "latent preconditioner" aligning embeddings with common dynamical motifs.

3. **Perceiver cross-attention**: Keys and values are computed from inputs via FNO mappings (1x1 convolutions), queries come from learned latent arrays. Output cross-attention reverses this.

4. **Codomain attention**: Dot product is computed over spatial dimensions (integral over domain) rather than between spatial positions, capturing function-space similarities.

## Assumptions and Unresolved Details

The following aspects are implemented based on reasonable interpretations of the paper:

1. **Mamba SSM specifics**: The paper does not provide exact Mamba hyperparameters. We use d_state=16, d_conv=4, expand=2 based on common practice.

2. **Perceiver IO details**: The paper mentions the Perceiver IO architecture but we implement a simplified version with FNO-based key/value projections.

3. **Swin-v2 configuration**: Window sizes, number of stages, and patch merging are adapted from standard Swin Transformer design.

4. **Training hyperparameters**: Learning rates, scheduler settings, and number of epochs are set to reasonable defaults. The paper reports results with approximately 10^6-10^9 parameters; we scale models accordingly.

5. **PDEBench data format**: The paper uses PDEBench [20] datasets. Our generators produce synthetic data in similar format using spectral methods.

6. **Exact experimental protocol**: The paper describes three scenarios but does not enumerate all exact parameter splits. We implement the general framework described.

## References

- [3] Kovachki et al., "Neural Operator: Learning Maps between Function Spaces", JMLR 2023
- [12] Herde et al., "POSEIDON: Efficient Foundation Models for PDEs", NeurIPS 2024
- [13] Rahman et al., "Pretraining Codomain Attention Neural Operators", NeurIPS 2024
- [16] Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2023
- [18] Jaegle et al., "Perceiver IO: A General Architecture for Structured Inputs & Outputs", 2021
- [19] Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022
- [20] Takamoto et al., "PDEBench: An Extensive Benchmark for Scientific Machine Learning", NeurIPS 2022
