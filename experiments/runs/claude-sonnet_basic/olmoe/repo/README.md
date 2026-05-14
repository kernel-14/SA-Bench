# OLMoE: Open Mixture-of-Experts Language Models - Reproduction

This repository reproduces the core contributions of the paper:
**"OLMoE: Open Mixture-of-Experts Language Models"** (Muennighoff et al., 2024)

## Overview

OLMoE-1B-7B is a fully open, state-of-the-art language model using sparse Mixture-of-Experts (MoE). It has 7B total parameters but uses only 1.3B per input token, achieving similar inference cost to 1B dense models while matching or exceeding 7B dense models on many benchmarks.

## Repository Structure

```
.
├── src/
│   ├── model.py           # OLMoE architecture (MoE transformer)
│   ├── train.py           # Pretraining script with optimizer/scheduler
│   ├── adaptation.py      # SFT and DPO fine-tuning
│   ├── sparse_upcycling.py # Sparse upcycling from dense models
│   └── expert_choice.py   # Expert Choice routing (ablation)
├── analysis/
│   ├── moe_analysis.py    # Router saturation, co-activation, specialization
│   └── visualize.py       # Plotting functions for analysis figures
├── configs/
│   ├── olmoe_1b_7b.yaml   # Full OLMoE-1B-7B configuration
│   ├── adaptation.yaml    # SFT and DPO configuration
│   └── ablations.yaml     # Ablation experiment configurations
├── scripts/
│   └── prepare_data.py    # Data filtering and preprocessing
└── requirements.txt
```

## Core Contributions Reproduced

### 1. OLMoE Architecture (`src/model.py`)

The OLMoE-1B-7B model with exact configuration from Table 10:

| Parameter | Value |
|-----------|-------|
| Hidden size | 2048 |
| Layers | 16 |
| Attention heads | 16 |
| Vocab size | 50,304 |
| Total experts | 64 |
| Activated experts | 8 |
| Expert FFN dim | 1,024 |
| Total params | ~6.9B |
| Active params | ~1.3B |

Key architectural features:
- **Dropless token choice routing**: Each token selects top-8 experts (no token dropping)
- **Fine-grained experts**: 64 small experts (1024 FFN dim) vs few large ones
- **RMSNorm**: Parametric, applied to all layers including QK projections
- **QK-Norm**: Applied after Q and K projections for stability
- **RoPE**: Rotary position embeddings with theta=10000
- **SwiGLU**: Activation function for expert FFNs
- **Truncated normal init**: std=0.02, clip at ±3*std=±0.06

### 2. Training Losses (`src/model.py`)

The total training loss is:
```
L = L_CE + alpha * L_LB + beta * L_RZ
```

**Load Balancing Loss** (alpha=0.01, from Shazeer et al. 2017):
```
L_LB = N_E * sum_{i=1}^{N_E} f_i * P_i
```
where f_i = fraction of tokens routed to expert i, P_i = routing probability for expert i.

**Router Z-Loss** (beta=0.001, from Zoph et al. 2022 ST-MoE):
```
L_RZ(x) = (1/B) * sum_{i=1}^{B} (log sum_{j=1}^{N_E} exp(x_j^(i)))^2
```
Penalizes large logits to prevent numeric overflows.

### 3. Pretraining Setup (`src/train.py`, `configs/olmoe_1b_7b.yaml`)

- **Optimizer**: AdamW with epsilon=1e-8 (key: reduced from OLMo's 1e-5)
- **LR schedule**: Cosine with 2500 warmup steps, then linear annealing to 0 for final 100B tokens
- **Peak LR**: 4e-4, minimum LR: 4e-5
- **Batch size**: ~4M tokens (1024 samples × 4096 seq len)
- **Weight decay**: 0.1 applied to ALL parameters (including embeddings and RMSNorm)
- **Gradient clipping**: 1.0 (global norm)
- **Training duration**: 5.133T tokens (1.3 epochs)
- **Mixed precision**: BF16

### 4. Pretraining Data (`scripts/prepare_data.py`, `configs/olmoe_1b_7b.yaml`)

OLMoE-Mix composition (Table 2):
| Source | Tokens (B) |
|--------|-----------|
| DCLM-Baseline | 3,860 |
| StarCoder | 101 |
| peS2o | 57.2 |
| arXiv | 21.1 |
| OpenWebMath | 12.7 |
| Algebraic Stack | 12.6 |
| Wikipedia+Wikibooks | 3.69 |
| **Total** | **~4,060** |

Data filtering:
- Remove documents with 32+ repeated n-grams (n=1 to 13 tokens)
- StarCoder: remove repos with <2 GitHub stars, top-1 word >30%, top-2 words >50%

### 5. Adaptation (`src/adaptation.py`, `configs/adaptation.yaml`)

**Instruction Tuning (SFT)**:
- Dataset: Tulu 2 SFT Mix + No Robots + CodeFeedback + MetaMathQA + Daring Anteater
- 2 epochs, constant LR=2e-5, global batch size 128
- Token-level loss aggregation (improves AlpacaEval)
- **Key finding**: NO load balancing loss during SFT

**Preference Tuning (DPO)**:
- Dataset: UltraFeedback binarized (filtered for TruthfulQA contamination)
- 3 epochs, LR=5e-7, DPO beta=0.1, global batch size 32
- **Key finding**: NO load balancing loss during DPO

### 6. Design Experiments (`configs/ablations.yaml`)

Key findings from Section 4:

| Experiment | Finding |
|-----------|---------|
| MoE vs Dense | MoE reaches dense performance with ~3x fewer tokens |
| Expert granularity | 64 experts > 32 > 8 (diminishing returns) |
| Shared experts | No shared expert is better (reduces combinations) |
| EC vs TC routing | Token choice outperforms expert choice |
| Sparse upcycling | From-scratch MoE catches up after ~500B tokens |
| Load balancing loss | Essential to prevent expert collapse |
| Router Z-loss | Improves stability and quality |
| Initialization | Truncated normal prevents divergence at ~450B tokens |
| RMSNorm | Better than non-parametric LayerNorm |
| QK-Norm | Improves stability |
| AdamW epsilon | 1e-8 significantly better than 1e-5 |

### 7. MoE Analysis (`analysis/moe_analysis.py`, `analysis/visualize.py`)

Four analysis metrics from Section 5:

**Router Saturation** (Section 5.1):
```
Router Saturation(t) = (1/N) * sum_{i=1}^{N} |E_i^(t) ∩ E_i^(T)| / k
```
- After 1% of pretraining: ~60% of top-8 routing saturated
- After 40%: ~80% saturated
- Later layers saturate earlier; Layer 0 is an outlier

**Expert Co-activation** (Section 5.2):
```
Expert co-activation(E_i, E_j) = N_{E_i, E_j} / N_{E_i}
```
- No strong co-activation (little redundancy across experts)

**Domain Specialization** (Section 5.3):
```
Domain specialization(E_i, D) = N_{E_i, D}^(k) / N_D
```
- Strong specialization for specific domains (arXiv, GitHub)
- Generic domains (C4) show balanced activations
- Mixtral shows little specialization (likely due to upcycling)

**Vocabulary Specialization** (Section 5.4):
```
Vocabulary specialization(E_i, x) = N_{x, E_i}^(k) / N_x
```
- Higher specialization in later layers
- Later layers specialize more on predicted output tokens

### 8. Sparse Upcycling (`src/sparse_upcycling.py`)

Implements the sparse upcycling experiment from Section 4.1.5:
- Clone dense FFN for each expert
- Initialize new router from scratch
- Continue pretraining

Finding: From-scratch MoE catches up with upcycled model after ~500B tokens (much faster than the 120% compute budget reported in prior work).

## Usage

### Create OLMoE-1B-7B Model

```python
from src.model import create_olmoe_1b_7b

model = create_olmoe_1b_7b()
print(f"Total params: {model.get_num_params() / 1e9:.2f}B")
```

### Training

```python
from src.model import create_olmoe_1b_7b
from src.train import TrainingConfig, OLMoETrainer

model = create_olmoe_1b_7b()
config = TrainingConfig()
trainer = OLMoETrainer(model, config, train_dataloader)
trainer.train()
```

### Adaptation (SFT + DPO)

```python
from src.adaptation import SFTConfig, DPOConfig, SFTTrainer, DPOTrainer

# SFT
sft_config = SFTConfig()
sft_trainer = SFTTrainer(model, sft_config, sft_dataloader)
sft_trainer.train()

# DPO
dpo_config = DPOConfig()
dpo_trainer = DPOTrainer(policy_model, reference_model, dpo_config, dpo_dataloader)
dpo_trainer.train()
```

### MoE Analysis

```python
from analysis.moe_analysis import (
    compute_router_saturation,
    compute_expert_coactivation,
    compute_domain_specialization,
    compute_vocabulary_specialization,
)

# Router saturation
saturation = compute_router_saturation(model_intermediate, model_final, dataset)

# Expert co-activation
coactivation = compute_expert_coactivation(model, dataset, layer_idx=7)

# Domain specialization
domain_spec = compute_domain_specialization(model, domain_datasets)

# Vocabulary specialization
vocab_spec = compute_vocabulary_specialization(model, dataset, k=1)
```

## Assumptions and Unresolved Details

1. **Exact data mixing ratios**: The paper gives token counts but not exact sampling weights for multi-epoch training. We use proportional weights.

2. **Tokenizer**: The paper uses GPT-NeoX tokenizer (vocab size 50,304). We assume this is the same as used in OLMo.

3. **FSDP configuration**: The paper uses PyTorch FSDP with ZeRO. Exact sharding strategy not specified.

4. **Evaluation setup**: The paper uses OLMES for evaluation. We document the setup but don't reproduce the full evaluation harness.

5. **Intermediate checkpoints**: The paper saves every 5000 steps. Our implementation matches this.

6. **Annealing data**: During annealing, the paper reshuffles the entire dataset. Implementation details not fully specified.

7. **Expert Choice routing**: The paper's EC implementation uses dropless MoE. Our implementation may differ slightly in how token dropping is handled.

## Key Differences from OLMo

OLMoE-1B-7B differs from OLMo-1B in:
1. MoE architecture (64 experts, 8 activated)
2. RMSNorm instead of non-parametric LayerNorm
3. QK-Norm for stability
4. AdamW epsilon: 1e-8 (vs 1e-5)
5. Truncated normal initialization
6. Load balancing loss + Router Z-loss
7. Different training data (OLMoE-Mix vs Dolma 1.7)
8. Weight decay applied to ALL parameters

## Results

After pretraining (Table 4):
- MMLU: 54.1% (best among ~1B active parameter models)
- HellaSwag: 80.0%
- ARC-Challenge: 62.1%
- ARC-Easy: 84.2%
- PIQA: 79.8%
- WinoGrande: 70.2%

After adaptation (Table 5, OLMOE-1B-7B-INSTRUCT):
- MMLU: 51.9%
- GSM8k: 45.5%
- BBH: 37.0%
- HumanEval: 54.8%
- AlpacaEval: 84.0%
- XSTest: 82.6%
- IFEval: 48.1%
- Average: 57.7%
