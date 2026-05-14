# OLMoE: Open Mixture-of-Experts Language Models

This repository contains a reproduction of the OLMoE paper:

> **OLMoE: Open Mixture-of-Experts Language Models**
> Muennighoff et al. (2024)
> [Paper](https://arxiv.org/abs/2409.02060) | [Model](https://hf.co/allenai/OLMoE-1B-7B-0924) | [Data](https://hf.co/datasets/allenai/OLMoE-mix-0924) | [Code](https://github.com/allenai/OLMoE)

## Overview

OLMoE-1B-7B is a fully open, state-of-the-art language model leveraging sparse Mixture-of-Experts (MoE). It has 7 billion total parameters but uses only 1.3B per input token. The model is pretrained on 5 trillion tokens and can be adapted via instruction tuning and preference optimization.

## Repository Structure

```
olmoe/
├── models/
│   ├── __init__.py
│   ├── configuration.py    # OLMoEConfig with all hyperparameters from Table 10
│   ├── moe.py              # MoE module with load balancing loss & router z-loss
│   └── transformer.py      # Full decoder-only transformer with MoE layers
├── training/
│   ├── __init__.py
│   └── trainer.py          # Training loop, SFT trainer, DPO trainer
├── analysis/
│   ├── __init__.py
│   └── routing.py          # Router saturation, co-activation, domain/vocab specialization
├── data/
│   ├── __init__.py
│   ├── pretraining.py      # Pretraining data processing (OLMoE-MIX)
│   └── adaptation.py       # SFT and preference data processing
scripts/
├── train.py                # Pretraining script
├── adapt.py                # SFT/DPO adaptation script
└── analyze.py              # MoE analysis script
```

## Core Contributions Reproduced

### 1. MoE Architecture (§2, §4.1)

The core MoE module is implemented in `olmoe/models/moe.py`:

- **Dropless token choice routing**: Each token selects top-k experts via learned router
- **Fine-grained experts**: 64 small experts per layer (FFN dim=1024), 8 activated
- **Load balancing loss** (§4.1.6): `L_LB = N_E * Σ f_i * P_i` with weight α=0.01
- **Router z-loss** (§4.1.7): `L_RZ = 1/B * Σ (log Σ exp(x_j))^2` with weight β=0.001
- **No shared experts** (§4.1.3): Experiments showed shared experts reduced flexibility

Total training loss: `L = L_CE + α*L_LB + β*L_RZ`

### 2. Model Configuration (Table 10, Appendix B)

The full model configuration is in `olmoe/models/configuration.py`:

| Parameter | Value |
|-----------|-------|
| Hidden dimension | 2048 |
| Layers | 16 |
| Attention heads | 16 |
| Vocabulary size | 50,304 |
| Max sequence length | 4,096 |
| FFN dimension (per expert) | 1,024 |
| Experts per layer | 64 |
| Activated experts | 8 |
| Activations | SwiGLU |
| Layer norm | RMSNorm (eps=1e-5) |
| QK-Norm | Yes |
| Position embedding | RoPE (θ=10,000) |
| Initialization | Truncated normal (std=0.02, trunc=±3σ) |
| Optimizer | AdamW (eps=1e-8) |
| LR schedule | Cosine with linear warmup (2500 steps) |
| Peak LR | 4.0e-4 |
| Min LR | 4.0e-5 |
| Weight decay | 0.1 (all params including embeddings & RMSNorm) |
| Gradient clipping | 1.0 |
| Batch size | ~4M tokens |
| Total tokens | 5.133T |
| Annealing tokens | 100B (linear decay to 0) |

### 3. Key Design Choices (§4.1)

Based on controlled experiments:

- **MoE vs. Dense** (§4.1.1): MoE reaches dense performance ~2× faster in training time
- **Expert granularity** (§4.1.2): 64 experts with 8 activated (fine-grained) outperforms fewer, larger experts
- **Routing algorithm** (§4.1.4): Token choice outperforms expert choice for text
- **Sparse upcycling** (§4.1.5): Training from scratch catches up to upcycled model after ~500B tokens

### 4. Stability and Performance Improvements (§4.2)

- **Truncated normal init** (§4.2.2): Prevents divergence at ~450B tokens
- **RMSNorm** (§4.2.3): Replaces non-parametric LayerNorm for better gradient behavior
- **Weight decay on all params** (§4.2.3, §4.2.4): Includes RMSNorm and embedding params
- **QK-Norm** (§4.2.5): Added for stability, prevents large attention logits
- **AdamW epsilon** (§4.2.6): Reduced to 1e-8 for faster convergence

### 5. Adaptation (§2, §4.3)

- **SFT**: No load balancing loss, LR=2e-5, 2 epochs, batch size 128
- **DPO**: No load balancing loss, LR=5e-7, 3 epochs, batch size 32, β=0.1
- **KTO**: Alternative to DPO with similar performance
- **Post-annealing checkpoint** used for adaptation

### 6. MoE Analysis (§5)

Implementation in `olmoe/analysis/routing.py`:

- **Router saturation** (§5.1): Proportion of expert activations matching final checkpoint routing. Saturates early (~60% at 1% of training)
- **Expert co-activation** (§5.2): Proportion of times two experts are simultaneously activated. Little co-activation found, suggesting low redundancy
- **Domain specialization** (§5.3): How often tokens from a domain route to each expert. OLMoE shows high specialization vs. Mixtral
- **Vocabulary specialization** (§5.4): How often specific tokens route to each expert. Higher in later layers; expert 27 specializes on non-alphabetic tokens

### 7. Data (§2)

- **OLMoE-MIX**: DCLM-Baseline + StarCoder + peS2o + arXiv + OpenWebMath + Algebraic Stack + Wikipedia/Wikibooks
- **Filtering**: Remove documents with 32+ repeated n-grams; StarCoder quality filters
- **Total**: 4.06T GPT-NeoX tokens, 17.4TB UTF-8 bytes, 3.08B documents

## Usage

### Installation

```bash
pip install -e .
```

### Creating a Model

```python
from olmoe.models import create_olmoe_model

model = create_olmoe_model()
print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
```

### Training

```bash
python scripts/train.py --data_path /path/to/data/*.jsonl --output_dir ./checkpoints
```

### Adaptation

```bash
# SFT
python scripts/adapt.py --mode sft --model_path checkpoint.pt --data_path sft_data.jsonl

# DPO
python scripts/adapt.py --mode dpo --model_path sft_checkpoint.pt --data_path dpo_data.jsonl
```

### Analysis

```bash
# Run all analyses
python scripts/analyze.py --model_path checkpoint.pt --data_path data.jsonl --analysis all

# Individual analyses
python scripts/analyze.py --model_path checkpoint.pt --data_path data.jsonl --analysis saturation
python scripts/analyze.py --model_path checkpoint.pt --data_path data.jsonl --analysis coactivation
python scripts/analyze.py --model_path checkpoint.pt --data_path data.jsonl --analysis domain
python scripts/analyze.py --model_path checkpoint.pt --data_path data.jsonl --analysis vocabulary
```

## Assumptions & Unresolved Details

1. **Data**: The actual OLMoE-MIX dataset is ~17TB. Our data processing code provides the pipeline but requires the actual dataset files.
2. **Training Scale**: The paper trains on 256 H100 GPUs for ~10 days. This reproduction provides the training loop but does not include distributed training (FSDP) setup.
3. **Efficient MoE Implementation**: The current MoE forward pass is a simple loop-based implementation. Production use requires efficient implementations like MegaBlocks or grouped GEMM operations.
4. **Checkpoint Conversion**: Loading the actual OLMoE-1B-7B weights from HuggingFace would require model weight mapping.
5. **Tokenization**: Uses the GPT-NeoX tokenizer (EleutherAI/gpt-neox-20b). The repository includes a fallback dummy tokenizer for development.
6. **Evaluation**: The paper uses OLMES, DCLM, and adaptation evaluation suites. Evaluation code is not included in this reproduction.
7. **KTO Implementation**: The KTO algorithm is referenced but not fully implemented; DPO is the primary preference method.

## Differences from the Original Paper

- The original uses PyTorch FSDP with ZeRO optimization for distributed training across 256 GPUs. This reproduction provides a simpler single-device implementation.
- The MoE forward pass uses a loop over experts rather than optimized grouped operations.
- Training hyperparameters are fully specified but the actual training scripts assume single-device execution.

## License

This code is released under Apache 2.0 license, matching the original OLMoE release.
