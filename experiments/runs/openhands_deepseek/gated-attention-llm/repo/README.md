# Gated Attention for Large Language Models

Reproduction of the paper "Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free".

## Overview

This repository implements gated attention mechanisms in standard softmax attention for transformer language models. The key finding is that applying a head-specific sigmoid gate after Scaled Dot-Product Attention (SDPA) consistently improves performance, training stability, and eliminates the attention sink phenomenon.

## Codebase Structure

```
repo/
├── config.py           # Model and training hyperparameters
├── gated_attention.py  # Core gated attention variants (G1-G5)
├── model.py            # Full transformer (dense + MoE)
├── moe.py              # Mixture of Experts components
├── train.py            # Training loop with distributed support
├── evaluate.py         # Evaluation on paper benchmarks
├── data.py             # Data loading and preprocessing
└── requirements.txt    # Dependencies
```

## Key Components

### Gating Variants (`gated_attention.py`)
- **G1**: Sigmoid gating after SDPA output (best performing)
- **G2**: Gating after value projection
- **G3**: Gating after key projection
- **G4**: Gating after query projection
- **G5**: Gating after output projection
- Elementwise and headwise granularity
- Head-specific and head-shared modes
- Multiplicative and additive gating
- Sigmoid, SiLU, and Identity activations

### Model Architectures (`model.py`)
- Dense transformer (1.7B params, 28 or 48 layers)
- MoE transformer (15B total, 2.54B activated, 128 experts, top-8 gating)
- GQA (Grouped Query Attention) support
- Pre-norm architecture with RMSNorm

### MoE Components (`moe.py`)
- Fine-grained experts (DeepSeekMoE style)
- Top-k softmax routing
- Global-batch Load Balancing Loss (Qiu et al., 2025)
- Z-loss stabilization (ST-MoE)

## Usage

### Training
```bash
# Dense model with SDPA elementwise gating
python train.py \
    --model_type dense \
    --gating_position G1 \
    --gating_granularity elementwise \
    --gating_activation sigmoid \
    --n_layers 28 \
    --d_model 2048 \
    --max_lr 4e-3 \
    --batch_size 1024 \
    --data_path data/tokens \
    --output_dir outputs

# MoE 15A2B model with SDPA headwise gating
python train.py \
    --model_type moe \
    --gating_position G1 \
    --gating_granularity headwise \
    --n_layers 48 \
    --d_model 4096 \
    --max_lr 2e-3 \
    --batch_size 1024 \
    --data_path data/tokens
```

### Evaluation
```python
from config import ModelConfig
from evaluate import run_full_evaluation

config = ModelConfig(...)
run_full_evaluation(
    model_path="outputs/gated_attention_final.pt",
    model_config=config,
    data_dir="data/tokens",
)
```

## Benchmarks

The evaluation suite includes:
- **Perplexity**: English, Chinese, Code, Math, Law, Literature held-out sets
- **Hellaswag**: English commonsense reasoning
- **MMLU**: Multi-task language understanding (5-shot)
- **GSM8k**: Math reasoning (5-shot)
- **HumanEval**: Code generation (pass@1)
- **C-Eval**: Chinese evaluation (5-shot)
- **CMMLU**: Chinese multi-task
- **RULER**: Long-context evaluation (4k-128k)

## Paper Results Summary

| Method | Avg PPL | Hellaswag | MMLU | GSM8k |
|--------|---------|-----------|------|-------|
| Baseline (MoE 15A2B) | 6.026 | 73.07 | 58.79 | 52.92 |
| SDPA Elementwise G1 | 5.761 | 74.64 | 60.82 | 55.27 |
| SDPA Headwise G1 | 5.792 | 74.50 | 60.05 | 54.44 |
| V Elementwise G2 | 5.820 | 74.38 | 59.17 | 53.97 |

## Requirements

- Python 3.10+
- PyTorch 2.0+
- NVIDIA GPU with CUDA support

## References

- Qiu et al., 2025. "Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free"
- Vaswani et al., 2017. "Attention Is All You Need"
- Dai et al., 2024. "DeepSeekMoE: Towards Ultimate Expert Specialization"
- Qiu et al., 2025. "Demons in the Detail: On Implementing Load Balancing Loss"
- Zoph et al., 2022. "ST-MoE: Designing Stable and Transferable Sparse Expert Models"
