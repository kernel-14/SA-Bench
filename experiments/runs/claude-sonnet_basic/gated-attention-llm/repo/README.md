# Gated Attention for Large Language Models: Reproduction

Reproduction of the paper: **"Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free"**

## Overview

This repository implements the core contributions of the paper, which systematically investigates gating mechanisms in standard softmax attention. The central finding is that applying a **head-specific sigmoid gate after the Scaled Dot-Product Attention (SDPA)** consistently improves performance, training stability, and eliminates attention sinks.

## Key Contributions Implemented

### 1. Gated Attention Variants (Table 1, Sec 2.2)

Five gating positions investigated:
- **G1 (SDPA output)**: Best performing — `gating_position="sdpa_output"`
- **G2 (Value projection)**: Second best — `gating_position="value"`
- **G3 (Key projection)**: `gating_position="key"`
- **G4 (Query projection)**: `gating_position="query"`
- **G5 (Dense output)**: `gating_position="dense_output"`

Two granularities:
- **Elementwise**: Per-dimension gating scores (n × q × d_k)
- **Headwise**: Per-head scalar gating (n × q)

Head-specific vs head-shared gating, multiplicative vs additive, sigmoid vs SiLU activation.

### 2. Non-linearity Analysis (Table 3, Sec 4.1)

The paper explains that W_V and W_O form a low-rank linear mapping. Adding non-linearity between them (via gating at G1 or G2) increases expressiveness. Implemented variants:
- Sigmoid gate (multiplicative)
- SiLU gate (multiplicative/additive)
- RMSNorm (non-linearity without parameters)
- Identity additive gate
- NS-sigmoid (non-sparse ablation): `0.5 + 0.5 * sigmoid(x)`

### 3. Sparsity Analysis (Sec 4.2)

The paper shows that effective gating scores are sparse (concentrated near 0). Key findings:
- SDPA output gating has the lowest mean scores (~0.15)
- Value gating has higher scores (~0.35)
- Head-shared gating reduces sparsity and hurts performance
- NS-sigmoid ablation confirms sparsity is crucial

### 4. Attention Sink Elimination (Sec 4.3)

Baseline models allocate 46.7% of attention to the first token on average. With G1 gating, this drops to 4.8%. The `AttentionSinkAnalyzer` class provides tools to measure this.

### 5. Context Length Extension (Table 5, Sec 4.4)

Gated models significantly outperform baseline when extending context with YaRN:
- At 128k context: SDPA-Gate achieves 58.82 vs baseline 31.65 on RULER
- `YaRNRoPE` class implements the context extension

### 6. Training Stability (Table 2, Sec 3.2.2)

Gating enables:
- Stable training at higher learning rates (8e-3 vs 4e-3)
- Larger batch sizes
- Deeper models (48 layers)
- Sandwich normalization comparison

## Repository Structure

```
submission/
├── models/
│   ├── gated_attention.py    # Core gated attention implementation
│   ├── transformer.py        # Dense transformer model (1.7B)
│   └── moe_transformer.py    # MoE transformer model (15A2B)
├── analysis/
│   ├── attention_sink.py     # Attention sink analysis tools
│   └── sparsity_analysis.py  # Gating score sparsity analysis
├── scripts/
│   ├── train.py              # Training script
│   └── context_extension.py  # YaRN context extension
├── configs/
│   ├── moe_experiments.yaml      # Table 1 experiment configs
│   ├── dense_experiments.yaml    # Table 2 experiment configs
│   └── nonlinearity_experiments.yaml  # Table 3 experiment configs
├── tests/
│   └── test_gated_attention.py   # Unit tests
└── requirements.txt
```

## Quick Start

### Install Dependencies

```bash
pip install torch numpy pyyaml
```

### Run Tests

```bash
python tests/test_gated_attention.py
```

### Create a Gated Attention Model

```python
from models.transformer import GatedTransformerModel, TransformerConfig

# Best-performing variant: SDPA elementwise sigmoid gating (G1)
config = TransformerConfig(
    d_model=2048,
    num_layers=28,
    num_heads=16,
    num_kv_heads=8,
    head_dim=128,
    ffn_intermediate_dim=11008,
    vocab_size=151936,
    gating_position="sdpa_output",    # G1 - best position
    gating_granularity="elementwise", # Per-dimension gating
    head_specific=True,               # Head-specific (not shared)
    gating_type="multiplicative",     # Y' = Y * sigmoid(X * W)
    gating_activation="sigmoid",      # Sigmoid activation
)
model = GatedTransformerModel(config)

# Baseline (no gating)
baseline_config = TransformerConfig(
    d_model=2048, num_layers=28, num_heads=16, num_kv_heads=8,
    head_dim=128, ffn_intermediate_dim=11008, vocab_size=151936,
    gating_position=None,  # No gating
)
baseline = GatedTransformerModel(baseline_config)
```

### Training

```bash
# Train with SDPA gating (best variant)
python scripts/train.py \
    --model_type dense \
    --gating_position sdpa_output \
    --gating_granularity elementwise \
    --head_specific \
    --gating_type multiplicative \
    --gating_activation sigmoid \
    --max_lr 4e-3 \
    --batch_size 1024

# Train baseline (no gating)
python scripts/train.py \
    --model_type dense \
    --gating_position none \
    --max_lr 4e-3

# MoE model with gating
python scripts/train.py \
    --model_type moe \
    --gating_position sdpa_output \
    --max_lr 2e-3 \
    --batch_size 1024
```

## Implementation Details

### Gating Mechanism

The gating mechanism is formalized as:
```
Y' = Y ⊙ σ(X * W_θ)  [multiplicative]
Y' = Y + σ(X * W_θ)  [additive]
```

where Y is the tensor to be modulated, X is the input for computing gating scores, W_θ are learnable parameters (initialized to zero), and σ is the activation function.

**Key design choices:**
- Zero initialization of gate projection ensures identity at initialization
- Head-specific gating: each head has its own gating scores
- Sigmoid activation produces sparse scores in [0, 1]

### Model Architecture

Both dense (1.7B) and MoE (15A2B) models follow the Qwen2.5 architecture:
- Pre-norm with RMSNorm
- SwiGLU FFN
- Group Query Attention (GQA)
- RoPE positional embeddings
- AdamW optimizer with cosine LR schedule

### MoE Architecture

The 15A2B MoE model uses:
- 128 total experts, top-8 routing
- Fine-grained expert segmentation (DeepSeekMoE style)
- Z-loss for router stability
- Load balancing loss

## Assumptions and Unresolved Details

1. **Exact model dimensions**: The paper mentions 15A2B MoE and 1.7B dense models but doesn't provide exact layer counts, hidden dimensions, etc. We approximate based on Qwen2.5 architecture.

2. **Training data**: The paper uses a proprietary 3.5T token dataset. This reproduction provides the training infrastructure but not the data.

3. **Gate initialization**: The paper doesn't specify initialization. We use zero initialization for the gate projection, which ensures the model starts as the baseline.

4. **Table 4 exact values**: The paper references Table 4 for gating score statistics but the table isn't fully reproduced in the markdown. We approximate based on the text descriptions.

5. **RULER benchmark**: The context extension experiments use RULER (Hsieh et al., 2024). The benchmark evaluation code is not included but the model infrastructure supports it.

## Results Summary

From the paper (Table 1, MoE 15A2B, 400B tokens):

| Method | PPL | MMLU | GSM8k | Hellaswag |
|--------|-----|------|-------|-----------|
| Baseline | 6.026 | 58.79 | 52.92 | 73.07 |
| SDPA Elementwise G1 | **5.761** | **60.82** | **55.27** | **74.64** |
| Value Elementwise G2 | 5.820 | 59.17 | 53.97 | 74.38 |
| SDPA Headwise G1 | 5.792 | 60.05 | 54.44 | 74.50 |

From Table 5 (Context Extension, RULER scores):

| Method | 4k | 32k | 64k | 128k |
|--------|-----|-----|-----|------|
| Baseline | 88.89 | 79.50 | - | - |
| SDPA-Gate | 90.56 | 79.77 | - | - |
| Baseline + YaRN | 82.90 | 37.94 | 37.51 | 31.65 |
| SDPA-Gate + YaRN | **88.13** | **72.88** | **66.60** | **58.82** |
