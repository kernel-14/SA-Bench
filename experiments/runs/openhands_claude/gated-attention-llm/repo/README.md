# Gated Attention for Large Language Models

Reproduction of the paper:
> **Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free**
> Zihan Qiu, Zekun Wang, Bo Zheng, Zeyu Huang, et al. (Qwen Team, Alibaba Group)

## Overview

The paper systematically investigates gating mechanisms in standard softmax attention. The central finding is that applying a **head-specific sigmoid gate after the Scaled Dot-Product Attention (SDPA) output** (position G1) consistently improves performance, enhances training stability, and eliminates the attention sink phenomenon.

## Repository Structure

```
repo/
├── config.py       # All hyperparameters, model configs, and gating variant presets
├── modules.py      # Core gating module (Eq. 5), RoPE, GatedMultiHeadAttention
├── layers.py       # SwiGLU FFN, MoE layer (DeepSeekMoE-style), TransformerBlock
├── model.py        # GatedTransformerLM (dense + MoE), factory functions
├── data.py         # Dataset loading (binary shards, HuggingFace), DataLoader factory
├── train.py        # Training loop with cosine LR schedule, AdamW, AMP
├── evaluate.py     # Perplexity, few-shot benchmarks (lm-eval), RULER
├── analysis.py     # Gating score stats, attention sink, massive activation analysis
├── requirements.txt
└── README.md
```

## Key Implementations

### Gating Mechanism (Eq. 5)

```
Y' = Y ⊙ σ(X W_θ)
```

Five positions investigated (Fig. 1):
- **G1** – after SDPA output ← best performing
- **G2** – after value projection
- **G3** – after key projection
- **G4** – after query projection
- **G5** – after final dense output layer

All variants from Table 1 are implemented in `config.GATING_VARIANTS`:

| Key | Description |
|-----|-------------|
| `baseline` | No gating |
| `G1_elementwise` | SDPA elementwise sigmoid (best) |
| `G1_headwise` | SDPA headwise sigmoid |
| `G1_head_shared` | SDPA head-shared elementwise |
| `G1_additive_silu` | SDPA additive SiLU |
| `G1_elementwise_silu` | SDPA elementwise SiLU |
| `G1_rmsnorm` | SDPA RMSNorm (Table 3 row 5) |
| `G1_ns_sigmoid` | NS-sigmoid (non-sparse, Eq. 9) |
| `G2_elementwise` | Value elementwise sigmoid |
| `G2_headwise` | Value headwise sigmoid |
| `G2_head_shared` | Value head-shared |
| `G3_elementwise` | Key elementwise sigmoid |
| `G4_elementwise` | Query elementwise sigmoid |
| `G5_elementwise` | Dense output sigmoid |

### Model Architectures

**Dense 1.7B** (Sec. 3.2.2):
- 28-layer or 48-layer variants
- GQA, RoPE, SwiGLU FFN, RMSNorm
- Optional sandwich norm (Ding et al., 2021)

**MoE 15A2B** (Sec. 3.2.1):
- 15B total / 2.54B activated parameters
- 128 experts, top-8 softmax routing (DeepSeekMoE fine-grained)
- Z-loss + global-batch load-balancing loss
- GQA: q=32, k=4, d_k=128

### Non-Sparse Sigmoid (Eq. 9)

```python
NS-sigmoid(x) = 0.5 + 0.5 * sigmoid(x)  # constrains scores to [0.5, 1.0]
```

Used in ablation to study the effect of removing sparsity while keeping non-linearity.

## Training

```bash
# MoE 15A2B with best gating variant (Table 1 row 5)
python train.py \
    --variant G1_elementwise \
    --model moe_15a2b \
    --train_data_dir /data/train \
    --eval_data_dir /data/eval

# Dense 1.7B baseline (28 layers, 400B tokens)
python train.py \
    --variant baseline \
    --model dense_1_7b_28l \
    --train_data_dir /data/train

# Dense 1.7B with gating, higher LR (Table 2 row 10)
python train.py \
    --variant G1_elementwise \
    --model dense_1_7b_48l \
    --max_lr 8e-3 \
    --train_data_dir /data/train

# Dense 1.7B with sandwich norm (Table 2 row 7)
python train.py \
    --variant baseline \
    --model dense_1_7b_48l \
    --sandwich_norm \
    --max_lr 8e-3 \
    --train_data_dir /data/train
```

### Training Hyperparameters

| Setting | MoE 15A2B | Dense 1.7B (400B) | Dense 1.7B (3.5T) | Dense 1.7B 48L (1T) |
|---------|-----------|-------------------|-------------------|---------------------|
| Max LR | 2e-3 | 4e-3 | 4.5e-3 | 5.3e-3 |
| Min LR | 3e-5 | 4e-5 | 4.5e-5 | 5.3e-5 |
| Warmup | 1k steps | 1k steps | 2k steps | 2k steps |
| Batch size | 1024 | 1024 | 2048 | 4096 |
| Total steps | 100k | ~98k | ~854k | ~61k |
| Seq len | 4096 | 4096 | 4096 | 4096 |

## Evaluation

```bash
# Perplexity + few-shot benchmarks
python evaluate.py \
    --checkpoint checkpoints/moe_15a2b_G1_elementwise/step_0100000.pt \
    --eval_data_dir /data/eval \
    --benchmarks hellaswag mmlu gsm8k ceval-valid cmmlu \
    --num_fewshot 5

# RULER long-context evaluation (Sec. 4.4)
python evaluate.py \
    --checkpoint checkpoints/dense_1_7b_G1_elementwise/step_final.pt \
    --ruler \
    --ruler_lengths 4096 8192 16384 32768 65536 131072
```

## Analysis

```bash
# Gating score statistics + attention sink + massive activations (Sec. 4.2-4.3)
python analysis.py \
    --checkpoint checkpoints/moe_15a2b_G1_elementwise/step_0100000.pt \
    --baseline_checkpoint checkpoints/moe_15a2b_baseline/step_0100000.pt \
    --eval_data_dir /data/eval \
    --output_dir analysis_results \
    --analyses gating attention_sink massive_act sparsity
```

## Data Format

Training data should be pre-tokenised binary shards (uint16 token arrays):
```
/data/train/
    train_00000.bin
    train_00001.bin
    ...
/data/eval/
    english.bin
    chinese.bin
    code.bin
    math.bin
    law.bin
    literature.bin
    val.bin
```

Each `.bin` file is a flat array of uint16 token IDs. Sequences are packed
end-to-end; the dataloader slices them into `seq_len + 1` chunks.

Alternatively, use the HuggingFace adapter in `data.py`:
```python
from data import HFTextDataset, load_tokenizer
tokenizer = load_tokenizer("Qwen/Qwen2-7B")
dataset = HFTextDataset("allenai/c4", tokenizer, seq_len=4096)
```

## Dependencies

```
pip install -r requirements.txt
```

Key dependencies: PyTorch ≥ 2.1, Transformers ≥ 4.40, lm-eval ≥ 0.4.2, DeepSpeed ≥ 0.14.
