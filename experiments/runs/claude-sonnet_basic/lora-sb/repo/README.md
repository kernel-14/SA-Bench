# LoRA-SB: Initialization using Update Approximation for Efficient Low-Rank Fine-Tuning

Reproduction of the paper: **"Initialization Using Update Approximation is a Silver Bullet for Extremely Efficient Low-Rank Fine-Tuning"** (Ponkshe et al., 2024)

## Overview

LoRA-SB is a parameter-efficient fine-tuning method that achieves full fine-tuning performance while using 27-90x fewer trainable parameters than LoRA. It uses the LoRA-XS architecture (W = W₀ + B·R·A) with a carefully designed initialization strategy.

### Key Contributions Implemented

1. **LoRA-SB Architecture** (`lora_sb.py`): The core method using LoRA-XS architecture where only R (r×r) is trainable, while B (m×r) and A (r×n) are fixed orthonormal matrices.

2. **Initialization via Update Approximation** (`lora_sb.py`): 
   - Computes ΔW_avg = -sign(Σ ∇_W L(W₀, xᵢ)) to approximate the first AdamW step
   - Uses truncated SVD to get optimal rank-r approximation (Eckart-Young theorem)
   - Sets B_init = U[:, :r], A_init = V[:r, :], R_init = diag(S[:r])

3. **Scaling-Factor Independence** (Theorem 5): With orthonormal B and A (B^T·B = A·A^T = I), the scaling factor s=1 is optimal, eliminating hyperparameter tuning.

4. **Optimal Gradient Approximation** (Theorem 3): With orthonormal B and A, the optimal gradient simplifies to g^R = g^R_LoRA-XS, meaning standard gradient descent already achieves optimal approximation.

5. **Guaranteed Loss Reduction** (Theorem 4): Orthonormal B and A ensure ΔL ≤ 0 at each step.

6. **O(1) Memory Initialization**: Layerwise gradient computation with immediate discarding.

## Repository Structure

```
├── lora_sb.py              # Core LoRA-SB implementation
├── lora_xs.py              # LoRA-XS baseline
├── lora_baselines.py       # LoRA, rsLoRA, PiSSA, DoRA baselines
├── train_math.py           # Training on MetaMathQA (Mistral-7B, Gemma-2 9B)
├── train_commonsense.py    # Training on COMMONSENSE170K (Llama-3.2 3B)
├── train_glue.py           # Training on GLUE (RoBERTa-large)
├── evaluate_math.py        # Evaluation on GSM8K and MATH
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Arithmetic Reasoning (Table 1 in paper)

Fine-tune Mistral-7B on MetaMathQA:
```bash
# LoRA-SB (rank 96) - paper's best result: 63.38 GSM8K, 17.44 MATH
python train_math.py \
    --model mistralai/Mistral-7B-v0.1 \
    --method lora_sb \
    --rank 96 \
    --n_train_samples 50000 \
    --num_epochs 1 \
    --batch_size 1 \
    --grad_accum_steps 32 \
    --learning_rate 1e-4 \
    --max_seq_len 512

# LoRA-XS baseline (rank 96)
python train_math.py \
    --model mistralai/Mistral-7B-v0.1 \
    --method lora_xs \
    --rank 96

# LoRA baseline (rank 32)
python train_math.py \
    --model mistralai/Mistral-7B-v0.1 \
    --method lora \
    --rank 32
```

Fine-tune Gemma-2 9B:
```bash
python train_math.py \
    --model google/gemma-2-9b \
    --method lora_sb \
    --rank 64
```

### Commonsense Reasoning (Table 2 in paper)

Fine-tune Llama-3.2 3B on COMMONSENSE170K:
```bash
python train_commonsense.py \
    --model meta-llama/Llama-3.2-3B \
    --method lora_sb \
    --rank 96 \
    --num_epochs 2 \
    --batch_size 6 \
    --grad_accum_steps 24 \
    --learning_rate 2e-3
```

### Natural Language Understanding (Table 3 in paper)

Fine-tune RoBERTa-large on GLUE:
```bash
# LoRA-SB rank 24 - paper's best result: 87.16 avg
python train_glue.py \
    --model roberta-large \
    --task cola \
    --method lora_sb \
    --rank 24

# Run all GLUE tasks
for task in cola rte mrpc stsb qnli sst2; do
    python train_glue.py --task $task --method lora_sb --rank 24
done
```

### Evaluation

```bash
# Evaluate on GSM8K
python evaluate_math.py \
    --model_path ./outputs/math/Mistral-7B-v0.1_lora_sb_r96 \
    --base_model mistralai/Mistral-7B-v0.1 \
    --benchmark gsm8k

# Evaluate on MATH
python evaluate_math.py \
    --model_path ./outputs/math/Mistral-7B-v0.1_lora_sb_r96 \
    --base_model mistralai/Mistral-7B-v0.1 \
    --benchmark math
```

## Method Details

### LoRA-SB Algorithm

```python
# 1. Apply LoRA-SB architecture to model
model = apply_lora_sb(model, target_modules=["q_proj", "v_proj", ...], rank=r)

# 2. Compute gradient estimate (0.1% of training data)
delta_w_dict = compute_gradient_estimate(model, init_dataloader, n_samples=50)

# 3. Initialize B, R, A via truncated SVD
initialize_lora_sb_from_gradients(model, delta_w_dict)

# 4. Train (only R matrices are updated)
trainer.train()
```

### Initialization Details

For AdamW optimizer, the first update step is approximately:
```
θ₁ = θ₀ - α · g₁/√(g₁² + ε) ≈ -α · sign(g₁)
```

So we compute:
```
ΔW_avg = -sign(Σᵢ ∇_W L(W₀, xᵢ))
U, S, V^T = SVD(ΔW_avg)
B_init = U[:, :r]    # orthonormal columns
A_init = V[:r, :]    # orthonormal rows  
R_init = diag(S[:r]) # singular values (s=1)
```

### Why Orthonormal B and A Matter

With B^T·B = A·A^T = I:
- **Theorem 3**: Optimal g^R = (1/s²)(B^T B)^{-1} g^R_LoRA-XS (A A^T)^{-1} = g^R_LoRA-XS
- **Theorem 5**: Equivalent gradient is s-independent → set s=1
- **Theorem 4**: ΔL = -η‖g^R_LoRA-XS‖²_F + o(η) ≤ 0

## Experimental Results (from paper)

### Table 1: Arithmetic Reasoning

| Method | Rank | #Params (Mistral-7B) | GSM8K | MATH |
|--------|------|---------------------|-------|------|
| LoRA | 32 | 83.88M | 61.94 | 15.98 |
| LoRA-XS | 96 | 2.06M | 58.53 | 16.42 |
| **LoRA-SB** | **96** | **2.06M** | **63.38** | **17.44** |

### Table 3: GLUE (RoBERTa-large)

| Method | Rank | #Params | CoLA | RTE | MRPC | STS-B | QNLI | SST-2 | Avg |
|--------|------|---------|------|-----|------|-------|------|-------|-----|
| LoRA | 8 | 2162K | 68.02 | 82.98 | 90.05 | 91.43 | 93.42 | 95.98 | 86.98 |
| LoRA-XS | 24 | 55.2K | 66.27 | 80.14 | 88.48 | 90.77 | 93.21 | 95.89 | 85.79 |
| **LoRA-SB** | **24** | **55.2K** | **68.28** | **83.03** | **90.12** | **91.65** | **93.75** | **96.11** | **87.16** |

## Hyperparameters

### Arithmetic/Commonsense (Table 8)
| Setting | Mistral-7B / Gemma-2 9B | Llama-3.2 3B |
|---------|------------------------|--------------|
| Optimizer | AdamW | AdamW |
| Batch size | 1 | 6 |
| Max seq len | 512 | 256 |
| Grad accum | 32 | 24 |
| Epochs | 1 | 2 |
| Dropout | 0 | 0.05 |
| Learning rate | 1×10⁻⁴ | 2×10⁻³ |
| LR scheduler | Cosine | Linear |
| Warmup ratio | 0.02 | 0.02 |

### GLUE (Table 9)
| Setting | Value |
|---------|-------|
| Optimizer | AdamW |
| Batch size | 32 |
| Epochs | 30 (CoLA/RTE/MRPC/STS-B), 15 (QNLI/SST-2) |
| Learning rate | 1×10⁻³ |
| LR scheduler | Linear |
| Warmup ratio | 0.06 |

## Assumptions and Notes

1. **COMMONSENSE170K**: This dataset is from the LLM-Adapters paper (Hu et al., 2023). It needs to be downloaded separately from the original repository.

2. **Initialization samples**: The paper uses 0.1% of training data (50 samples for 50K MetaMathQA). We use `init_fraction=0.001` by default.

3. **LoRA-XS initialization**: The paper uses PiSSA-style initialization (SVD of pre-trained weights) for LoRA-XS. Our implementation follows this.

4. **Alpha for LoRA-XS**: The paper sets α=r for arithmetic/commonsense and α=16 for GLUE.

5. **Target modules**: For causal LMs, we target q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj. For RoBERTa, we target only self-attention layers (query, key, value, dense).

6. **Memory efficiency**: The gradient estimation uses layerwise computation with immediate discarding, achieving O(1) memory usage independent of the number of layers.

## Citation

```bibtex
@article{ponkshe2024lorasb,
  title={Initialization Using Update Approximation is a Silver Bullet for Extremely Efficient Low-Rank Fine-Tuning},
  author={Ponkshe, Kaustubh and Singhal, Raghav and Gorbunov, Eduard and Tumanov, Alexey and Horvath, Samuel and Vepakomma, Praneeth},
  journal={arXiv preprint},
  year={2024}
}
```
