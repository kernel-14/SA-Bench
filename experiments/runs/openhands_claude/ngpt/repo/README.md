# nGPT: Normalized Transformer with Representation Learning on the Hypersphere

Reproduction of [Loshchilov et al. (2024)](https://arxiv.org/abs/2410.01131).

## Repository Structure

```
repo/
├── config.py       — ModelConfig and TrainConfig dataclasses; all hyperparameters
├── layers.py       — RoPE, l2_norm, normalize_matrix_rows, RMSNorm
├── modules.py      — Attention and MLP blocks for GPT and nGPT; ScaledParameter
├── model.py        — Full GPT and nGPT models; normalize_weights()
├── data.py         — OpenWebText tokenization and DataLoader construction
├── train.py        — Training loop (single-GPU and DDP)
├── evaluate.py     — Downstream task evaluation and analysis utilities
├── requirements.txt
└── README.md
```

## Key Architectural Differences: nGPT vs GPT

| Component | GPT | nGPT |
|---|---|---|
| Normalization | RMSNorm (pre-norm) | None (all vectors unit-norm) |
| Hidden state update | h ← h + block(h) | h ← Norm(h + α(h_block − h)) |
| Weight matrices | Unconstrained | Normalized after every step |
| Softmax scale | 1/√d_k | √d_k |
| QK normalization | No | Yes, with learnable s_qk |
| MLP scaling | No | s_u, s_v·√d_model |
| Logit scaling | No | Learnable s_z per token |
| Weight decay | 0.1 | 0.0 |
| LR warmup | 2000 steps | None |

## Model Sizes (Table 2)

| Size | Layers | d_model | Heads | Parameters |
|---|---|---|---|---|
| 0.5B | 24 | 1024 | 16 | ~468M |
| 1B | 36 | 1280 | 20 | ~1026M |

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare data

```bash
# Tokenize OpenWebText with the LLaMA-2 tokenizer
python data.py --tokenize \
    --data_dir openwebtext \
    --output_dir data/openwebtext \
    --tokenizer meta-llama/Llama-2-7b-hf
```

### 3. Train nGPT (single GPU)

```bash
python train.py \
    --model_type ngpt \
    --model_size 0.5B \
    --seq_len 4096 \
    --lr 2e-3 \
    --max_steps 100000 \
    --dataset_path data/openwebtext
```

### 4. Train baseline GPT (single GPU)

```bash
python train.py \
    --model_type gpt \
    --model_size 0.5B \
    --seq_len 4096 \
    --lr 2e-3 \
    --max_steps 100000 \
    --dataset_path data/openwebtext
```

### 5. Multi-GPU training (8 GPUs, as in paper)

```bash
torchrun --nproc_per_node=8 train.py \
    --model_type ngpt \
    --model_size 1B \
    --seq_len 4096 \
    --global_batch_size 512 \
    --lr 1e-3 \
    --max_steps 200000
```

### 6. Evaluate

```bash
python evaluate.py \
    --checkpoint checkpoints/ckpt_0100000.pt \
    --tasks hellaswag piqa winogrande arc_easy arc_challenge \
    --data_dir data/eval \
    --val_bin data/openwebtext/val.bin
```

## nGPT Implementation Notes

### Scaling parameter trick (Section 2.5)

All learnable scaling parameters (α_A, α_M, s_qk, s_u, s_v, s_z) use a
two-scalar representation: the parameter is stored at `s_scale` and the
actual value is recovered during the forward pass as `param * (s_init / s_scale)`.
This controls the Adam effective learning rate independently of the global LR.

| Parameter | s_init | s_scale | Effective LR |
|---|---|---|---|
| α_A, α_M | 0.05 | 1/√d_model | ~1/√d_model |
| s_qk | 1.0 | 1/√d_model | ~1/√d_model |
| s_u, s_v | 1.0 | 1.0 | ~1.0 |
| s_z | 1.0 | 1/√d_model | ~1/√d_model |

### Weight normalization

After every optimizer step, `model.normalize_weights()` projects all weight
matrix rows back onto the unit hypersphere. This is the critical step that
keeps all vectors on the hypersphere and must not be skipped.

### Eigen learning rates

α_A and α_M are per-dimension vectors (shape: d_model) applied as |α| during
the forward pass to ensure positive contributions. They control how much each
layer's attention/MLP output displaces the hidden state h.

## Paper Results

nGPT achieves the same validation loss as GPT with:
- ~4× fewer tokens at 1k context length
- ~10× fewer tokens at 4k context length  
- ~20× fewer tokens at 8k context length

## Citation

```bibtex
@article{loshchilov2024ngpt,
  title={nGPT: Normalized Transformer with Representation Learning on the Hypersphere},
  author={Loshchilov, Ilya and Hsieh, Cheng-Ping and Sun, Simeng and Ginsburg, Boris},
  journal={arXiv preprint arXiv:2410.01131},
  year={2024}
}
```
