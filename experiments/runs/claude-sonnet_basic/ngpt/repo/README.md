# nGPT: Normalized Transformer with Representation Learning on the Hypersphere

Reproduction of **"nGPT: Normalized Transformer with Representation Learning on the Hypersphere"** by Loshchilov, Hsieh, Sun & Ginsburg (NVIDIA, 2024).

---

## What was reproduced

### Core architecture (`model.py`)

The full nGPT model and the baseline GPT model are implemented side-by-side.

**nGPT key changes vs. standard GPT (Section 2, Table 1):**

| Component | Standard GPT | nGPT |
|-----------|-------------|------|
| Pre-block norm | RMSNorm(h) | — (removed) |
| Residual update | h ← h + block(h) | h ← Norm(h + α⊙(Norm(block(h)) − h)) |
| Post-block norm | — | Norm(·) (retraction to hypersphere) |
| Final norm | RMSNorm(h) | — (not needed) |
| Weight matrices | Unconstrained | Normalised along embedding dim after each step |
| Attention scale | 1/√d_k | √d_k |
| Q, K | Unconstrained | Norm(·) × s_qk (optional) |
| MLP intermediate | u, v | u × s_u, v × s_v × √d_model |
| Logits | E_out · h | (E_out · h) × s_z |

**Learnable scaling parameters (Section 2.5 init scheme):**

Each parameter `s` is stored at `s_scale` and the forward pass uses `s_param × (s_init / s_scale)`. This decouples the effective Adam learning rate from the parameter value.

| Parameter | init value | scale | Meaning |
|-----------|-----------|-------|---------|
| α_A, α_M | 0.05 | 1/√d_model | Eigen learning rates (per d_model dim) |
| s_qk | 1.0 | 1/√d_model | QK scaling (per d_head dim) |
| s_u, s_v | 1.0 | 1.0 | MLP gate scaling (per d_mlp dim) |
| s_z | 1.0 | 1/√d_model | Logit scaling (per vocab token) |

### Training (`train.py`)

Implements the full training loop with:
- **nGPT**: Adam (no weight decay), no warmup, cosine annealing to 0
- **GPT**: AdamW (wd=0.1), 2000-step warmup, cosine annealing to 0
- `model.normalize_weights()` called after every `optimizer.step()` for nGPT
- Distributed training via PyTorch DDP
- bfloat16 mixed precision

### Data preparation (`prepare_data.py`)

Downloads and tokenises OpenWebText using the LLaMA-2 tokeniser (32k vocab) or GPT-2 tokeniser as fallback. Saves memory-mapped `.bin` files.

### Evaluation (`evaluate.py`)

Log-likelihood scoring for multiple-choice downstream tasks:
- HellaSwag, PIQA, WinoGrande, ARC-Easy, ARC-Challenge
- Perplexity computation on held-out data

### Analysis (`analysis.py`)

Reproduces Figures 4, 5, 6 from the paper:
- **Figure 4**: Embedding norm distributions, covariance eigenvalues, pairwise dot products
- **Figure 5**: Median condition numbers of attention/MLP matrices per layer
- **Figure 6**: Learned values of α_A, α_M, s_qk, s_u, s_v, s_z after training

### Tests (`test_model.py`)

Verifies all key invariants:
- All weight matrices are unit-norm after init and after `normalize_weights()`
- Hidden states remain on the unit hypersphere through every layer
- Scaling parameters initialise to their intended values
- nGPT uses √d_k attention scale; GPT uses 1/√d_k
- Forward pass produces correct shapes and finite loss
- Parameter counts match Table 2 in the paper

---

## Model sizes (Table 2, Appendix A.6)

| Model | Layers | d_model | Heads | Params (paper) |
|-------|--------|---------|-------|----------------|
| 0.5B nGPT | 24 | 1024 | 16 | 468.4M |
| 0.5B GPT  | 24 | 1024 | 16 | 468.2M |
| 1B nGPT   | 36 | 1280 | 20 | 1026.1M |
| 1B GPT    | 36 | 1280 | 20 | 1025.7M |

---

## Quick start

```bash
# 1. Install dependencies
pip install torch numpy tqdm datasets sentencepiece tiktoken matplotlib

# 2. Prepare data (requires LLaMA-2 tokenizer or falls back to GPT-2)
python prepare_data.py --output_dir data/ --tokenizer_path /path/to/tokenizer.model

# 3. Train nGPT (0.5B, 4k context)
python train.py \
    --model_type ngpt \
    --n_layers 24 --d_model 1024 --n_heads 16 \
    --seq_len 4096 --batch_size 8 \
    --max_steps 200000 --lr 1e-3 \
    --train_data data/train.bin --val_data data/val.bin \
    --out_dir out_ngpt_0.5B_4k/

# 4. Train GPT baseline (same size)
python train.py \
    --model_type gpt \
    --n_layers 24 --d_model 1024 --n_heads 16 \
    --seq_len 4096 --batch_size 8 \
    --max_steps 200000 --lr 1e-3 \
    --train_data data/train.bin --val_data data/val.bin \
    --out_dir out_gpt_0.5B_4k/

# 5. Multi-GPU training (64 GPUs, 8 nodes as in the paper)
torchrun --nproc_per_node=8 --nnodes=8 train.py \
    --model_type ngpt --n_layers 36 --d_model 1280 --n_heads 20 \
    --seq_len 4096 --batch_size 8 \
    --max_steps 200000 --lr 1e-3 \
    --out_dir out_ngpt_1B_4k/

# 6. Run tests
python test_model.py

# 7. Analyse trained models
python analysis.py \
    --ngpt_ckpt out_ngpt_0.5B_4k/best_model.pt \
    --gpt_ckpt  out_gpt_0.5B_4k/best_model.pt \
    --log_ngpt  out_ngpt_0.5B_4k/training_log.json \
    --log_gpt   out_gpt_0.5B_4k/training_log.json \
    --out_dir   figures/
```

---

## Key implementation notes

### Weight normalisation dimension

The paper says to normalise matrices "along their embedding dimension". For `nn.Linear(in, out)` the weight tensor has shape `(out, in)`. The embedding dimension is the *input* dimension, so we normalise along `dim=1` (each row becomes a unit vector in R^{in_features}). For embedding matrices `(vocab, d_model)` we normalise along `dim=-1`.

### The normalise-after-step requirement

The paper warns (Appendix A.6): *"it is very important to note that when implementing nGPT in training libraries, one should make sure that not only instantiated model parameters are normalized but also the ones which are used by the optimizer. Missing the latter is a common bug."*

In our implementation `normalize_weights()` modifies `weight.data` in-place with `torch.no_grad()`, which updates the tensor that the optimizer's momentum buffers reference. This is the correct approach.

### Attention scaling

Standard attention uses `1/√d_k` to prevent dot products from growing too large. In nGPT, q and k are unit-normalised before the dot product, so their expected variance is `1/d_k`. To restore variance 1, the scale should be `√d_k` (Section 2.3.2).

### SiLU non-linearity (Appendix A.1)

The v projection in the MLP produces values in `[-1, 1]` (cosine similarities). For very small inputs, `SiLU(x) ≈ x/2`, losing non-linearity. Scaling v by `√d_model` restores the expected variance to ~1, making SiLU behave like ReLU for large values.

---

## Assumptions and unresolved details

1. **Tokeniser**: The paper uses the LLaMA-2 tokeniser (32k vocab). We fall back to GPT-2 (50k vocab) if the LLaMA-2 tokeniser is unavailable.

2. **s_qk sharing**: The paper says s_qk ∈ R^{d_k} is "a vector of trainable scaling factors for the i-th head". We implement a single shared vector per layer (not per-head), which is the most natural reading and matches the nanoGPT reference implementation.

3. **Global batch size**: The paper uses a global batch of 512 sequences across 64 GPUs. With 8 GPUs per node and 8 nodes, this is 8 sequences per GPU. Our `--batch_size` argument is per-GPU.

4. **Downstream task evaluation**: The paper evaluates on HellaSwag, PIQA, WinoGrande, ARC-Easy, ARC-Challenge, and WMT14-FR-EN. Our `evaluate.py` implements the first five; WMT14 BLEU requires a separate translation pipeline.

5. **Internal Megatron-LM library**: The paper's experiments used an internal Megatron-LM-based library. Our implementation uses standard PyTorch DDP, which should produce equivalent results but may differ in throughput.
