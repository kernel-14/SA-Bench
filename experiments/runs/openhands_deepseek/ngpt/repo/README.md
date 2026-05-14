# nGPT: Normalized Transformer with Representation Learning on the Hypersphere

Reproduction of Loshchilov et al. (2025) "nGPT: Normalized Transformer with Representation Learning on the Hypersphere".

## Codebase Structure

```
repo/
├── config.py       # All hyperparameters (Tables 2, 3; Section 2.6; Appendix A.6)
├── modules.py      # Core components: ScaledParameter, NormalizedLinear, Attention, MLP, blocks
├── model.py        # Full GPT and nGPT model implementations
├── data.py         # OpenWebText dataset loading with LLaMA-2 tokenizer
├── train.py        # Training loop with Adam/AdamW, cosine schedule, evaluation
├── requirements.txt
└── README.md
```

## Key Implementation Details

### nGPT Architecture
- **All vectors normalized**: embeddings, attention matrices, MLP matrices lie on unit hypersphere
- **No LayerNorm/RMSNorm**: normalization layers removed entirely
- **Eigen learning rates** (α_A, α_M): per-dimension learnable parameters controlling hidden state updates
- **QK normalization**: optional normalization of query/key with learnable scaling s_qk
- **MLP rescaling**: v scaled by √d_model for SiLU non-linearity (Appendix A.1)
- **Logit scaling**: s_z scales logits element-wise to control softmax temperature
- **LERP/SLERP**: hidden state updated via interpolation on hypersphere
- **Post-step normalization**: all weight matrices normalized after each optimizer step

### GPT Baseline
- Standard decoder-only Transformer with RMSNorm
- Learned positional embeddings
- Tied input/output embeddings
- Weight decay + warmup

## Usage

```bash
# Train 0.5B nGPT with 4k context
python train.py --preset 0.5B --use-ngpt --seq-len 4096

# Train 1B GPT baseline with 1k context
python train.py --preset 1.0B --no-ngpt --seq-len 1024

# Train 1B nGPT with 8k context
python train.py --preset 1.0B --use-ngpt --seq-len 8192
```

## Model Presets (Table 2)

| Preset | Layers | d_model | n_heads | d_mlp | Parameters |
|--------|--------|---------|---------|-------|------------|
| 0.5B   | 24     | 1024    | 16      | 4096  | ~468M      |
| 1.0B   | 36     | 1280    | 20      | 5120  | ~1026M     |

## Training Recipe (Section 2.6)

1. Remove all normalization layers (RMSNorm/LayerNorm)
2. Normalize all matrix parameters along embedding dimension after each step
3. Use LERP/SLERP update with eigen learning rates (α_A_init = 0.05, scale = 1/√d_model)
4. Attention: √d_k softmax scaling, QK normalization + scaling (s_qk_init = 1, scale = 1/√d_model)
5. MLP: s_u and s_v scaling (init = 1, scale = 1) with √d_model rescaling of v
6. Logits: s_z scaling (init = 1, scale = 1/√d_model)
7. Remove weight decay and learning rate warmup

## References

- Loshchilov et al. (2025) "nGPT: Normalized Transformer with Representation Learning on the Hypersphere"
- Official implementation: https://github.com/NVIDIA/ngpt
