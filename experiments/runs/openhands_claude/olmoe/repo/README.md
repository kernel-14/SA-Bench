# OLMoE: Open Mixture-of-Experts Language Models

Reproduction of [OLMoE: Open Mixture-of-Experts Language Models](https://arxiv.org/abs/2409.02060) (Muennighoff et al., 2024).

## Architecture

OLMoE-1B-7B is a decoder-only transformer with sparse Mixture-of-Experts layers:
- 16 transformer layers, d_model=2048, 16 attention heads
- Every layer uses MoE with 64 experts, 8 activated per token (top-k token choice)
- Expert FFN dim=1024 with SwiGLU activation
- RMSNorm (parametric, weight-decayed), QK-Norm, RoPE (θ=10000)
- Truncated normal initialization (std=0.02, truncated at ±0.06)
- No weight tying, no biases
- 6.9B total parameters, 1.3B active per token

## Training

Pretrained on 5.133T tokens (1.3 epochs of OLMoE-Mix):
- DCLM-Baseline (3,860B), StarCoder (101B), peS2o (57.2B), arXiv (21.1B),
  OpenWebMath (12.7B), Algebraic Stack (12.6B), Wikipedia/Wikibooks (3.69B)

Loss: `L = L_CE + 0.01 * L_LB + 0.001 * L_RZ`
- `L_CE`: cross-entropy language modeling loss
- `L_LB`: load balancing loss (Shazeer et al. 2017)
- `L_RZ`: router Z-loss (Zoph et al. 2022)

Optimizer: AdamW (β1=0.9, β2=0.95, ε=1e-8, weight_decay=0.1)
LR schedule: cosine warmup (2500 steps) to 4e-4, then cosine decay to 4e-5, then linear annealing to 0 over final 100B tokens

## Repository Structure

```
repo/
├── config.py       # All hyperparameters (ModelConfig, TrainConfig, AdaptConfig)
├── layers.py       # RMSNorm, RoPE, Attention with QK-Norm, SwiGLU
├── modules.py      # MoE router, ExpertFFN, MoEModule, TransformerBlock
│                   # load_balance_loss, router_z_loss
├── model.py        # Full OLMoE decoder-only LM, build_olmoe_1b_7b()
├── data.py         # Pretraining dataset (OLMoE-Mix), SFT/DPO/KTO datasets
│                   # n-gram repetition filter, StarCoder quality filters
├── train.py        # Pretraining loop with cosine+annealing LR schedule
├── adapt.py        # SFT, DPO (Rafailov et al. 2023), KTO (Ethayarajh et al. 2024)
├── analysis.py     # Router saturation, expert co-activation,
│                   # domain specialization, vocabulary specialization
├── evaluate.py     # Perplexity, multiple-choice (CF/MCF), MMLU, GSM8k,
│                   # HumanEval Pass@k
└── requirements.txt
```

## Key Design Choices (Table 1)

| Choice | OLMoE-1B-7B |
|--------|-------------|
| Expert granularity | 64 small experts (FFN dim=1024), 8 activated |
| Expert sharing | No shared expert |
| Routing algorithm | Dropless token choice (MegaBlocks) |
| Sparse upcycling | Not used (train from scratch) |
| Load balancing loss | Used, weight=0.01 |
| Router Z-loss | Used, weight=0.001 |

## Adaptation

OLMOE-1B-7B-INSTRUCT is created via:
1. **SFT**: 2 epochs, LR=2e-5, batch=128, no load balancing loss
   - Tulu 2 SFT Mix, No Robots, CodeFeedback, MetaMathQA, Daring Anteater
2. **DPO**: 3 epochs, LR=5e-7, batch=32, β=0.1, no load balancing loss
   - UltraFeedback binarized (filtered for TruthfulQA contamination)

## MoE Analysis (§5)

Four analysis metrics implemented in `analysis.py`:
- **Router Saturation**: ~60% of top-8 routing saturates after just 1% of pretraining
- **Expert Co-activation**: Low co-activation suggests little expert redundancy
- **Domain Specialization**: Experts specialize in arXiv, GitHub, books, etc.
- **Vocabulary Specialization**: Later layers specialize more on predicted output tokens

## Usage

```python
from model import build_olmoe_1b_7b
import torch

model = build_olmoe_1b_7b()
input_ids = torch.randint(0, 50304, (1, 128))
out = model(input_ids=input_ids, labels=input_ids)
print(f"Loss: {out['loss'].item():.4f}")
print(f"CE: {out['ce_loss'].item():.4f}")
print(f"LB: {out['load_balance_loss'].item():.4f}")
print(f"RZ: {out['router_z_loss'].item():.6f}")
```

```bash
# Pretraining
python train.py --data_path /path/to/olmoe_mix --save_dir checkpoints/

# Adaptation
python adapt.py --base_checkpoint checkpoints/final/checkpoint.pt \
                --sft_data_path /path/to/sft_mix \
                --dpo_data_path /path/to/dpo_mix \
                --mode dpo
```

## Citation

```bibtex
@article{muennighoff2024olmoe,
  title={OLMoE: Open Mixture-of-Experts Language Models},
  author={Muennighoff, Niklas and Soldaini, Luca and Groeneveld, Dirk and Lo, Kyle and
          Morrison, Jacob and Min, Sewon and Shi, Weijia and Walsh, Pete and Tafjord, Oyvind and
          Lambert, Nathan and Gu, Yuling and Arora, Shane and Bhagia, Akshita and Schwenk, Dustin and
          Wadden, David and Wettig, Alexander and Hui, Binyuan and Dettmers, Tim and Kiela, Douwe and
          Farhadi, Ali and Smith, Noah A. and Koh, Pang Wei and Singh, Amanpreet and Hajishirzi, Hannaneh},
  journal={arXiv preprint arXiv:2409.02060},
  year={2024}
}
```
