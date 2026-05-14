# OLMoE: Open Mixture-of-Experts Language Models

Reproduction of the OLMoE paper (Muennighoff et al., 2024).

## Structure

- `config.yaml` - All hyperparameters from Table 10 (Appendix B)
- `model/` - Neural network components
  - `layers.py` - RMSNorm, SwiGLU MLP, RoPE, Multi-Head Attention, QK-Norm
  - `moe.py` - MoE module with dropless token choice routing, load balancing loss, router z-loss
  - `olmoe_model.py` - Full OLMoE-1B-7B decoder-only transformer
- `training/`
  - `train.py` - Pretraining loop with AdamW, cosine LR, annealing
  - `losses.py` - Cross-entropy, load balancing loss, router z-loss, combined loss
  - `sft_train.py` - SFT instruction tuning
  - `dpo_train.py` - DPO preference tuning
- `data/`
  - `pretraining_data.py` - OLMoE-Mix dataset loading
  - `adaptation_data.py` - SFT/DPO dataset loading
- `analysis/`
  - `router_saturation.py` - Router saturation analysis
  - `co_activation.py` - Expert co-activation analysis
  - `specialization.py` - Domain and vocabulary specialization analysis
- `scripts/` - Entry points for training and evaluation

## Key Design Choices (Table 1)

| Setting | Value |
|---|---|
| Active params | 1.3B |
| Total params | 6.9B |
| Expert granularity | 64 experts, 8 activated |
| Shared expert | None |
| Routing | Dropless token choice |
| Sparse upcycling | Not used |
| Load balancing loss | weight 0.01 |
| Router z-loss | weight 0.001 |
