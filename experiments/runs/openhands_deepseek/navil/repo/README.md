# NaViL: Rethinking Scaling Properties of Native Multimodal Large Language Models under Data Constraints

Reproduction of the NaViL paper. This codebase implements a native MLLM trained end-to-end
with modality-specific Mixture-of-Experts, a bidirectional visual encoder, and joint
vision-language scaling.

## Code Structure

- `layers.py` — Primitive layers: RMSNorm, 1D/2D RoPE, MultiHeadAttention, SwiGLU FFN, PatchEmbedding
- `modules.py` — Compound modules: VisualEncoder, ModalityMoE Attention/FFN, MoEDecoderLayer, Connector
- `model.py` — NaViL model combining visual encoder, connector, and MoE-extended LLM
- `config.py` — Model and training hyperparameter configurations
- `data.py` — Dataset loading, image preprocessing, multimodal sequence packing
- `train.py` — Three-stage training loop (pretrain, high-quality alignment, SFT)

## Key Features

- End-to-end native MLLM training
- Modality-specific MoE (attention + FFN experts)
- Visual multi-scale packing for any-resolution inference
- Scaling law exploration between visual encoder and LLM
