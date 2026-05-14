# Ca2-VDM: Efficient Autoregressive Video Diffusion Model with Causal Generation and Cache Sharing

Reproduction of the paper "Ca2-VDM: Efficient Autoregressive Video Diffusion Model with Causal Generation and Cache Sharing" (Gao et al.).

## Codebase Structure

- `config.py` — All hyperparameters and configuration
- `layers.py` — Core attention layers (causal temporal attention, prefix-enhanced spatial attention, cross attention)
- `modules.py` — Transformer blocks, timestep embeddings, positional embeddings (including Cyclic-TPEs)
- `model.py` — Full Ca2-VDM model with noise prediction and VLB loss
- `data.py` — Dataset loading, partial noising, and preprocessing
- `train.py` — Training loop with combined loss
- `inference.py` — Autoregressive inference with KV-cache sharing and queue
- `requirements.txt` — Python dependencies
