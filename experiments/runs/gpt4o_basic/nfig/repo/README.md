# NFIG: Multi-Scale Autoregressive Image Generation

## Repository Overview
This repository replicates the core contributions of the paper "NFIG: Multi-Scale Autoregressive Image Generation via Frequency Ordering." The primary implementation includes:

1. **Models**:
   - Frequency-guided Residual-quantized VAE (FR-VAE): focuses on frequency decomposition, tokenization, and residual quantization.
   - Autoregressive Image Generator: a transformer-based model employing block-wise causal attention for progressive image generation.

2. **Datasets**:
   - ImageNet loader tailored for the subset described in the paper.

3. **Evaluation**:
   - Metrics such as Fréchet Inception Distance (FID) and Inception Score (IS) are included for assessing generated image quality and diversity.

---

## Assumptions
- The implementation assumes PyTorch as the deep learning framework.
- Frequency masks and quantization strategies are placeholders and require specific details from the paper.
- Dataset structure aligns with ImageNet preprocessed image folders.

## Next Steps
- Refinement and validation of FR-VAE and quantization strategies.
- Integration of detailed experimental setups and comparisons.

---

## Execution
Ensure that ImageNet data is available in the correct structure. Instructions for training and evaluation remain under progress.
