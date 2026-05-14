# Reproducing "Lessons and Insights from a Unifying Study of Parameter-Efficient Fine-Tuning (PEFT) in Visual Recognition"

## 1. Project Overview

This project aims to systematically reproduce the empirical study of Parameter-Efficient Fine-Tuning (PEFT) methods in visual recognition, as described in the paper "Lessons and Insights from a Unifying Study of Parameter-Efficient Fine-Tuning (PEFT) in Visual Recognition". The reproduction efforts will cover:

*   **Comparison of diverse PEFT methods:** Including Prompt-based (VPT-Shallow, VPT-Deep), Adapter-based (Houlsby, Pfeiffer, AdaptFormer, ConvPass, RepAdapter), Direct Selective Tuning (BitFit, LayerNorm, DiffFit, SSF), and Efficient Selective Tuning (LoRA, FacT_TT, FacT_TK).
*   **Baselines:** Linear Probing and Full Fine-Tuning.
*   **Experimental Regimes:** Low-shot learning on VTAB-1K, many-shot learning on full-size datasets (CIFAR-100, RESISC, Clevr-Distance), and robustness evaluation to distribution shifts using CLIP and ImageNet variants (ImageNet-V2, -R, -S, -A).
*   **Key Analyses:** Accuracy comparison, prediction diversity (overlap), ensemble performance, method ranking frequency, and Weight-space ensembles (WiSE) for robustness.

## 2. Project Structure

The project is organized into a modular structure to facilitate clarity, maintainability, and reproducibility.

