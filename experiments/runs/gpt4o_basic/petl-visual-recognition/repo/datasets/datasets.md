# Dataset Overview

This section summarizes the datasets utilized in the paper for empirical evaluations.

## VTAB-1K
VTAB-1K consists of 19 tasks organized into three groups:
1. **Natural:** Example - Flower classification.
2. **Specialized:** Example - Remote sensing images.
3. **Structured:** Example - Depth classifications.

Setup: Training uses 1000 images, with hyperparameter tuning on an 80/20 split.

## Robustness Evaluation
Diverse datasets introduced via CLIP for domain shifts.
- Variants such as ImageNet-V2, R, A, and S are analyzed with weight-space ensembles.

Full dataset configurations will be provided in implementation.
