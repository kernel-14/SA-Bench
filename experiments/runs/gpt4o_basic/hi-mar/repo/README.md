# Hi-MAR: Hierarchical Masked Autoregressive Models with Low-Resolution Token Pivots

## Summary
This repository reproduces the core contributions of the paper 'Hi-MAR: Hierarchical Masked Autoregressive Models with Low-Resolution Token Pivots.' The focus is on implementing Hi-MAR's hierarchical visual token prediction approach, including:

1. Hierarchical Masked Autoregressive Transformer.
2. Scale-Aware Transformer Blocks.
3. Diffusion Transformer Heads.

Please refer to the paper for additional theoretical insights.

## Directory Structure
- src/: Contains Python source code for model, training, and evaluation pipelines.
- models/: Definitions of Hi-MAR variants (Base, Large, Huge).
- config/: Configurations for hyperparameters and dataset settings.
- data/: Placeholder for preprocessing utilities and datasets.

## Tasks Addressed
- Class-conditional image generation on ImageNet.
- Text-to-image generation on MS-COCO.

## Next Steps
1. Implement hierarchical visual tokenization.
2. Define architecture for Hi-MAR variants.
3. Create pipelines for token prediction across phases (global + local refinement).
4. Write configurations and integrate evaluation metrics.

## Assumptions and Uncertainties
Given no addendum file:
- All hyperparameter and training details are derived from the markdown documentation.
- Training results use provided configurations but cannot be benchmarked.
