# nGPT: Normalized Transformer

This repository contains the code to reproduce the implementation of the Normalized Transformer (nGPT) as described in the paper 'nGPT: Normalized Transformer with Representation Learning on the Hypersphere'.

## Structure
- 'ngpt/models': Contains the implementation of models including embedding, normalized layers, attention blocks, etc.
- 'ngpt/utils': Utility functions including normalization operations and scaling factor utilities.
- 'scripts': Placeholder for training and evaluation scripts (to be implemented).

## Features
- Hyperspherical representation learning for embeddings, attention, and MLP blocks.
- Faster convergence and improved training stability.
- Removal of LayerNorm and RMSNorm, replaced by unit hypersphere normalization.
- Scaling factors to ensure smooth transitions during training.

## Progress
Currently, this repository provides a working implementation of the core nGPT components, pending final integration with training pipelines and evaluation settings.
