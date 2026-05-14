# Reproduction: Improving Consistency Models with Generator-Augmented Flows

## Overview
This repository contains the implementation of the core methodologies introduced in the paper 'Improving Consistency Models with Generator-Augmented Flows'. The focus is on the development and training of consistency models using generator-augmented flows (GC).

## Implemented Contributions
1. **Consistency Model Architecture**: Implemented as described, with convolutional layers for feature extraction and fully connected layers for endpoint prediction.
2. **Dataset Utilities**: Includes functions for loading CIFAR-10 and similar datasets with transformations.
3. **Training Pipeline**:
   - Supports training consistency models using standard IC-based approaches.
   - Placeholder design for GC losses with joint learning parameter µ for flexibility.

## Limitations
- Current implementation uses MSE loss rather than the precise GC-based loss described, due to time constraints.
- Pre-training strategies for endpoint predictors (Section 5.1) are deferred but programmable based on the modular structure.

## How to Use
1. Place datasets in the paths defined in .
2. Run  for training using IC methodologies.

## Future Work
- Implement GC and joint learning in detail using the theoretical framework.
- Apply to additional datasets like ImageNet and CelebA and integrate evaluation metrics (FID, IS, KID).

