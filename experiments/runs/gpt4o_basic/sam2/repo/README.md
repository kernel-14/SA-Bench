# SAM 2: Segment Anything in Images and Videos

## Overview
This repository attempts to reproduce the core contributions of the paper "SAM 2: Segment Anything in Images and Videos."

## Structure
The project is divided into the following modules:
- **models/**: Contains the implementation of the SAM 2 architecture, including components like image encoder, memory attention modules, and mask decoder.
- **dataset/**: Scripts for dataset loading and preprocessing with a focus on simulating annotations described in the paper.
- **evaluation/**: Scripts for benchmarking the SAM 2 architecture on zero-shot segmentation tasks described in the paper.

## Implementation Plan
1. Build the SAM 2 architecture, focusing on:
   - Memory attention and prompt encoder components.
   - Streaming design for handling video segmentation, iterative prompting, and propagation.

2. Use publicly available datasets for evaluation to replicate some aspects of the SA-V dataset.

3. Design benchmarking scripts to validate zero-shot performance, comparing SAM 2 to existing baselines.

**Note**: Due to time constraints, we'll focus primarily on model implementation and evaluation simulations.

## Progress
- Project structure established.
- Architecture and dataset design underway.

