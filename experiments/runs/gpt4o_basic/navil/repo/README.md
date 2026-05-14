# Reproduction of NaViL Paper

## Overview
This repository aims to reproduce the main contributions of the paper titled 'NaViL: Rethinking Scaling Properties of Native Multimodal LLMs under Data Constraints.'

### Key Contributions: 
1. Architectures optimized for native multimodal large language models (MLLMs), with emphasis on visual encoders and mixture-of-experts (MoEs).
2. Empirical scaling analysis of visual encoders and linguistic large language models, and their log-proportional relationship.
3. Implementation of the NaViL model with end-to-end training techniques utilizing multi-modal generative pre-training and supervised fine-tuning.
4. Performance evaluation across 14 multimodal benchmarks spanning image captioning, OCR, visual question answering, and more. 

### Implementation Goals: 
The repository shall include:
- Source code for the NaViL MLLM architecture.
- Preprocessing pipelines for web-scale multimodal datasets as described.
- Scripts for generative multi-modal pre-training and supervised fine-tuning.
- Benchmarking scripts to replicate results on evaluation datasets (Image Captioning, OCR, etc.).

### Structure
- `src/`: Implementation of the NaViL architecture and training logic.
- `data/`: Scripts to generate and process web-scale multimodal datasets.
- `benchmarks/`: Code for multimodal evaluations and comparisons.
- `README.md`: Guide to the repository setup, assumptions, and notes on progress.

#### Next Steps
- Populate `src/` with architecture modules based on Section 4.
- Implement preprocessing from Stage 1 (datasets and synthesis) as described in Section 4.2.

