# SAM 2: Segment Anything in Images and Videos - Reproduction

This repository aims to reproduce the core contributions of the paper "SAM 2: Segment Anything in Images and Videos".

## 1. Introduction

SAM 2 extends the concept of promptable segmentation from images to videos, introducing a unified model capable of handling both modalities. Key contributions include:
- A novel transformer-based architecture with streaming memory for real-time video processing.
- A data engine for collecting the large-scale SA-V video segmentation dataset.
- Significant performance improvements in zero-shot video and image segmentation tasks.

## 2. Reproduction Scope

This reproduction attempt focuses on the following core aspects of the paper:

### 2.1. Model Architecture
The primary goal is to replicate the SAM 2 model architecture as described in Section 4 of the paper and further detailed in Appendix D. This includes:
- **Image Encoder**: Hiera (MAE pre-trained) for unconditioned frame feature embeddings.
- **Memory Attention**: Stacked transformer blocks with self-attention and cross-attention to a memory bank.
- **Prompt Encoder**: Identical to SAM's, handling clicks, boxes, and masks.
- **Mask Decoder**: Two-way transformer blocks, similar to SAM, predicting multiple masks and an object presence head.
- **Memory Encoder**: Generates memories by downsampling output masks and fusing with frame embeddings.
- **Memory Bank**: FIFO queues for recent frames' memories and prompted frames' information, along with object pointers.

### 2.2. Data Engine and SA-V Dataset
While the actual data collection cannot be reproduced, this section will document the methodology of the data engine (Section 5.1 and Appendix E) and the characteristics of the SA-V dataset (Section 5.2). This will include descriptions of:
- Phase 1: SAM per frame annotation.
- Phase 2: SAM + SAM 2 Mask for temporal propagation.
- Phase 3: Fully-featured SAM 2 with various prompts.
- Quality verification and auto masklet generation.
- SA-V dataset composition, splits, and comparison to other VOS datasets.

### 2.3. Training Methodology
The training process (Section 4.5 and Appendix D) will be outlined, including:
- Joint training on image and video data.
- Simulation of interactive prompting (sampling frames, corrective clicks, initial prompt types).

### 2.4. Evaluation Protocols
A description of the zero-shot experiments (Section 6) will be provided, covering:
- Promptable Video Segmentation (offline and online evaluation).
- Semi-supervised Video Object Segmentation (first-frame prompts).
- Image Segmentation (1-click and 5-click mIoUs).
- Metrics used ($\mathcal{J}\& \mathcal{F}$, mIoU).

## 3. Assumptions and Limitations

- **Static Reproduction**: This reproduction is static; no code execution, model training, or data generation will be performed. The goal is to provide a codebase and documentation that accurately reflects the paper's descriptions.
- **Appendix Details**: Specific implementation details often reside in the appendices (e.g., Appendix D for model details, Appendix E for data engine, Appendix F for experimental details). These will be incorporated as much as possible.
- **External Resources**: Where the paper references external resources or prior work (e.g., MAE, Hiera, SAM), it is assumed that these components would be available or implemented based on their respective publications.
- **Computational Resources**: The original paper implies significant computational resources for data collection and model training. This reproduction does not account for these practical constraints.
- **Blacklisted Resources**: No blacklisted resources (e.g., the paper's official codebase) were used.

## 4. Codebase Structure

The repository will be structured as follows:


