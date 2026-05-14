
# Ca2-VDM: Efficient Autoregressive Video Diffusion Model with Causal Generation and Cache Sharing

This repository provides a faithful reproduction of the Ca2-VDM paper.

## Project Structure

The codebase is organized as follows:

- `ca2_vdm/`: Contains the core implementation of the Ca2-VDM model.
    - `config.py`: Defines all hyperparameters and configuration settings for the model, diffusion process, training, data, and system.
    - `modules.py`: Implements essential building blocks such as TimestepEmbedding, PositionalEncoding, CausalTemporalAttention, PrefixEnhancedSpatialAttention, ResNetBlock, FeedForward, and placeholders for VAE and T5TextEncoder.
    - `model.py`: Defines the main `Ca2VDM` architecture, which is a spatial-temporal Transformer-based UNet-like model, integrating the causal attention mechanisms and handling KV-cache.
    - `data.py`: Handles dataset loading and preprocessing. It includes a `VideoDataset` class that simulates video data loading and prepares training samples with clean prefixes, denoising targets, and cyclic temporal positional embeddings.
    - `train.py`: Implements the training loop, including the `GaussianDiffusion` process for noise scheduling and denoising steps, the training objective (simplified loss + VLB), and a two-stage training strategy as described in the paper.
    - `inference.py`: Implements the autoregressive inference pipeline with cache sharing. It includes a `KVCacheQueue` for managing temporal and spatial KV-caches and a `VideoGenerator` for orchestrating the denoising and cache writing stages.

- `requirements.txt`: Lists all necessary Python dependencies for running the code.

## Key Features Implemented

- **Causal Generation**: Implemented via `CausalTemporalAttention` which uses an attention mask to ensure each frame only attends to its preceding frames.
- **Cache Sharing**: Supported by the `KVCacheQueue` in `inference.py`, allowing temporal KV-caches to be shared across all denoising timesteps and a single spatial KV-cache for the most recent chunk.
- **Prefix-Enhanced Spatial Attention**: Implemented in `PrefixEnhancedSpatialAttention` to enhance guidance from prefix frames by spatially concatenating a sub-prefix to the denoising target.
- **Cyclic-TPEs**: Integrated into `PositionalEncoding` and used in both training and inference to enable long-term context by cyclically shifting temporal positional embeddings.
- **Two-Stage Training**: For text-to-video generation, the training follows a two-stage approach: first, causal modeling without a clean prefix, then training with a clean prefix and cyclic TPEs.
- **Autoregressive Inference**: The `VideoGenerator` orchestrates the autoregressive process, including denoising a chunk, computing new KV-caches from the denoised chunk, and updating the KV-cache queues.

## Setup and Usage (Conceptual)

1.  **Environment Setup**: Install the required packages using `pip install -r requirements.txt`.
2.  **Dataset Preparation**: Prepare your video datasets (e.g., InternVid, SkyTimelapse, MSR-VTT, UCF-101) according to the paths specified in `config.py`. The current `data.py` uses dummy data; actual video loading would need to be integrated.
3.  **Pre-trained Components**: Obtain and integrate pre-trained VAE (e.g., from Stable Diffusion) and T5 Text Encoder as mentioned in the paper. Placeholders are currently used.
4.  **Training**: Run `train.py` to start the training process. Modify `config.py` for specific task types (text-to-video or video prediction) and hyperparameters.
5.  **Inference**: After training, use `inference.py` to generate long videos autoregressively.

## Disclaimer

This codebase is a reproduction based solely on the provided paper text. It aims for faithfulness to the described methods and architectural details. Actual performance may vary depending on the specific pre-trained VAE, text encoder, and large-scale video datasets used for training, which are outside the scope of this direct code reproduction.
</p>
