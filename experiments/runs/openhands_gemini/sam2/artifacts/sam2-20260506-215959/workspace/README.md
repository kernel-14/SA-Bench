# SAM 2: Segment Anything in Images and Videos

This repository contains a faithful reproduction of the Segment Anything Model 2 (SAM 2) as described in the paper "SAM 2: Segment Anything in Images and Videos" by Ravi et al.

## Project Structure

- `main.py`: Entry point for training and evaluation.
- `config.py`: Contains all hyperparameters and configuration settings.
- `requirements.txt`: Lists all Python dependencies.
- `model/`:
    - `sam2.py`: Main SAM 2 model architecture.
    - `image_encoder.py`: Implements the Hiera image encoder.
    - `memory_attention.py`: Implements the memory attention module.
    - `prompt_encoder.py`: Implements the prompt encoder (similar to SAM).
    - `mask_decoder.py`: Implements the mask decoder (similar to SAM with extensions).
    - `memory_encoder.py`: Implements the memory encoder.
    - `layers.py`: Common neural network layers and utility functions.
- `data/`:
    - `datasets.py`: Handles dataset loading (SA-V, SA-1B, VOS datasets).
    - `transforms.py`: Defines data augmentation and preprocessing pipelines.
- `training/`:
    - `trainer.py`: Contains the training loop and evaluation logic.
    - `losses.py`: Implements the loss functions used for training.
- `utils/`:
    - `metrics.py`: Implements evaluation metrics (e.g., T&F, mIoU).
    - `misc.py`: Contains miscellaneous utility functions.

## Reproduction Details

This implementation aims to reproduce the core contributions of the paper, including:
- The unified architecture for image and video segmentation.
- The streaming memory mechanism for video processing.
- Training strategies including alternating image/video data, interactive prompting simulation, and various augmentations.
- Hyperparameters and configurations are set according to the paper's descriptions and appendix.
