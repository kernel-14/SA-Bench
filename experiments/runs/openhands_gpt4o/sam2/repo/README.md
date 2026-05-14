# SAM 2: Segment Anything in Images and Videos

This repository contains the implementation of the Segment Anything Model 2 (SAM 2) as described in the paper. SAM 2 is a unified model for video and image segmentation, capable of interactively segmenting regions through prompts (clicks, boxes, or masks) on one or multiple video frames.

## Codebase Structure

- `model.py`: Contains the implementation of the SAM 2 model architecture, including the image encoder, memory attention, prompt encoder, mask decoder, memory encoder, and memory bank.
- `train.py`: Implements the training loop, including loss functions and training strategies.
- `data.py`: Handles dataset loading and preprocessing, including support for the SA-V dataset.
- `config.yaml`: Contains all hyperparameters and configuration settings.
- `requirements.txt`: Lists all dependencies required for the implementation.
- `README.md`: Provides an overview of the repository and instructions for usage.

## Getting Started

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure the hyperparameters in `config.yaml` as needed.

3. Run the training script:
   ```bash
   python train.py
   ```

## License

This project is licensed under the Apache 2.0 License.