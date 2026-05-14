# Ca2-VDM: Efficient Autoregressive Video Diffusion Model with Causal Generation and Cache Sharing

This repository contains the implementation of the Ca2-VDM model as described in the paper "Ca2-VDM: Efficient Autoregressive Video Diffusion Model with Causal Generation and Cache Sharing".

## Codebase Structure

- `model.py`: Defines the main model architecture, including causal temporal attention and prefix-enhanced spatial attention.
- `modules.py`: Contains reusable modules for the model.
- `layers.py`: Implements custom layers used in the model.
- `train.py`: Training loop with objectives and hyperparameters.
- `data.py`: Dataset loading and preprocessing for MSR-VTT, UCF-101, and Sky Timelapse.
- `config.py`: Configuration file with all hyperparameters and settings.
- `requirements.txt`: Lists all dependencies required to run the code.
- `README.md`: Provides an overview of the project and instructions for use.

## Datasets

The following datasets are supported:
- **MSR-VTT**: A large video description dataset.
- **UCF-101**: A dataset of 101 human action classes from videos in the wild.
- **Sky Timelapse**: A time-lapse dataset showing dynamic sky scenes.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Configure the settings in `config.py`.
2. Train the model:
   ```bash
   python train.py
   ```
3. Evaluate the model:
   ```bash
   python evaluate.py
   ```

## Citation

If you use this code, please cite the original paper:
```
@article{gao2026ca2vdm,
  title={Ca2-VDM: Efficient Autoregressive Video Diffusion Model with Causal Generation and Cache Sharing},
  author={Kaifeng Gao and Jiaxin Shi and Hanwang Zhang and Chunping Wang and Jun Xiao and Long Chen},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```