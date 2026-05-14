# Pyramidal Flow Matching for Efficient Video Generative Modeling

This repository contains the implementation of the paper "Pyramidal Flow Matching for Efficient Video Generative Modeling." The codebase is structured to reproduce the core contributions of the paper, including the pyramidal flow matching algorithm and its application to video generation.

## Codebase Structure

- `model.py`: Defines the main model architecture, including the spatial and temporal pyramids.
- `modules.py`: Contains reusable modules such as the spatial pyramid, temporal pyramid, and diffusion transformer.
- `layers.py`: Implements utility layers like upsampling and downsampling.
- `train.py`: Implements the training loop with all hyperparameters from the paper.
- `data.py`: Handles dataset loading and preprocessing.
- `config.py`: Contains all hyperparameters and configurations.
- `requirements.txt`: Lists all dependencies required to run the codebase.

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare the dataset:
   Place your video dataset in the `./data/videos` directory.

3. Train the model:
   ```bash
   python train.py
   ```

## Notes

- The implementation is based on the methods and algorithms described in the paper.
- Ensure that the dataset is preprocessed as required by the `VideoDataset` class in `data.py`.

## Citation

If you use this code, please cite the original paper:

```
@article{jin2026pyramidal,
  title={Pyramidal Flow Matching for Efficient Video Generative Modeling},
  author={Yang Jin and Zhicheng Sun and Ningyuan Li and Kun Xu and Hao Jiang and Nan Zhuang and Quzhe Huang and Yang Song and Yadong Mu and Zhouchen Lin},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```