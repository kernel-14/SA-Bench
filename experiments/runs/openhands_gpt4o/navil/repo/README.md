# NaViL: Native Multimodal Large Language Model

This repository contains the implementation of NaViL, a native multimodal large language model designed for end-to-end training of vision and language modalities. The implementation is based on the paper "NaViL: Rethinking Scaling Properties of Native Multimodal Large Language Models under Data Constraints."

## Codebase Structure

- `model.py`: Contains the implementation of the NaViL model, including the visual encoder, LLM, and MoE components.
- `train.py`: Implements the training loop with all hyperparameters and strategies described in the paper.
- `data.py`: Handles dataset loading and preprocessing for web-scale noisy image-caption pairs and high-quality multimodal data.
- `config.py`: Contains all hyperparameters and configuration settings.
- `requirements.txt`: Lists all dependencies required for the implementation.
- `README.md`: Provides an overview of the codebase and usage instructions.

## Getting Started

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure the settings in `config.py` as needed.

4. Run the training script:
   ```bash
   python train.py
   ```

## Citation

If you use this codebase, please cite the original paper:

```
@article{tian2026navil,
  title={NaViL: Rethinking Scaling Properties of Native Multimodal Large Language Models under Data Constraints},
  author={Changyao Tian and Hao Li and Gen Luo and others},
  journal={arXiv preprint arXiv:2504.07951},
  year={2026}
}
```