# Reproduction of "Lessons Learned from a Unifying Empirical Study of PETL in Visual Recognition"

This repository aims to reproduce the core contributions of the paper "Lessons Learned from a Unifying Empirical Study of PETL in Visual Recognition". The focus is on implementing representative Parameter-Efficient Fine-Tuning (PEFT) methods for Vision Transformers (ViT) and evaluating their performance across various scenarios described in the paper.

## Implemented Components

### 1. Vision Transformer (ViT) Backbone
- **`models/vit.py`**: Implements a base Vision Transformer model using `HuggingFace Transformers` library (`google/vit-base-patch16-224-in21k`). It includes functionalities to initialize the model with a classification head, freeze the backbone, and reset the classification head for different tasks.
  - **Note**: The paper uses `ViT-B/16` pre-trained on `ImageNet-21K`. `google/vit-base-patch16-224-in21k` is used as a suitable public proxy. For robustness experiments involving CLIP, a dedicated CLIP-pretrained ViT would ideally be used; for this reproduction, the same base ViT is adapted, and this is an acknowledged simplification.

### 2. Parameter-Efficient Fine-Tuning (PEFT) Methods
- **`models/peft_modules.py`**: Contains implementations of several key PEFT methods discussed in the paper, categorized as follows:
  - **LoRA (Low-Rank Adaptation)**: Implemented as `LoRALinear` and `apply_lora_to_linear`, which replaces `nn.Linear` layers with LoRA-augmented versions.
  - **BitFit**: Implemented as `apply_bitfit`, which unfreezes only the bias terms in the backbone and the classification head.
  - **LayerNorm Tuning**: Implemented as `apply_layernorm_tuning`, which unfreezes only the Layer Normalization parameters and the classification head.
  - **Houl. Adapter**: Implemented as `Adapter` and `apply_adapter_to_vit`, inserting bottleneck-structured adapters after the Multi-Head Self-Attention (MSA) and Multi-Layer Perceptron (MLP) blocks in each Transformer layer.
  - **VPT-Deep (Visual Prompt Tuning - Deep)**: Implemented as `VPTDeepModel`, which prepends learnable prompts to the input tokens of each Transformer layer.
- **`get_peft_model`**: A utility function to apply the chosen PEFT method to the base ViT model.

### 3. Data Loading and Preprocessing
- **`data_utils/transformations.py`**: Defines standard image transformations (resizing, cropping, normalization) for training and evaluation, consistent with ViT models.
- **`data_utils/vtab_datasets.py`**: Simulates the VTAB-1K benchmark for low-shot learning. Due to the complexity and proprietary nature of some VTAB-1K tasks, representative tasks are simulated using `CIFAR-100` (for natural images) and `torchvision.datasets.FakeData` for specialized and structured tasks. The 1000-shot, 80/20 train/validation split logic is implemented.
- **`data_utils/many_shot_datasets.py`**: Provides data loaders for many-shot scenarios, using `CIFAR-100`, and `torchvision.datasets.FakeData` to simulate `RESISC` and `Clevr-Distance` datasets. Training is done on the full dataset.
- **`data_utils/robustness_datasets.py`**: Handles datasets for robustness evaluation. `torchvision.datasets.FakeData` is used to simulate `ImageNet-1K` (100-shot target) and distribution shift datasets (`ImageNet-V2`, `ImageNet-R`, `ImageNet-S`, `ImageNet-A`).
  - **Note**: The FakeData approach aims to replicate the dataset *structure* (number of classes, approximate size) but does not use actual image data from these benchmarks. This is a significant simplification for static code reproduction.

### 4. Training and Evaluation Pipeline
- **`training/utils.py`**: Contains helper functions like `calculate_accuracy` and `get_device`.
- **`training/evaluate.py`**: Implements the model evaluation loop, calculating loss and accuracy on a given data loader.
- **`training/train.py`**: The main training loop. It initializes the model, applies the chosen PEFT method, sets up the optimizer and loss function, and iterates through epochs and tasks. It handles different data loading scenarios (VTAB-1K, many-shot, robustness) and performs validation/testing.

### 5. Experiment Management
- **`configs/default_config.yaml`**: A YAML configuration file defining default hyperparameters and experiment settings (scenario, PEFT method, model, data, training parameters).
- **`scripts/run_experiment.py`**: The entry point for running experiments. It loads a configuration, sets up an output directory, runs the training process, and saves the results.

## Missing Implementations / Future Work
- **Full Fine-Tuning (Full FT)**: Currently, only linear probing (freezing the backbone entirely and training only the head) is implicitly handled by setting `peft.method: none`. A distinct option for full fine-tuning, where the entire backbone is unfrozen and trained, needs to be added.
- **Weight-Space Ensembles (WiSE)**: Section 7 discusses WiSE for PEFT robustness. This feature is not yet implemented.
- **Prediction Diversity Analysis**: Section 4 details analysis of prediction diversity. This analysis code is not yet implemented.
- **Comprehensive Hyperparameter Tuning**: The paper emphasizes systematic hyperparameter tuning. While the config allows setting hyperparameters, a full tuning framework (e.g., using Weights & Biases, Optuna) is beyond the scope of this static reproduction but is crucial for achieving reported results.
- **Actual Dataset Integration**: The use of `torchvision.datasets.FakeData` for many datasets is a simplification. For a complete reproduction, actual dataset downloading and processing for VTAB-1K tasks, RESISC, Clevr-Distance, and ImageNet variants would be required.
- **DINOv2 and larger backbones (ViT-L, ViT-H)**: Mentioned in Appendix C, but not implemented.
- **Other PEFT methods**: Only a subset of the 14 PEFT methods are implemented (LoRA, BitFit, LayerNorm, Houl. Adapter, VPT-Deep). Others like Pfeif. Adapter, AdaptFormer, ConvPass, RepAdapter, DiffFit, SSF, FacT_TT, FacT_TK are not yet implemented.

## How to Run (Conceptual)

To run an experiment, you would typically execute `run_experiment.py` with a path to a configuration file:

```bash
python scripts/run_experiment.py --config configs/my_experiment_config.yaml
```

Example `my_experiment_config.yaml` for LoRA on VTAB-1K:

```yaml
scenario: vtab_1k
seed: 42
model:
  name: google/vit-base-patch16-224-in21k
peft:
  method: lora
  kwargs: {rank: 4, lora_alpha: 1}
data:
  image_size: 224
  num_workers: 4
training:
  epochs: 50 # Example: based on paper, more epochs are common
  batch_size: 64
  learning_rate: 0.0001 # Example
  weight_decay: 0.01
```


## Acknowledgements

This reproduction attempt is based on the insights and methodologies presented in the original paper:

Zheda Mai, Ping Zhang, Cheng-Hao Tu, Hong-You Chen, Quang-Huy Nguyen, Li Zhang, Wei-Lun Chao. "Lessons Learned from a Unifying Empirical Study of PETL in Visual Recognition."

