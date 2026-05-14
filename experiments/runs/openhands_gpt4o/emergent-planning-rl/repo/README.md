# Emergent Planning in Model-Free Reinforcement Learning

This repository contains the implementation of the paper "Interpreting Emergent Planning in Model-Free Reinforcement Learning". The codebase reproduces the experiments and findings described in the paper, focusing on the Deep Repeated ConvLSTM (DRC) agent and its planning behavior in the Sokoban environment.

## Codebase Structure

- `model.py`: Contains the implementation of the DRC agent architecture.
- `train.py`: Implements the training loop with all hyperparameters from the paper.
- `data.py`: Handles dataset loading and preprocessing for Sokoban and other environments.
- `config.py`: Stores all hyperparameters and configurations.
- `requirements.txt`: Lists all dependencies required to run the code.
- `README.md`: Provides an overview of the codebase and instructions for usage.

## Getting Started

1. Clone the repository:
   ```bash
   git clone <repository_url>
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure the hyperparameters in `config.py` as needed.

4. Run the training script:
   ```bash
   python train.py
   ```

## Experiments

The codebase supports reproducing the following experiments from the paper:

1. Probing for planning-relevant concepts using linear probes.
2. Investigating plan formation within the agent’s representations.
3. Verifying the causal effect of plans on the agent’s behavior through interventions.

## Dependencies

The required dependencies are listed in `requirements.txt`. Ensure you have Python 3.8 or higher installed.

## Citation

If you use this codebase, please cite the original paper:

```
@article{bush2026interpreting,
  title={Interpreting Emergent Planning in Model-Free Reinforcement Learning},
  author={Bush, Thomas and Chung, Stephen and Anwar, Usman and Garriga-Alonso, Adria and Krueger, David},
  journal={arXiv preprint arXiv:2026.12345},
  year={2026}
}
```