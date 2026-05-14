# SCoRe: Self-Correction via Reinforcement Learning

This repository contains a faithful reproduction of the paper "Training Language Models to Self-Correct via Reinforcement Learning" (Kumar et al., 2024).

## Project Structure

- `config.py`: Configuration for hyperparameters, model, and training settings.
- `data.py`: Handles dataset loading, preprocessing, and tokenization for MATH and HumanEval/MBPP datasets.
- `model.py`: Defines the LLM architecture and its components.
- `modules.py`: Contains specific neural network modules or custom layers if required by the model.
- `metrics.py`: Implements the evaluation metrics described in the paper (Accuracy@t1, Accuracy@t2, Delta metrics).
- `prompts.py`: Stores the zero-shot and self-correction prompts used for MATH and HumanEval/MBPP.
- `train.py`: Contains the main training loop for SFT and SCoRe (Stage I and Stage II)
- `requirements.txt`: Lists all Python dependencies.
