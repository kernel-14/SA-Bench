
# MA-RLHF: Reinforcement Learning from Human Feedback with Macro Actions

This repository contains a faithful reproduction of the core contributions of the paper "MA-RLHF: Reinforcement Learning from Human Feedback with Macro Actions" by Chai et al. (Baidu Inc.). The paper proposes MA-RLHF, a framework that incorporates macro actions (sequences of tokens or higher-level language constructs) into the RLHF process to address the credit assignment problem in long sequences, enhance learning efficiency, and stabilize policy gradient estimates.

## Project Structure

The codebase is organized as follows:

- `config.py`: Defines all hyperparameters and configuration settings for SFT, RM, PPO, Macro Actions, Model, Data, and general training/evaluation. Hyperparameters are derived directly from the paper's main text and Appendix B.2 and D.1.
- `model.py`: Implements the neural network architectures for the Policy Model (CausalLM), Reward Model (AutoModel with regression head), and Value Model (Critic, AutoModel with regression head). Supports LoRA for efficient fine-tuning.
- `modules.py`: Contains the core logic for Macro Actions, including various termination conditions (`fixed_ngram`, `randomized_ngram`, `parsing`, `perplexity`) and value estimation assignments (`equal`, `unit`, `position_decayed`). It also includes the PPO policy and critic loss calculations adapted for macro actions, and Generalized Advantage Estimation (GAE).
- `data.py`: Handles dataset loading, preprocessing, and data collation for the Supervised Fine-Tuning (SFT), Reward Modeling (RM), and Proximal Policy Optimization (PPO) stages. It includes data loaders for TL;DR, HH-RLHF, WebGPT Comparisons, and APPS datasets, with appropriate data splits as described in the paper.
- `train.py`: Orchestrates the multi-stage training process: SFT, RM, and PPO (MA-PPO). It uses the `accelerate` library for distributed training and manages optimizers, learning rate schedulers, and model saving.
- `evaluation.py`: Provides functionalities for evaluating the trained models using metrics such as Reward Model scores, and simulated GPT-4 and human pairwise win rates. For code generation (APPS), it includes a placeholder for `pass@k` evaluation.
- `requirements.txt`: Lists all necessary Python dependencies to run the codebase.

## Key Features Implemented

- **MA-RLHF Framework**: Full implementation of the three stages: SFT, RM, and MA-PPO.
- **Macro Action Logic**:
    - **Termination Conditions**: `fixed_ngram`, `randomized_ngram`, `parsing` (placeholder/fallback), `perplexity` (placeholder for PPL calculation).
    - **Value Estimation Assignments**: `equal`, `unit`, `position_decayed` for macro action values.
    - **MA-PPO Algorithm**: PPO policy and critic losses adapted for macro actions, using GAE.
- **Model Architectures**: Policy (CausalLM), Reward, and Value models based on `transformers`. Support for LoRA.
- **Dataset Handling**: Loaders and collators for TL;DR, HH-RLHF, WebGPT Comparisons, and APPS, with specified data splits.
- **Evaluation Metrics**: Reward Model scores, simulated GPT-4 and human evaluations, and placeholder for `pass@k` for code.

## How to Run

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Configure**: Adjust settings in `config.py` as needed, especially `model_name_or_path` and dataset paths.
3.  **Run Training**:
    ```bash
    accelerate launch train.py --task_name [tldr|hh_rlhf|webgpt|apps]
    ```
    (Note: The `train.py` script needs to be modified to accept `task_name` argument and initialize `Config` appropriately. Currently, it's set up for direct instantiation within `if __name__ == "__main__":` block for demonstration.)

## Notes on Reproducibility

- **Placeholders**: Some functionalities, particularly for `parsing`-based and `perplexity`-based macro action termination, and the `pass@k` evaluation for APPS, are implemented as placeholders or simplified simulations. A full implementation would require external parsing libraries, real-time perplexity calculations from model logits, and an isolated code execution environment, respectively. These are outside the scope of a pure code reproduction from paper text alone.
- **Dataset Loading**: The `data.py` uses `load_dataset` from Hugging Face for common datasets (TL;DR, HH-RLHF). For WebGPT Comparisons and APPS, dummy dataset structures are provided, as direct Hugging Face datasets might not exist or require specific loading methods not fully detailed in the paper's main text. Users should replace these with actual loading logic for their specific data.
- **KL Coefficient for 7B TL;DR**: The paper mentions reducing the KL coefficient to 0.01 for 7B TL;DR model for stability. This kind of dynamic adjustment would typically be handled by a more sophisticated configuration system or command-line overrides during an actual training run.
