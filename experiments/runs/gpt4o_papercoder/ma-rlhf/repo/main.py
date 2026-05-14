# main.py

import os
import yaml
import torch
from dataset_loader import DatasetLoader
from model import Model
from termination import Termination
from ppo import PPO
from trainer import Trainer
from evaluation import Evaluation

class Main:
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the main experiment workflow by loading configuration and setting up components.

        Args:
            config_path (str): Path to the configuration file.
        """
        # Load configuration
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        # Set up essential paths
        self.task_type = self.config.get("training", {}).get("task_type", "tl_dr")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Logging and checkpoints directories
        self.log_dir = self.config.get("training", {}).get("log_dir", "./logs")
        self.checkpoint_dir = self.config.get("training", {}).get("checkpoint_dir", "./checkpoints")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def run_experiment(self):
        """
        Execute the main workflow of the experiment: dataset loading, training, and evaluation.
        """
        print("Starting MA-RLHF Experiment...")

        # Step 1: Load dataset
        dataset_loader = DatasetLoader(self.config)
        train_dataset, metadata = dataset_loader.load_data()
        print(f"Dataset loaded: {metadata['task_type']} with {metadata['train_size']} training samples.")
        
        # Step 2: Initialize the models
        pretrained_model_name = self.config.get("models", {}).get("pretrained_model", "gemma-2b")
        model_params = self.config.get("models", {})
        policy_model = Model(pretrained_model_name, model_params)
        critic_model = Model(pretrained_model_name, model_params)
        print(f"Initialized models: Policy model and Critic model loaded using {pretrained_model_name}.")

        # Step 3: Configure termination strategy
        termination_type = self.config.get("termination", {}).get("type", "fixed_ngram")
        termination_params = self.config.get("termination", {}).get("params", {})
        termination = Termination(termination_type, termination_params)
        print(f"Initialized termination strategy: {termination_type}.")

        # Step 4: Set up PPO optimizer
        ppo_params = self.config.get("ppo", {}).get("hyperparameters", {})
        ppo_optimizer = PPO(policy_model, critic_model, termination, ppo_params)
        print("PPO optimizer configured.")

        # Step 5: Set up the training pipeline
        trainer = Trainer(policy_model, dataset_loader, termination, ppo_optimizer, self.config)
        trainer.train()

        # Step 6: Evaluate the trained models
        evaluation = Evaluation(policy_model, dataset_loader, self.config)
        
        # Run RM score evaluation
        rm_results = evaluation.evaluate_reward_model()
        print("Reward model evaluation results:", rm_results)
        
        # Run GPT-4 evaluation if prompts are provided in config
        gpt4_enabled = self.config.get("evaluation", {}).get("enable_gpt4", False)
        if gpt4_enabled:
            responses = self.get_sample_responses(train_dataset)
            gpt4_results = evaluation.evaluate_gpt4(self.task_type, responses)
            print("GPT-4 evaluation results:", gpt4_results)

        # Run human evaluation if enabled
        human_evaluation_enabled = self.config.get("evaluation", {}).get("enable_human", False)
        if human_evaluation_enabled:
            human_results = evaluation.evaluate_human(self.task_type, responses)
            print("Human evaluation results:", human_results)

        print("MA-RLHF Experiment completed successfully.")

    def get_sample_responses(self, dataset) -> List[Tuple[str, str]]:
        """
        Extract sample responses for paired evaluation (e.g., GPT-4 and human evaluation).

        Args:
            dataset: Loaded dataset object.

        Returns:
            List[Tuple[str, str]]: List of paired responses for evaluation.
        """
        sample_responses = []
        for entry in dataset[:50]:  # Randomly sample first 50 responses for evaluation
            sample_responses.append((entry["chosen_response"], entry["rejected_response"]))
        return sample_responses


if __name__ == "__main__":
    # Instantiate and run the experiment
    main_experiment = Main(config_path="config.yaml")
    main_experiment.run_experiment()
