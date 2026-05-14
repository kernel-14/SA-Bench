## main.py

import yaml
import torch
from trainer import Trainer
from model import Model
from dataset_loader import DatasetLoader
from adaptive_inference import AdaptiveInference
from evaluation import Evaluation
from utils import save_model

class Main:
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initializes the Main class, loads configurations, and sets up the pipeline.

        Args:
            config_path (str): Path to the configuration file.
        """
        # Load configuration file
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        # GPU/CPU device setup
        self.device = torch.device("cuda" if self.config["hardware"]["use_gpu"] and torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def run_experiment(self):
        """
        Orchestrates the full workflow for training, inference, and evaluation.
        """
        # Step 1: Prepare Dataset
        print("Loading and preprocessing datasets...")
        loader = DatasetLoader(self.config)
        datasets = loader.load_data()
        preprocessed_datasets = loader.preprocess_data(datasets)

        # Step 2: Initialize Model
        print("Initializing model...")
        model_params = {
            "transformer_layers": self.config["model"]["transformer_layers"],
            "hidden_dim": self.config["model"]["hidden_dim"],
            "num_attention_heads": self.config["model"]["num_attention_heads"],
            "dropout_rate": self.config["model"]["dropout_rate"],
            "max_sequence_length": self.config["model"]["max_sequence_length"],
            "positional_encoding": self.config["model"]["positional_encoding"],
        }
        model = Model(model_params).to(self.device)

        # Step 3: Train Model
        print("Training model...")
        trainer = Trainer(model, preprocessed_datasets["slim_pajama"], self.config)
        trainer.train()

        # Save the trained model
        save_model(model, "./checkpoints/trained_model.pt")

        # Step 4: Perform Inference
        print("Starting inference...")
        inference = AdaptiveInference(model, self.config)
        
        print("Performing Vanilla Inference...")
        vanilla_outputs = inference.apply_adaptive_strategy(preprocessed_datasets["slim_pajama"]["masked_sequences"], strategy_name="vanilla")
        
        print("Performing Adaptive Top Probability Inference...")
        top_probability_outputs = inference.apply_adaptive_strategy(preprocessed_datasets["slim_pajama"]["masked_sequences"], strategy_name="top_probability")
        
        print("Performing Adaptive Top Probability Margin Inference...")
        top_probability_margin_outputs = inference.apply_adaptive_strategy(preprocessed_datasets["slim_pajama"]["masked_sequences"], strategy_name="top_probability_margin")
        
        # Step 5: Evaluate Model
        print("Evaluating model...")
        evaluation = Evaluation(model, preprocessed_datasets, self.config["evaluation"]["metrics"])
        
        print("Evaluation on SlimPajama Dataset (Text Data)...")
        text_metrics = evaluation.evaluate_text_data()
        print(f"Text Evaluation Metrics: {text_metrics}")
        
        print("Evaluation on Sudoku and Zebra Puzzle Datasets (Logic Puzzles)...")
        puzzle_metrics = evaluation.evaluate_logic_puzzles()
        print(f"Puzzle Evaluation Metrics: {puzzle_metrics}")

        # Final summary
        print("Experiment completed successfully!")
        print(f"Text Metrics: {text_metrics}")
        print(f"Puzzle Metrics: {puzzle_metrics}")

# Entry point
if __name__ == "__main__":
    main = Main(config_path="config.yaml")
    main.run_experiment()
