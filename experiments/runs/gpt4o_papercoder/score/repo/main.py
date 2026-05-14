## main.py

import os
from utils import load_config, set_seed, log_results
from dataset_loader import DatasetLoader
from model import Model
from stage1_trainer import Stage1Trainer
from stage2_trainer import Stage2Trainer
from evaluation import Evaluation

def run_experiment(config_path: str = "config.yaml") -> None:
    """
    Main entry point for executing the SCoRe reproducibility pipeline. 
    This includes dataset loading, model training in two stages, and evaluation.

    Args:
        config_path (str): Path to the configuration file (default is 'config.yaml').
    """
    print("Loading Configuration...")
    # Load configuration and set the random seed
    config = load_config(config_path)
    set_seed(config["logging"]["seed"])

    # Step 1: Initialize DatasetLoader and Load Datasets
    print("Initializing Dataset Loader...")
    dataset_loader = DatasetLoader(config)
    
    print("Loading and Preprocessing Data...")
    train_dataset, test_dataset = dataset_loader.load_data()  # Raw loading
    train_dataset = dataset_loader.preprocess_data(train_dataset)  # Preprocess for training
    test_dataset = dataset_loader.preprocess_data(test_dataset)  # Preprocess for evaluation

    # Step 2: Initialize Pre-trained Model
    print("Initializing Pre-trained Model...")
    pretrained_model_name = config["models"]["reasoning_model"]  # Use Gemini 1.5 Flash for MATH
    model = Model(pretrained_model=pretrained_model_name, config=config)

    # Step 3: Stage I Training (Policy Initialization)
    print("Starting Stage I Training...")
    stage1_trainer = Stage1Trainer(model=model, train_dataset=train_dataset, config=config)
    model = stage1_trainer.train()  # Update model with Stage I-trained weights

    # Step 4: Save Stage I Checkpoint
    print("Saving Stage I Checkpoint...")
    stage1_checkpoint_path = "stage1_final_checkpoint.pth"
    model.save_model_weights(stage1_checkpoint_path)

    # Step 5: Stage II Training (Reward Shaping and Refinement)
    print("Starting Stage II Training...")
    stage2_trainer = Stage2Trainer(model=model, train_dataset=train_dataset, config=config)
    model = stage2_trainer.train()  # Update model with Stage II-trained weights

    # Step 6: Save Stage II Checkpoint
    print("Saving Stage II Checkpoint...")
    stage2_checkpoint_path = "stage2_final_checkpoint.pth"
    model.save_model_weights(stage2_checkpoint_path)

    # Step 7: Evaluation
    print("Starting Evaluation...")
    evaluator = Evaluation(model=model, test_dataset=test_dataset, config=config)
    evaluation_results = evaluator.evaluate()

    # Step 8: Log Results
    print("Logging Results...")
    log_results(evaluation_results, config["logging"]["log_path"])

    # Final Output
    print("Experiment Complete! Results Logged to:", config["logging"]["log_path"])

if __name__ == "__main__":
    run_experiment()
