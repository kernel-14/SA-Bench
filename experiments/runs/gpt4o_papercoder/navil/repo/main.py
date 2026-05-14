"""
main.py: Entry point for the NaViL experiment pipeline.

This script orchestrates configuration setup, dataset loading, model initialization, 
training processes, and evaluation, adhering to the methodology described in the NaViL paper.

Dependencies:
- Config: For loading and managing configuration settings from a YAML file.
- DatasetLoader: For loading and preprocessing datasets.
- Model: For building the end-to-end NaViL architecture.
- Trainer: For executing training workflows (pretraining and fine-tuning stages).
- Evaluation: For benchmarking and validating model performance on multimodal tasks.
- Utils: Shared helper functions for logging and other operations.
"""

import sys
from config import Config
from dataset_loader import DatasetLoader
from model import VisualEncoder, LLM, Model
from trainer import Trainer
from evaluation import Evaluation
import utils

def main(config_path: str = "config.yaml") -> None:
    """
    Main function to run the NaViL experiment pipeline.

    Args:
        config_path (str): Path to the configuration file (default: "config.yaml").

    Raises:
        Exception: If any stage of the pipeline fails.
    """
    try:
        # Step 1: Load and validate configurations
        print("Loading configuration...")
        config = Config(config_path).get_config()

        # Step 2: Initialize DatasetLoader for datasets
        print("Initializing datasets...")
        dataset_loader = DatasetLoader(config)
        pretraining_data = dataset_loader.load_pretraining_data()
        finetuning_data = dataset_loader.load_finetuning_data()

        # Step 3: Build the NaViL model
        print("Building the NaViL model...")
        visual_encoder_config = config["model"]["visual_encoder"]
        llm_config = config["model"]["llm"]

        visual_encoder = VisualEncoder(
            depth=visual_encoder_config["depth"],
            width=visual_encoder_config["width"],
            patch_size=visual_encoder_config["patch_size"],
            mlp_width=visual_encoder_config["mlp_width"],
            attention_heads=visual_encoder_config["attention_heads"]
        )

        llm = LLM(
            model_name=llm_config["model_name"],
            use_moe=llm_config["use_moe"],
            num_experts=llm_config["num_experts"],
            depth=llm_config["depth"],
            width=llm_config["width"],
            mlp_width=llm_config["mlp_width"],
            attention_heads=llm_config["attention_heads"]
        )

        model = Model(visual_encoder, llm)

        # Step 4: Train the model
        print("Starting training pipeline...")
        trainer = Trainer(config=config, model=model, data_loader=dataset_loader)
        trainer.train_pretraining_stage()
        trainer.train_finetuning_stage()

        # Step 5: Evaluate the model
        print("Evaluating the model on multimodal tasks...")
        evaluator = Evaluation(model=model, dataset_loader=dataset_loader, config=config)

        # Evaluate on multimodal benchmarks
        tasks_to_evaluate = [
            "OCRBench", 
            "MMVet", 
            "ChartQA", 
            "DocVQA", 
            "InfographicVQA",
            "MMMU"
        ]
        evaluation_results = evaluator.evaluate_multimodal(tasks_to_evaluate)

        # Generate attention map visualization (if enabled)
        if config["logging"].get("log_metrics", True):
            sample_input = {
                "image": pretraining_data[:1]["image"],  # Example: Use a sample image from pretraining dataset
                "text": pretraining_data[:1]["text"]    # Example: Use a sample text from pretraining dataset
            }
            evaluator.generate_visualization(sample_input, layer=12)  # Visualize attention for a specific layer

        # Log the evaluation results to a file
        evaluator.log_evaluation_results(evaluation_results)

        print("Experiment completed successfully!")

    except Exception as e:
        print(f"Error during experiment execution: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    # Entry point for running the pipeline
    main("config.yaml")
