import argparse
import yaml
import os
import json
import torch
import numpy as np

# Import custom modules
from config import Config
from utils import adjust_ffn_width_for_gating, calculate_model_parameters
from data_loader import DataLoader
from model.gated_transformer import GatedTransformer
from trainer import Trainer
from evaluator import Evaluator

from accelerate import Accelerator


def main(args: argparse.Namespace) -> None:
    """
    Main function to orchestrate the Gated Attention LLM reproduction pipeline.

    Args:
        args: Command-line arguments parsed by argparse.
    """
    # 1. Load Configuration from YAML file
    with open(args.config_path, 'r') as f:
        config_dict: Dict[str, Any] = yaml.safe_load(f)

    # Override config_dict with command-line arguments where specified
    # General arguments
    if args.experiment_name is not None:
        config_dict["experiment_name"] = args.experiment_name
    if args.output_dir is not None:
        config_dict["output_dir"] = args.output_dir
    if args.seed is not None:
        config_dict["seed"] = args.seed

    # Model arguments
    if args.model_type is not None:
        config_dict["model"]["type"] = args.model_type
    if args.num_layers is not None:
        config_dict["model"]["num_layers"] = args.num_layers
    if args.d_model is not None:
        config_dict["model"]["d_model"] = args.d_model
    if args.q_heads is not None:
        config_dict["model"]["q_heads"] = args.q_heads
    if args.kv_heads is not None:
        config_dict["model"]["kv_heads"] = args.kv_heads
    if args.head_dim is not None:
        config_dict["model"]["head_dim"] = args.head_dim

    # Gating arguments
    if args.gating_enabled is not None:
        config_dict["gating"]["enabled"] = args.gating_enabled
    if args.gating_position is not None:
        config_dict["gating"]["position"] = args.gating_position
    if args.gating_granularity is not None:
        config_dict["gating"]["granularity"] = args.gating_granularity
    if args.gating_head_specific is not None:
        config_dict["gating"]["head_specific"] = args.gating_head_specific
    if args.gating_type is not None:
        config_dict["gating"]["type"] = args.gating_type
    if args.gating_activation_fn is not None:
        config_dict["gating"]["activation_fn"] = args.gating_activation_fn
    if args.gating_ns_sigmoid_factor is not None:
        config_dict["gating"]["ns_sigmoid_factor"] = args.gating_ns_sigmoid_factor

    # Training arguments
    if args.max_learning_rate is not None:
        config_dict["training"]["max_learning_rate"] = args.max_learning_rate
    if args.global_batch_size is not None:
        config_dict["training"]["global_batch_size"] = args.global_batch_size
    if args.total_train_tokens is not None:
        config_dict["training"]["total_train_tokens"] = args.total_train_tokens
    if args.mixed_precision is not None:
        config_dict["training"]["mixed_precision"] = args.mixed_precision
    if args.run_training is not None:
        config_dict["run_training"] = args.run_training
    if args.resume_from_checkpoint is not None:
        config_dict["resume_from_checkpoint"] = args.resume_from_checkpoint

    # Evaluation arguments
    if args.run_evaluation is not None:
        config_dict["run_evaluation"] = args.run_evaluation
    if args.run_analysis is not None:
        config_dict["run_analysis"] = args.run_analysis

    # Create the Config object, which also validates and derives parameters
    config = Config(config_dict)

    # 2. Accelerator Initialization
    accelerator = Accelerator(
        mixed_precision=config.training.mixed_precision,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        log_with="tensorboard",  # Logging with TensorBoard
        project_dir=config.output_dir,
    )

    # 3. Set Seed for reproducibility
    accelerator.set_seed(config.seed)

    # 4. Output Directory Management
    experiment_output_dir: str = os.path.join(config.output_dir, config.experiment_name)
    os.makedirs(experiment_output_dir, exist_ok=True)

    # Save the final resolved config to the experiment directory (only from main process)
    if accelerator.is_main_process:
        with open(os.path.join(experiment_output_dir, "final_config.yaml"), 'w') as f:
            yaml.dump(config.to_dict(), f, sort_keys=False) # sort_keys=False to preserve order
        with open(os.path.join(experiment_output_dir, "final_config.json"), 'w') as f:
            json.dump(config.to_dict(), f, indent=4)
        accelerator.print(f"Final configuration saved to {experiment_output_dir}")

    # 5. Adjust FFN Width (if applicable for dense models with gating)
    # This ensures parameter count matches baseline for dense models as per paper.
    if config.gating_enabled and config.model_type == "dense":
        original_d_ff: int = config.d_ff
        config = adjust_ffn_width_for_gating(config)
        if accelerator.is_main_process:
            accelerator.print(f"FFN width adjusted from {original_d_ff} to {config.d_ff} for parameter parity.")

    # 6. Instantiate Components
    data_loader: DataLoader = DataLoader(config)
    model: GatedTransformer = GatedTransformer(config)

    # Calculate and log model parameters (unwrapped model for true count)
    total_params: int = calculate_model_parameters(model)
    if accelerator.is_main_process:
        accelerator.print(f"Model initialized with approximately {total_params / 1e6:.2f}M trainable parameters.")
        if config.model_type == "dense":
            accelerator.print(f"Target parameters: {config.total_parameters_target / 1e6:.2f}M.")

    # Initialize Trainer and Evaluator. Model preparation (accelerator.prepare)
    # is handled within Trainer's __init__ if training, or explicitly here for eval-only.
    trainer: Optional[Trainer] = None
    if args.run_training:
        trainer = Trainer(model, data_loader, config)
        # The prepared model from trainer is now assigned back to the `model` variable for evaluator.
        model = trainer.model 
        if args.resume_from_checkpoint:
            accelerator.load_state(args.resume_from_checkpoint)
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
    else: # If not training, ensure model is prepared for evaluation
        model = accelerator.prepare(model)
        model.eval()  # Ensure model is in eval mode if no training loop will run

    evaluator: Evaluator = Evaluator(model, data_loader, config)

    # 7. Training Phase Orchestration
    if args.run_training and trainer is not None:
        accelerator.print("Starting training...")
        trainer.train()
        accelerator.print("Training finished.")

    # 8. Evaluation and Analysis Phase Orchestration
    eval_results: Dict[str, Any] = {}
    analysis_results: Dict[str, Any] = {}

    if args.run_evaluation:
        accelerator.print("Starting evaluation...")
        ppl_score: float = evaluator.evaluate_ppl()
        eval_results["perplexity"] = ppl_score
        accelerator.print(f"Final Evaluation PPL: {ppl_score:.4f}")

        benchmark_scores: Dict[str, float] = evaluator.evaluate_benchmarks()
        eval_results["benchmarks"] = benchmark_scores
        if accelerator.is_main_process:
            accelerator.print(f"Benchmark Results: {json.dumps(benchmark_scores, indent=2)}")

        accelerator.print("Evaluation finished.")

    if args.run_analysis:
        accelerator.print("Starting analysis...")
        analysis_results = evaluator.analyze_metrics()
        if accelerator.is_main_process:
            accelerator.print(f"Analysis Results: {json.dumps(analysis_results, indent=2)}")
        accelerator.print("Analysis finished.")

    # 9. Results Reporting and Saving (only from main process)
    if accelerator.is_main_process:
        final_report: Dict[str, Any] = {
            "config": config.to_dict(),
            "final_evaluation_results": eval_results,
            "analysis_results": analysis_results,
        }
        report_path: str = os.path.join(experiment_output_dir, "final_results.json")
        with open(report_path, 'w') as f:
            json.dump(final_report, f, indent=4)
        accelerator.print(f"Full results report saved to {report_path}")

    # End Accelerate training session
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce Gated Attention LLM experiments.")

    # Basic configuration arguments
    parser.add_argument("--config_path", type=str, default="config.yaml",
                        help="Path to the YAML configuration file.")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Name for the current experiment. Overrides config.yaml.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Base directory for experiment outputs. Overrides config.yaml.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility. Overrides config.yaml.")

    # Model configuration overrides
    parser.add_argument("--model_type", type=str, choices=["dense", "moe"], default=None,
                        help="Type of model (dense or moe). Overrides config.yaml.")
    parser.add_argument("--num_layers", type=int, default=None,
                        help="Number of transformer layers. Overrides config.yaml.")
    parser.add_argument("--d_model", type=int, default=None,
                        help="Model dimension. Overrides config.yaml.")
    parser.add_argument("--q_heads", type=int, default=None,
                        help="Number of query heads. Overrides config.yaml.")
    parser.add_argument("--kv_heads", type=int, default=None,
                        help="Number of key-value heads. Overrides config.yaml.")
    parser.add_argument("--head_dim", type=int, default=None,
                        help="Dimension of each attention head. Overrides config.yaml.")

    # Gating configuration overrides
    parser.add_argument("--gating_enabled", type=lambda x: (str(x).lower() == 'true'), default=None,
                        help="Enable or disable gating. Overrides config.yaml.")
    parser.add_argument("--gating_position", type=str,
                        choices=["G1", "G2", "G3", "G4", "G5"], default=None,
                        help="Position of gating application. Overrides config.yaml.")
    parser.add_argument("--gating_granularity", type=str,
                        choices=["elementwise", "headwise"], default=None,
                        help="Granularity of gating scores. Overrides config.yaml.")
    parser.add_argument("--gating_head_specific", type=lambda x: (str(x).lower() == 'true'), default=None,
                        help="Whether gating parameters are head-specific. Overrides config.yaml.")
    parser.add_argument("--gating_type", type=str,
                        choices=["multiplicative", "additive"], default=None,
                        help="Type of gating (multiplicative or additive). Overrides config.yaml.")
    parser.add_argument("--gating_activation_fn", type=str,
                        choices=["sigmoid", "silu", "identity", "ns_sigmoid"], default=None,
                        help="Activation function for gating scores. Overrides config.yaml.")
    parser.add_argument("--gating_ns_sigmoid_factor", type=float, default=None,
                        help="Factor for non-sparse sigmoid (0.5 + factor * sigmoid(x)). Overrides config.yaml.")

    # Training configuration overrides
    parser.add_argument("--max_learning_rate", type=float, default=None,
                        help="Maximum learning rate. Overrides config.yaml.")
    parser.add_argument("--global_batch_size", type=int, default=None,
                        help="Effective global batch size. Overrides config.yaml.")
    parser.add_argument("--total_train_tokens", type=float, default=None,
                        help="Total tokens to train on (e.g., 400e9 for 400B). Overrides config.yaml.")
    parser.add_argument("--mixed_precision", type=str, choices=["no", "fp16", "bf16"], default=None,
                        help="Mixed precision training mode. Overrides config.yaml.")
    parser.add_argument("--run_training", type=lambda x: (str(x).lower() == 'true'), default=True,
                        help="Whether to run the training phase. Default to True.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to a checkpoint to resume training from.")

    # Evaluation and Analysis flags
    parser.add_argument("--run_evaluation", type=lambda x: (str(x).lower() == 'true'), default=True,
                        help="Whether to run the evaluation phase. Default to True.")
    parser.add_argument("--run_analysis", type=lambda x: (str(x).lower() == 'true'), default=True,
                        help="Whether to run the analysis phase. Default to True.")

    args = parser.parse_args()
    main(args)

