## main.py

import argparse
import torch
import os
from config import Config, get_config
from dataset_loader import DatasetLoader
from model import SpatialTemporalTransformer
from trainer import Trainer
from inference import Inference
from evaluation import Evaluation

def parse_arguments() -> argparse.Namespace:
    """
    Parse user-provided command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments including mode and checkpoint path.
    """
    parser = argparse.ArgumentParser(description="Reproducing the Ca2-VDM methodology and experiments.")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["train", "evaluate", "generate"], 
        default="train", 
        help="Mode of operation: 'train' for training, 'evaluate' for model evaluation, 'generate' for inference."
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default="", 
        help="Path to a pre-trained model checkpoint for evaluation or generation."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="./outputs", 
        help="Path to save generated videos or evaluation metrics."
    )
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_arguments()

    # Load Configurations
    config = get_config()

    print("Configuration loaded successfully.")

    # Step 1: Dataset Preparation
    print("Initializing dataset loader...")
    dataset_loader = DatasetLoader(config)

    train_loader = dataset_loader.load_data("train")
    val_loader = dataset_loader.load_data("val")
    test_loader = dataset_loader.load_data("test")
    print("Datasets loaded: train, validation, test.")

    # Step 2: Initialize Model
    print("Initializing the Ca2-VDM model...")
    model = SpatialTemporalTransformer(config.config_dict)

    if torch.cuda.is_available():
        model = model.cuda()
    print("Model initialized.")

    # Step 3: Depending on Mode, Perform Relevant Operations
    if args.mode == "train":
        # Training Mode
        print("Entering training mode...")
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config
        )

        trainer.train()  # Train the model

    elif args.mode == "generate":
        # Generation Mode (Inference)
        if not args.checkpoint:
            raise ValueError("Checkpoint must be provided for the 'generate' mode.")

        print(f"Loading model checkpoint from {args.checkpoint}...")
        model.load_state_dict(torch.load(args.checkpoint)["model_state_dict"])

        print("Initializing inference engine...")
        inference_engine = Inference(model=model, config=config)

        # Prepare an initial frame. For demonstration, using a dummy tensor.
        initial_frame = torch.zeros((1, 3, config.get("vae.resolution"), config.get("vae.resolution")))
        generated_video = inference_engine.generate_video(initial_frame=initial_frame, num_frames=64)

        # Save Generated Video
        os.makedirs(args.output, exist_ok=True)
        output_path = os.path.join(args.output, "generated_video.pt")
        torch.save(generated_video, output_path)
        print(f"Generated video saved at {output_path}.")

    elif args.mode == "evaluate":
        # Evaluation Mode
        if not args.checkpoint:
            raise ValueError("Checkpoint must be provided for the 'evaluate' mode.")

        print(f"Loading model checkpoint from {args.checkpoint}...")
        model.load_state_dict(torch.load(args.checkpoint)["model_state_dict"])

        print("Initializing inference for evaluation...")
        inference_engine = Inference(model=model, config=config)

        print("Generating video chunks for evaluation...")
        generated_videos = []
        for test_batch in test_loader:
            initial_frame = test_batch["video"][:, 0, :, :, :]  # Assuming the first frame is the starting frame
            generated_video = inference_engine.generate_video(initial_frame=initial_frame, num_frames=64)
            generated_videos.append(generated_video)

        # Prepare data loaders for evaluation
        generated_loader = torch.utils.data.DataLoader(
            generated_videos, batch_size=1, shuffle=False
        )

        print("Initializing evaluator...")
        evaluator = Evaluation(ground_truth=test_loader, generated=generated_loader, config=config)

        # Compute FVD Metric
        fvd_score = evaluator.compute_fvd(pretrained_i3d=config.get("evaluation.fvd.pretrained_i3d"))
        print(f"Frechet Video Distance (FVD): {fvd_score}")

        # Compute Time and Memory Efficiency
        time_metrics = evaluator.compute_time_efficiency()
        memory_usage = evaluator.compute_memory_usage()

        # Save Evaluation Results
        results = {
            "FVD Score": fvd_score,
            **time_metrics,
            "Peak Memory Usage (GB)": memory_usage,
        }

        os.makedirs(args.output, exist_ok=True)
        output_path = os.path.join(args.output, "evaluation_metrics.txt")
        evaluator.save_results(output_path, results)

        print(f"Evaluation results saved to {output_path}.")

if __name__ == "__main__":
    main()
