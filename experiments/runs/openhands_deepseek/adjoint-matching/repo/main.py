"""
Main entry point for running Adjoint Matching experiments.

Reproduces the experiments from:
"Adjoint Matching: Fine-tuning Flow and Diffusion Generative Models
with Memoryless Stochastic Optimal Control"
"""
import torch
import torch.nn as nn
import argparse
import yaml
import os
import random
import numpy as np
from typing import Dict, Any

from models.unet import UNetModel
from models.flow_matching import FlowMatchingModel
from soc.memoryless_schedule import MemorylessNoiseSchedule
from soc.adjoint_matching import AdjointMatchingLoss, LeanAdjointSolver
from training.train_adjoint_matching import train_adjoint_matching
from training.train_baselines import (
    train_draft,
    train_refl,
    train_dpo,
    train_continuous_adjoint,
    train_discrete_adjoint,
)
from data.dataset import create_dataloader, load_prompts_from_file, split_prompts
from evaluation.metrics import evaluate_model


def set_seed(seed: int = 42):
    """Set all random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def build_models(config: Dict[str, Any], device: str = "cuda"):
    """Build Flow Matching model and reward model."""
    model_cfg = config["model"]
    
    # Build U-Net
    unet = UNetModel(
        in_channels=model_cfg.get("in_channels", 4),
        model_channels=model_cfg.get("model_channels", 320),
        out_channels=model_cfg.get("out_channels", 4),
        num_res_blocks=model_cfg.get("num_res_blocks", 2),
        attention_resolutions=model_cfg.get("attention_resolutions", [4, 2, 1]),
        dropout=model_cfg.get("dropout", 0.0),
        channel_mult=model_cfg.get("channel_mult", [1, 2, 4, 4]),
        num_heads=model_cfg.get("num_heads", 8),
        transformer_depth=model_cfg.get("transformer_depth", 1),
        context_dim=model_cfg.get("context_dim", 768),
        use_linear_in_transformer=model_cfg.get("use_linear_in_transformer", True),
        image_size=model_cfg.get("image_size", 64),
    )
    
    # Build Flow Matching model
    fm_model = FlowMatchingModel(unet)
    
    return fm_model


def build_reward_model(config: Dict[str, Any], device: str = "cuda"):
    """
    Build reward model (ImageReward or similar).
    
    This is a placeholder - actual implementation depends on the specific
    reward model being used (ImageReward, HPSv2, etc.)
    """
    reward_type = config.get("reward", {}).get("type", "imagereward")
    
    if reward_type == "imagereward":
        # Placeholder: ImageReward model
        # In practice, this would load the ImageReward checkpoint
        # from https://github.com/THUDM/ImageReward
        class DummyRewardModel(nn.Module):
            def forward(self, x):
                # x is latent, would need to decode through VAE first
                return torch.randn(x.shape[0], device=x.device)
        
        reward_model = DummyRewardModel()
    else:
        raise ValueError(f"Unknown reward model type: {reward_type}")
    
    return reward_model.to(device)


def run_experiment(args):
    """Run a single experiment (fine-tuning method)."""
    config = load_config(args.config)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set seed
    set_seed(args.seed)
    
    # Load prompts
    all_prompts = load_prompts_from_file(
        args.prompt_file,
        num_prompts=config.get("data", {}).get("prompt_dataset_size", 100000),
    )
    
    # Split into train/test
    train_prompts, test_prompts = split_prompts(
        all_prompts,
        train_size=config.get("training", {}).get("finetune_prompts", 40000),
        test_size=config.get("training", {}).get("test_prompts", 1000),
    )
    
    # Create dataloaders
    train_loader = create_dataloader(
        prompts=train_prompts,
        batch_size=config.get("training", {}).get("batch_size", 20),
        shuffle=True,
    )
    
    test_loader = create_dataloader(
        prompts=test_prompts,
        batch_size=config.get("training", {}).get("batch_size", 20),
        shuffle=False,
    )
    
    # Build models
    base_model = build_models(config, device)
    finetune_model = build_models(config, device)
    
    # Initialize fine-tune model weights from base model
    finetune_model.load_state_dict(base_model.state_dict())
    
    reward_model = build_reward_model(config, device)
    
    # Get training config
    train_cfg = config.get("training", {})
    adj_cfg = config.get("adjoint_matching", {})
    
    # Run fine-tuning based on method
    method = args.method
    
    if method == "adjoint_matching":
        for lam in adj_cfg.get("lambda_values", [12500]):
            print(f"\n{'='*60}")
            print(f"Running Adjoint Matching with lambda={lam}")
            print(f"{'='*60}\n")
            
            finetune_model = train_adjoint_matching(
                base_model=base_model,
                finetune_model=finetune_model,
                reward_model=reward_model,
                dataloader=train_loader,
                config=config,
                num_epochs=train_cfg.get("num_epochs", 25),
                lambda_reward=lam,
                num_steps=train_cfg.get("num_timesteps", 40),
                lr=train_cfg.get("learning_rate", 2e-5),
                adam_betas=(train_cfg.get("adam_beta1", 0.95), train_cfg.get("adam_beta2", 0.999)),
                weight_decay=train_cfg.get("weight_decay", 1e-2),
                grad_clip=train_cfg.get("grad_clip", 1.0),
                precision=train_cfg.get("precision", "bfloat16"),
                lct_constant=adj_cfg.get("lct_constant", 1.6),
                grad_timesteps_first=adj_cfg.get("gradient_timesteps_subset_first", 10),
                grad_timesteps_last=adj_cfg.get("gradient_timesteps_subset_last", 10),
                dt_offset=True,
                device=device,
                save_dir=config.get("logging", {}).get("save_dir", "./checkpoints"),
            )
            
            # Evaluate
            results = evaluate_model(
                model=finetune_model,
                prompt_list=test_prompts,
                vae_decoder=None,  # placeholder
                num_generations_per_prompt=40,
                num_diversity_prompts=25,
                guidance_weight=0.0,
                num_steps=train_cfg.get("num_inference_steps", 40),
                device=device,
            )
            print(f"Results for lambda={lam}: {results}")
    
    elif method == "draft-1":
        finetune_model = train_draft(
            base_model=base_model,
            finetune_model=finetune_model,
            reward_model=reward_model,
            dataloader=train_loader,
            config=config,
            K=1,
            num_epochs=train_cfg.get("num_epochs", 25),
            num_steps=train_cfg.get("num_timesteps", 40),
            lr=train_cfg.get("learning_rate", 2e-5),
            adam_betas=(train_cfg.get("adam_beta1", 0.95), train_cfg.get("adam_beta2", 0.999)),
            weight_decay=train_cfg.get("weight_decay", 1e-2),
            grad_clip=train_cfg.get("grad_clip", 1.0),
            precision=train_cfg.get("precision", "bfloat16"),
            device=device,
            save_dir=config.get("logging", {}).get("save_dir", "./checkpoints"),
        )
    
    elif method == "draft-40":
        finetune_model = train_draft(
            base_model=base_model,
            finetune_model=finetune_model,
            reward_model=reward_model,
            dataloader=train_loader,
            config=config,
            K=40,
            num_epochs=train_cfg.get("num_epochs", 25),
            num_steps=train_cfg.get("num_timesteps", 40),
            lr=train_cfg.get("learning_rate", 2e-5),
            adam_betas=(train_cfg.get("adam_beta1", 0.95), train_cfg.get("adam_beta2", 0.999)),
            weight_decay=train_cfg.get("weight_decay", 1e-2),
            grad_clip=train_cfg.get("grad_clip", 1.0),
            precision=train_cfg.get("precision", "bfloat16"),
            device=device,
            save_dir=config.get("logging", {}).get("save_dir", "./checkpoints"),
        )
    
    elif method == "refl":
        finetune_model = train_refl(
            base_model=base_model,
            finetune_model=finetune_model,
            reward_model=reward_model,
            dataloader=train_loader,
            config=config,
            num_epochs=train_cfg.get("num_epochs", 25),
            num_steps=train_cfg.get("num_timesteps", 40),
            lr=train_cfg.get("learning_rate", 2e-5),
            adam_betas=(train_cfg.get("adam_beta1", 0.95), train_cfg.get("adam_beta2", 0.999)),
            weight_decay=train_cfg.get("weight_decay", 1e-2),
            grad_clip=train_cfg.get("grad_clip", 1.0),
            precision=train_cfg.get("precision", "bfloat16"),
            device=device,
            save_dir=config.get("logging", {}).get("save_dir", "./checkpoints"),
        )
    
    elif method == "dpo":
        finetune_model = train_dpo(
            base_model=base_model,
            finetune_model=finetune_model,
            reward_model=reward_model,
            dataloader=train_loader,
            config=config,
            num_epochs=train_cfg.get("num_epochs", 25),
            num_steps=train_cfg.get("num_timesteps", 40),
            beta_dpo=5000.0,
            lr=train_cfg.get("learning_rate", 2e-5),
            adam_betas=(train_cfg.get("adam_beta1", 0.95), train_cfg.get("adam_beta2", 0.999)),
            weight_decay=train_cfg.get("weight_decay", 1e-2),
            grad_clip=train_cfg.get("grad_clip", 1.0),
            precision=train_cfg.get("precision", "bfloat16"),
            device=device,
            save_dir=config.get("logging", {}).get("save_dir", "./checkpoints"),
        )
    
    elif method == "continuous_adjoint":
        for lam in adj_cfg.get("lambda_values", [12500]):
            finetune_model = train_continuous_adjoint(
                base_model=base_model,
                finetune_model=finetune_model,
                reward_model=reward_model,
                dataloader=train_loader,
                config=config,
                lambda_reward=lam,
                num_epochs=train_cfg.get("num_epochs", 25),
                num_steps=train_cfg.get("num_timesteps", 40),
                lr=train_cfg.get("learning_rate", 2e-5),
                adam_betas=(train_cfg.get("adam_beta1", 0.95), train_cfg.get("adam_beta2", 0.999)),
                weight_decay=train_cfg.get("weight_decay", 1e-2),
                grad_clip=train_cfg.get("grad_clip", 1.0),
                precision=train_cfg.get("precision", "bfloat16"),
                lct_constant=1600.0,
                dt_offset=True,
                device=device,
                save_dir=config.get("logging", {}).get("save_dir", "./checkpoints"),
            )
    
    elif method == "discrete_adjoint":
        for lam in adj_cfg.get("lambda_values", [12500]):
            finetune_model = train_discrete_adjoint(
                base_model=base_model,
                finetune_model=finetune_model,
                reward_model=reward_model,
                dataloader=train_loader,
                config=config,
                lambda_reward=lam,
                num_epochs=train_cfg.get("num_epochs", 25),
                num_steps=train_cfg.get("num_timesteps", 40),
                lr=1e-5,  # lower LR for discrete adjoint
                adam_betas=(train_cfg.get("adam_beta1", 0.95), train_cfg.get("adam_beta2", 0.999)),
                weight_decay=train_cfg.get("weight_decay", 1e-2),
                grad_clip=train_cfg.get("grad_clip", 1.0),
                precision=train_cfg.get("precision", "bfloat16"),
                dt_offset=True,
                device=device,
                save_dir=config.get("logging", {}).get("save_dir", "./checkpoints"),
            )
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return finetune_model


def main():
    parser = argparse.ArgumentParser(description="Adjoint Matching Experiments")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="adjoint_matching",
        choices=[
            "adjoint_matching",
            "draft-1",
            "draft-40",
            "refl",
            "dpo",
            "continuous_adjoint",
            "discrete_adjoint",
        ],
        help="Fine-tuning method to use",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        required=True,
        help="Path to file containing text prompts",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory for checkpoints and logs",
    )
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    run_experiment(args)


if __name__ == "__main__":
    main()
