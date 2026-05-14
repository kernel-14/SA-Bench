
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from accelerate import Accelerator
from tqdm import tqdm
import os
import argparse

from config import NaViLConfig, get_config
from model import NaViL
from data import get_dataloader

def get_lr_scheduler(optimizer, total_steps, warmup_steps, schedule_type, peak_lr):
    if schedule_type == "constant_warmup":
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif schedule_type == "cosine_decay":
        # CosineAnnealingLR usually goes from max_lr to min_lr.
        # For warmup, we can combine a LambdaLR for warmup and then a CosineAnnealingLR.
        # A common practice is to use a Linear warmup then Cosine Decay.
        # For simplicity here, we'll implement a linear warmup and then switch to cosine decay.
        # This requires careful handling of total_steps for CosineAnnealingLR.
        
        # A more robust approach would be to use a combined scheduler like in HuggingFace transformers
        # For now, let's just make it linear warmup and then a simple constant LR if not fully implemented.
        # Given the paper's description, it implies linear warmup then cosine decay.
        
        # This is a simplified cosine decay, a full implementation often uses warmup in conjunction
        # For now, if we reach this, we assume warmup is handled.
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=0)
    else:
        raise ValueError(f"Unknown LR schedule type: {schedule_type}")

def set_trainable_params(model: NaViL, stage: str):
    """
    Sets which parameters are trainable based on the training stage.
    """
    if stage == "s1_1": # Stage 1.1: Visual params trainable, textual frozen.
        for name, param in model.named_parameters():
            if "visual_encoder" in name or "mlp_projector" in name or ("llm_blocks" in name and "MoE" in name): # Assuming MoE parameters are newly added vision-specific
                param.requires_grad = True
            else:
                param.requires_grad = False
        print("Stage 1.1: Only visual encoder, projector, and MoE experts are trainable.")
    elif stage == "s1_2": # Stage 1.2: Textual self-attention unfrozen, all other vision-specific params remain trainable
        for name, param in model.named_parameters():
            if "llm_blocks" in name and ("attn" in name or "MoE" in name): # Unfreeze attention and MoE within LLM blocks
                param.requires_grad = True
            elif "visual_encoder" in name or "mlp_projector" in name:
                param.requires_grad = True
            else: # Freeze other LLM parameters (e.g., FFN if not MoE, embeddings, head)
                param.requires_grad = False
        print("Stage 1.2: Visual params and LLM self-attention (incl. MoE) trainable.")
    elif stage == "s2": # Stage 2: All parameters unfrozen
        for param in model.parameters():
            param.requires_grad = True
        print("Stage 2: All parameters are trainable.")
    else:
        raise ValueError(f"Unknown training stage: {stage}")

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M / Total parameters: {total_params / 1e6:.2f}M")


def train():
    parser = argparse.ArgumentParser(description="NaViL Training Script")
    parser.add_argument("--model_size", type=str, default="2B", choices=["2B", "9B"],
                        help="Specify the NaViL model size (e.g., '2B', '9B').")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Directory to save checkpoints and logs.")
    args = parser.parse_args()

    # Initialize Accelerator
    accelerator = Accelerator(
        mixed_precision=get_config(args.model_size).training["numerical_precision"]
    )

    config = get_config(args.model_size)
    accelerator.print(f"Starting training for NaViL-{args.model_size} with config: \n{config}")

    # Initialize Model
    model = NaViL(config)
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.training["s1_1_peak_learning_rate"], # Initial LR for S1.1
        betas=config.training["optimizer_betas"],
        eps=config.training["optimizer_eps"],
        weight_decay=config.training["s1_1_weight_decay"]
    )

    # Prepare for distributed training
    model, optimizer = accelerator.prepare(model, optimizer)

    # Training Stages
    stages = ["s1_1", "s1_2", "s2"]
    for stage_name in stages:
        accelerator.print(f"\n{'='*20} Starting {stage_name} {'='*20}")
        stage_config = config.training[stage_name]
        
        # Set trainable parameters for the current stage
        set_trainable_params(model, stage_name)

        # Learning Rate Scheduler for the current stage
        total_steps = stage_config["training_steps"]
        warmup_steps = stage_config["warmup_steps"]
        peak_lr = stage_config["peak_learning_rate"]
        lr_schedule_type = stage_config["learning_rate_schedule"]
        weight_decay = stage_config["weight_decay"]

        # Update optimizer's learning rate and weight decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = peak_lr
            param_group['weight_decay'] = weight_decay

        lr_scheduler = get_lr_scheduler(optimizer, total_steps, warmup_steps, lr_schedule_type, peak_lr)
        lr_scheduler = accelerator.prepare(lr_scheduler) # Prepare scheduler if it has state

        # Data Loader for the current stage
        # This is a conceptual data loading. In a real scenario, data_paths need to be dynamically determined.
        if stage_name == "s1_1":
            data_paths = ["conceptual_web_scale_data_part1", "conceptual_web_scale_data_part2", "conceptual_synthesized_data"]
        elif stage_name == "s1_2":
            data_paths = ["conceptual_high_quality_multimodal", "conceptual_pure_language"]
        elif stage_name == "s2":
            data_paths = ["conceptual_high_quality_multimodal_finetuning"]
        
        train_dataloader = get_dataloader(
            data_paths, config, model.tokenizer,
            batch_size=stage_config["global_batch_size"],
            shuffle=True,
            is_train=True
        )
        train_dataloader = accelerator.prepare(train_dataloader)

        progress_bar = tqdm(range(total_steps), disable=not accelerator.is_main_process)

        for step in range(total_steps):
            model.train()
            batch = next(iter(train_dataloader)) # Get a batch

            # Forward pass
            # Handle multi-scale images for input
            images_input = batch["images"]
            
            outputs = model(
                images=images_input,
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            loss = outputs["loss"]

            # Backward pass and optimization
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

            # Log periodically (e.g., every 100 steps)
            if (step + 1) % 100 == 0:
                accelerator.print(f"Step {step+1}/{total_steps}, Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        accelerator.print(f"\n{'='*20} {stage_name} Finished {'='*20}")
        # Save checkpoint at the end of each stage if needed
        # accelerator.save_state(os.path.join(args.output_dir, f"checkpoint_{stage_name}"))

    accelerator.print("Training complete!")

if __name__ == "__main__":
    train()
