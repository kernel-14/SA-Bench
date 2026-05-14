import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
import math
import random
import os

from config import parse_args, get_model_config
from model import HiMAR
from data import ImageNetDataset, MSCOCODataset, BaseScheduler

def setup_model(args, accelerator):
    model_config = get_model_config(args.model_type)
    
    # Determine text conditioning based on dataset
    text_conditioning = (args.dataset == "mscoco")

    model = HiMAR(
        vae_path=args.vae_path,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        patch_size=args.patch_size,
        num_classes=args.num_classes,
        conditioning_dim=model_config["transformer_hidden_size"],
        transformer_layers=model_config["transformer_layers"],
        transformer_hidden_size=model_config["transformer_hidden_size"],
        num_heads=args.num_heads,
        diff_head1_layers=model_config["diff_head1_layers"],
        diff_head1_hidden_size=model_config["diff_head1_hidden_size"],
        diff_head2_layers=model_config["diff_head2_layers"],
        diff_head2_hidden_size=model_config["diff_head2_hidden_size"],
        low_res_image_size=args.low_res_image_size,
        image_size=args.image_size,
        text_conditioning=text_conditioning,
        clip_model_name=args.clip_model_name,
    )
    return model

def get_dataloader(args):
    if args.dataset == "imagenet":
        dataset = ImageNetDataset(args.data_path, args.image_size, args.low_res_image_size)
    elif args.dataset == "mscoco":
        dataset = MSCOCODataset(args.data_path, args.image_size, args.low_res_image_size, args.clip_model_name)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return dataloader

def get_optimizer(args, model):
    if args.optimizer == "AdamW":
        if args.dataset == "imagenet":
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
        elif args.dataset == "mscoco":
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_t2i, weight_decay=args.weight_decay_t2i)
        else:
            raise ValueError(f"Unknown dataset for optimizer: {args.dataset}")
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    return optimizer

def train():
    args = parse_args()
    accelerator = Accelerator(
        mixed_precision="fp16" if torch.cuda.is_available() else "no",
        log_with="wandb" # if args.use_wandb else None
    )
    
    if accelerator.is_main_process:
        accelerator.init_trackers("himar_experiment", config=vars(args))

    # Setup model, optimizer, scheduler, etc.
    model = setup_model(args, accelerator)
    optimizer = get_optimizer(args, model)
    dataloader = get_dataloader(args)
    diffusion_scheduler = BaseScheduler()

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(dataloader, disable=not accelerator.is_main_process)
        for batch in pbar:
            # Extract data
            high_res_images = batch["high_res_image"]
            low_res_images = batch["low_res_image"]
            
            if args.dataset == "imagenet":
                conditions = batch["label"]
            elif args.dataset == "mscoco":
                conditions = batch["text_input_ids"]
            
            B = high_res_images.shape[0]

            # 1. Encode images to latent space using VAE
            # NOTE: model.vae is not prepared by accelerator because we don't train it.
            # It should be moved to the correct device by default if AutoencoderKL is used.
            with torch.no_grad():
                low_res_latents = model.encode_vae(low_res_images, low_res=True)
                high_res_latents = model.encode_vae(high_res_images, low_res=False)

            # 2. Sample timestep for diffusion
            timesteps = torch.randint(0, diffusion_scheduler.num_train_timesteps, (B,), device=accelerator.device).long()
            
            # 3. Add noise to low-res latents
            noise_low_res = torch.randn_like(low_res_latents)
            noisy_low_res_latents = diffusion_scheduler.add_noise(low_res_latents, noise_low_res, timesteps)

            # 4. Phase 1: Predict low-resolution conditional tokens (Z^s)
            # Masking for Phase 1: uniformly sampled in [0.7, 1.0]
            mask_ratio_phase1 = diffusion_scheduler.get_masking_ratio(
                current_step=0, total_steps=0, strategy="uniform", 
                r_min=args.masking_ratio_phase1_min, r_max=args.masking_ratio_phase1_max
            )
            # Apply masking (simplified: replace a fraction with zeros or a special token)
            # A more sophisticated masking strategy would involve a mask token or learnable embedding.
            num_masked_tokens_low_res = int(noisy_low_res_latents.shape[1] * noisy_low_res_latents.shape[2] * mask_ratio_phase1)
            
            # Convert to sequence (B, N, C) for masking
            noisy_low_res_latents_seq = noisy_low_res_latents.view(B, -1, model.latent_channels)
            
            # Randomly select tokens to mask
            indices = torch.randperm(noisy_low_res_latents_seq.shape[1])[:num_masked_tokens_low_res]
            masked_low_res_latents_seq = noisy_low_res_latents_seq.clone()
            masked_low_res_latents_seq[:, indices] = 0.0 # Replace with zero for simplicity

            # Ensure model input is (B, C, H, W) for HiMAR.forward's initial processing
            masked_low_res_latents_input = masked_low_res_latents_seq.permute(0, 2, 1).view(B, model.latent_channels, model.low_res_latent_size, model.low_res_latent_size)

            predicted_noise_low_res_latent, conditional_tokens_low_res = model(
                x_low_res=masked_low_res_latents_input,
                x_high_res=None, # Not used in phase 1
                t=timesteps,
                y=conditions,
                phase=1
            )
            
            # Loss calculation for Phase 1
            loss_phase1 = F.mse_loss(predicted_noise_low_res_latent, noise_low_res)

            # 5. Add noise to high-res latents
            noise_high_res = torch.randn_like(high_res_latents)
            noisy_high_res_latents = diffusion_scheduler.add_noise(high_res_latents, noise_high_res, timesteps)

            # 6. Phase 2: Predict high-resolution tokens (Z^l)
            # Masking for Phase 2: cosine masking strategy
            mask_ratio_phase2 = diffusion_scheduler.get_masking_ratio(
                current_step=timesteps.float().mean().item(), # Use mean timestep for ratio calculation
                total_steps=diffusion_scheduler.num_train_timesteps,
                strategy=args.masking_strategy_phase2
            )
            num_masked_tokens_high_res = int(noisy_high_res_latents.shape[1] * noisy_high_res_latents.shape[2] * mask_ratio_phase2)

            noisy_high_res_latents_seq = noisy_high_res_latents.view(B, -1, model.latent_channels)
            indices_high_res = torch.randperm(noisy_high_res_latents_seq.shape[1])[:num_masked_tokens_high_res]
            masked_high_res_latents_seq = noisy_high_res_latents_seq.clone()
            masked_high_res_latents_seq[:, indices_high_res] = 0.0 # Replace with zero for simplicity

            masked_high_res_latents_input = masked_high_res_latents_seq.permute(0, 2, 1).view(B, model.latent_channels, model.high_res_latent_size, model.high_res_latent_size)

            predicted_noise_high_res_latent = model(
                x_low_res=None, # Not used in phase 2
                x_high_res=masked_high_res_latents_input,
                t=timesteps,
                y=conditions, # Original conditions
                phase=2,
                low_res_pivots=conditional_tokens_low_res # Z^s from phase 1
            )

            # Loss calculation for Phase 2
            loss_phase2 = F.mse_loss(predicted_noise_high_res_latent, noise_high_res)

            # Total Loss (can be weighted, for now sum)
            loss = loss_phase1 + loss_phase2

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            pbar.set_description(f"Epoch {epoch} Loss: {loss.item():.4f}")
            if accelerator.is_main_process:
                accelerator.log({"loss": loss.item(), "loss_phase1": loss_phase1.item(), "loss_phase2": loss_phase2.item()}, step=epoch)

        # Save model checkpoint periodically
        if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save_state(os.path.join(args.output_dir, f"checkpoint_{epoch:04d}"))
            accelerator.save(unwrapped_model.state_dict(), os.path.join(args.output_dir, f"himar_model_{epoch:04d}.pt"))

@torch.no_grad()
def evaluate(args, model, dataloader, accelerator, diffusion_scheduler):
    model.eval()
    total_fid = 0
    total_is = 0
    num_batches = 0

    # In a real evaluation, you'd generate a large number of images (e.g., 50K)
    # and use dedicated FID/IS calculation libraries.
    # This is a placeholder for the evaluation loop.
    for batch in tqdm(dataloader, disable=not accelerator.is_main_process):
        high_res_images = batch["high_res_image"]
        low_res_images = batch["low_res_image"]
        
        if args.dataset == "imagenet":
            conditions = batch["label"]
        elif args.dataset == "mscoco":
            conditions = batch["text_input_ids"]
        
        B = high_res_images.shape[0]

        # Simplified inference loop:
        # 1. Generate low-res pivots
        # 2. Generate high-res image conditioned on pivots

        # Initial noise for low-res generation
        # N_low = model.low_res_latent_size * model.low_res_latent_size
        # latent_shape_low_res = (B, model.latent_channels, model.low_res_latent_size, model.low_res_latent_size)
        # current_latents_low_res = torch.randn(latent_shape_low_res, device=accelerator.device)

        # Low-res generation (Phase 1 inference) - for simplicity, using ground truth low_res_images
        # In actual inference, we would start with pure noise and predict noise iteratively.
        # Here we're using ground_truth for evaluation setup.
        
        # To mimic inference, we would take `current_latents_low_res` and iterate `inference_steps_phase1` times,
        # predicting the noise, removing it, and optionally adding new noise.
        # For evaluation, we need to make sure this is done correctly, typically by
        # drawing 't' samples and running the diffusion reverse process.

        # To align with the paper's ablation on inference steps (Fig 4):
        # Phase 1: Fixed 32 steps, Phase 2: 4 steps.
        
        # Placeholder for actual inference logic which involves the reverse diffusion process
        # For now, let's just make dummy calls and assume a perfect generation.
        # This part requires a full diffusion sampler (DDPM, DDIM etc.) which is not defined here.
        # For faithful reproduction, the sampler would be implemented using the `BaseScheduler`.
        
        # Dummy inference:
        # Phase 1: Predict low-res pivots
        low_res_latents_from_vae = model.encode_vae(low_res_images, low_res=True)
        noise_lr = torch.randn_like(low_res_latents_from_vae)
        t_eval_lr = torch.randint(0, diffusion_scheduler.num_train_timesteps, (B,), device=accelerator.device).long()
        noisy_lr_latents = diffusion_scheduler.add_noise(low_res_latents_from_vae, noise_lr, t_eval_lr)
        
        predicted_noise_low_res_latent, conditional_tokens_low_res = model(
            x_low_res=noisy_lr_latents, # input to phase 1 transformer (can be noise)
            x_high_res=None,
            t=t_eval_lr,
            y=conditions,
            phase=1
        )
        # In a real diffusion sampler, we'd iteratively refine `noisy_lr_latents` using `predicted_noise_low_res_latent`
        # to get `generated_low_res_latents`.
        generated_low_res_latents = low_res_latents_from_vae # Placeholder: assuming perfect generation
        
        # Phase 2: Generate high-res image
        high_res_latents_from_vae = model.encode_vae(high_res_images, low_res=False)
        noise_hr = torch.randn_like(high_res_latents_from_vae)
        t_eval_hr = torch.randint(0, diffusion_scheduler.num_train_timesteps, (B,), device=accelerator.device).long()
        noisy_hr_latents = diffusion_scheduler.add_noise(high_res_latents_from_vae, noise_hr, t_eval_hr)
        
        predicted_noise_high_res_latent = model(
            x_low_res=None,
            x_high_res=noisy_hr_latents, # input to phase 2 transformer (can be noise)
            t=t_eval_hr,
            y=conditions,
            phase=2,
            low_res_pivots=conditional_tokens_low_res # Use Z^s from phase 1
        )
        # In a real diffusion sampler, we'd iteratively refine `noisy_hr_latents` using `predicted_noise_high_res_latent`
        generated_high_res_latents = high_res_latents_from_vae # Placeholder: assuming perfect generation
        
        # Decode to images (if needed for FID/IS)
        # generated_images = model.decode_vae(generated_high_res_latents)

        # Placeholder for FID/IS calculation
        # fid_score = calculate_fid(generated_images, real_images)
        # is_score = calculate_is(generated_images)
        fid_score = 0.0 # Dummy
        is_score = 0.0 # Dummy

        total_fid += fid_score
        total_is += is_score
        num_batches += 1

    avg_fid = total_fid / num_batches
    avg_is = total_is / num_batches
    
    if accelerator.is_main_process:
        accelerator.print(f"Evaluation - Avg FID: {avg_fid:.4f}, Avg IS: {avg_is:.4f}")
        accelerator.log({"avg_fid": avg_fid, "avg_is": avg_is})

def main():
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # For now, just run training. Evaluation will be integrated later.
    train()

if __name__ == "__main__":
    main()