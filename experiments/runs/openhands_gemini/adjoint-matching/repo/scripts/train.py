
import torch
import torch.optim as optim
from tqdm import tqdm
import os

from adjoint_matching.config import Config
from adjoint_matching.models.model import FlowMatchingModel, RewardModel
from adjoint_matching.losses.adjoint_matching_loss import AdjointMatchingLoss
from adjoint_matching.data import get_dummy_dataloader

def train():
    config = Config()

    # Initialize models
    # The paper assumes a pre-trained base model.
    # For this reproduction, we'll initialize a dummy base model.
    base_model = FlowMatchingModel(config).to(config.DEVICE, config.DTYPE)
    # The model to be fine-tuned starts as a copy of the base model
    finetuned_model = FlowMatchingModel(config).to(config.DEVICE, config.DTYPE)
    finetuned_model.load_state_dict(base_model.state_dict())

    # Reward model (dummy for now)
    reward_model = RewardModel().to(config.DEVICE, config.DTYPE)

    # Optimizer for the fine-tuned model
    optimizer = optim.AdamW(
        finetuned_model.parameters(),
        lr=config.LEARNING_RATE,
        betas=(config.ADAM_BETA1, config.ADAM_BETA2),
        eps=config.ADAM_EPS,
        weight_decay=config.WEIGHT_DECAY
    )

    # Adjoint Matching Loss
    adjoint_matching_criterion = AdjointMatchingLoss(config, base_model, reward_model)

    # Dummy DataLoader
    # Assuming latent_dim from UNET_CONFIG's in_channels and a spatial size (e.g., 64x64)
    # The paper mentions 512x512 images, then latent variables.
    # So if latent_dim is 4 channels, 64x64, that's a common latent size.
    latent_dim_example = (config.UNET_CONFIG["in_channels"], 64, 64) 
    text_embed_dim_example = config.UNET_CONFIG["cross_attention_dim"]
    
    dataloader = get_dummy_dataloader(
        num_samples=config.NUM_PROMPTS_PER_EPOCH,
        latent_dim=latent_dim_example,
        text_embed_dim=text_embed_dim_example,
        batch_size=config.BATCH_SIZE
    )

    print(f"Starting training on {config.DEVICE} with {config.DTYPE} precision.")
    print(f"Total fine-tuning iterations: {config.NUM_FINE_TUNE_ITERATIONS}")

    finetuned_model.train()
    for iteration in tqdm(range(config.NUM_FINE_TUNE_ITERATIONS)):
        try:
            latents, text_embeds = next(data_iter)
        except (StopIteration, NameError):
            data_iter = iter(dataloader)
            latents, text_embeds = next(data_iter)

        latents = latents.to(config.DEVICE, config.DTYPE)
        text_embeds = text_embeds.to(config.DEVICE, config.DTYPE)

        optimizer.zero_grad()

        # Compute Adjoint Matching Loss
        loss = adjoint_matching_criterion(finetuned_model, latents, text_embeds)
        
        loss.backward()
        
        # Gradient norm clipping
        torch.nn.utils.clip_grad_norm_(finetuned_model.parameters(), config.GRADIENT_NORM_CLIPPING)
        
        optimizer.step()

        if (iteration + 1) % 100 == 0:
            print(f"Iteration {iteration + 1}/{config.NUM_FINE_TUNE_ITERATIONS}, Loss: {loss.item():.4f}")
            
    print("Fine-tuning complete.")
    
    # Save the fine-tuned model
    output_dir = "checkpoints"
    os.makedirs(output_dir, exist_ok=True)
    torch.save(finetuned_model.state_dict(), os.path.join(output_dir, "finetuned_flow_matching_model.pth"))
    print(f"Fine-tuned model saved to {os.path.join(output_dir, 'finetuned_flow_matching_model.pth')}")

if __name__ == '__main__':
    train()
