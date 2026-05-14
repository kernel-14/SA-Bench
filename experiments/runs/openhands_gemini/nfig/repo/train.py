
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
from tqdm import tqdm
import lpips # For LPIPS perceptual loss

from nfig.config import get_config
from nfig.data import ImageNetDataset
from nfig.model import NFIGModel, FR_VAE
from nfig.modules import Discriminator

def train():
    config = get_config()

    # Set device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    train_dataset = ImageNetDataset(
        root=config.data_path,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train=True
    )
    train_dataloader = train_dataset.get_dataloader()

    # Initialize model
    model = NFIGModel(config).to(device)
    fr_vae_discriminator = Discriminator().to(device)

    # Optimizers
    optimizer_fr_vae = optim.Adam(model.fr_vae.parameters(), lr=config.learning_rate)
    optimizer_discriminator = optim.Adam(fr_vae_discriminator.parameters(), lr=config.learning_rate)
    optimizer_transformer = optim.Adam(model.nfig_transformer.parameters(), lr=config.learning_rate)

    # Loss functions
    l1_loss = nn.L1Loss().to(device) # For image reconstruction loss
    lpips_loss_fn = lpips.LPIPS(net='alex').to(device) # Perceptual loss
    mse_loss = nn.MSELoss().to(device) # For latent feature reconstruction and GAN

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, 'checkpoints'), exist_ok=True)

    # Training loop
    print("Starting training...")
    for epoch in range(config.epochs):
        model.train()
        fr_vae_discriminator.train()

        total_fr_vae_loss = 0
        total_transformer_loss = 0
        total_discriminator_loss = 0

        for batch_idx, (images, labels) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.epochs}")):
            images = images.to(device)
            labels = labels.to(device)

            # --- Train FR-VAE ---
            optimizer_fr_vae.zero_grad()
            x_recon, q_losses, _, f_latent, tilde_f_latent = model.fr_vae(images)

            # Reconstruction Loss: ||I - I_hat||_2^2 + ||f - f_hat||_2^2
            recon_pixel_loss = mse_loss(x_recon, images)
            recon_latent_loss = mse_loss(tilde_f_latent, f_latent) # Or l1_loss, paper says L2
            
            # LPIPS Perceptual Loss: L_p(I)
            perceptual_loss = lpips_loss_fn(x_recon, images).mean()

            # Quantization Loss (commitment loss from VQ)
            q_loss_sum = sum(q_losses)

            # Generator (FR-VAE) part of GAN loss
            fake_logits = fr_vae_discriminator(x_recon)
            generator_gan_loss = mse_loss(fake_logits, torch.ones_like(fake_logits))

            # Total FR-VAE Loss
            lambda_pixel = 1.0
            lambda_latent = 1.0
            lambda_perceptual = 1.0
            lambda_gan = 0.5 # As per paper's 0.5 * L_g(I)

            fr_vae_loss = (
                lambda_pixel * recon_pixel_loss +
                lambda_latent * recon_latent_loss +
                lambda_perceptual * perceptual_loss +
                lambda_gan * generator_gan_loss +
                q_loss_sum # Commitment loss is usually added directly
            )
            
            fr_vae_loss.backward()
            optimizer_fr_vae.step()
            total_fr_vae_loss += fr_vae_loss.item()

            # --- Train Discriminator ---
            optimizer_discriminator.zero_grad()

            # Real loss
            real_logits = fr_vae_discriminator(images.detach())
            loss_real = mse_loss(real_logits, torch.ones_like(real_logits))

            # Fake loss
            fake_logits = fr_vae_discriminator(x_recon.detach()) # Detach x_recon
            loss_fake = mse_loss(fake_logits, torch.zeros_like(fake_logits))

            discriminator_loss = 0.5 * (loss_real + loss_fake) # Standard GAN discriminator loss
            
            discriminator_loss.backward()
            optimizer_discriminator.step()
            total_discriminator_loss += discriminator_loss.item()

            # --- Train NFIG Transformer ---
            optimizer_transformer.zero_grad()
            
            # Use the NFIGModel forward pass to get tokens and transformer logits
            # model.forward expects images and class_labels
            _, _, transformer_logits, transformer_input_tokens, _, _ = model(images, labels)
            
            target_tokens = transformer_input_tokens[:, 1:] # Target is the next token in sequence
            
            # Cross-entropy loss for token prediction
            # Reshape logits and target for CrossEntropyLoss
            transformer_logits = rearrange(transformer_logits, 'b n c -> (b n) c')
            target_tokens = rearrange(target_tokens, 'b n -> (b n)')

            ce_loss = F.cross_entropy(transformer_logits, target_tokens)
            total_transformer_loss += ce_loss.item()

            ce_loss.backward()
            optimizer_transformer.step()
            
            if (batch_idx + 1) % config.log_interval == 0:
                print(f"  Batch {batch_idx+1}/{len(train_dataloader)}: "
                      f"FR-VAE Loss: {fr_vae_loss.item():.4f}, "
                      f"D Loss: {discriminator_loss.item():.4f}, "
                      f"T Loss: {ce_loss.item():.4f}")

        avg_fr_vae_loss = total_fr_vae_loss / len(train_dataloader)
        avg_discriminator_loss = total_discriminator_loss / len(train_dataloader)
        avg_transformer_loss = total_transformer_loss / len(train_dataloader)

        print(f"Epoch {epoch+1} finished. "
              f"Avg FR-VAE Loss: {avg_fr_vae_loss:.4f}, "
              f"Avg D Loss: {avg_discriminator_loss:.4f}, "
              f"Avg T Loss: {avg_transformer_loss:.4f}")

        # Save generated images
        if (epoch + 1) % config.save_interval == 0:
            model.eval()
            with torch.no_grad():
                # Generate a few sample images using the NFIGModel's inference method
                # Generate a few sample images
                num_generate_samples = 4
                # Create dummy class labels for generation
                # In a real scenario, these would be specific classes we want to generate.
                class_labels_for_generation = torch.randint(0, config.num_classes, (num_generate_samples,), device=device)
                
                generated_images = model.generate_image(
                    num_samples=num_generate_samples,
                    class_label_input=class_labels_for_generation,
                    temperature=1.0,
                    cfg_scale=config.cfg_scale,
                    top_k=config.top_k,
                    device=device
                )
                
                # Denormalize images for saving
                generated_images = (generated_images * 0.5) + 0.5
                save_image(generated_images, os.path.join(config.output_dir, 'images', f'epoch_{epoch+1}.png'))
                
                # Save model checkpoints
                torch.save(model.state_dict(), os.path.join(config.output_dir, 'checkpoints', f'nfig_model_epoch_{epoch+1}.pth'))
                torch.save(fr_vae_discriminator.state_dict(), os.path.join(config.output_dir, 'checkpoints', f'discriminator_epoch_{epoch+1}.pth'))
            model.train() # Set back to train mode

if __name__ == '__main__':
    train()
