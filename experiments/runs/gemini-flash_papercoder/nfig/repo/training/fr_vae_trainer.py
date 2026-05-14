import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import lpips
import os
from omegaconf import OmegaConf # For saving DictConfig

# Assuming these are defined in models/fr_vae.py, models/discriminator.py, datasets.py, and config.py
from models.fr_vae import FRVAE
from models.discriminator import Discriminator
from datasets import DatasetLoader
from config import Config
from utils import set_seed # Assuming set_seed is in utils.py


class FRVAETrainer:
    """
    Manages the training loop for the Frequency-guided Residual-quantized VAE (FR-VAE)
    and its associated Discriminator in an adversarial setting.
    """

    def __init__(self, fr_vae: FRVAE, discriminator: Discriminator, config: Config, data_loader: DatasetLoader):
        """
        Initializes the FRVAETrainer.

        Args:
            fr_vae (FRVAE): The Frequency-guided Residual-quantized VAE model (generator).
            discriminator (Discriminator): The discriminator model.
            config (Config): Project configuration object.
            data_loader (DatasetLoader): Data loader utility.
        """
        self.fr_vae = fr_vae
        self.discriminator = discriminator
        self.config = config
        self.data_loader = data_loader

        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fr_vae.to(self.device)
        self.discriminator.to(self.device)

        # Initialize LPIPS model for perceptual loss
        # The 'alex' network is a common choice for LPIPS.
        self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
        # Ensure LPIPS model is in evaluation mode
        self.lpips_model.eval()

        # Optimizers
        self.optimizer_G = optim.Adam(
            self.fr_vae.parameters(),
            lr=self.config.fr_vae_training.learning_rate
        )
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.fr_vae_training.learning_rate
        )

        # Training parameters from config
        self.epochs: int = self.config.fr_vae_training.epochs
        self.batch_size: int = self.config.fr_vae_training.batch_size
        self.loss_weights: OmegaConf = self.config.fr_vae_training.loss_weights

        self.start_epoch: int = 0
        self.best_rfid: float = float('inf') # Track best reconstruction FID for saving model

        # Setup checkpoint directory
        self.checkpoint_dir: str = os.path.join("checkpoints", "fr_vae")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        print(f"FRVAETrainer initialized on device: {self.device}")

    def _compute_losses(self, recons_output: dict, images: torch.Tensor, d_out_real: torch.Tensor, d_out_fake: torch.Tensor, is_generator_step: bool) -> dict:
        """
        Computes all loss components for FR-VAE training based on the current training step
        (generator or discriminator).

        Args:
            recons_output (dict): Dictionary containing outputs from FRVAE's forward pass.
                                  Expected keys: 'hat_I', 'tilde_f', 'f', 'commit_loss_list'.
            images (torch.Tensor): Original input images (real images).
            d_out_real (torch.Tensor): Discriminator output for real images.
            d_out_fake (torch.Tensor): Discriminator output for fake (reconstructed) images.
            is_generator_step (bool): True if computing losses for the generator, False for discriminator.

        Returns:
            dict: A dictionary of computed loss values.
        """
        # --- Reconstruction Losses ---
        # L2 reconstruction loss for the image (||I - hat_I||_2^2)
        loss_recon_image_l2 = F.mse_loss(images, recons_output['hat_I'])

        # L2 reconstruction loss for the feature map (||f - tilde_f||_2^2)
        loss_recon_feature_l2 = F.mse_loss(recons_output['f'], recons_output['tilde_f'])

        # Perceptual loss (LPIPS)
        # lpips_model expects inputs in [-1, 1], which our image_tensor and hat_I are.
        loss_perceptual = self.lpips_model(images, recons_output['hat_I']).mean()

        # Codebook loss (sum of commitment losses for each band)
        loss_codebook = sum(cb_loss for cb_loss in recons_output['commit_loss_list'])

        # --- GAN Losses (Hinge Loss for both D and G) ---
        if is_generator_step:
            # Generator wants fake images to be classified as real by the discriminator
            # (i.e., maximize D's output for fake images, thus minimize -d_out_fake.mean())
            loss_gan_generator = -d_out_fake.mean()
            
            total_generator_loss = (
                self.loss_weights.recon_image_L2 * loss_recon_image_l2 +
                self.loss_weights.recon_feature_L2 * loss_recon_feature_l2 +
                self.loss_weights.perceptual_loss * loss_perceptual +
                self.loss_weights.gan_loss_generator * loss_gan_generator +
                self.loss_weights.codebook_loss_beta * loss_codebook
            )
            return {
                'total_generator_loss': total_generator_loss,
                'recon_image_l2': loss_recon_image_l2.item(),
                'recon_feature_l2': loss_recon_feature_l2.item(),
                'perceptual': loss_perceptual.item(),
                'gan_generator': loss_gan_generator.item(),
                'codebook': loss_codebook.item()
            }
        else: # Discriminator step
            # Discriminator wants real images classified as real (d_out_real >= 1)
            # and fake images classified as fake (d_out_fake <= -1)
            loss_gan_discriminator = (
                F.relu(1.0 - d_out_real).mean() +  # Maximize d_out_real (make it > 1)
                F.relu(1.0 + d_out_fake).mean()    # Minimize d_out_fake (make it < -1)
            )
            return {
                'total_discriminator_loss': loss_gan_discriminator,
                'gan_discriminator': loss_gan_discriminator.item(),
                'd_out_real_mean': d_out_real.mean().item(),
                'd_out_fake_mean': d_out_fake.mean().item()
            }

    def _train_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> dict:
        """
        Performs a single training step for one batch, including discriminator and generator updates.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): A batch containing image tensors and their labels.
                                                     Labels are typically not used directly for FR-VAE's
                                                     forward pass but are part of the DataLoader output.

        Returns:
            dict: A dictionary of loss values for the current step.
        """
        images, _ = batch # We assume FR-VAE training is not directly class-conditional
        images = images.to(self.device)

        # --- Train Discriminator ---
        self.optimizer_D.zero_grad()
        self.discriminator.train()
        self.fr_vae.eval() # Keep FRVAE in eval mode during D training for stable output

        with torch.no_grad():
            recons_output_det = self.fr_vae(images)
        reconstructed_image_det = recons_output_det['hat_I'].detach() # Detach reconstructed image for D training

        d_out_real = self.discriminator(images.contiguous())
        d_out_fake = self.discriminator(reconstructed_image_det.contiguous())

        discriminator_losses = self._compute_losses(recons_output_det, images, d_out_real, d_out_fake, is_generator_step=False)
        total_discriminator_loss = discriminator_losses['total_discriminator_loss']
        
        total_discriminator_loss.backward()
        self.optimizer_D.step()

        # --- Train Generator (FRVAE) ---
        self.optimizer_G.zero_grad()
        self.fr_vae.train()
        self.discriminator.eval() # Keep Discriminator in eval mode for G training

        # Re-run FRVAE forward to ensure gradients flow for generator loss
        recons_output = self.fr_vae(images)
        
        # Discriminator output for reconstructed images (no detach, gradients flow to G)
        d_out_fake_for_gen = self.discriminator(recons_output['hat_I'].contiguous())

        generator_losses = self._compute_losses(recons_output, images, None, d_out_fake_for_gen, is_generator_step=True)
        total_generator_loss = generator_losses['total_generator_loss']
        
        total_generator_loss.backward()
        self.optimizer_G.step()
        
        # Combine and return all losses for logging.
        # Convert total loss tensors to scalar for reporting.
        step_losses = {**discriminator_losses, **generator_losses}
        step_losses['total_discriminator_loss'] = total_discriminator_loss.item()
        step_losses['total_generator_loss'] = total_generator_loss.item()
        
        return step_losses

    def train(self):
        """
        Runs the main training loop for the FR-VAE.
        """
        print(f"Starting FR-VAE training for {self.epochs} epochs.")
        train_dataloader = self.data_loader.get_train_dataloader(
            image_size=self.config.data.image_size,
            batch_size=self.batch_size,
            is_conditional=True # ImageNetDataset returns labels, even if not used directly by FR-VAE
        )
        
        for epoch in range(self.start_epoch, self.epochs):
            self.fr_vae.train()
            self.discriminator.train()
            
            # Initialize accumulators for average epoch losses
            epoch_losses_aggregator = {
                'total_discriminator_loss': [], 'gan_discriminator': [], 'd_out_real_mean': [], 'd_out_fake_mean': [],
                'total_generator_loss': [], 'recon_image_l2': [], 'recon_feature_l2': [], 'perceptual': [],
                'gan_generator': [], 'codebook': []
            }

            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{self.epochs}")
            for batch_idx, batch in enumerate(pbar):
                step_losses = self._train_step(batch)

                for k, v in step_losses.items():
                    if k in epoch_losses_aggregator:
                        epoch_losses_aggregator[k].append(v)
                
                # Update progress bar description with current losses
                pbar_postfix = {
                    'D_loss': f"{step_losses['total_discriminator_loss']:.3f}",
                    'G_loss': f"{step_losses['total_generator_loss']:.3f}",
                    'R_L2': f"{step_losses['recon_image_l2']:.3f}",
                    'Percep': f"{step_losses['perceptual']:.3f}"
                }
                pbar.set_postfix(pbar_postfix)

            # Log average epoch losses
            avg_epoch_losses = {k: sum(v) / len(v) for k, v in epoch_losses_aggregator.items() if v}
            print(f"\nEpoch {epoch+1} finished. Average losses:")
            for k, v in avg_epoch_losses.items():
                print(f"  {k}: {v:.4f}")

            # Save checkpoint
            checkpoint_path = os.path.join(self.checkpoint_dir, f"fr_vae_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'fr_vae_state_dict': self.fr_vae.state_dict(),
                'discriminator_state_dict': self.discriminator.state_dict(),
                'optimizer_G_state_dict': self.optimizer_G.state_dict(),
                'optimizer_D_state_dict': self.optimizer_D.state_dict(),
                'best_rfid': self.best_rfid, # Current best rFID (to be updated by Evaluator if integrated)
                'config': OmegaConf.to_container(self.config._cfg, resolve=True) # Save raw config dict
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

            # TODO: If evaluating rFID during training, the logic to call an evaluator and
            # update self.best_rfid and save the best model would go here.
            # For this task, we assume evaluation happens as a separate phase in main.py.

        print("FR-VAE training complete.")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Loads a checkpoint to resume training or for evaluation.

        Args:
            checkpoint_path (str): Path to the checkpoint file.
        """
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found at {checkpoint_path}. Starting from scratch.")
            return

        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.fr_vae.load_state_dict(checkpoint['fr_vae_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        self.start_epoch = checkpoint['epoch']
        self.best_rfid = checkpoint.get('best_rfid', float('inf')) # Safely get best_rfid

        # Optionally load config from checkpoint, though typically config is fixed
        # loaded_config_dict = checkpoint.get('config')
        # if loaded_config_dict:
        #     self.config._cfg = OmegaConf.create(loaded_config_dict)

        print(f"Loaded checkpoint from {checkpoint_path}. Resuming from epoch {self.start_epoch + 1}.")

