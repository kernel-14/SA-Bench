## utils/losses.py
"""Loss functions for the NFIG framework.

Centralizes all loss computations for both training phases:
  - FR-VAE training: reconstruction, frequency quantization, perceptual, GAN, commitment
  - NFIG Transformer training: cross-entropy over predicted token sequences

Paper reference (Appendix B.1):
    L = ||I - Î||₂² + ||f̂ - f̂_reconstructed||₂² + L_p(I) + 0.5 * L_g(I)

Transformer loss (Appendix B.1):
    L(T, T̃) = -Σᵢ tᵢ log(t̃ᵢ)

Config values used:
    frvae.gan_loss_weight:  0.5   (weight for GAN adversarial loss)
    frvae.lpips_weight:     1.0   (weight for LPIPS perceptual loss)
    frvae.commitment_loss_weight: 0.25  (applied by caller/trainer)
    frvae.reconstruction_weight:  1.0   (applied by caller/trainer)
    frvae.freq_quantization_weight: 1.0 (applied by caller/trainer)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    import lpips as lpips_lib
except ImportError as exc:
    raise ImportError(
        "lpips is required for perceptual loss computation. "
        "Install it with: pip install lpips>=0.1.4"
    ) from exc


class NFIGLosses(nn.Module):
    """Centralized loss module for FR-VAE and NFIG Transformer training.

    Implements all loss terms described in the paper. Inherits from nn.Module
    so that the LPIPS network (itself an nn.Module) is properly registered and
    moves with .to(device) calls from the trainer.

    The GAN weight (0.5) and LPIPS weight (1.0) are applied internally by the
    relevant methods. All other weights (reconstruction=1.0, freq_quantization=1.0,
    commitment=0.25) are applied externally by the trainer when assembling the
    total loss, keeping this class focused on per-term computation.

    Total FR-VAE loss assembled by FRVAETrainer:
        L_total = reconstruction_weight * reconstruction_loss(x, x_hat)
                + freq_quantization_weight * frequency_quantization_loss(f, f_tilde)
                + perceptual_loss(x, x_hat)          # lpips_weight applied here
                + gan_generator_loss(fake_logits)     # gan_weight applied here
                + commitment_loss_weight * commitment_loss(v, v_q)

    Attributes:
        lpips_fn: Frozen VGG-based LPIPS perceptual loss network.
        gan_weight: Weight for GAN adversarial loss (config.frvae.gan_loss_weight = 0.5).
        lpips_weight: Weight for LPIPS perceptual loss (config.frvae.lpips_weight = 1.0).
    """

    def __init__(
        self,
        gan_weight: float = 0.5,
        lpips_weight: float = 1.0,
    ) -> None:
        """Initialize all loss components.

        Args:
            gan_weight: Multiplicative weight applied to the GAN generator loss.
                From config.frvae.gan_loss_weight = 0.5.
                Explicitly stated in paper Appendix B.1: "0.5 * L_g(I)".
            lpips_weight: Multiplicative weight applied to the LPIPS perceptual loss.
                From config.frvae.lpips_weight = 1.0.
                The paper includes L_p(I) with implicit weight 1.0.

        Raises:
            ImportError: If the lpips package is not installed.
        """
        super().__init__()

        # Store loss weights from config.
        # gan_weight = 0.5: explicitly stated in paper Appendix B.1 ("0.5 * L_g(I)")
        # lpips_weight = 1.0: implicit weight 1.0 in paper formula
        self.gan_weight: float = gan_weight
        self.lpips_weight: float = lpips_weight

        # --- LPIPS perceptual loss network ---
        # VGG-based LPIPS is standard for VQ-GAN style training.
        # The network is frozen (eval mode, no gradient updates) — it is a
        # fixed perceptual metric, not a trainable component.
        # Registered as a submodule so it moves with .to(device) calls.
        self.lpips_fn: lpips_lib.LPIPS = lpips_lib.LPIPS(net="vgg")

        # Freeze LPIPS parameters: no gradient computation through the metric network.
        # Gradients flow through the input x_hat, not through LPIPS weights.
        self.lpips_fn.eval()
        for param in self.lpips_fn.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True) -> "NFIGLosses":
        """Override train() to always keep LPIPS in eval mode.

        The LPIPS network must remain in eval mode regardless of whether the
        losses module is in training or eval mode. This prevents any BatchNorm
        or Dropout layers in VGG from switching to training behavior.

        Args:
            mode: If True, sets the module to training mode. If False, sets
                it to evaluation mode. LPIPS always stays in eval().

        Returns:
            self (for method chaining, consistent with nn.Module.train()).
        """
        super().train(mode)
        # Always force LPIPS back to eval mode.
        self.lpips_fn.eval()
        return self

    def reconstruction_loss(self, x: Tensor, x_hat: Tensor) -> Tensor:
        """Compute pixel-space L2 reconstruction loss: ||I - Î||₂²

        Measures the mean squared error between the original image and its
        reconstruction in pixel space. Both inputs are normalized to [-1, 1]
        per config.yaml (mean=std=0.5).

        Paper formula (Appendix B.1): first term ||I - Î||₂²

        Args:
            x: Original image batch of shape (B, 3, H, W), values in [-1, 1].
               From the DataLoader with config normalization applied.
            x_hat: Reconstructed image batch of shape (B, 3, H, W), values in [-1, 1].
               Output of VQGANDecoder.forward() via FRVAE.forward().

        Returns:
            Scalar tensor: mean squared error averaged over all elements.
            Gradients flow through x_hat → decoder → quantizer → encoder.
        """
        return F.mse_loss(x_hat, x, reduction="mean")

    def frequency_quantization_loss(
        self,
        f_hat: Tensor,
        f_reconstructed: Tensor,
    ) -> Tensor:
        """Compute feature-space L2 loss for frequency-guided quantization.

        Measures the mean squared error between the original latent feature map
        (encoder output) and its quantized reconstruction (composed from all
        frequency-band quantized representations). This loss operates in the
        latent feature space (C=768 channels per config) rather than pixel space.

        Paper formula (Appendix B.1): second term ||f̂ - f̂_reconstructed||₂²

        Args:
            f_hat: Original latent feature map of shape (B, C, H', W').
                Output of VQGANEncoder.forward() before quantization.
                For default config: (B, 768, 16, 16).
            f_reconstructed: Composed quantized feature map of shape (B, C, H', W').
                f_tilde = Σᵢ T(vᵢ^q, H', W') from ResidualQuantizer.
                For default config: (B, 768, 16, 16).

        Returns:
            Scalar tensor: mean squared error in feature space.
            Gradients flow through f_reconstructed → quantizer (straight-through)
            → encoder.
        """
        return F.mse_loss(f_reconstructed, f_hat, reduction="mean")

    def perceptual_loss(self, x: Tensor, x_hat: Tensor) -> Tensor:
        """Compute LPIPS perceptual loss L_p(I) with stored weight.

        Uses a frozen VGG-based LPIPS network to measure perceptual similarity
        between the original and reconstructed images. LPIPS captures high-level
        semantic and structural differences that pixel-wise MSE misses.

        Paper formula (Appendix B.1): third term L_p(I)
        Config: frvae.lpips_weight = 1.0

        LPIPS expects inputs in [-1, 1], which matches our normalization
        (config.data.mean=0.5, config.data.std=0.5 → [-1, 1] range).
        No rescaling is needed.

        Args:
            x: Original image batch of shape (B, 3, H, W), values in [-1, 1].
            x_hat: Reconstructed image batch of shape (B, 3, H, W), values in [-1, 1].

        Returns:
            Scalar tensor: lpips_weight * mean LPIPS score over the batch.
            Gradients flow through x_hat only; LPIPS parameters are frozen.

        Note:
            The lpips_fn must be on the same device as x and x_hat.
            The trainer is responsible for calling losses.to(device) after
            instantiation to ensure device alignment.
        """
        # lpips_fn returns shape [B, 1, 1, 1] — per-image perceptual distances.
        # .mean() averages over the batch dimension.
        perceptual_scores: Tensor = self.lpips_fn(x_hat, x)
        return self.lpips_weight * perceptual_scores.mean()

    def gan_generator_loss(self, fake_logits: Tensor) -> Tensor:
        """Compute the generator's adversarial loss with stored GAN weight.

        The generator wants the discriminator to classify reconstructed/generated
        images as real. Uses the non-saturating (hinge) GAN formulation:
            L_gen = -mean(fake_logits)

        This is equivalent to minimizing -log(D(G(z))) in the non-saturating
        formulation, and is the standard generator loss for VQ-GAN style training.

        Paper formula (Appendix B.1): fourth term 0.5 * L_g(I)
        Config: frvae.gan_loss_weight = 0.5

        Args:
            fake_logits: Discriminator logits on generated/reconstructed images.
                Shape: (B, num_patches) from DINODiscriminator.forward().
                These are raw (un-activated) logits; higher values indicate
                the discriminator judges the patch as real.
                Must NOT be detached — gradients flow through to the generator.

        Returns:
            Scalar tensor: gan_weight * (-mean(fake_logits)).
            Gradients flow through fake_logits → discriminator (frozen during
            generator step) → x_hat → decoder → quantizer → encoder.
        """
        # Non-saturating generator loss: maximize discriminator output on fakes.
        # Equivalent to minimizing -log(sigmoid(fake_logits)) in the limit.
        generator_loss: Tensor = -fake_logits.mean()
        return self.gan_weight * generator_loss

    def gan_discriminator_loss(
        self,
        real_logits: Tensor,
        fake_logits: Tensor,
    ) -> Tensor:
        """Compute the discriminator's hinge adversarial loss.

        The discriminator wants to correctly classify real images as real
        (logits > 1) and fake/reconstructed images as fake (logits < -1).
        Uses the hinge loss formulation standard for VQ-GAN discriminators:

            L_real = mean(ReLU(1 - real_logits))   # penalize real logits < 1
            L_fake = mean(ReLU(1 + fake_logits))   # penalize fake logits > -1
            L_disc = 0.5 * (L_real + L_fake)

        The discriminator loss does NOT apply gan_weight — that weight is for
        the generator's use of the adversarial signal. The discriminator is
        trained with its own optimizer (optimizer_d in FRVAETrainer).

        Args:
            real_logits: Discriminator logits on real images from the DataLoader.
                Shape: (B, num_patches) from DINODiscriminator.forward().
                Gradients flow through these to update the discriminator head.
            fake_logits: Discriminator logits on reconstructed images.
                Shape: (B, num_patches) from DINODiscriminator.forward().
                Must be detached from the generator graph in the trainer
                (x_hat.detach() before passing to discriminator).
                Gradients flow through these to update the discriminator head.

        Returns:
            Scalar tensor: 0.5 * (hinge_real_loss + hinge_fake_loss).
            Gradients flow through both real_logits and fake_logits to update
            the discriminator head parameters.
        """
        # Hinge loss for real images: penalize when real_logits < 1.
        # ReLU(1 - real_logits): zero when real_logits >= 1, positive otherwise.
        real_loss: Tensor = F.relu(1.0 - real_logits).mean()

        # Hinge loss for fake images: penalize when fake_logits > -1.
        # ReLU(1 + fake_logits): zero when fake_logits <= -1, positive otherwise.
        fake_loss: Tensor = F.relu(1.0 + fake_logits).mean()

        # Standard hinge discriminator loss: average of real and fake components.
        discriminator_loss: Tensor = 0.5 * (real_loss + fake_loss)

        return discriminator_loss

    def commitment_loss(self, v: Tensor, v_q: Tensor) -> Tensor:
        """Compute the VQ commitment loss (encoder → codebook alignment).

        Encourages the encoder outputs to stay close to their assigned codebook
        entries. The codebook entry is treated as a fixed target (detached) so
        gradients only flow through the encoder output v.

        Standard VQ-VAE commitment loss (Oord et al. 2017):
            L_commit = ||v - sg(v_q)||²

        where sg(·) is the stop-gradient operator (v_q.detach()).

        The commitment loss weight beta = 0.25 (config.frvae.commitment_loss_weight)
        is applied externally by the trainer when assembling the total loss.
        This method returns the unweighted MSE to keep the weight application
        explicit and configurable in the trainer.

        Note on codebook loss: The codebook loss ||sg(v) - v_q||² is handled
        separately by ResidualQuantizer.get_codebook_loss() which combines both
        terms. This method provides the commitment component for cases where
        the trainer wants to compute it independently.

        Args:
            v: Continuous encoder output (pre-quantization) at scale resolution.
                Shape: (B, C, h_i, w_i) for frequency band i.
                Gradients flow through this tensor to update the encoder.
            v_q: Quantized representation (codebook lookup result).
                Shape: (B, C, h_i, w_i), same as v.
                Treated as a fixed target — detached inside this method.
                Can be either the raw quantized tensor or the STE version;
                the detach() call inside ensures correct gradient behavior.

        Returns:
            Scalar tensor: mean squared error between v and detached v_q.
            Gradients flow through v only (encoder update direction).
            The caller (FRVAETrainer) multiplies by commitment_loss_weight=0.25.
        """
        # Detach v_q to prevent gradients from flowing to the codebook through
        # this loss term. The codebook is updated via the codebook loss term
        # (||sg(v) - v_q||²) in ResidualQuantizer.get_codebook_loss().
        return F.mse_loss(v, v_q.detach(), reduction="mean")

    def transformer_ce_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute cross-entropy loss for NFIG Transformer token prediction.

        Implements the transformer training loss from paper Appendix B.1:
            L(T, T̃) = -Σᵢ tᵢ log(t̃ᵢ)

        This is the standard cross-entropy loss over the codebook vocabulary
        (K=4096 entries per config.frvae.codebook_size) for all token positions
        in the frequency-ordered sequence (total_tokens=680 per config).

        F.cross_entropy applies log-softmax internally for numerical stability,
        which is equivalent to the paper's formula when tᵢ is a one-hot target.

        Args:
            logits: Raw (unnormalized) logits from the transformer output head.
                Shape: (B, total_tokens, K) where:
                  - B: batch size
                  - total_tokens: 680 (config.nfig.total_tokens = 680)
                  - K: 4096 (config.nfig.codebook_size = 4096)
                Gradients flow through these to update the transformer.
            targets: Ground-truth token indices from FR-VAE encoding.
                Shape: (B, total_tokens), dtype torch.long.
                Values in [0, K-1] = [0, 4095].
                Produced by NFIGTrainer._tokenize_batch() via FRVAE.get_tokens().

        Returns:
            Scalar tensor: mean cross-entropy loss over all (B * total_tokens)
            token positions. Gradients flow through logits → transformer head
            → transformer blocks → token/positional embeddings.

        Raises:
            RuntimeError: If logits and targets have incompatible shapes
                (propagated from F.cross_entropy).
        """
        # Retrieve vocabulary size from logits for reshape.
        # logits: (B, total_tokens, K) → (B * total_tokens, K)
        # targets: (B, total_tokens) → (B * total_tokens,)
        batch_size: int = logits.shape[0]
        total_tokens: int = logits.shape[1]
        vocab_size: int = logits.shape[2]  # K = 4096

        # Flatten batch and sequence dimensions for F.cross_entropy.
        # F.cross_entropy expects (N, C) logits and (N,) targets.
        logits_flat: Tensor = logits.reshape(batch_size * total_tokens, vocab_size)
        targets_flat: Tensor = targets.reshape(batch_size * total_tokens)

        # Standard cross-entropy with mean reduction.
        # F.cross_entropy applies log-softmax internally (numerically stable).
        # No label smoothing — the paper does not mention it.
        # No ignore_index — all 680 token positions have valid targets.
        ce_loss: Tensor = F.cross_entropy(
            logits_flat,
            targets_flat,
            reduction="mean",
        )

        return ce_loss

    def extra_repr(self) -> str:
        """Return a human-readable string with key loss configuration.

        Returns:
            String describing the loss weights and LPIPS network type.
        """
        return (
            f"gan_weight={self.gan_weight}, "
            f"lpips_weight={self.lpips_weight}, "
            f"lpips_net='vgg'"
        )
