from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False


class PerceptualLoss(nn.Module):
    """LPIPS perceptual loss."""

    def __init__(self, net: str = "vgg"):
        super().__init__()
        if not LPIPS_AVAILABLE:
            raise ImportError("lpips package required: pip install lpips")
        self.loss_fn = lpips.LPIPS(net=net)
        for param in self.loss_fn.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(x, y).mean()


class FRVAELoss(nn.Module):
    """
    Total loss for FR-VAE training (Appendix B.1):

    L = ||I - I_hat||_2^2 + ||f - f_tilde||_2^2 + L_p(I) + 0.5 * L_g(I)

    where:
    - ||I - I_hat||_2^2: pixel-space reconstruction loss
    - ||f - f_tilde||_2^2: feature-space frequency-guided quantization loss
    - L_p: LPIPS perceptual loss
    - L_g: GAN loss (hinge)
    """

    def __init__(
        self,
        rec_loss_weight: float = 1.0,
        freq_loss_weight: float = 1.0,
        perceptual_loss_weight: float = 1.0,
        gan_loss_weight: float = 0.5,
        codebook_loss_weight: float = 1.0,
        disc_start: int = 50001,
        use_perceptual: bool = True,
    ):
        super().__init__()
        self.rec_loss_weight = rec_loss_weight
        self.freq_loss_weight = freq_loss_weight
        self.perceptual_loss_weight = perceptual_loss_weight
        self.gan_loss_weight = gan_loss_weight
        self.codebook_loss_weight = codebook_loss_weight
        self.disc_start = disc_start

        if use_perceptual and LPIPS_AVAILABLE:
            self.perceptual = PerceptualLoss(net="vgg")
        else:
            self.perceptual = None

    def generator_loss(
        self,
        x: torch.Tensor,
        x_rec: torch.Tensor,
        f: torch.Tensor,
        f_tilde: torch.Tensor,
        vq_loss: torch.Tensor,
        disc_outputs: Optional[List[torch.Tensor]],
        global_step: int,
        last_layer_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute generator (VAE) loss.

        Args:
            x: (B, 3, H, W) original image
            x_rec: (B, 3, H, W) reconstructed image
            f: (B, C, H', W') original feature map
            f_tilde: (B, C, H', W') reconstructed feature map
            vq_loss: scalar VQ loss
            disc_outputs: list of discriminator logits on fake (x_rec)
            global_step: current training step
            last_layer_weight: decoder last layer weight for adaptive weighting

        Returns:
            total_loss, loss_dict
        """
        # Pixel reconstruction loss
        rec_loss = F.mse_loss(x_rec, x)

        # Feature-space frequency quantization loss
        freq_loss = F.mse_loss(f_tilde, f.detach())

        # Perceptual loss
        if self.perceptual is not None:
            perc_loss = self.perceptual(x_rec, x)
        else:
            perc_loss = torch.tensor(0.0, device=x.device)

        nll_loss = (
            self.rec_loss_weight * rec_loss
            + self.freq_loss_weight * freq_loss
            + self.perceptual_loss_weight * perc_loss
        )

        # GAN generator loss
        g_loss = torch.tensor(0.0, device=x.device)
        if disc_outputs is not None and global_step >= self.disc_start:
            for logits_fake in disc_outputs:
                g_loss = g_loss + (-logits_fake.mean())
            g_loss = g_loss / len(disc_outputs)

            # Adaptive weight
            if last_layer_weight is not None:
                try:
                    nll_grads = torch.autograd.grad(
                        nll_loss, last_layer_weight, retain_graph=True
                    )[0]
                    g_grads = torch.autograd.grad(
                        g_loss, last_layer_weight, retain_graph=True
                    )[0]
                    d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
                    d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
                except RuntimeError:
                    d_weight = torch.tensor(1.0, device=x.device)
            else:
                d_weight = torch.tensor(1.0, device=x.device)

            gan_weight = d_weight * self.gan_loss_weight
        else:
            gan_weight = torch.tensor(0.0, device=x.device)

        total_loss = nll_loss + self.codebook_loss_weight * vq_loss + gan_weight * g_loss

        loss_dict = {
            "rec_loss": rec_loss.item(),
            "freq_loss": freq_loss.item(),
            "perc_loss": perc_loss.item(),
            "vq_loss": vq_loss.item(),
            "g_loss": g_loss.item(),
            "total_loss": total_loss.item(),
        }
        return total_loss, loss_dict

    def discriminator_loss(
        self,
        disc_outputs_real: List[torch.Tensor],
        disc_outputs_fake: List[torch.Tensor],
        global_step: int,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute discriminator hinge loss.

        Args:
            disc_outputs_real: list of discriminator logits on real images
            disc_outputs_fake: list of discriminator logits on fake images
            global_step: current training step

        Returns:
            d_loss, loss_dict
        """
        if global_step < self.disc_start:
            d_loss = torch.tensor(0.0, device=disc_outputs_real[0].device)
            return d_loss, {"d_loss": 0.0}

        d_loss = torch.tensor(0.0, device=disc_outputs_real[0].device)
        for logits_real, logits_fake in zip(disc_outputs_real, disc_outputs_fake):
            loss_real = F.relu(1.0 - logits_real).mean()
            loss_fake = F.relu(1.0 + logits_fake).mean()
            d_loss = d_loss + 0.5 * (loss_real + loss_fake)
        d_loss = d_loss / len(disc_outputs_real)

        return d_loss, {"d_loss": d_loss.item()}


class NFIGTransformerLoss(nn.Module):
    """
    Cross-entropy loss for NFIG transformer training (Appendix B.1, Eq. 8).

    L(T, T_tilde) = -sum_i t_i * log(t_tilde_i)

    Computed between predicted logits and FR-VAE ground truth tokens.
    """

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        targets_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            logits: (B, total_tokens, vocab_size) predicted logits
            targets_list: list of (B, h_i*w_i) target token indices

        Returns:
            loss, loss_dict
        """
        # Concatenate all target tokens
        targets = torch.cat(targets_list, dim=1)  # (B, total_tokens)
        B, L, V = logits.shape
        loss = F.cross_entropy(
            logits.reshape(B * L, V),
            targets.reshape(B * L),
            label_smoothing=self.label_smoothing,
        )
        return loss, {"ce_loss": loss.item()}
