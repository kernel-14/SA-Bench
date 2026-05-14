## models/vae_tokenizer.py
"""VAE tokenizer wrapper for Hi-MAR.

This module provides ``VAETokenizer``, a frozen wrapper around the pre-trained
KL-16 variational autoencoder from MAR. It bridges pixel space and the
continuous latent token space that Hi-MAR operates in.

The VAE is **never trained** — all parameters are frozen from construction.
All public methods are decorated with ``@torch.no_grad()`` as a safety
guarantee, but callers should also wrap invocations in ``torch.no_grad()``
contexts for clarity.

Configuration alignment (config.yaml → vae section):
    vae.ckpt              = "pretrained/kl16.ckpt"
    vae.scale_factor      = 0.2325
    vae.latent_channels   = 16
    vae.downsample_factor = 16
    vae.freeze            = true

Resolution alignment (config.yaml → resolution section):
    resolution.high_res   = 256  →  latent: 16×16 = 256 tokens
    resolution.low_res    = 128  →  latent:  8×8  =  64 tokens
    resolution.hr_seq_len = 256
    resolution.lr_seq_len = 64

Paper reference (Section 4.2):
    "We employ the variational autoencoder (KL-16 version) trained by MAR to
    encode low-resolution (128×128) and high-resolution (256×256) images into
    latent representations for the two phases."
"""

import os
from typing import Tuple

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL


class VAETokenizer:
    """Frozen KL-16 VAE wrapper that converts images to/from latent token sequences.

    This class is a leaf node in the project dependency graph — it imports
    nothing from other project modules. It is instantiated once and shared
    across the Trainer, Generator, and Evaluator.

    The KL-16 VAE has a spatial downsampling factor of 16, so:
        - 256×256 images → 16×16 latent → 256 tokens  (Phase 2, high-res)
        - 128×128 images →  8×8 latent →  64 tokens  (Phase 1, low-res)

    All latents are multiplied by ``scale_factor = 0.2325`` during encoding
    and divided during decoding. This normalises the latent distribution to
    approximately unit variance, which is required for the diffusion loss to
    function correctly. This constant must be consistent across all usages.

    Attributes:
        vae: The underlying ``AutoencoderKL`` model, frozen and in eval mode.
        device: Compute device on which the VAE resides.
        scale_factor: Latent scaling constant (0.2325 per MAR / config).
        latent_channels: Number of latent channels (16 for KL-16 per config).
    """

    def __init__(
        self,
        vae_ckpt: str = "pretrained/kl16.ckpt",
        device: torch.device = torch.device("cpu"),
        scale_factor: float = 0.2325,
        latent_channels: int = 16,
    ) -> None:
        """Loads the KL-16 VAE, moves it to device, and freezes all parameters.

        Supports two checkpoint formats:
        1. **Directory / HuggingFace model ID**: If ``vae_ckpt`` is a directory
           (or a HuggingFace Hub model ID), ``AutoencoderKL.from_pretrained()``
           is used directly.
        2. **Raw checkpoint file** (``.ckpt`` / ``.pt`` / ``.pth``): If
           ``vae_ckpt`` is a file path, the KL-16 architecture is instantiated
           with the correct config and the state dict is loaded manually.

        Args:
            vae_ckpt: Path to the MAR KL-16 VAE checkpoint. Can be a directory
                (diffusers format), a HuggingFace model ID, or a raw ``.ckpt``
                / ``.pt`` file. Config default: ``"pretrained/kl16.ckpt"``.
            device: Compute device for the VAE and all tensor operations.
                Should match the device used by the Transformer backbone.
            scale_factor: Latent scaling constant applied after encoding and
                inverted before decoding. Config: ``vae.scale_factor = 0.2325``.
            latent_channels: Number of latent channels in the KL-16 VAE.
                Config: ``vae.latent_channels = 16``.

        Raises:
            FileNotFoundError: If ``vae_ckpt`` is a file path that does not
                exist on disk.
            RuntimeError: If the checkpoint file cannot be loaded or the state
                dict is incompatible with the KL-16 architecture.
        """
        self.device: torch.device = device
        self.scale_factor: float = scale_factor
        self.latent_channels: int = latent_channels

        # ------------------------------------------------------------------
        # Load the VAE from the checkpoint.
        # ------------------------------------------------------------------
        self.vae: AutoencoderKL = self._load_vae(vae_ckpt)

        # Move to the target device.
        self.vae = self.vae.to(self.device)

        # Set to evaluation mode — disables dropout and batch norm training
        # behaviour. Critical since the VAE is never trained.
        self.vae.eval()

        # Freeze all parameters unconditionally. The VAE is architecturally
        # never trained in Hi-MAR regardless of config.vae.freeze.
        for param in self.vae.parameters():
            param.requires_grad = False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_vae(self, vae_ckpt: str) -> AutoencoderKL:
        """Loads the KL-16 VAE from a checkpoint path or model ID.

        Dispatches to the appropriate loading strategy based on whether
        ``vae_ckpt`` is a directory/model-ID or a raw checkpoint file.

        Args:
            vae_ckpt: Checkpoint path or HuggingFace model ID.

        Returns:
            Loaded ``AutoencoderKL`` instance (not yet moved to device).

        Raises:
            FileNotFoundError: If ``vae_ckpt`` is a file path that does not
                exist.
            RuntimeError: If the checkpoint cannot be loaded or the state dict
                is incompatible.
        """
        # ------------------------------------------------------------------
        # Strategy 1: Directory or HuggingFace model ID.
        # A path is treated as a directory if it either IS a directory on disk
        # or does not have a recognised checkpoint file extension (suggesting
        # it is a HuggingFace Hub model ID like "stabilityai/sd-vae-ft-mse").
        # ------------------------------------------------------------------
        checkpoint_extensions: tuple[str, ...] = (".ckpt", ".pt", ".pth", ".bin")
        is_file_path: bool = any(
            vae_ckpt.endswith(ext) for ext in checkpoint_extensions
        )

        if not is_file_path or os.path.isdir(vae_ckpt):
            # diffusers from_pretrained handles both local directories and
            # HuggingFace Hub model IDs transparently.
            try:
                vae: AutoencoderKL = AutoencoderKL.from_pretrained(vae_ckpt)
                return vae
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load VAE from pretrained path/ID '{vae_ckpt}'. "
                    f"Original error: {exc}"
                ) from exc

        # ------------------------------------------------------------------
        # Strategy 2: Raw checkpoint file (.ckpt / .pt / .pth / .bin).
        # Instantiate the KL-16 architecture and load the state dict manually.
        # ------------------------------------------------------------------
        if not os.path.isfile(vae_ckpt):
            raise FileNotFoundError(
                f"VAE checkpoint file not found: '{vae_ckpt}'. "
                "Please download the MAR KL-16 VAE checkpoint and place it at "
                "the path specified by config.vae.ckpt."
            )

        # KL-16 architecture configuration matching MAR's pre-trained VAE.
        # These values are fixed by the KL-16 architecture and must not be
        # changed — they are not user-configurable.
        vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            down_block_types=(
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
            ),
            up_block_types=(
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
            ),
            block_out_channels=(128, 256, 512, 512),
            layers_per_block=2,
            act_fn="silu",
            latent_channels=self.latent_channels,  # 16 for KL-16
            norm_num_groups=32,
            sample_size=256,
        )

        # Load the state dict from the checkpoint file.
        try:
            checkpoint: dict = torch.load(
                vae_ckpt,
                map_location="cpu",
                weights_only=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load checkpoint file '{vae_ckpt}'. "
                f"Original error: {exc}"
            ) from exc

        # Handle various checkpoint formats:
        # - Plain state dict: {"encoder.weight": ..., ...}
        # - Nested under "state_dict" key (common in Lightning checkpoints)
        # - Nested under "model" key
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict: dict = checkpoint["state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
        else:
            raise RuntimeError(
                f"Unexpected checkpoint format in '{vae_ckpt}'. "
                "Expected a dict with keys 'state_dict', 'model', or a plain "
                "state dict."
            )

        # Strip common prefixes that appear in some checkpoint formats.
        # e.g., "first_stage_model." prefix from LDM checkpoints.
        cleaned_state_dict: dict = {}
        prefixes_to_strip: tuple[str, ...] = (
            "first_stage_model.",
            "vae.",
            "module.",
        )
        for key, value in state_dict.items():
            clean_key: str = key
            for prefix in prefixes_to_strip:
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]
                    break
            cleaned_state_dict[clean_key] = value

        # Load with strict=False to tolerate minor key mismatches (e.g.,
        # missing non-essential buffers). Log any missing/unexpected keys.
        missing_keys: list[str]
        unexpected_keys: list[str]
        missing_keys, unexpected_keys = vae.load_state_dict(
            cleaned_state_dict, strict=False
        )

        if missing_keys:
            # Filter out known non-critical keys (e.g., loss-related weights
            # that are not part of the encoder/decoder).
            critical_missing: list[str] = [
                k for k in missing_keys
                if not any(
                    skip in k
                    for skip in ("loss", "discriminator", "lpips")
                )
            ]
            if critical_missing:
                raise RuntimeError(
                    f"Critical keys missing from VAE checkpoint '{vae_ckpt}': "
                    f"{critical_missing[:10]}{'...' if len(critical_missing) > 10 else ''}"
                )

        return vae

    def _to_float32(self, tensor: torch.Tensor) -> torch.Tensor:
        """Casts a tensor to float32 if it is not already.

        The KL-16 VAE may not be numerically stable in float16. This helper
        ensures inputs are always float32 before passing to the VAE, even
        when the rest of training uses mixed precision (torch.cuda.amp).

        Args:
            tensor: Input tensor of any dtype.

        Returns:
            The same tensor cast to ``torch.float32``.
        """
        if tensor.dtype != torch.float32:
            return tensor.float()
        return tensor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encodes a batch of images to a flat sequence of continuous latent tokens.

        The encoding pipeline is:
            images [B,3,H,W] → VAE encoder → posterior → .mode() →
            × scale_factor → reshape → tokens [B, h*w, latent_channels]

        The posterior mode (mean) is used rather than a stochastic sample to
        ensure stable, deterministic conditioning signals during training.
        This matches MAR's implementation.

        Input contract: Images must be in ``[-1, 1]`` range, as produced by
        the dataset transforms (``Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])``
        applied to ``[0, 1]`` images). No additional normalisation is applied here.

        Args:
            images: Float tensor of shape ``[B, 3, H, W]`` in ``[-1, 1]``
                range. ``H`` and ``W`` must be divisible by
                ``vae.downsample_factor = 16``.

        Returns:
            Float tensor of shape ``[B, h*w, latent_channels]`` where
            ``h = H // 16``, ``w = W // 16``. For ``H=W=256``: ``[B, 256, 16]``.
            For ``H=W=128``: ``[B, 64, 16]``.
        """
        # Ensure float32 for VAE stability under mixed precision training.
        images = self._to_float32(images)

        # Ensure images are on the correct device.
        images = images.to(self.device)

        # Encode to posterior distribution.
        # diffusers AutoencoderKL.encode() returns AutoencoderKLOutput with
        # .latent_dist attribute of type DiagonalGaussianDistribution.
        posterior = self.vae.encode(images).latent_dist

        # Use the mode (mean) of the posterior for deterministic, stable
        # conditioning. Shape: [B, latent_channels, h, w].
        latent: torch.Tensor = posterior.mode()

        # Apply scale factor to normalise latent distribution to ~unit variance.
        # This constant (0.2325) is fixed by the MAR KL-16 VAE training and
        # must be consistent with decode().
        latent = latent * self.scale_factor

        # Reshape from [B, C, h, w] to [B, h*w, C] (channel-last sequence format).
        # Step 1: [B, C, h, w] → [B, C, h*w] via flatten(2)
        # Step 2: [B, C, h*w] → [B, h*w, C] via transpose(1, 2)
        batch_size: int = latent.shape[0]
        tokens: torch.Tensor = latent.flatten(2).transpose(1, 2)
        # tokens shape: [B, h*w, latent_channels]

        return tokens

    @torch.no_grad()
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Converts predicted latent tokens back to pixel images.

        The decoding pipeline is:
            tokens [B, N, latent_channels] → reshape → ÷ scale_factor →
            VAE decoder → clamp → images [B, 3, H, W]

        This is the exact inverse of ``encode()``. The spatial dimensions are
        inferred from the sequence length ``N``:
            N = 256 → h = w = 16 → H = W = 256
            N =  64 → h = w =  8 → H = W = 128

        Args:
            tokens: Float tensor of shape ``[B, N, latent_channels]``.
                ``N`` must be a perfect square (256 or 64 for standard usage).

        Returns:
            Float tensor of shape ``[B, 3, H, W]`` in ``[-1, 1]`` range,
            clamped to prevent out-of-range values.

        Raises:
            ValueError: If ``N`` is not a perfect square.
        """
        # Ensure float32 for VAE stability.
        tokens = self._to_float32(tokens)
        tokens = tokens.to(self.device)

        batch_size: int = tokens.shape[0]
        n_tokens: int = tokens.shape[1]

        # Infer spatial dimensions from sequence length.
        h: int = int(n_tokens ** 0.5)
        if h * h != n_tokens:
            raise ValueError(
                f"Token sequence length {n_tokens} is not a perfect square. "
                "Expected 256 (16×16 for 256×256 images) or 64 (8×8 for "
                "128×128 images)."
            )
        w: int = h  # Square latent grids only.

        # Reshape from [B, N, C] to [B, C, h, w].
        # Step 1: [B, N, C] → [B, C, N] via transpose(1, 2)
        # Step 2: [B, C, N] → [B, C, h, w] via reshape
        latent: torch.Tensor = (
            tokens.transpose(1, 2).reshape(batch_size, self.latent_channels, h, w)
        )

        # Invert the scale factor applied during encoding.
        latent = latent / self.scale_factor

        # Decode latent to pixel space.
        # diffusers AutoencoderKL.decode() returns DecoderOutput with .sample.
        images: torch.Tensor = self.vae.decode(latent).sample

        # Clamp to [-1, 1] to prevent out-of-range pixel values that would
        # corrupt FID computation or visualisation.
        images = images.clamp(-1.0, 1.0)

        return images

    @torch.no_grad()
    def encode_dual_resolution(
        self,
        images_256: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes 256×256 images to both high-res and low-res token sequences.

        This is the primary method called during training. It produces both
        the high-resolution tokens for Phase 2 and the low-resolution tokens
        for Phase 1 from a single 256×256 image batch.

        The low-resolution tokens are obtained by first downsampling the
        256×256 images to 128×128 in pixel space (using bilinear interpolation)
        and then encoding with the same VAE. This ensures the low-resolution
        tokens genuinely represent a lower-resolution view of the image, not
        a subsampled latent.

        Paper reference (Section 4.2):
            "We employ the variational autoencoder (KL-16 version) trained by
            MAR to encode low-resolution (128×128) and high-resolution (256×256)
            images into latent representations for the two phases."

        Config alignment:
            resolution.high_res   = 256  →  hr_tokens: [B, 256, 16]
            resolution.low_res    = 128  →  lr_tokens: [B,  64, 16]
            resolution.hr_seq_len = 256
            resolution.lr_seq_len = 64

        Args:
            images_256: Float tensor of shape ``[B, 3, 256, 256]`` in
                ``[-1, 1]`` range, as produced by the dataset transforms.

        Returns:
            Tuple of:
                - ``hr_tokens``: High-resolution tokens, shape ``[B, 256, 16]``.
                  Used as Phase 2 training targets.
                - ``lr_tokens``: Low-resolution tokens, shape ``[B, 64, 16]``.
                  Used as Phase 1 training targets.

            Return order is ``(hr_tokens, lr_tokens)`` — high-res first.
            This matches the unpacking convention in ``trainer.py``:
            ``tokens_hr, tokens_lr = vae.encode_dual_resolution(images_256)``
        """
        # Ensure float32 and correct device.
        images_256 = self._to_float32(images_256).to(self.device)

        # ------------------------------------------------------------------
        # Phase 2 tokens: encode 256×256 images directly.
        # Output: [B, 256, 16]
        # ------------------------------------------------------------------
        hr_tokens: torch.Tensor = self.encode(images_256)

        # ------------------------------------------------------------------
        # Phase 1 tokens: downsample to 128×128 in pixel space, then encode.
        # Downsampling in pixel space (not latent space) ensures the low-res
        # tokens represent a genuinely lower-resolution view of the image.
        # Output: [B, 64, 16]
        # ------------------------------------------------------------------
        images_128: torch.Tensor = F.interpolate(
            images_256,
            size=(128, 128),
            mode="bilinear",
            align_corners=False,
        )
        lr_tokens: torch.Tensor = self.encode(images_128)

        return hr_tokens, lr_tokens
