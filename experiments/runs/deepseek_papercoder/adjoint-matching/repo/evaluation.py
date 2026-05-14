## evaluation.py

import math
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

# ---------------------------------------------------------------------------
#  Metric‑specific imports (optional: guard against missing packages)
# ---------------------------------------------------------------------------
try:
    import open_clip
    HAS_OPEN_CLIP = True
except ImportError:
    HAS_OPEN_CLIP = False

try:
    from transformers import CLIPProcessor, CLIPModel
    HAS_TRANSFORMERS_CLIP = True
except ImportError:
    HAS_TRANSFORMERS_CLIP = False

# PickScore model is usually loaded via transformers
try:
    from transformers import AutoProcessor, AutoModelForImageTextToText   # placeholder
    # Actually PickScore is from yuvalkirstain/pick_score_v1; we'll use that.
    _ = AutoModelForImageTextToText  # avoid error
    HAS_PICKSCORE = True
except Exception:
    HAS_PICKSCORE = False

try:
    import hpsv2
    HAS_HPSV2 = True
except ImportError:
    HAS_HPSV2 = False

try:
    from dreamsim import dreamsim
    HAS_DREAMSIM = True
except ImportError:
    HAS_DREAMSIM = False


class Evaluator:
    """
    Post‑training evaluation suite for reward‑fine‑tuned Flow Matching models.

    Computes ClipScore, PickScore, HPSv2, and DreamSim diversity following
    the protocol described in the paper (Section 7, Appendix G.4).

    Args:
        fine_model:   The fine‑tuned velocity field U‑Net (FineTunedModel).
        base_models:  An instance of BaseModels holding the VAE,
                      CLIP text encoder, tokenizer, and optionally the
                      reward model (not used for standard metrics).
        config:       Application configuration (Config object).

    All heavy models are moved to the same device as ``fine_model`` and set
    to evaluation mode.
    """

    def __init__(
        self,
        fine_model: torch.nn.Module,
        base_models: Any,      # BaseModels from models.py
        config: Any,
    ) -> None:
        self.fine_model = fine_model
        self.vae = base_models.vae
        self.clip_text_encoder = base_models.clip_text_encoder
        self.tokenizer = base_models.tokenizer
        self.config = config

        # Determine device from the fine‑tuned model
        self.device = next(fine_model.parameters()).device

        # Move base components to the same device (if not already there)
        self.vae = self.vae.to(self.device)
        self.clip_text_encoder = self.clip_text_encoder.to(self.device)

        # Set all components to evaluation mode & disable gradients
        self.fine_model.eval()
        self.vae.eval()
        self.clip_text_encoder.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.clip_text_encoder.parameters():
            p.requires_grad_(False)

        # Pre‑compute unconditional text embedding for classifier‑free guidance
        padding_token_id = self.tokenizer.pad_token_id
        if padding_token_id is None:
            padding_token_id = self.tokenizer.eos_token_id
        empty_inputs = self.tokenizer(
            [""],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        empty_inputs = {k: v.to(self.device) for k, v in empty_inputs.items()}
        with torch.no_grad():
            encoder_output = self.clip_text_encoder(**empty_inputs)
            self.uncond_embedding = encoder_output.last_hidden_state  # (1, seq_len, dim)

        # Extract key evaluation settings
        self.sampling_steps = getattr(config.evaluation, "sampling_steps", 40)
        self.default_guidance_weight = getattr(
            config.evaluation, "classifier_free_guidance_weight", 1.0
        )
        self.num_samples_per_prompt = getattr(
            config.evaluation, "num_test_samples", 40
        )
        self.diversity_prompts_count = getattr(
            config.evaluation, "diversity_prompts_count", 25
        )
        # List of guidance weights to test (default single value)
        self.guidance_weights = getattr(
            config.evaluation, "guidance_weights_to_test", [self.default_guidance_weight]
        )
        # Make sure it is iterable
        if isinstance(self.guidance_weights, (float, int)):
            self.guidance_weights = [self.guidance_weights]

        # Internal latent size (assumed from VAE config; typical: 4×64×64)
        self.latent_channels = self.vae.config.latent_channels
        # Compression factor (e.g., 8) – we only need spatial size
        self.latent_size = self.vae.config.sample_size  # e.g., 64 or 96
        if self.latent_size is None:
            # fallback for some VAEs
            self.latent_size = 64

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------
    def generate_images(
        self,
        prompts: List[str],
        guidance_weight: Optional[float] = None,
        num_samples: Optional[int] = None,
    ) -> List[Image.Image]:
        """
        Produce a flat list of PIL images by running the ODE sampler with the
        fine‑tuned velocity field and classifier‑free guidance.

        For each prompt, ``num_samples`` images are generated (defaults to
        ``self.num_samples_per_prompt``).  The list is ordered: all samples
        for prompt 0, then prompt 1, …, and so on.

        Args:
            prompts:          List of text prompts.
            guidance_weight:  Classifier‑free guidance strength.
                              If ``None``, ``self.default_guidance_weight``
                              is used.
            num_samples:      Images per prompt (default from config).

        Returns:
            List of ``PIL.Image`` objects, length
            ``len(prompts) * num_samples``.
        """
        if guidance_weight is None:
            guidance_weight = self.default_guidance_weight
        if num_samples is None:
            num_samples = self.num_samples_per_prompt

        # ---- 1. Prepare condition embeddings ----
        # We process prompts in mini‑batches to keep memory manageable.
        # A single forward pass over thousands of latents can be heavy.
        # We adopt a simple loop over prompts, generating images one prompt
        # at a time.  This is slower but safe.
        all_images = []
        # For efficiency we could batch several prompts together, but for
        # simplicity we iterate per prompt.
        for prompt in prompts:
            # Encode prompt for the full batch of this prompt.
            cond_emb = self._encode_prompts([prompt])      # (1, seq_len, dim)
            # Expand to num_samples copies
            cond_emb = cond_emb.expand(num_samples, -1, -1)  # (S, seq_len, dim)
            # Unconditional embedding (expanded)
            uncond_emb = self.uncond_embedding.expand(num_samples, -1, -1)

            # ---- 2. Initial noise ----
            z = torch.randn(
                num_samples,
                self.latent_channels,
                self.latent_size,
                self.latent_size,
                device=self.device,
                dtype=cond_emb.dtype,
            )

            # ---- 3. ODE integration (σ = 0) ----
            dt = 1.0 / self.sampling_steps
            x = z
            with torch.no_grad():
                for k in range(self.sampling_steps):
                    t = k * dt
                    # Time tensor (shape (num_samples,), float in [0,1])
                    t_tensor = torch.full(
                        (num_samples,), t, device=self.device, dtype=z.dtype
                    )
                    # Conditional velocity
                    v_cond = self.fine_model(
                        x, t_tensor, encoder_hidden_states=cond_emb
                    )
                    # Unconditional velocity
                    v_uncond = self.fine_model(
                        x, t_tensor, encoder_hidden_states=uncond_emb
                    )
                    # Guided velocity
                    v_guided = v_uncond + guidance_weight * (v_cond - v_uncond)
                    # Euler step
                    x = x + dt * v_guided

            # Final latent at t=1 (approximately)
            # ---- 4. Decode to pixel space ----
            latents = x  # (num_samples, C, H, W)
            # VAE decoding expects latents scaled appropriately.
            # We assume the fine‑tuned model predicts latents in the same
            # scaled space as the VAE expects (no extra scaling).
            decoded = self.vae.decode(latents).sample  # (num_samples, 3, 512, 512)
            # Clamp to [-1, 1] then shift to [0,1] for PIL
            decoded = (decoded / 2 + 0.5).clamp(0, 1)
            # Convert to PIL images
            for i in range(num_samples):
                img_tensor = decoded[i].cpu()
                pil_img = T.ToPILImage()(img_tensor)
                all_images.append(pil_img)

        return all_images

    # ------------------------------------------------------------------
    #  Metric computation
    # ------------------------------------------------------------------
    def compute_clip_score(
        self,
        prompts: List[str],
        images: List[Image.Image],
    ) -> float:
        """
        Compute the CLIP Score: average cosine similarity between image and
        text CLIP embeddings over corresponding pairs.

        Args:
            prompts: List of strings (length N), one per image.
            images:  List of PIL images (length N).

        Returns:
            Scalar ClipScore (higher is better).
        """
        if not HAS_OPEN_CLIP:
            raise ImportError("open_clip is required for ClipScore. Install via `pip install open_clip_torch`.")
        # Load CLIP model (cache friendly)
        model_name = "ViT-B-32"   # commonly used; matches paper's open_clip usage
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained="laion400m_e32"
        )
        clip_model = clip_model.to(self.device)
        clip_model.eval()

        # Preprocess images to tensors
        image_batch = torch.stack(
            [clip_preprocess(img).to(self.device) for img in images]
        )  # (N, 3, 224, 224)

        # Tokenize prompts
        text_tokens = open_clip.tokenize(prompts).to(self.device)  # (N, max_len)

        with torch.no_grad():
            image_features = clip_model.encode_image(image_batch)
            text_features = clip_model.encode_text(text_tokens)

            # Normalise
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)

            # Pairwise cosine similarities – diagonal corresponds to correct pairs
            cos_sim = (image_features * text_features).sum(dim=-1)  # (N,)
            clip_score = cos_sim.mean().item()

        return clip_score

    def compute_pick_score(
        self,
        prompts: List[str],
        images: List[Image.Image],
    ) -> float:
        """
        Compute the PickScore (human preference alignment) using the
        Pick‑a‑Pic model.

        Args:
            prompts: List of strings (length N).
            images:  List of PIL images (length N).

        Returns:
            Average PickScore (higher is better).
        """
        if not HAS_PICKSCORE:
            raise ImportError(
                "transformers‑based PickScore not found. "
                "Install with: pip install transformers"
            )
        from transformers import AutoProcessor, AutoModelForImageTextToText

        model_name = "yuvalkirstain/pick_score_v1"
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForImageTextToText.from_pretrained(model_name)
        model = model.to(self.device)
        model.eval()

        scores = []
        with torch.no_grad():
            for img, prompt in zip(images, prompts):
                # The processor can handle a single image‑text pair
                inputs = processor(
                    text=prompt,
                    images=img,
                    return_tensors="pt",
                    padding=True,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                # The model outputs a logit (scalar)
                output = model(**inputs)
                scores.append(output.logits.item())   # type: ignore

        return np.mean(scores)

    def compute_hpsv2(
        self,
        prompts: List[str],
        images: List[Image.Image],
    ) -> float:
        """
        Compute Human Preference Score v2 (HPSv2).

        Args:
            prompts: List of strings (length N).
            images:  List of PIL images (length N).

        Returns:
            Average HPSv2 (higher is better).
        """
        if not HAS_HPSV2:
            raise ImportError("hpsv2 not found. Install via `pip install hpsv2`")
        scores = []
        for img, prompt in zip(images, prompts):
            # hpsv2.score expects PIL image and string
            score = hpsv2.score(img, prompt)
            scores.append(score)
        return np.mean(scores)

    def compute_dreamsim_diversity(
        self,
        prompts: List[str],
        images: List[Image.Image],
    ) -> float:
        """
        Compute the DreamSim diversity score.

        For the first ``diversity_prompts_count`` prompts, take the
        corresponding ``num_samples_per_prompt`` images, embed them,
        compute the average pairwise squared Euclidean distance among the
        embeddings, and return the mean over prompts.

        Args:
            prompts: List of distinct prompts (length P).
            images:  Flat list of images of length ``P * num_samples_per_prompt``
                     ordered by prompt.

        Returns:
            DreamSim diversity (higher means more diverse).
        """
        if not HAS_DREAMSIM:
            raise ImportError(
                "dreamsim not found. Install from "
                "github.com/ssundaram21/dreamsim"
            )
        # Load the DreamSim model
        model, preprocess = dreamsim.load_model(device=self.device)

        P = len(prompts)
        S = self.num_samples_per_prompt
        # Only use a subset of prompts
        n_prompts = min(self.diversity_prompts_count, P)
        promt_avg_divs = []

        with torch.no_grad():
            for i in range(n_prompts):
                # Extract the slice of images for this prompt
                start = i * S
                end = start + S
                group_imgs = images[start:end]
                # Preprocess and stack
                img_tensors = torch.stack(
                    [preprocess(img).to(self.device) for img in group_imgs]
                )  # (S, 3, 224, 224) – dreamsim preprocess resizes/normalizes
                # Get embeddings
                feats = model(img_tensors)  # (S, dim)
                # Pairwise squared Euclidean distances
                # Using torch.cdist
                dists = torch.cdist(feats, feats, p=2)  # (S, S)
                # Only upper triangle (excluding diagonal)
                mask = torch.triu(
                    torch.ones(S, S, device=self.device), diagonal=1
                ).bool()
                upper_dists = dists[mask]
                avg_dist = upper_dists.pow(2).mean().item()   # squared L2
                promt_avg_divs.append(avg_dist)

        return np.mean(promt_avg_divs)

    def evaluate_all(self, prompt_file: str) -> Dict[str, float]:
        """
        Run the full evaluation pipeline: load prompts, generate images
        for each configured guidance weight, compute all metrics, and
        return a dictionary of results.

        Args:
            prompt_file: Path to a text file with one prompt per line.

        Returns:
            Dict mapping metric names (including guidance weight) to
            scalar scores.
        """
        # Load test prompts
        with open(prompt_file, "r", encoding="utf-8") as f:
            all_prompts = [line.strip() for line in f if line.strip()]

        # Optionally truncate to the number used in the paper (1000)
        max_test_prompts = getattr(self.config.evaluation, "max_test_prompts", None)
        if max_test_prompts is not None and len(all_prompts) > max_test_prompts:
            all_prompts = all_prompts[:max_test_prompts]

        results = {}

        for w in self.guidance_weights:
            # Generate images for this guidance weight
            images = self.generate_images(all_prompts, guidance_weight=w)
            # Create repeated prompt list matching images
            repeated_prompts = [
                p for p in all_prompts for _ in range(self.num_samples_per_prompt)
            ]
            # Compute metrics
            clip_score = self.compute_clip_score(repeated_prompts, images)
            pick_score = self.compute_pick_score(repeated_prompts, images)
            hpsv2_score = self.compute_hpsv2(repeated_prompts, images)

            # Diversity only on the first diversity_prompts_count prompts
            diversity_prompts = all_prompts[: self.diversity_prompts_count]
            diversity_images = images[
                : self.diversity_prompts_count * self.num_samples_per_prompt
            ]
            dreamsim_div = self.compute_dreamsim_diversity(
                diversity_prompts, diversity_images
            )

            # Store with a key that includes the guidance weight
            prefix = f"w_{w}_" if w != 1.0 else ""
            results.update(
                {
                    f"{prefix}ClipScore": clip_score,
                    f"{prefix}PickScore": pick_score,
                    f"{prefix}HPSv2": hpsv2_score,
                    f"{prefix}DreamSimDiversity": dreamsim_div,
                }
            )

        return results

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------
    def _encode_prompts(self, prompts: List[str]) -> torch.Tensor:
        """
        Encode a list of prompts into CLIP last hidden states.

        Args:
            prompts: List of strings.

        Returns:
            Tensor of shape ``(len(prompts), max_length, embed_dim)``.
        """
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        with torch.no_grad():
            encoder_output = self.clip_text_encoder(**text_inputs)
        return encoder_output.last_hidden_state

