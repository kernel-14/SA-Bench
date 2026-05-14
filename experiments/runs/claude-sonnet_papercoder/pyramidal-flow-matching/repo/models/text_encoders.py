```python
## models/text_encoders.py
"""Text encoders for Pyramidal Flow Matching.

Wraps T5-XXL and CLIP ViT-L/14 text encoders following the FLUX.1 approach
described in the paper (Appendix B): "Following the more recent FLUX.1,
both T5 and CLIP encoders are employed for prompts embedding."

Both encoders are frozen during DiT training. T5 sequence embeddings feed
into cross-attention layers in MM-DiT, while CLIP pooled embeddings are
added to the timestep/stage conditioning signal.

Usage:
    from models.text_encoders import TextEncoders

    text_encoders = TextEncoders(config)
    text_encoders = text_encoders.to(device)

    # Conditional encoding
    text_cond = text_encoders.encode(["A beautiful sunset over the ocean"])
    # text_cond['t5_embeds']:      [B, 256, 4096]
    # text_cond['clip_embeds']:    [B, 768]
    # text_cond['attention_mask']: [B, 256]

    # Unconditional encoding for CFG
    null_cond = text_encoders.null_embed(batch_size=2)
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utils.logging import get_logger

## ---------------------------------------------------------------------------
## Module-level logger
## ---------------------------------------------------------------------------
logger = get_logger(__name__)

## ---------------------------------------------------------------------------
## Optional dependency availability flags
## ---------------------------------------------------------------------------
_TRANSFORMERS_AVAILABLE: bool = False
_OPEN_CLIP_AVAILABLE: bool = False
_FTFY_AVAILABLE: bool = False

try:
    import transformers  # type: ignore[import]
    from transformers import AutoTokenizer, T5EncoderModel  # type: ignore[import]
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning(
        "transformers not available. T5 encoder will not load. "
        "Install with: pip install transformers==4.40.0"
    )

try:
    import open_clip  # type: ignore[import]
    _OPEN_CLIP_AVAILABLE = True
except ImportError:
    logger.warning(
        "open_clip not available. CLIP encoder will not load. "
        "Install with: pip install open-clip-torch==2.24.0"
    )

try:
    import ftfy  # type: ignore[import]
    _FTFY_AVAILABLE = True
except ImportError:
    logger.warning(
        "ftfy not available. Text preprocessing will be skipped. "
        "Install with: pip install ftfy==6.2.0"
    )

## ---------------------------------------------------------------------------
## Mapping from config model names to open_clip naming convention
## ---------------------------------------------------------------------------
_CLIP_MODEL_NAME_MAP: Dict[str, Tuple[str, str]] = {
    "openai/clip-vit-large-patch14": ("ViT-L-14", "openai"),
    "openai/clip-vit-base-patch32": ("ViT-B-32", "openai"),
    "openai/clip-vit-base-patch16": ("ViT-B-16", "openai"),
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K": ("ViT-H-14", "laion2b_s32b_b79k"),
    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k": ("ViT-bigG-14", "laion2b_39b_b160k"),
}


def _preprocess_text(text: str) -> str:
    """Cleans a text string using ftfy for unicode normalization.

    Handles encoding artifacts common in web-scraped captions from
    LAION-5B, WebVid-10M, and other training datasets.

    Args:
        text: Raw text string, possibly containing unicode artifacts.

    Returns:
        Cleaned text string with normalized unicode and whitespace.
        Returns the original string unchanged if ftfy is not available.
    """
    if not _FTFY_AVAILABLE:
        return text
    try:
        return ftfy.fix_text(text)  # type: ignore[name-defined]
    except Exception:
        return text


class TextEncoders(nn.Module):
    """Frozen T5-XXL and CLIP ViT-L/14 text encoders for MM-DiT conditioning.

    Implements the dual-encoder text conditioning described in the paper
    (Appendix B): T5 provides rich sequence embeddings for cross-attention,
    while CLIP provides a compact pooled embedding for global conditioning.

    Both encoders are frozen (no gradient updates) during DiT training.
    The ``train()`` method is overridden to keep encoders in eval mode
    even when the parent model is set to training mode.

    Null (unconditional) embeddings are precomputed at initialization and
    stored as buffers for efficient CFG inference.

    Attributes:
        t5_model_name: HuggingFace model name for T5 encoder.
        clip_model_name: Config model name for CLIP (mapped to open_clip).
        t5_max_length: Maximum T5 tokenization length (256 from config).
        t5_embed_dim: T5 hidden state dimension (4096 from config).
        clip_embed_dim: CLIP pooled embedding dimension (768 from config).
        freeze_encoders: Whether to freeze encoder weights (True from config).
        model_dtype: PyTorch dtype for encoder weights (bfloat16 from config).
        t5_model: T5EncoderModel (frozen).
        t5_tokenizer: T5 tokenizer.
        clip_model: CLIP model (frozen).
        clip_tokenizer: CLIP tokenizer.
        null_t5_embeds: Precomputed null T5 embeddings [1, t5_max_length, 4096].
        null_t5_mask: Precomputed null T5 attention mask [1, t5_max_length].
        null_clip_embeds: Precomputed null CLIP embeddings [1, 768].
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initializes TextEncoders from the project config.

        Loads T5-XXL and CLIP ViT-L/14, freezes both, and precomputes
        null embeddings for CFG inference.

        Args:
            config: Project configuration dictionary. Expected keys under
                config['model']['text_encoder']:
                - t5_model (str): "google/t5-v1_1-xxl"
                - clip_model (str): "openai/clip-vit-large-patch14"
                - t5_max_length (int): 256
                - t5_embed_dim (int): 4096
                - clip_embed_dim (int): 768
                - freeze_encoders (bool): True
                Also reads:
                - config['model']['dtype'] (str): "bfloat16"
                - config['paths']['cache_dir'] (str): ".cache"
        """
        super().__init__()

        # ----------------------------------------------------------------
        # Parse configuration
        # ----------------------------------------------------------------
        model_cfg: Dict[str, Any] = config.get("model", {})
        text_enc_cfg: Dict[str, Any] = model_cfg.get("text_encoder", {})
        paths_cfg: Dict[str, Any] = config.get("paths", {})

        self.t5_model_name: str = str(
            text_enc_cfg.get("t5_model", "google/t5-v1_1-xxl")
        )
        self.clip_model_name: str = str(
            text_enc_cfg.get("clip_model", "openai/clip-vit-large-patch14")
        )
        self.t5_max_length: int = int(text_enc_cfg.get("t5_max_length", 256))
        self.t5_embed_dim: int = int(text_enc_cfg.get("t5_embed_dim", 4096))
        self.clip_embed_dim: int = int(text_enc_cfg.get("clip_embed_dim", 768))
        self.freeze_encoders: bool = bool(
            text_enc_cfg.get("freeze_encoders", True)
        )

        # Determine model dtype
        dtype_str: str = str(model_cfg.get("dtype", "bfloat16"))
        self.model_dtype: torch.dtype = (
            torch.bfloat16 if dtype_str == "bfloat16"
            else torch.float16 if dtype_str == "float16"
            else torch.float32
        )

        # Cache directory for HuggingFace downloads
        self.cache_dir: str = str(paths_cfg.get("cache_dir", ".cache"))

        # ----------------------------------------------------------------
        # Load T5 encoder
        # ----------------------------------------------------------------
        self.t5_model: Optional[nn.Module] = None
        self.t5_tokenizer: Optional[Any] = None
        self._load_t5()

        # ----------------------------------------------------------------
        # Load CLIP encoder
        # ----------------------------------------------------------------
        self.clip_model: Optional[nn.Module] = None
        self.clip_tokenizer: Optional[Any] = None
        self._load_clip()

        # ----------------------------------------------------------------
        # Precompute null (unconditional) embeddings for CFG
        # ----------------------------------------------------------------
        # These are registered as buffers so they move with the module
        # when .to(device) is called.
        self._precompute_null_embeddings()

        logger.info(
            "TextEncoders initialized: t5=%s, clip=%s, "
            "t5_max_length=%d, t5_embed_dim=%d, clip_embed_dim=%d, "
            "freeze=%s, dtype=%s",
            self.t5_model_name,
            self.clip_model_name,
            self.t5_max_length,
            self.t5_embed_dim,
            self.clip_embed_dim,
            self.freeze_encoders,
            dtype_str,
        )

    # -----------------------------------------------------------------------
    # Private loading methods
    # -----------------------------------------------------------------------

    def _load_t5(self) -> None:
        """Loads the T5-XXL encoder model and tokenizer.

        Uses only the encoder half of T5 (T5EncoderModel) since we only
        need text embeddings, not generation. Applies dtype conversion and
        freezing as configured.
        """
        if not _TRANSFORMERS_AVAILABLE:
            logger.error(
                "Cannot load T5 encoder: transformers package not available. "
                "Install with: pip install transformers==4.40.0"
            )
            return

        logger.info(
            "Loading T5 encoder: %s (cache_dir=%s)",
            self.t5_model_name,
            self.cache_dir,
        )

        try:
            # Load tokenizer
            self.t5_tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[name-defined]
                self.t5_model_name,
                cache_dir=self.cache_dir,
                use_fast=False,  # SentencePiece tokenizer
            )

            # Load encoder-only T5 model
            # T5EncoderModel loads only the encoder stack, saving ~50% memory
            # compared to the full T5ForConditionalGeneration
            t5_model: nn.Module = T5EncoderModel.from_pretrained(  # type: ignore[name-defined]
                self.t5_model_name,
                cache_dir=self.cache_dir,
                torch_dtype=self.model_dtype,
                low_cpu_mem_usage=True,
            )

            # Freeze encoder weights if configured
            if self.freeze_encoders:
                t5_model.requires_grad_(False)
                t5_model.eval()
                logger.info("T5 encoder frozen (requires_grad=False, eval mode).")

            self.t5_model = t5_model

            # Log parameter count
            num_params: int = sum(p.numel() for p in t5_model.parameters())
            logger.info(
                "T5 encoder loaded: %d parameters (%.1f B)",
                num_params,
                num_params / 1e9,
            )

        except Exception as exc:
            logger.error(
                "Failed to load T5 encoder '%s': %s. "
                "T5 conditioning will be unavailable.",
                self.t5_model_name,
                exc,
            )
            self.t5_model = None
            self.t5_tokenizer = None

    def _load_clip(self) -> None:
        """Loads the CLIP ViT-L/14 model and tokenizer.

        Maps the config model name to open_clip's naming convention and
        loads the model. Applies dtype conversion and freezing as configured.
        """
        if not _OPEN_CLIP_AVAILABLE:
            logger.error(
                "Cannot load CLIP encoder: open_clip package not available. "
                "Install with: pip install open-clip-torch==2.24.0"
            )
            return

        # Map config model name to open_clip (model_name, pretrained) tuple
        clip_name_key: str = self.clip_model_name
        if clip_name_key in _CLIP_MODEL_NAME_MAP:
            open_clip_name, open_clip_pretrained = _CLIP_MODEL_NAME_MAP[clip_name_key]
        else:
            # Fallback: try to use the config name directly
            logger.warning(
                "CLIP model name '%s' not in known mapping. "
                "Attempting to use as open_clip model name directly. "
                "Known mappings: %s",
                clip_name_key,
                list(_CLIP_MODEL_NAME_MAP.keys()),
            )
            open_clip_name = clip_name_key
            open_clip_pretrained = "openai"

        logger.info(
            "Loading CLIP encoder: %s (pretrained=%s)",
            open_clip_name,
            open_clip_pretrained,
        )

        try:
            # create_model_and_transforms returns (model, preprocess_train, preprocess_val)
            # We only need the model; image transforms are not used here
            clip_model, _, _ = open_clip.create_model_and_transforms(  # type: ignore[name-defined]
                model_name=open_clip_name,
                pretrained=open_clip_pretrained,
                cache_dir=self.cache_dir,
                precision="bf16" if self.model_dtype == torch.bfloat16 else "fp32",
            )

            # Load CLIP tokenizer
            self.clip_tokenizer = open_clip.get_tokenizer(open_clip_name)  # type: ignore[name-defined]

            # Freeze encoder weights if configured
            if self.freeze_encoders:
                clip_model.requires_grad_(False)
                clip_model.eval()
                logger.info("CLIP encoder frozen (requires_grad=False, eval mode).")

            self.clip_model = clip_model

            # Log parameter count
            num_params: int = sum(p.numel() for p in clip_model.parameters())
            logger.info(
                "CLIP encoder loaded: %s (pretrained=%s), %d parameters (%.1f M)",
                open_clip_name,
                open_clip_pretrained,
                num_params,
                num_params / 1e6,
            )

        except Exception as exc:
            logger.error(
                "Failed to load CLIP encoder '%s' (pretrained=%s): %s. "
                "CLIP conditioning will be unavailable.",
                open_clip_name,
                open_clip_pretrained,
                exc,
            )
            self.clip_model = None
            self.clip_tokenizer = None

    def _precompute_null_embeddings(self) -> None:
        """Precomputes null (unconditional) embeddings for CFG inference.

        Encodes an empty string through both T5 and CLIP encoders and
        stores the results as module buffers. Buffers automatically move
        to the correct device when the module is moved via .to(device).

        If encoders are not available, registers zero-filled buffers of
        the correct shape as fallbacks.
        """
        logger.info("Precomputing null embeddings for CFG inference...")

        # ----------------------------------------------------------------
        # Null T5 embeddings
        # ----------------------------------------------------------------
        if self.t5_model is not None and self.t5_tokenizer is not None:
            try:
                with torch.no_grad():
                    null_t5_embeds, null_t5_mask = self.encode_t5([""])
                    # null_t5_embeds: [1, t5_max_length, t5_embed_dim]
                    # null_t5_mask:   [1, t5_max_length]
            except Exception as exc:
                logger.warning(
                    "Failed to precompute null T5 embeddings: %s. "
                    "Using zero fallback.",
                    exc,
                )
                null_t5_embeds = torch.zeros(
                    1, self.t5_max_length, self.t5_embed_dim,
                    dtype=self.model_dtype,
                )
                null_t5_mask = torch.zeros(
                    1, self.t5_max_length, dtype=torch.long
                )
        else:
            logger.warning(
                "T5 encoder not available. Using zero null T5 embeddings."
            )
            null_t5_embeds = torch.zeros(
                1, self.t5_max_length, self.t5_embed_dim,
                dtype=self.model_dtype,
            )
            null_t5_mask = torch.zeros(
                1, self.t5_max_length, dtype=torch.long
            )

        # ----------------------------------------------------------------
        # Null CLIP embeddings
        # ----------------------------------------------------------------
        if self.clip_model is not None and self.clip_tokenizer is not None:
            try:
                with torch.no_grad():
                    null_clip_embeds = self.encode_clip([""])
                    # null_clip_embeds: [1, clip_embed_dim]
            except Exception as exc:
                logger.warning(
                    "Failed to precompute null CLIP embeddings: %s. "
                    "Using zero fallback.",
                    exc,
                )
                null_clip_embeds = torch.zeros(
                    1, self.clip_embed_dim, dtype=self.model_dtype
                )
        else:
            logger.warning(
                "CLIP encoder not available. Using zero null CLIP embeddings."
            )
            null_clip_embeds = torch.zeros(
                1, self.clip_embed_dim, dtype=self.model_dtype
            )

        # ----------------------------------------------------------------
        # Register as buffers (move with module, not updated by optimizer)
        # ----------------------------------------------------------------
        self.register_buffer("null_t5_embeds", null_t5_embeds)
        self.register_buffer("null_t5_mask", null_t5_mask)
        self.register_buffer("null_clip_embeds", null_clip_embeds)

        logger.info(
            "Null embeddings precomputed: "
            "t5_embeds=%s, t5_mask=%s, clip_embeds=%s",
            tuple(null_t5_embeds.shape),
            tuple(null_t5_mask.shape),
            tuple(null_clip_embeds.shape),
        )

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _get_device(self) -> torch.device:
        """Returns the device of the module's parameters/buffers.

        Uses the null_t5_embeds buffer as a proxy for the module's device,
        since it is always registered (even as a zero fallback).

        Returns:
            The torch.device where the module's tensors reside.
        """
        return self.null_t5_embeds.device  # type: ignore[attr-defined]

    def _preprocess_prompts(self, prompts: List[str]) -> List[str]:
        """Applies ftfy text cleaning to a list of prompt strings.

        Args:
            prompts: List of raw text prompts.

        Returns:
            List of cleaned text prompts with normalized unicode.
        """
        return [_preprocess_text(p) for p in prompts]

    # -----------------------------------------------------------------------
    # Public encoding API
    # -----------------------------------------------------------------------

    def encode_t5(
        self,
        prompts: List[str],
    ) -> Tuple[Tensor, Tensor]:
        """Encodes text prompts using the T5-XXL encoder.

        Tokenizes prompts with padding to t5_max_length=256, runs the
        T5 encoder, and returns the last hidden states as sequence
        embeddings along with the attention mask.

        Args:
            prompts: List of text strings to encode. Length B.
                Empty strings produce null/zero-like embeddings.

        Returns:
            Tuple of:
                - last_hidden_state: Tensor [B, t5_max_length, t5_embed_dim]
                  containing T5 sequence embeddings. Dtype matches
                  self.model_dtype (bfloat16 during training).
                - attention_mask: Tensor [B, t5_max_length] of dtype torch.long.
                  1 for real tokens, 0 for padding. Used in MM-DiT
                  cross-attention to ignore padding positions.

        Raises:
            RuntimeError: If T5 encoder or tokenizer is not available.

        Example:
            >>> embeds, mask = text_encoders.encode_t5(["A sunset over the ocean"])
            >>> embeds.shape
            torch.Size([1, 256, 4096])
            >>> mask.shape
            torch.Size([1, 256])
        """
        if self.t5_model is None or self.t5_tokenizer is None:
            raise RuntimeError(
                "T5 encoder is not available. "
                "Ensure transformers is installed and the model loaded correctly."
            )

        device: torch.device = self._get_device()

        # Apply text preprocessing (ftfy unicode normalization)
        cleaned_prompts: List[str] = self._preprocess_prompts(prompts)

        # Tokenize with padding to max_length=256 (from config)
        # padding="max_length" ensures uniform [B, 256] tensor shapes
        # truncation=True silently truncates prompts longer than max_length
        encoding = self.t5_tokenizer(
            cleaned_prompts,
            max_length=self.t5_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids: Tensor = encoding["input_ids"].to(device)
        attention_mask: Tensor = encoding["attention_mask"].to(device)

        # Run T5 encoder under no_grad (frozen weights)
        with torch.no_grad():
            outputs = self.t5_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # T5EncoderModel returns BaseModelOutput with last_hidden_state
        last_hidden_state: Tensor = outputs.last_hidden_state
        # Shape: [B, t5_max_length, t5_embed_dim] = [B, 256, 4096]

        # Ensure output dtype matches configured model dtype
        last_hidden_state = last_hidden_state.to(dtype=self.model_dtype)

        return last_hidden_state, attention_mask

    def encode_clip(
        self,
        prompts: List[str],
    ) -> Tensor:
        """Encodes text prompts using the CLIP ViT-L/14 encoder.

        Tokenizes prompts using CLIP's tokenizer (77-token limit with
        automatic truncation), runs the CLIP text encoder, and returns
        L2-normalized pooled text embeddings.

        Args:
            prompts: List of text strings to encode. Length B.
                Prompts longer than 77 tokens are silently truncated.

        Returns:
            Tensor [B, clip_embed_dim] = [B, 768] containing L2-normalized
            CLIP pooled text embeddings. Dtype matches self.model_dtype.

        Raises:
            RuntimeError: If CLIP encoder or tokenizer is not available.

        Example:
            >>> embeds = text_encoders.encode_clip(["A sunset over the ocean"])
            >>> embeds.shape
            torch.Size([1, 768])
            >>> # L2-normalized: norm should be ~1.0
            >>> embeds.norm(dim=-1)
            tensor([1.0000], ...)
        """
        if self.clip_model is None or self.clip_tokenizer is None:
            raise RuntimeError(
                "CLIP encoder is not available. "
                "Ensure open_clip is installed and the model loaded correctly."
            )

        device: torch.device = self._get_device()

        # Apply text preprocessing (ftfy unicode normalization)
        cleaned_prompts: List[str] = self._preprocess_prompts(prompts)

        # Tokenize using CLIP tokenizer
        # open_clip tokenizer handles truncation to 77 tokens internally
        tokens: Tensor = self.clip_tokenizer(cleaned_prompts)
        # Shape: [B, 77] (CLIP's fixed context length)
        tokens = tokens.to(device)

        # Run CLIP text encoder under no_grad (frozen weights)
        with torch.no_grad():
            # encode_text returns pooled text features [B, clip_embed_dim]
            # CLIP uses the [EOS] token representation as the pooled embedding
            clip_embeds: Tensor = self.clip_model.encode_text(tokens)
            # Shape: [B, 768]

        # L2-normalize CLIP embeddings (standard CLIP usage)
        # This ensures consistent scale regardless of prompt length/content
        clip_embeds = F.normalize(clip_embeds, p=2, dim=-1)

        # Ensure output dtype matches configured model dtype
        clip_embeds = clip_embeds.to(dtype=self.model_dtype)

        return clip_embeds

    def encode(
        self,
        prompts: List[str],
    ) -> Dict[str, Tensor]:
        """Encodes text prompts using both T5 and CLIP encoders.

        Primary interface called by PyramidFlowModel.forward() and
        InferenceSampler.sample_video(). Runs both encoders sequentially
        and packages results into a dict consumed by MMDiT.forward().

        Args:
            prompts: List of text strings to encode. Length B.
                All prompts in the list are encoded together as a batch.

        Returns:
            Dictionary with keys:
                - 't5_embeds': Tensor [B, t5_max_length, t5_embed_dim]
                  = [B, 256, 4096]. T5 sequence embeddings for cross-attention.
                - 'clip_embeds': Tensor [B, clip_embed_dim] = [B, 768].
                  L2-normalized CLIP pooled embeddings for global conditioning.
                - 'attention_mask': Tensor [B, t5_max_length] = [B, 256].
                  T5 padding mask (1=real token, 0=padding).

        Raises:
            RuntimeError: If either encoder is not available.

        Example:
            >>> text_cond = text_encoders.encode(["A beautiful sunset"])
            >>> text_cond['t5_embeds'].shape
            torch.Size([1, 256, 4096])
            >>> text_cond['clip_embeds'].shape
            torch.Size([1, 768])
            >>> text_cond['attention_mask'].shape
            torch.Size([1, 256])
        """
        # Encode with T5 (sequence embeddings + attention mask)
        t5_embeds, attention_mask = self.encode_t5(prompts)

        # Encode with CLIP (pooled embeddings)
        clip_embeds = self.encode_clip(prompts)

        return {
            "t5_embeds": t5_embeds,          # [B, 256, 4096]
            "clip_embeds": clip_embeds,       # [B, 768]
            "attention_mask": attention_mask, # [B, 256]
        }

    def null_embed(
        self,
        batch_size: int,
    ) -> Dict[str, Tensor]:
        """Returns null (unconditional) embeddings for CFG inference.

        Expands the precomputed null embeddings (from empty string encoding)
        to the requested batch size. Used for the unconditional forward pass
        in classifier-free guidance (Section 3.4).

        The null embeddings are precomputed at initialization and stored as
        module buffers, so this method involves no encoder forward passes —
        only tensor expansion, which is very cheap.

        Args:
            batch_size: Number of samples in the batch. The null embeddings
                are expanded (not copied) to this batch size.

        Returns:
            Dictionary with the same structure as encode():
                - 't5_embeds': Tensor [batch_size, t5_max_length, t5_embed_dim]
                - 'clip_embeds': Tensor [batch_size, clip_embed_dim]
                - 'attention_mask': Tensor [batch_size, t5_max_length]

        Example:
            >>> null_cond = text_encoders.null_embed(batch_size=4)
            >>> null_cond['t5_embeds'].shape
            torch.Size([4, 256, 4096])
            >>> null_cond['clip_embeds'].shape
            torch.Size([4, 768])
        """
        # Expand precomputed null embeddings from [1, ...] to [batch_size, ...]
        # .expand() creates a view without copying data (memory-efficient)
        null_t5: Tensor = self.null_t5_embeds.expand(  # type: ignore[attr-defined]
            batch_size, -1, -1
        ).contiguous()
        # Shape: [batch_size, t5_max_length, t5_embed_dim]

        null_mask: Tensor = self.null_t5_mask.expand(  # type: ignore[attr-defined]
            batch_size, -1
        ).contiguous()
        # Shape: [batch_size, t5_max_length]

        null_clip: Tensor = self.null_clip_embeds.expand(  # type: ignore[attr-defined]
            batch_size, -1
        ).contiguous()
        # Shape: [batch_size, clip_embed_dim]

        return {
            "t5_embeds": null_t5,
            "clip_embeds": null_clip,
            "attention_mask": null_mask,
        }

    # -----------------------------------------------------------------------
    # Training mode override
    # -----------------------------------------------------------------------

    def train(self, mode: bool = True) -> "TextEncoders":
        """Sets the module to training mode, keeping encoders in eval mode.

        Overrides nn.Module.train() to prevent accidentally enabling dropout
        and batch norm training behavior in the frozen T5 and CLIP encoders
        when the parent PyramidFlowModel is set to training mode.

        The DiT transformer (MMDiT) will be set to train mode normally;
        only the text encoders are kept in eval mode.

        Args:
            mode: If True, sets to training mode. If False, sets to eval mode.
                Regardless of this value, T5 and CLIP are always kept in eval.

        Returns:
            self (for method chaining, following nn.Module convention).
        """
        # Set the TextEncoders module itself to the requested mode
        super().train(mode)

        # Always keep frozen encoders in eval mode to prevent:
        # 1. Dropout activation during training
        # 2. BatchNorm using batch statistics instead of running stats
        # 3. Any other training-mode-specific behavior
        if self.t5_model is not None and self.freeze_encoders:
            self.t5_model.eval()

        if self.clip_model is not None and self.freeze_encoders: