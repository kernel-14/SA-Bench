## evaluation/metrics.py
import math
import os
import random
from typing import List, Any, Callable, Dict

import numpy as np
import open_clip
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq # For PickScore

import hpsv2
from dreamsim import dreamsim


class Metrics:
  """
  A class providing static methods to calculate various evaluation metrics
  for generative models, including ClipScore, PickScore, HPSv2, and diversity
  metrics based on different embedding spaces.

  Models for metrics are loaded lazily upon their first use and moved to the
  specified device.
  """

  # Class-level variables to store loaded models and preprocessors for lazy loading
  _clip_model: Any = None
  _clip_preprocess: Any = None
  _clip_tokenizer: Any = None # OpenCLIP also has its own tokenizer

  _pickscore_processor: Any = None
  _pickscore_model: Any = None

  _hpsv2_model: Any = None

  _dreamsim_model: Any = None

  _device: str = "cuda"  # Default device, will be set by Evaluator based on config

  @classmethod
  def set_device(cls, device: str):
    """Sets the device for all metric models."""
    cls._device = device

  @classmethod
  def _load_clip_model(cls):
    """
    Loads the OpenCLIP model, preprocessor, and tokenizer if not already loaded.
    Using 'ViT-L-14' with 'openai' weights for ClipScore, a common and strong choice.
    """
    if cls._clip_model is None:
      # create_model_and_transforms returns model, preprocess_train, preprocess_val
      model, _, preprocess = open_clip.create_model_and_transforms(
          "ViT-L-14", pretrained="openai"
      )
      cls._clip_model = model.eval().to(cls._device)
      cls._clip_preprocess = preprocess
      cls._clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")
      print(f"Loaded OpenCLIP model on device: {cls._device}")

  @classmethod
  def _load_pickscore_model(cls):
    """
    Loads the PickScore-related model and processor using HuggingFace Transformers.
    The paper mentions using the transformers library for PickScore.
    'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k' is a common backbone for such tasks.
    For direct scoring, we assume the model's forward pass outputs a relevant score.
    For embeddings, we access its vision encoder.
    """
    if cls._pickscore_model is None:
      model_name = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
      cls._pickscore_processor = AutoProcessor.from_pretrained(model_name)
      # For scoring, many preference models expose a scoring head.
      # AutoModelForVision2Seq is a generic base; a more specific preference model
      # like "OpenAssistant/reward-model-deberta-v3-large" might be better for actual scores.
      # Given the prompt for AutoModelForVision2Seq and the general nature of PickScore,
      # we will treat this as a CLIP-like model that can extract features and compute similarity.
      # If a specific PickScore model's scoring head is needed, this would need to be adapted.
      cls._pickscore_model = AutoModelForVision2Seq.from_pretrained(model_name)
      cls._pickscore_model.eval().to(cls._device)
      print(f"Loaded PickScore model/processor on device: {cls._device}")

  @classmethod
  def _load_hpsv2_model(cls):
    """Loads the HPSv2 model from the hpsv2 library."""
    if cls._hpsv2_model is None:
      cls._hpsv2_model = hpsv2.HPSv2(device=cls._device)
      print(f"Loaded HPSv2 model on device: {cls._device}")

  @classmethod
  def _load_dreamsim_model(cls):
    """Loads the DreamSim model from the dreamsim library."""
    if cls._dreamsim_model is None:
      cls._dreamsim_model = dreamsim(pretrained=True).to(cls._device)
      cls._dreamsim_model.eval()
      print(f"Loaded DreamSim model on device: {cls._device}")

  @classmethod
  @torch.no_grad()
  def calculate_clip_score(cls, images: List[Image.Image], prompts: List[str]) -> float:
    """
    Calculates the average ClipScore for a list of images and corresponding prompts.
    ClipScore (Hessel et al., 2021) is the cosine similarity between CLIP image
    and text embeddings, averaged over samples.

    Args:
        images: A list of PIL.Image.Image objects.
        prompts: A list of strings, corresponding to each image.

    Returns:
        The average ClipScore as a float.
    """
    cls._load_clip_model()

    if not images or not prompts or len(images) != len(prompts):
      raise ValueError("Images and prompts lists must be non-empty and of equal length.")

    # Preprocess images and tokenize prompts
    image_inputs = torch.stack([cls._clip_preprocess(img) for img in images]).to(
        cls._device
    )
    text_tokens = cls._clip_tokenizer(prompts).to(cls._device)

    # Compute embeddings
    image_features = cls._clip_model.encode_image(image_inputs)
    text_features = cls._clip_model.encode_text(text_tokens)

    # Normalize features to unit length
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Compute cosine similarity (dot product of normalized features)
    similarity = (image_features * text_features).sum(dim=-1)

    return similarity.mean().item()

  @classmethod
  @torch.no_grad()
  def calculate_pick_score(cls, images: List[Image.Image], prompts: List[str]) -> float:
    """
    Calculates the average PickScore for a list of images and corresponding prompts.
    This implementation assumes that the loaded `_pickscore_model` can compute
    a direct scalar preference score for an image-text pair. This part
    is subject to the specific API of the chosen PickScore implementation.

    Args:
        images: A list of PIL.Image.Image objects.
        prompts: A list of strings, corresponding to each image.

    Returns:
        The average PickScore as a float.
    """
    cls._load_pickscore_model()

    if not images or not prompts or len(images) != len(prompts):
      raise ValueError("Images and prompts lists must be non-empty and of equal length.")

    pick_scores = []
    # Process in batches if `_pickscore_processor` and `_pickscore_model` support it,
    # otherwise, iterate one by one. Assuming batch processing for efficiency.
    batch_size = len(images) # Can be adjusted for memory constraints
    
    # Preprocess all images and texts in one go
    inputs = cls._pickscore_processor(images=images, text=prompts, return_tensors="pt", padding=True, truncation=True).to(cls._device)
    
    # Hypothetical scoring. The actual API of preference models can vary.
    # Often, they output logits representing preference for one item over another.
    # For a single image-text pair, we might interpret a specific output logit as a score.
    # This is a best-effort interpretation based on common patterns.
    # If a dedicated 'score' method is available, it should be used.
    # If the model is CLIP-like, cosine similarity could be a fallback.
    
    # For AutoModelForVision2Seq, directly getting a scalar 'score' is not standard.
    # A common proxy would be to compute CLIP-like similarity from its features.
    # The original PickScore (Kirstain et al., 2023) is a preference model, which usually means
    # it predicts log-odds for chosen over rejected, not a direct score for a single pair.
    # Without the exact PickScore model, using CLIP similarity as a proxy for the 'score' term
    # is a reasonable interpretation of "PickScore" in a generative context for consistency.
    
    # Fallback to CLIP-like similarity using PickScore's CLIP backbone:
    image_features = cls._pickscore_model.get_image_features(pixel_values=inputs.pixel_values)
    text_features = cls._pickscore_model.get_text_features(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    similarity_scores = (image_features * text_features).sum(dim=-1)
    
    pick_scores = similarity_scores.tolist() # Convert tensor of scores to list

    return np.mean(pick_scores)

  @classmethod
  @torch.no_grad()
  def calculate_hpsv2(cls, images: List[Image.Image], prompts: List[str]) -> float:
    """
    Calculates the average Human Preference Score v2 (HPSv2) for a list of
    images and corresponding prompts. (Wu et al., 2023b).

    Args:
        images: A list of PIL.Image.Image objects.
        prompts: A list of strings, corresponding to each image.

    Returns:
        The average HPSv2 score as a float.
    """
    cls._load_hpsv2_model()

    if not images or not prompts or len(images) != len(prompts):
      raise ValueError("Images and prompts lists must be non-empty and of equal length.")

    hps_scores = []
    for img, prompt in zip(images, prompts):
      # hpsv2.score directly takes a PIL image and a string prompt, returning a float.
      score = cls._hpsv2_model.score(img, prompt)
      hps_scores.append(score)

    return np.mean(hps_scores)

  @classmethod
  @torch.no_grad()
  def _get_dreamsim_embeddings(cls, images: List[Image.Image]) -> torch.Tensor:
    """
    Computes DreamSim embeddings for a list of PIL images.

    Args:
        images: A list of PIL.Image.Image objects.

    Returns:
        A torch.Tensor of embeddings, shape (num_images, embedding_dim).
    """
    cls._load_dreamsim_model()
    # dreamsim.embed expects a list of PIL images or a tensor (B, C, H, W).
    # It returns a tensor of embeddings (B, D).
    # ensure input to dreamsim is a list of PIL Images
    return cls._dreamsim_model.embed(images)

  @classmethod
  @torch.no_grad()
  def _get_clip_embeddings(cls, images: List[Image.Image]) -> torch.Tensor:
    """
    Computes CLIP image embeddings for a list of PIL images.

    Args:
        images: A list of PIL.Image.Image objects.

    Returns:
        A torch.Tensor of normalized CLIP image features, shape (num_images, embedding_dim).
    """
    cls._load_clip_model()
    image_inputs = torch.stack([cls._clip_preprocess(img) for img in images]).to(
        cls._device
    )
    image_features = cls._clip_model.encode_image(image_inputs)
    # Normalize features as is standard for CLIP embeddings for cosine similarity
    return image_features / image_features.norm(dim=-1, keepdim=True)

  @classmethod
  @torch.no_grad()
  def _get_pickscore_embeddings(cls, images: List[Image.Image]) -> torch.Tensor:
    """
    Computes image embeddings from the PickScore-related model (CLIP backbone).
    This assumes that the `_pickscore_model` (AutoModelForVision2Seq) can
    provide vision features directly.

    Args:
        images: A list of PIL.Image.Image objects.

    Returns:
        A torch.Tensor of normalized image features, shape (num_images, embedding_dim).
    """
    cls._load_pickscore_model()
    
    # The processor can handle a list of PIL images directly
    inputs = cls._pickscore_processor(images=images, return_tensors="pt").to(cls._device)
    
    # Accessing the vision model within AutoModelForVision2Seq
    # This is a common pattern for transformers models with a vision backbone.
    # The specific attribute name (`vision_model`, `get_image_features`) might vary.
    # Here, we assume `vision_model` is accessible and `pooler_output` provides features.
    vision_encoder_output = cls._pickscore_model.vision_model(
        pixel_values=inputs.pixel_values,
        output_hidden_states=True, # Ensure hidden states are available
        return_dict=True
    )
    # Typically, pooler_output or the last_hidden_state's CLS token are used as image features.
    image_features = vision_encoder_output.pooler_output
    
    # Normalize features as is standard for cosine similarity
    return image_features / image_features.norm(dim=-1, keepdim=True)

  # Dictionary to map embedding type strings to their respective getter methods
  _embedding_getters: Dict[str, Callable[[List[Image.Image]], torch.Tensor]] = {
      "dreamsim": _get_dreamsim_embeddings,
      "clip": _get_clip_embeddings,
      "pickscore": _get_pickscore_embeddings,
  }

  @classmethod
  @torch.no_grad()
  def _calculate_diversity_metric(
      cls,
      images: List[Image.Image],
      num_samples_per_prompt: int,
      num_eval_prompts: int,
      embedding_type: str,
  ) -> float:
    """
    Generic method to calculate diversity for a given embedding type.
    Diversity is computed as the average pairwise squared Euclidean distance of embeddings
    for images generated for the same prompt, averaged across multiple prompts.
    Formula from paper (Appendix G.4):
    Diversity = (1/num_eval_prompts) * sum_{k=1}^{num_eval_prompts}
                (2 / (N_gen * (N_gen-1))) * sum_{1 <= i < j <= N_gen} ||Embedding(g_i^k) - Embedding(g_j^k)||^2
    where N_gen = num_samples_per_prompt.

    Args:
        images: A flat list of all generated PIL.Image.Image objects. It is
                assumed that images for the same prompt are grouped sequentially.
        num_samples_per_prompt: The number of images generated per prompt (N_gen).
        num_eval_prompts: The total number of prompts used for diversity calculation.
        embedding_type: A string indicating which embedding space to use
                        ("dreamsim", "clip", "pickscore").

    Returns:
        The calculated diversity score as a float.
    """
    if num_eval_prompts == 0 or num_samples_per_prompt < 2:
      # Cannot compute diversity with insufficient samples/prompts.
      # If only 1 sample per prompt, no pairs for distance.
      return 0.0

    if len(images) != num_eval_prompts * num_samples_per_prompt:
      raise ValueError(
          f"Expected {num_eval_prompts * num_samples_per_prompt} images for diversity, "
          f"but got {len(images)}. Check image generation/grouping in Evaluator."
      )

    embedding_getter: Callable[[List[Image.Image]], torch.Tensor] = cls._embedding_getters.get(embedding_type)
    if embedding_getter is None:
      raise ValueError(
          f"Unknown or unimplemented embedding type for diversity: {embedding_type}"
      )

    total_diversity_score = 0.0

    for i in range(num_eval_prompts):
      start_idx = i * num_samples_per_prompt
      end_idx = start_idx + num_samples_per_prompt
      prompt_images = images[start_idx:end_idx]

      if not prompt_images:
        continue  # Should not happen if initial image count check is correct

      # Compute embeddings for all images of the current prompt
      embeddings = embedding_getter(prompt_images)  # Shape: (num_samples_per_prompt, embedding_dim)

      # Calculate pairwise squared Euclidean distances for this prompt
      sum_sq_dist_for_prompt = 0.0
      num_pairs_for_prompt = 0

      # Iterate through all unique pairs of embeddings (i, j) where i < j
      for j in range(num_samples_per_prompt):
        for k in range(j + 1, num_samples_per_prompt):
          diff = embeddings[j] - embeddings[k]
          # Sum of squared differences across embedding dimensions for Euclidean distance
          sum_sq_dist_for_prompt += (diff * diff).sum().item()
          num_pairs_for_prompt += 1

      if num_pairs_for_prompt > 0:
        # According to paper's formula (2 / (N_gen * (N_gen-1))) * sum ||...||^2
        # `sum_sq_dist_for_prompt` is already `sum ||...||^2`
        # `num_pairs_for_prompt` is N_gen * (N_gen-1) / 2
        # So `sum_sq_dist_for_prompt / num_pairs_for_prompt` is the average pairwise squared distance.
        # The paper's constant (2 / (N_gen * (N_gen-1))) simplifies to 1 / num_pairs_for_prompt (if sum is over all pairs, not just unique)
        # Let's stick to the "average of pairwise distances" interpretation, which is `sum_sq_dist_for_prompt / num_pairs_for_prompt`
        # and then average that over prompts.
        # The formula in G.4: (1/40) * sum_k=1^40 (2 / (25 * 24)) * sum_1<=i<j<=25 ||Clip(g_i^k)-Clip(g_j^k)||^2
        # This formula is slightly confusing. "Average values of ImageReward (reward function), control cost ..., and ClipScore vs. wall-clock time" (Figure 6)
        # and "ClipScore diversity as the variance of Clip embeddings of 40 generations for a given prompt, averaged across 25 prompts" (G.4)
        # The formula given is actually a variance-like quantity, specifically 2 * Average_Pairwise_Squared_Distance.
        # Let's directly implement the given formula for `ClipScore_Diversity` as:
        # `(1/N_prompts) * sum_{k=1}^{N_prompts} ( (1/N_gen^2) * sum_{i,j} ||Emb(g_i^k) - Emb(g_j^k)||^2 )`
        # or simplified: average of `||Emb(g_i^k) - Emb(g_j^k)||^2` for all `i,j` from `1` to `N_gen`.
        # The provided formula `(2 / (25 * 24)) * sum_{1 <= i < j <= 25}` simplifies to `1 / num_pairs` * `sum_{unique pairs}`.
        # So, `sum_sq_dist_for_prompt / num_pairs_for_prompt` is the per-prompt diversity.

        avg_pairwise_sq_dist_for_prompt = sum_sq_dist_for_prompt / num_pairs_for_prompt
        total_diversity_score += avg_pairwise_sq_dist_for_prompt

    # Average the diversity scores across all prompts
    return total_diversity_score / num_eval_prompts

  @classmethod
  def calculate_dreamsim_diversity(
      cls,
      images: List[Image.Image],
      prompts: List[str], # Prompts not used for embedding in this method, but part of interface
      num_samples_per_prompt: int,
      num_eval_prompts: int,
  ) -> float:
    """Calculates DreamSim diversity based on the paper's formula (Appendix G.4)."""
    return cls._calculate_diversity_metric(
        images, num_samples_per_prompt, num_eval_prompts, "dreamsim"
    )

  @classmethod
  def calculate_clip_diversity(
      cls,
      images: List[Image.Image],
      prompts: List[str], # Prompts not used for embedding in this method, but part of interface
      num_samples_per_prompt: int,
      num_eval_prompts: int,
  ) -> float:
    """Calculates Clip embedding diversity based on the paper's formula (Appendix G.4)."""
    return cls._calculate_diversity_metric(
        images, num_samples_per_prompt, num_eval_prompts, "clip"
    )

  @classmethod
  def calculate_pick_diversity(
      cls,
      images: List[Image.Image],
      prompts: List[str], # Prompts not used for embedding in this method, but part of interface
      num_samples_per_prompt: int,
      num_eval_prompts: int,
  ) -> float:
    """
    Calculates PickScore-related embedding diversity.
    Relies on the assumption that PickScore model can provide general image embeddings.
    """
    return cls._calculate_diversity_metric(
        images, num_samples_per_prompt, num_eval_prompts, "pickscore"
    )

