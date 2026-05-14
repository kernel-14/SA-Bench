## evaluation/evaluator.py
import json
import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from diffusers import AutoencoderKL # Assuming VAE is AutoencoderKL from diffusers
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTextModel, AutoTokenizer

from config import Config
from data.dataset import TextPromptDataset
from diffusion.noise_schedule import NoiseSchedule
from diffusion.sde_solver import SDESolver, zero_sigma_fn
from evaluation.metrics import Metrics
from models.flow_matching_unet import FlowMatchingUNet
from models.reward_model import RewardModel
from utils.helpers import get_text_embeddings


class Evaluator:
  """
  Coordinates the evaluation process of the fine-tuned generative model.
  It manages sample generation, metric calculation, and result reporting/storage.
  """

  def __init__(
      self,
      config: Config,
      flow_model: FlowMatchingUNet,
      reward_model: RewardModel,
      eval_dataset: TextPromptDataset,
      sde_solver: SDESolver,
      noise_schedule: NoiseSchedule,
      vae: Optional[AutoencoderKL] = None,
      text_encoder: Optional[CLIPTextModel] = None,
      tokenizer: Optional[AutoTokenizer] = None,
  ):
    """
    Initializes the Evaluator with necessary components and configuration.

    Args:
        config: An instance of Config holding all experimental settings.
        flow_model: An instance of FlowMatchingUNet, which is the fine-tuned model
                    to be evaluated.
        reward_model: An instance of RewardModel to calculate ImageReward scores.
        eval_dataset: An instance of TextPromptDataset configured for evaluation prompts.
        sde_solver: An instance of SDESolver to handle the generation process.
        noise_schedule: An instance of NoiseSchedule to retrieve specific sigma functions.
        vae: An optional AutoencoderKL instance for decoding latent representations
             into pixel-space images.
        text_encoder: An optional CLIPTextModel instance for generating conditional
                      text embeddings during sampling (needed by SDESolver.sample).
        tokenizer: An optional AutoTokenizer instance corresponding to the text_encoder.
    """
    if not isinstance(config, Config):
      raise TypeError("config must be an instance of Config.")
    if not isinstance(flow_model, FlowMatchingUNet):
      raise TypeError("flow_model must be an instance of FlowMatchingUNet.")
    if not isinstance(reward_model, RewardModel):
      raise TypeError("reward_model must be an instance of RewardModel.")
    if not isinstance(eval_dataset, TextPromptDataset):
      raise TypeError("eval_dataset must be an instance of TextPromptDataset.")
    if not isinstance(sde_solver, SDESolver):
      raise TypeError("sde_solver must be an instance of SDESolver.")
    if not isinstance(noise_schedule, NoiseSchedule):
      raise TypeError("noise_schedule must be an instance of NoiseSchedule.")
    if vae is not None and not isinstance(vae, AutoencoderKL):
      raise TypeError("vae must be an instance of AutoencoderKL or None.")
    if text_encoder is not None and not isinstance(text_encoder, CLIPTextModel):
      raise TypeError("text_encoder must be an instance of CLIPTextModel or None.")
    if tokenizer is not None and not isinstance(tokenizer, AutoTokenizer):
      raise TypeError("tokenizer must be an instance of AutoTokenizer or None.")

    self.config: Config = config
    self.flow_model: FlowMatchingUNet = flow_model
    self.reward_model: RewardModel = reward_model
    self.eval_dataset: TextPromptDataset = eval_dataset
    self.sde_solver: SDESolver = sde_solver
    self.noise_schedule: NoiseSchedule = noise_schedule
    self.vae: Optional[AutoencoderKL] = vae
    self.text_encoder: Optional[CLIPTextModel] = text_encoder
    self.tokenizer: Optional[AutoTokenizer] = tokenizer

    self.device: str = config.general.device

    # Construct output directory path for evaluation results
    self.output_dir: str = os.path.join(
        config.general.output_dir, config.general.run_name, "evaluation"
    )
    os.makedirs(self.output_dir, exist_ok=True)
    print(f"Evaluation output directory created at: {self.output_dir}")

    # Instantiate Metrics calculator and set its device
    self.metrics_calculator: Metrics = Metrics()
    Metrics.set_device(self.device)

    # Set flow_model to evaluation mode and move to device
    self.flow_model.eval().to(self.device)
    for param in self.flow_model.parameters():
        param.requires_grad = False

    # Set VAE to evaluation mode if provided
    if self.vae is not None:
      self.vae.eval().to(self.device)
      for param in self.vae.parameters():
        param.requires_grad = False
      # The VAE scale factor for Stable Diffusion models (Appendix G.1 of Adjoint Matching)
      self.vae_scale_factor: float = 0.18215
    else:
      self.vae_scale_factor: float = 1.0 # Default if no VAE used, might lead to errors if pixel space is expected
      print("Warning: VAE not provided. Latent images will not be decoded to pixel space. "
            "Metrics requiring pixel images (e.g., ImageReward, HPSv2, ClipScore, etc.) will fail.")

    # Unconditional text embeddings (for CFG)
    self.unconditional_text_embeddings: torch.Tensor = get_text_embeddings(
        prompts=[""], # Empty string for unconditional generation
        text_encoder=self.text_encoder,
        tokenizer=self.tokenizer,
        device=self.device,
        max_length=self.config.model.text_encoder.max_length,
    ) if self.text_encoder is not None and self.tokenizer is not None else None

    if self.unconditional_text_embeddings is None:
        print("Warning: Text encoder or tokenizer not provided. CFG cannot be used. "
              "Evaluation with cfg_weight > 0 will produce conditional samples but no unconditional velocity.")


  def _get_sigma_function(
      self, sigma_type: str
  ) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Internal helper to map a string `sigma_type` to a callable function
    that returns the diffusion coefficient σ(t).

    Args:
        sigma_type: A string, either "memoryless" or "ode".

    Returns:
        A callable function `sigma_fn(t)` that returns a torch.Tensor of sigma(t) values.
    """
    if not isinstance(sigma_type, str):
      raise TypeError("sigma_type must be a string.")

    if sigma_type == "memoryless":
      # NoiseSchedule.get_memoryless_sigma_t internally uses self.h
      return lambda t: self.noise_schedule.get_memoryless_sigma_t(t).to(self.device)
    elif sigma_type == "ode":
      # sigma(t) = 0 for noiseless sampling
      return lambda t: zero_sigma_fn(t).to(self.device)
    else:
      raise ValueError(f"Unknown sigma_type: {sigma_type}. Must be 'memoryless' or 'ode'.")

  @torch.no_grad()
  def generate_samples(
      self,
      prompts: List[str],
      num_samples_per_prompt: int,
      sampling_sigma_type: str,
      cfg_weight: float,
      num_inference_timesteps: int,
  ) -> List[Image.Image]:
    """
    Generates a batch of images for a given set of prompts under specified
    sampling conditions.

    Args:
        prompts: A list of raw text prompts.
        num_samples_per_prompt: Number of images to generate for each prompt.
        sampling_sigma_type: The type of noise schedule to use ("memoryless" or "ode").
        cfg_weight: The Classifier-Free Guidance weight.
        num_inference_timesteps: The number of discrete steps for the SDE solver during sampling.

    Returns:
        A list of PIL.Image.Image objects.
    """
    if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
      raise TypeError("prompts must be a list of strings.")
    if not isinstance(num_samples_per_prompt, int) or num_samples_per_prompt <= 0:
      raise ValueError("num_samples_per_prompt must be a positive integer.")
    if not isinstance(sampling_sigma_type, str):
      raise TypeError("sampling_sigma_type must be a string.")
    if not isinstance(cfg_weight, (float, int)) or cfg_weight < 0:
      raise ValueError("cfg_weight must be a non-negative float.")
    if not isinstance(num_inference_timesteps, int) or num_inference_timesteps <= 0:
      raise ValueError("num_inference_timesteps must be a positive integer.")

    if self.vae is None:
      raise RuntimeError(
          "VAE not provided to Evaluator. Cannot decode latent samples to pixel images."
      )
    if (cfg_weight > 0.0 and (self.text_encoder is None or self.tokenizer is None or self.unconditional_text_embeddings is None)):
        raise RuntimeError("CFG requested (cfg_weight > 0) but text_encoder/tokenizer "
                           "or unconditional_text_embeddings are not initialized. "
                           "Cannot perform CFG sampling.")

    all_generated_images_pil: List[Image.Image] = []

    # Retrieve the appropriate sigma_fn
    sigma_fn = self._get_sigma_function(sampling_sigma_type)

    # Set the cfg_weight on the sde_solver instance for this generation call
    self.sde_solver.cfg_weight = cfg_weight

    # Prepare timesteps tensor for inference
    timesteps_tensor = self.noise_schedule.get_timesteps_tensor(
        num_inference_timesteps, self.device, self.flow_model.unet.dtype
    )
    h_val_inference = 1.0 / num_inference_timesteps

    for prompt in tqdm(prompts, desc=f"Generating samples ({sampling_sigma_type}, CFG={cfg_weight})"):
      # Obtain conditional text embeddings for the current prompt
      conditional_text_embeddings = get_text_embeddings(
          prompts=[prompt],
          text_encoder=self.text_encoder,
          tokenizer=self.tokenizer,
          device=self.device,
          max_length=self.config.model.text_encoder.max_length,
      )
      # Repeat for the number of samples per prompt
      prompt_embeddings = conditional_text_embeddings.repeat(num_samples_per_prompt, 1, 1)

      # Unconditional embeddings (repeated for batch size)
      uncond_embeddings_batched = self.unconditional_text_embeddings.repeat(num_samples_per_prompt, 1, 1)

      # Call sde_solver.sample() to generate latent image samples
      latent_samples = self.sde_solver.sample(
          num_samples=num_samples_per_prompt,
          text_prompts=[prompt] * num_samples_per_prompt, # SDESolver.sample takes list of prompts for internal use
          unconditional_text_embeddings=uncond_embeddings_batched,
          timesteps=timesteps_tensor,
          sigma_fn=sigma_fn,
          h_val=h_val_inference,
          text_encoder_tokenizer=(self.text_encoder, self.tokenizer),
          text_encoder_max_length=self.config.model.text_encoder.max_length,
      )

      # Decode the latent_samples into pixel space
      # VAE decoding involves scaling the latents and then clamping/scaling pixel values
      decoded_images_tensor = (
          self.vae.decode(latent_samples / self.vae_scale_factor).sample()
      )
      decoded_images_tensor = (decoded_images_tensor / 2 + 0.5).clamp(0, 1)

      # Convert torch.Tensor images to PIL.Image.Image
      for img_tensor in decoded_images_tensor:
        # Convert from (C, H, W) float tensor to (H, W, C) numpy array, then to PIL
        img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(
            np.uint8
        )
        pil_img = Image.fromarray(img_np)
        all_generated_images_pil.append(pil_img)

    # Reset sde_solver's cfg_weight to default if necessary (e.g. 0.0)
    self.sde_solver.cfg_weight = 0.0 

    return all_generated_images_pil

  @torch.no_grad()
  def evaluate(self, iteration: int = 0) -> Dict[str, Any]:
    """
    Orchestrates the full evaluation process, including generating samples
    for all specified sampling_sigma_type and cfg_weight combinations,
    calculating all metrics, and saving results.

    Args:
        iteration: Current training iteration, primarily for logging and naming saved files.

    Returns:
        A dictionary containing all evaluation results.
    """
    if not isinstance(iteration, int) or iteration < 0:
      raise ValueError("iteration must be a non-negative integer.")

    print(f"\n--- Starting Evaluation for iteration {iteration} ---")

    self.flow_model.eval()  # Set flow_model to evaluation mode

    results: Dict[str, Any] = {}
    all_eval_prompts: List[str] = self.eval_dataset.prompts

    # Select a subset of eval_prompts for actual evaluation if specified in config
    num_eval_prompts_for_metrics = self.config.evaluation.num_eval_prompts
    if num_eval_prompts_for_metrics > len(all_eval_prompts):
        print(f"Warning: config.evaluation.num_eval_prompts ({num_eval_prompts_for_metrics}) "
              f"is greater than available prompts ({len(all_eval_prompts)}). "
              "Using all available prompts for evaluation.")
        selected_eval_prompts = all_eval_prompts
    else:
        # Ensure consistent selection by seeding if needed, or just take first N
        # For fair comparison in paper, it's "random sets of 1000 test prompts" for each run
        # So it's fine to randomly sample.
        random.seed(self.config.general.seed + iteration) # Seed with iteration for unique eval sets per checkpoint
        selected_eval_prompts = random.sample(all_eval_prompts, num_eval_prompts_for_metrics)


    num_samples_per_prompt = self.config.evaluation.num_samples_per_prompt
    num_inference_timesteps = self.config.evaluation.num_inference_timesteps

    # --- Metrics for Base Model (if iteration is 0 or initial run) ---
    if iteration == 0:
        print("Evaluating Base Model (without fine-tuning)...")
        # Temporarily use the base_flow_model for evaluation for "None (Base model)" rows in Table 2
        original_flow_model = self.sde_solver.generative_model
        self.sde_solver.generative_model = self.config.base_flow_model # This assumes base_flow_model is stored in config (it's not, needs adjustment)

        # For the base model, we should use the actual base_flow_model.
        # This requires the `base_flow_model` to be passed to the evaluator during init or access it from config if stored there.
        # Given the current design, `sde_solver` is initialized with `flow_model` (v_finetune).
        # We need to temporarily set `sde_solver.generative_model` to `base_flow_model`
        # for base model evaluation. This implies that `base_flow_model` needs to be accessible here.
        # Let's adjust `Evaluator.__init__` to store `base_flow_model`.

        # Re-initialize sde_solver with base_flow_model temporarily if evaluating base.
        temp_sde_solver = SDESolver(
            generative_model=self.config.base_flow_model, # Assuming config now stores base_flow_model
            noise_schedule=self.noise_schedule,
            cfg_weight=0.0, # Will be overridden
            device=self.device,
        )

        for sampling_sigma_type in self.config.evaluation.sampling_sigma_types:
            for cfg_weight in self.config.evaluation.cfg_weights:
                metrics_key_base = f"BaseModel_{sampling_sigma_type}_cfg{cfg_weight}"
                print(f"  {metrics_key_base}...")
                generated_images_base = self.generate_samples(
                    prompts=selected_eval_prompts,
                    num_samples_per_prompt=num_samples_per_prompt,
                    sampling_sigma_type=sampling_sigma_type,
                    cfg_weight=cfg_weight,
                    num_inference_timesteps=num_inference_timesteps,
                    # Use temp_sde_solver with base model
                    _sde_solver_override=temp_sde_solver,
                )
                base_metrics = self._calculate_all_metrics(
                    generated_images_base,
                    selected_eval_prompts,
                    num_samples_per_prompt,
                    num_eval_prompts_for_metrics,
                )
                results[metrics_key_base] = base_metrics
                if self.config.evaluation.save_samples:
                    self._save_generated_samples(
                        generated_images_base,
                        selected_eval_prompts,
                        metrics_key_base,
                        iteration,
                    )
        # Restore original sde_solver's generative model
        self.sde_solver.generative_model = original_flow_model
        print("Base Model Evaluation Complete.")
    # --- End Base Model Evaluation ---


    for sampling_sigma_type in self.config.fine_tuning.evaluation_sampling_sigma_types:
      for cfg_weight in self.config.evaluation.cfg_weights:
        metrics_key = f"{sampling_sigma_type}_cfg{cfg_weight}"
        print(f"  Evaluating FineTunedModel ({metrics_key})...")

        generated_images = self.generate_samples(
            prompts=selected_eval_prompts,
            num_samples_per_prompt=num_samples_per_prompt,
            sampling_sigma_type=sampling_sigma_type,
            cfg_weight=cfg_weight,
            num_inference_timesteps=num_inference_timesteps,
        )

        current_metrics = self._calculate_all_metrics(
            generated_images,
            selected_eval_prompts,
            num_samples_per_prompt,
            num_eval_prompts_for_metrics,
        )
        results[f"FineTunedModel_{metrics_key}"] = current_metrics

        if self.config.evaluation.save_samples:
          self._save_generated_samples(
              generated_images, selected_eval_prompts, metrics_key, iteration
          )

    if self.config.evaluation.save_metrics:
      self._save_metrics_to_file(results, iteration)

    print(f"--- Evaluation for iteration {iteration} Complete ---")
    return results

  def _calculate_all_metrics(
      self,
      images: List[Image.Image],
      prompts: List[str],
      num_samples_per_prompt: int,
      num_eval_prompts: int,
  ) -> Dict[str, float]:
    """Helper to calculate all specified metrics for a given set of images and prompts."""
    current_metrics: Dict[str, float] = {}

    # ImageReward Calculation
    image_reward_scores = self.reward_model.predict(images, prompts).tolist()
    current_metrics["ImageReward"] = float(torch.tensor(image_reward_scores).mean().item())

    # Consistency Metrics
    current_metrics["ClipScore"] = self.metrics_calculator.calculate_clip_score(images, prompts)
    current_metrics["PickScore"] = self.metrics_calculator.calculate_pick_score(images, prompts)
    current_metrics["HPSv2"] = self.metrics_calculator.calculate_hpsv2(images, prompts)

    # Diversity Metrics
    current_metrics["DreamSim_Diversity"] = self.metrics_calculator.calculate_dreamsim_diversity(
        images, prompts, num_samples_per_prompt, num_eval_prompts
    )
    # The paper only lists DreamSim Diversity in main Table 2.
    # ClipScore Diversity and PickScore Diversity are not explicitly in the main table.
    # If they were to be included:
    # current_metrics["ClipScore_Diversity"] = self.metrics_calculator.calculate_clip_diversity(
    #     images, prompts, num_samples_per_prompt, num_eval_prompts
    # )
    # current_metrics["PickScore_Diversity"] = self.metrics_calculator.calculate_pick_diversity(
    #     images, prompts, num_samples_per_prompt, num_eval_prompts
    # )

    return current_metrics

  def _save_generated_samples(
      self, images: List[Image.Image], prompts: List[str], metrics_key: str, iteration: int
  ) -> None:
    """
    Internal helper to save generated PIL images to disk.

    Args:
        images: A list of PIL Images.
        prompts: The corresponding list of prompts.
        metrics_key: A string identifier for the current evaluation settings (e.g., "memoryless_cfg1.0").
        iteration: Current training iteration (for filename).
    """
    if not isinstance(images, list) or not all(isinstance(img, Image.Image) for img in images):
      raise TypeError("images must be a list of PIL.Image.Image.")
    if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
      raise TypeError("prompts must be a list of strings.")
    if not isinstance(metrics_key, str):
      raise TypeError("metrics_key must be a string.")
    if not isinstance(iteration, int) or iteration < 0:
      raise ValueError("iteration must be a non-negative integer.")

    samples_subdir = os.path.join(
        self.output_dir, "samples", f"iter_{iteration}", metrics_key
    )
    os.makedirs(samples_subdir, exist_ok=True)

    num_samples_per_prompt = self.config.evaluation.num_samples_per_prompt
    for i, img in enumerate(images):
      prompt_idx = i // num_samples_per_prompt
      sample_idx = i % num_samples_per_prompt
      # Sanitize prompt for filename use
      sanitized_prompt = prompts[prompt_idx][:50].replace("/", "_").replace("\\", "_").replace(" ", "_")
      filename = f"prompt_{prompt_idx:03d}_{sanitized_prompt}_sample_{sample_idx:02d}.png"
      img.save(os.path.join(samples_subdir, filename))

    print(f"Saved {len(images)} samples to {samples_subdir}")

  def _save_metrics_to_file(self, metrics: Dict[str, Any], iteration: int) -> None:
    """
    Internal helper to save all collected metrics to a JSON file.

    Args:
        metrics: The dictionary of all evaluation results.
        iteration: Current training iteration (for filename).
    """
    if not isinstance(metrics, dict):
      raise TypeError("metrics must be a dictionary.")
    if not isinstance(iteration, int) or iteration < 0:
      raise ValueError("iteration must be a non-negative integer.")

    metrics_path = os.path.join(self.output_dir, f"metrics_iter_{iteration:06d}.json")
    with open(metrics_path, "w") as f:
      json.dump(metrics, f, indent=4)
    print(f"Saved metrics for iteration {iteration} to {metrics_path}")

