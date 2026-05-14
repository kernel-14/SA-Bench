import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
from typing import List, Tuple, Any, Optional, Dict, Union
import numpy as np
import os
import json
import logging
import tempfile
import shutil

# Required packages for metrics
import cleanfid.fid as cleanfid_fid # for FID
from torch_fidelity import calculate_metrics # for IS, Precision/Recall

# Local imports
from config import Config
from generator import Generator # Assuming generator is importable without circular dependency issues

logger = logging.getLogger(__name__)

class Evaluator:
    """
    Evaluator class for computing various metrics (FID, IS, Precision, Recall, T2I-CompBench)
    to assess the quality of generated images.
    """
    def __init__(self, generator: Generator, config: Dict[str, Any], device: str):
        """
        Initializes the Evaluator.

        Args:
            generator: The Generator instance to use for generating samples.
            config: The full loaded configuration dictionary from config.py.
            device: The computational device ('cuda' or 'cpu').
        """
        self.generator = generator
        self.device = device
        self.global_config = config # Store full config to extract relevant parts

        # Determine the active evaluation task (ImageNet or MS-COCO)
        if self.global_config['training']['imagenet']['enabled']: # Use training config to infer task
            self.task_type = 'imagenet'
        elif self.global_config['training']['mscoco']['enabled']:
            self.task_type = 'mscoco'
        else:
            # Fallback if no training task is enabled, default to imagenet for evaluation context
            logger.warning("No training task (imagenet or mscoco) is enabled in config.yaml. "
                           "Defaulting Evaluator context to 'imagenet'. Ensure evaluation config is correct.")
            self.task_type = 'imagenet'

        self.evaluation_cfg: Dict[str, Any] = Config.get_evaluation_config(self.task_type)
        self.generation_cfg: Dict[str, Any] = Config.get_generation_config()
        
        self.inception_model: Optional[nn.Module] = None # Placeholder as per design

        logger.info(f"Evaluator initialized for task: {self.task_type}.")
        # The design specifies `_load_inception_model` exists.
        # For `torch_fidelity` and `cleanfid`, they typically manage their own Inception loading.
        # So, this call ensures the method exists, but `self.inception_model` is not directly used
        # by `_calculate_all_metrics` for FID/IS/PR when using `torch_fidelity`'s internal mechanism.
        self._load_inception_model()

    def _load_inception_model(self) -> nn.Module:
        """
        Loads a pre-trained InceptionV3 model, specifically configured for feature extraction.
        This method ensures `self.inception_model` is set. While `torch_fidelity` and `cleanfid`
        often use their own internal Inception models, this adheres to the design specification
        and allows for custom feature extraction if needed.
        """
        if self.inception_model is None:
            logger.info("Loading InceptionV3 model for potential custom feature extraction...")
            # `transform_input=False` means we handle normalization ourselves.
            model = models.inception_v3(pretrained=True, transform_input=False)
            model.eval()
            # For feature extraction typically, the final classification layer is removed or ignored.
            # Here, we keep it but it would be ignored if we extracted features from an earlier layer.
            # If a direct feature extraction function was used, `model.fc = nn.Identity()`
            # would be a common step to ensure the model outputs features directly.
            self.inception_model = model.to(self.device)
            logger.info("InceptionV3 model loaded successfully to `self.inception_model`.")
        return self.inception_model

    def _get_real_image_paths_or_dir(self, task_type: str) -> str:
        """
        Returns the path to the real image directory for a given task type.
        Assumes `real_features_path` in config points to a directory of images
        for `torch_fidelity` and `cleanfid` to use.
        """
        if task_type not in self.evaluation_cfg:
            raise ValueError(f"Task type '{task_type}' not found in evaluation config.")
        
        real_path = self.evaluation_cfg[task_type]['real_features_path']
        if not os.path.isdir(real_path):
            raise FileNotFoundError(f"Real image directory not found at: {real_path}. "
                                    f"Please ensure '{task_type}.real_features_path' in config.yaml "
                                    "points to a directory containing real validation images for metric calculation.")
        return real_path

    def _save_images_to_temp_dir(self, images: List[Image.Image], prefix: str = "gen_") -> str:
        """
        Saves a list of PIL images to a temporary directory.

        Args:
            images: List of PIL Image objects.
            prefix: Prefix for generated image filenames.

        Returns:
            Path to the temporary directory where images are saved.
        """
        temp_dir = tempfile.mkdtemp(prefix=prefix)
        for i, img in enumerate(images):
            # Convert to RGB if not already to prevent issues with saving (e.g., RGBA)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(os.path.join(temp_dir, f"{i:05d}.png"))
        logger.info(f"Saved {len(images)} images to temporary directory: {temp_dir}")
        return temp_dir

    def evaluate_imagenet(self, conditions: List[int], num_samples: int, guidance_scale: Optional[float]) -> Dict[str, float]:
        """
        Evaluates the Hi-MAR model on the ImageNet dataset for class-conditional image generation.

        Args:
            conditions: List of class IDs for class-conditional generation (e.g., range(1000)).
            num_samples: Total number of samples to generate for evaluation.
            guidance_scale: The Classifier-Free Guidance scale for generated samples.

        Returns:
            A dictionary containing FID, IS, Precision, and Recall scores for both
            'with CFG' and 'without CFG' settings.
        """
        eval_cfg = self.evaluation_cfg['imagenet']
        metrics_to_compute = eval_cfg['metrics']
        
        results: Dict[str, float] = {}
        real_image_dir = self._get_real_image_paths_or_dir('imagenet')

        # Evaluate with CFG (w/CFG)
        if eval_cfg.get('eval_with_cfg', True): # Default to True if not specified
            logger.info(f"Generating {num_samples} samples with CFG (guidance_scale={guidance_scale})...")
            generated_images_w_cfg = self.generator.generate_samples(
                conditions=conditions,
                num_samples=num_samples,
                guidance_scale=guidance_scale,
                # For ImageNet 'w/CFG' evaluation, Phase 2 CFG is active.
                phase2_cfg_off_for_eval=False, 
                low_res_steps=self.generation_cfg['inference_steps']['phase1'],
                high_res_steps=self.generation_cfg['inference_steps']['phase2']
            )
            
            gen_dir_w_cfg = self._save_images_to_temp_dir(generated_images_w_cfg, prefix="gen_w_cfg_")
            metrics_dict_w_cfg = self._calculate_all_metrics(real_image_dir, gen_dir_w_cfg, metrics_to_compute)
            for k, v in metrics_dict_w_cfg.items():
                results[f"w_cfg_{k}"] = v
            shutil.rmtree(gen_dir_w_cfg) # Clean up temp directory

        # Evaluate without CFG for dense tokens (w/o CFG in paper's Table 2)
        if eval_cfg.get('eval_without_cfg', True): # Default to True if not specified
            logger.info(f"Generating {num_samples} samples without CFG for dense tokens (Phase 2)...")
            generated_images_wo_cfg = self.generator.generate_samples(
                conditions=conditions,
                num_samples=num_samples,
                guidance_scale=guidance_scale, 
                # This explicitly disables CFG for Phase 2, matching paper's description for "w/o CFG"
                phase2_cfg_off_for_eval=True,  
                low_res_steps=self.generation_cfg['inference_steps']['phase1'],
                high_res_steps=self.generation_cfg['inference_steps']['phase2']
            )
            
            gen_dir_wo_cfg = self._save_images_to_temp_dir(generated_images_wo_cfg, prefix="gen_wo_cfg_")
            metrics_dict_wo_cfg = self._calculate_all_metrics(real_image_dir, gen_dir_wo_cfg, metrics_to_compute)
            for k, v in metrics_dict_wo_cfg.items():
                results[f"wo_cfg_{k}"] = v
            shutil.rmtree(gen_dir_wo_cfg) # Clean up temp directory

        return results

    def evaluate_mscoco(self, prompts: List[str], num_samples: int, guidance_scale: Optional[float]) -> Dict[str, float]:
        """
        Evaluates the Hi-MAR model on the MS-COCO dataset for text-to-image generation.

        Args:
            prompts: List of text prompts for generation.
            num_samples: Number of samples to generate.
            guidance_scale: The Classifier-Free Guidance scale for generated samples.

        Returns:
            A dictionary containing FID and T2I-CompBench scores.
        """
        eval_cfg = self.evaluation_cfg['mscoco']
        metrics_to_compute = eval_cfg['metrics']

        # Ensure we have enough prompts or adjust num_samples
        if len(prompts) > num_samples:
            prompts = prompts[:num_samples]
        elif len(prompts) < num_samples:
            logger.warning(f"Number of prompts ({len(prompts)}) is less than requested num_samples ({num_samples}). "
                           "Adjusting num_samples to match available prompts.")
            num_samples = len(prompts) 
        
        if not prompts: # If no prompts are available after adjustment
            logger.error("No prompts available for MS-COCO evaluation.")
            return {"FID": float('nan'), "T2I_CompBench_Color": float('nan')} # Return dummy scores

        real_image_dir = self._get_real_image_paths_or_dir('mscoco')

        logger.info(f"Generating {num_samples} samples for MS-COCO with CFG (guidance_scale={guidance_scale})...")
        generated_images = self.generator.generate_samples(
            conditions=prompts,
            num_samples=num_samples,
            guidance_scale=guidance_scale,
            # For MS-COCO, paper implies CFG is generally ON for best results.
            phase2_cfg_off_for_eval=False, 
            low_res_steps=self.generation_cfg['inference_steps']['phase1'],
            high_res_steps=self.generation_cfg['inference_steps']['phase2']
        )

        results: Dict[str, float] = {}
        gen_dir = self._save_images_to_temp_dir(generated_images, prefix="gen_mscoco_")
        
        metrics_dict = self._calculate_all_metrics(real_image_dir, gen_dir, metrics_to_compute)
        for k, v in metrics_dict.items():
            results[k] = v

        # Specific T2I-CompBench evaluation
        if "T2I-CompBench" in metrics_to_compute:
            t2i_compbench_scores = self.compute_t2icompbench(prompts, generated_images)
            results.update(t2i_compbench_scores)

        shutil.rmtree(gen_dir) # Clean up temp directory
        return results

    def _calculate_all_metrics(self, real_image_dir: str, gen_image_dir: str, metrics_to_compute: List[str]) -> Dict[str, float]:
        """
        Calculates FID, IS, Precision, and Recall using `cleanfid` for FID and `torch_fidelity` for others.
        """
        fidelity_metrics_torch_fidelity = []
        if "IS" in metrics_to_compute: fidelity_metrics_torch_fidelity.append("is")
        if "Precision" in metrics_to_compute or "Recall" in metrics_to_compute:
            fidelity_metrics_torch_fidelity.append("pr") # `torch_fidelity` calculates both together

        results: Dict[str, float] = {}

        # Use cleanfid for FID calculation
        if "FID" in metrics_to_compute:
            try:
                fid_score = cleanfid_fid.compute_fid(fdir1=real_image_dir, fdir2=gen_image_dir,
                                                    mode="clean", num_workers=os.cpu_count() // 2,
                                                    device=self.device)
                results["FID"] = fid_score
                logger.info(f"  FID: {fid_score:.4f}")
            except Exception as e:
                logger.error(f"Error calculating FID with cleanfid: {e}")
                results["FID"] = float('nan')
        
        # Use torch_fidelity for IS and PR
        if fidelity_metrics_torch_fidelity:
            try:
                torch_fidelity_results = calculate_metrics(
                    input1=real_image_dir, # Real images are input1 for PR calculation
                    input2=gen_image_dir, # Generated images are input2 for PR calculation, and input1 for IS
                    cuda=self.device.startswith('cuda'), # Use CUDA if device is cuda
                    metrics=fidelity_metrics_torch_fidelity,
                    save_cpu_ram=True, # Reduces RAM usage for inception model
                    verbose=False,
                    csv_file=None # Don't save to csv
                )
                if "is" in fidelity_metrics_torch_fidelity: # Check if IS was requested via `metrics_to_compute`
                    results["IS"] = torch_fidelity_results["inception_score_mean"]
                    # results["IS_std"] = torch_fidelity_results["inception_score_std"] # Can add if needed
                    logger.info(f"  IS: {results['IS']:.4f}")
                if "pr" in fidelity_metrics_torch_fidelity: # Check if PR was requested
                    results["Precision"] = torch_fidelity_results["pr_precision"]
                    results["Recall"] = torch_fidelity_results["pr_recall"]
                    logger.info(f"  Precision: {results['Precision']:.4f}, Recall: {results['Recall']:.4f}")
            except Exception as e:
                logger.error(f"Error calculating IS/Precision/Recall with torch_fidelity: {e}")
                if "is" in fidelity_metrics_torch_fidelity: results["IS"] = float('nan')
                if "pr" in fidelity_metrics_torch_fidelity:
                    results["Precision"] = float('nan')
                    results["Recall"] = float('nan')

        return results

    def compute_fid(self, real_features: torch.Tensor, generated_features: torch.Tensor) -> float:
        """
        [DEPRECATED] Computes the Frechet Inception Distance (FID).
        This function is kept to match the design but its functionality is subsumed by `_calculate_all_metrics`
        which leverages `cleanfid` directly from image paths/directories.
        """
        logger.warning("compute_fid in Evaluator is deprecated. Use _calculate_all_metrics for integrated metric calculation.")
        # Placeholder implementation for design adherence.
        # If real features are provided as (mu, sigma) from a .npz, and generated as (mu, sigma),
        # cleanfid.fid.compute_fid_from_stats could be used.
        # Given the input signature `torch.Tensor`, this implies raw features.
        if real_features.shape[0] == 0 or generated_features.shape[0] == 0:
            return float('nan')
        # Here, one would typically convert `real_features` and `generated_features` (numpy arrays)
        # to their respective mu and sigma, then use the FID formula.
        return float('nan') # Dummy value

    def compute_is(self, images: List[Image.Image]) -> Tuple[float, float]:
        """
        [DEPRECATED] Computes the Inception Score (IS) for a set of generated images.
        This function is kept to match the design but its functionality is subsumed by `_calculate_all_metrics`
        which leverages `torch_fidelity` from image paths/directories.
        """
        logger.warning("compute_is in Evaluator is deprecated. Use _calculate_all_metrics for integrated metric calculation.")
        # Placeholder implementation for design adherence.
        if not images:
            return float('nan'), float('nan')
        # If implemented manually, this would involve running images through InceptionV3
        # to get classification logits, then calculating IS.
        return float('nan'), float('nan') # Dummy values for mean and std

    def compute_precision_recall(self, real_features: torch.Tensor, generated_features: torch.Tensor) -> Tuple[float, float]:
        """
        [DEPRECATED] Computes Precision and Recall.
        This function is kept to match the design but its functionality is subsumed by `_calculate_all_metrics`
        which leverages `torch_fidelity` from image paths/directories.
        """
        logger.warning("compute_precision_recall in Evaluator is deprecated. Use _calculate_all_metrics for integrated metric calculation.")
        # Placeholder implementation for design adherence.
        if real_features.shape[0] == 0 or generated_features.shape[0] == 0:
            return float('nan'), float('nan')
        # If implemented manually, this would involve K-nearest neighbors search in feature space.
        return float('nan'), float('nan') # Dummy values for Precision and Recall

    def compute_t2icompbench(self, prompts: List[str], generated_images: List[Image.Image]) -> Dict[str, float]:
        """
        Computes scores for the T2I-CompBench benchmark.
        This is a placeholder as a full implementation requires specialized external tools
        not typically covered by standard metric libraries like cleanfid/torch_fidelity.
        """
        logger.warning("T2I-CompBench evaluation is a placeholder. Full implementation requires specialized tools.")
        # Returns dummy values as a placeholder.
        return {
            "T2I_CompBench_Color": 0.0,
            "T2I_CompBench_Shape": 0.0,
            "T2I_CompBench_Texture": 0.0,
            "T2I_CompBench_Spatial": 0.0,
            "T2I_CompBench_NonSpatial": 0.0,
            "T2I_CompBench_Complex": 0.0,
        }

