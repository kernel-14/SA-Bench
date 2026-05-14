import argparse
import logging
import os
import torch
from typing import Dict, Any, List

# Import modules from the project
from config import Config
from tokenizer import VAETokenizer, CLIPTextEncoder
from data import DataModule
from model import HiMARModel
from trainer import Trainer
from generator import Generator
from evaluation import Evaluator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Main:
    """
    Main class to orchestrate the entire Hi-MAR reproduction experiment.
    Handles configuration loading, component initialization, training, generation, and evaluation.
    """
    def __init__(self):
        """
        Initializes the Main class, sets up argument parsing and basic logging.
        """
        parser = argparse.ArgumentParser(description="Reproduce Hi-MAR experiments.")
        parser.add_argument(
            "--config_path",
            type=str,
            default="config.yaml",
            help="Path to the YAML configuration file."
        )
        self.args = parser.parse_args()
        logger.info(f"Initialized Main with config path: {self.args.config_path}")

        # Placeholder for instantiated components
        self.config: Dict[str, Any] = {}
        self.device: str = "cpu"
        self.vae_tokenizer: VAETokenizer | None = None
        self.clip_encoder: CLIPTextEncoder | None = None
        self.data_module: DataModule | None = None
        self.model: HiMARModel | None = None
        self.trainer: Trainer | None = None
        self.generator: Generator | None = None
        self.evaluator: Evaluator | None = None

    def run_experiment(self, config_path: str) -> None:
        """
        Executes the main experiment pipeline: loads configuration, initializes components,
        runs training, and performs evaluation.

        Args:
            config_path: Path to the YAML configuration file.
        """
        logger.info(f"Starting Hi-MAR experiment with configuration from: {config_path}")

        # 1. Load Configuration
        self.config = Config.load_config(config_path)
        global_cfg = Config.get_global_config()
        tokenizer_cfg = Config.get_tokenizer_config()
        clip_cfg = Config.get_clip_text_encoder_config()

        # Set random seed for reproducibility
        torch.manual_seed(global_cfg['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(global_cfg['seed'])
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.info(f"Random seed set to {global_cfg['seed']}.")

        # 2. Setup Device
        self.device = global_cfg['device']
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA device requested but not available. Falling back to CPU.")
            self.device = "cpu"
        logger.info(f"Using device: {self.device}")

        # Create output directory if it doesn't exist
        output_dir = os.path.join(global_cfg['output_dir'], global_cfg['project_name'])
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Experiment output will be saved to: {output_dir}")

        # 3. Initialize Tokenizers
        self.vae_tokenizer = VAETokenizer(
            vae_path=tokenizer_cfg['vae_path'],
            latent_channels=tokenizer_cfg['latent_channels'],
            high_res_image_size=tokenizer_cfg['high_res_image_size'],
            low_res_image_size=tokenizer_cfg['low_res_image_size'],
            device=self.device
        )
        logger.info(f"VAETokenizer initialized with latent_channels: {self.vae_tokenizer.latent_channels}")

        # Check if MS-COCO related tasks are enabled
        mscoco_training_enabled = self.config['training']['mscoco']['enabled']
        mscoco_evaluation_enabled = self.config['evaluation']['mscoco']['enabled']
        if mscoco_training_enabled or mscoco_evaluation_enabled:
            self.clip_encoder = CLIPTextEncoder(
                model_name=clip_cfg['model_name'],
                device=self.device
            )
            logger.info(f"CLIPTextEncoder initialized with model: {clip_cfg['model_name']}")
        else:
            self.clip_encoder = None
            logger.info("CLIPTextEncoder not required/initialized as MS-COCO tasks are disabled.")

        # 4. Initialize Data Module
        active_training_config: Dict[str, Any] = {}
        if self.config['training']['imagenet']['enabled']:
            active_training_config = Config.get_training_config('imagenet')
            logger.info("ImageNet training is enabled.")
        elif self.config['training']['mscoco']['enabled']:
            active_training_config = Config.get_training_config('mscoco')
            logger.info("MS-COCO training is enabled.")
        else:
            logger.warning("No training task is enabled. DataModule will be initialized for evaluation context if needed.")
        
        # DataModule is initialized with the full global config (for easier access to nested parameters)
        # and the specific training config parameters are extracted within DataModule.
        self.data_module = DataModule(
            config=self.config, # Pass the full config object
            tokenizer=self.vae_tokenizer,
            clip_encoder=self.clip_encoder
        )
        logger.info("DataModule initialized.")

        # 5. Instantiate Hi-MAR Model
        model_variant = self.config['model_config']['variant']
        model_config_for_init = Config.get_model_config(model_variant)
        # Add VAE downsampling factor to model config, for DiffusionTransformerHead
        # It's an internal property of VAETokenizer, which needs to be passed to model.py
        model_config_for_init["vae_downsampling_factor"] = self.vae_tokenizer.vae_downsampling_factor

        self.model = HiMARModel(
            config=model_config_for_init,
            tokenizer_latent_channels=self.vae_tokenizer.latent_channels,
            num_scales=model_config_for_init['num_scales']
        ).to(self.device)
        logger.info(f"HiMARModel ({model_variant}) initialized with {sum(p.numel() for p in self.model.parameters() if p.requires_grad)/1e6:.2f}M trainable parameters.")

        # 6. Initialize and Run Trainer (if enabled)
        if self.config['training']['imagenet']['enabled'] or self.config['training']['mscoco']['enabled']:
            self.trainer = Trainer(
                model=self.model,
                data_module=self.data_module,
                global_config=self.config, # Pass full config to trainer
                device=self.device
            )
            logger.info("Trainer initialized. Starting training...")
            self.trainer.train()
            logger.info("Training complete.")
            # Use EMA model from trainer for subsequent steps
            model_for_generation = self.trainer.ema_model.ema_model 
        else:
            logger.info("Training is disabled. Skipping trainer initialization.")
            # If no training, use the initialized model directly (assuming it's a pre-trained checkpoint if evaluation is enabled)
            # In a real scenario, you'd load a checkpoint here. For this exercise, we assume evaluation
            # without training uses the freshly initialized model, or an external mechanism loads weights.
            # The prompt does not specify checkpoint loading if training is skipped.
            model_for_generation = self.model
        
        # 7. Initialize Generator
        self.generator = Generator(
            model=model_for_generation,
            tokenizer=self.vae_tokenizer,
            clip_encoder=self.clip_encoder,
            config=self.config, # Pass full config to generator
            device=self.device
        )
        logger.info("Generator initialized.")

        # 8. Initialize and Run Evaluator (if enabled)
        imagenet_eval_enabled = self.config['evaluation']['imagenet']['enabled']
        mscoco_eval_enabled = self.config['evaluation']['mscoco']['enabled']

        if imagenet_eval_enabled or mscoco_eval_enabled:
            self.evaluator = Evaluator(
                generator=self.generator,
                config=self.config, # Pass full config to evaluator
                device=self.device
            )
            logger.info("Evaluator initialized. Starting evaluation...")

            if imagenet_eval_enabled:
                imagenet_eval_cfg = Config.get_evaluation_config('imagenet')
                imagenet_conditions: List[int] = list(range(1000)) # ImageNet has 1000 classes
                num_samples_imagenet = imagenet_eval_cfg['num_samples']
                guidance_scale = Config.get_generation_config()['guidance_scale']

                # Perform evaluation as per paper's Table 2 structure
                # The `Evaluator.evaluate_imagenet` method is designed to handle both w/CFG and w/o CFG settings internally.
                eval_results_imagenet = self.evaluator.evaluate_imagenet(
                    conditions=imagenet_conditions,
                    num_samples=num_samples_imagenet,
                    guidance_scale=guidance_scale
                )
                logger.info(f"ImageNet Evaluation Results: {eval_results_imagenet}")

            if mscoco_eval_enabled:
                mscoco_eval_cfg = Config.get_evaluation_config('mscoco')
                num_samples_mscoco = mscoco_eval_cfg['num_samples']
                guidance_scale = Config.get_generation_config()['guidance_scale']
                prompts_file = mscoco_eval_cfg['evaluation_prompts_file']

                if not os.path.exists(prompts_file):
                    logger.error(f"MS-COCO evaluation prompts file not found at {prompts_file}. Skipping MS-COCO evaluation.")
                else:
                    with open(prompts_file, 'r') as f:
                        mscoco_prompts: List[str] = [p.strip() for p in f.readlines()]
                        # If a specific number of samples is required, truncate or pad prompts.
                        # The evaluator handles this within its `evaluate_mscoco` method.
                    logger.info(f"Loaded {len(mscoco_prompts)} prompts for MS-COCO evaluation.")

                    eval_results_mscoco = self.evaluator.evaluate_mscoco(
                        prompts=mscoco_prompts,
                        num_samples=num_samples_mscoco,
                        guidance_scale=guidance_scale
                    )
                    logger.info(f"MS-COCO Evaluation Results: {eval_results_mscoco}")
            logger.info("Evaluation complete.")
        else:
            logger.info("Evaluation is disabled. Skipping evaluator initialization.")

        logger.info("Experiment run finished.")


if __name__ == "__main__":
    main_app = Main()
    main_app.run_experiment(main_app.args.config_path)

