import yaml
from typing import Optional, Dict, Any, Tuple

class Config:
    """
    Configuration class to load and manage hyperparameters and settings
    from a YAML file.
    """

    def __init__(self, config_path: str, task_name: Optional[str] = None, stage_name: Optional[str] = None):
        """
        Initializes the Config object by loading settings from the specified YAML file.

        Args:
            config_path (str): Path to the YAML configuration file.
            task_name (Optional[str]): The name of the active training/evaluation task.
                                       If provided, `set_active_task` will be called immediately.
                                       Example: "t2v_internvid", "vp_skytimelapse".
            stage_name (Optional[str]): The specific stage within the active task.
                                        If provided, `set_active_task` will be called immediately.
                                        Example: "ca2_vdm_stage1", "os_fix_baseline".
        """
        self._raw_config: Dict[str, Any] = self._load_config_file(config_path)

        # General settings
        self.vae_model_name: str = self._get_config_value(["general", "vae_model_name"], str)
        self.text_encoder_name: str = self._get_config_value(["general", "text_encoder_name"], str)
        self.open_sora_config_path: str = self._get_config_value(["general", "open_sora_config_path"], str)
        self.image_size: int = self._get_config_value(["general", "image_size"], int)
        self.diffusion_steps: int = self._get_config_value(["general", "diffusion_steps"], int)
        self.beta_schedule: str = self._get_config_value(["general", "beta_schedule"], str)
        self.beta_start: float = self._get_config_value(["general", "beta_start"], float)
        self.beta_end: float = self._get_config_value(["general", "beta_end"], float)
        self.num_inference_steps: int = self._get_config_value(["general", "num_inference_steps"], int)
        self.guidance_scale: float = self._get_config_value(["general", "guidance_scale"], float)
        self.use_prefix_enhancement: bool = self._get_config_value(["general", "use_prefix_enhancement"], bool)
        self.prefix_enhancement_sub_len: int = self._get_config_value(["general", "prefix_enhancement_sub_len"], int)
        self.data_path: str = self._get_config_value(["general", "data_path"], str)
        self.save_path: str = self._get_config_value(["general", "save_path"], str)

        # General training settings
        self.optimizer: str = self._get_config_value(["training", "optimizer"], str)
        self.learning_rate: float = self._get_config_value(["training", "learning_rate"], float)

        # Evaluation settings
        self.num_generated_samples_fvd: int = self._get_config_value(["evaluation", "num_generated_samples_fvd"], int)
        self.fvd_chunk_size: int = self._get_config_value(["evaluation", "fvd_chunk_size"], int)

        # Task-specific attributes, initialized to None and set by set_active_task
        self.chunk_length: Optional[int] = None
        self.max_prefix_length: Optional[int] = None
        self.max_train_video_length: Optional[int] = None # For Ca2-VDM stages (P_max + l)
        self.train_video_length: Optional[int] = None    # For OS-Fix baselines (fixed L_train)
        self.prefix_length: Optional[int] = None         # For OS-Fix baselines (fixed P)
        self.batch_size: Optional[int] = None
        self.training_steps: Optional[int] = None

        # Store raw task configurations for dynamic access
        self._t2v_internvid_config: Dict[str, Any] = self._raw_config["training"].get("t2v_internvid", {})
        self._vp_skytimelapse_config: Dict[str, Any] = self._raw_config["training"].get("vp_skytimelapse", {})

        if task_name and stage_name:
            self.set_active_task(task_name, stage_name)
        elif task_name and not stage_name:
            # For VP SkyTimelapse, stages are implied in how parameters are structured
            if task_name == "vp_skytimelapse":
                # Assuming 'ca2_vdm' or 'os_ext' for these parameters for VP.
                # In config.yaml, batch_size and training_steps are directly under vp_skytimelapse,
                # and chunk_length, max_prefix_length, max_train_video_length are also there.
                # This suggests these are the 'main' VP task parameters, not tied to a specific sub-stage.
                self.set_active_task(task_name, "ca2_vdm") # Use a generic stage name for base VP
            else:
                raise ValueError(f"Stage name is required for task '{task_name}' if not 'vp_skytimelapse'")


    def _load_config_file(self, config_path: str) -> Dict[str, Any]:
        """Loads the YAML configuration file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")

    def _get_config_value(self, path: list, expected_type: type) -> Any:
        """
        Retrieves a value from the nested config dictionary based on a path.
        Raises an error if the path does not exist or the type is incorrect.
        """
        current_config = self._raw_config
        for key in path:
            if not isinstance(current_config, dict) or key not in current_config:
                raise KeyError(f"Missing configuration key: {'->'.join(path)}")
            current_config = current_config[key]
        if not isinstance(current_config, expected_type):
            raise TypeError(f"Configuration key {'->'.join(path)} has unexpected type. "
                            f"Expected {expected_type}, got {type(current_config)}")
        return current_config

    @property
    def betas(self) -> Tuple[float, float]:
        """Returns the beta start and end values as a tuple."""
        return (self.beta_start, self.beta_end)

    def set_active_task(self, task_name: str, stage_name: Optional[str] = None):
        """
        Dynamically sets task-specific configuration parameters.

        Args:
            task_name (str): The name of the active task (e.g., "t2v_internvid", "vp_skytimelapse").
            stage_name (Optional[str]): The specific stage within the task (e.g., "ca2_vdm_stage1", "os_fix_baseline").
                                        Can be None if the task doesn't have explicit stages for its primary settings.
        Raises:
            ValueError: If an unknown task_name or stage_name is provided.
        """
        if task_name == "t2v_internvid":
            task_config = self._t2v_internvid_config
            if stage_name == "ca2_vdm_stage1":
                self.chunk_length = 0 # No chunking concept for stage1, or effectively 0 for conditional frames
                self.max_prefix_length = 0 # No clean prefix for stage1
                self.max_train_video_length = 32 # Train on 32-frame videos
                self.batch_size = task_config["ca2_vdm_stage1"]["batch_size"]
                self.training_steps = task_config["ca2_vdm_stage1"]["training_steps"]
                self.train_video_length = None # Not applicable for stage1
                self.prefix_length = None      # Not applicable for stage1
            elif stage_name == "ca2_vdm_stage2":
                self.chunk_length = task_config["chunk_length"]
                self.max_prefix_length = task_config["max_prefix_length"]
                self.max_train_video_length = task_config["max_train_video_length"]
                self.batch_size = task_config["ca2_vdm_stage2"]["batch_size"]
                self.training_steps = task_config["ca2_vdm_stage2"]["training_steps"]
                self.train_video_length = None
                self.prefix_length = None
            elif stage_name == "os_fix_baseline":
                os_fix_config = task_config["os_fix_baseline"]
                self.train_video_length = os_fix_config["train_video_length"]
                self.prefix_length = os_fix_config["prefix_length"]
                self.batch_size = os_fix_config["batch_size"]
                self.training_steps = os_fix_config["training_steps"]
                self.chunk_length = None
                self.max_prefix_length = None
                self.max_train_video_length = None
            else:
                raise ValueError(f"Unknown stage_name '{stage_name}' for task 't2v_internvid'. "
                                 f"Expected 'ca2_vdm_stage1', 'ca2_vdm_stage2', or 'os_fix_baseline'.")
        elif task_name == "vp_skytimelapse":
            task_config = self._vp_skytimelapse_config
            # For VP SkyTimelapse, batch_size and training_steps are general for the task
            # and chunk_length, max_prefix_length, max_train_video_length apply to Ca2-VDM/OS-Ext,
            # while os_fix_baseline has its own train_video_length and prefix_length.
            self.batch_size = task_config["batch_size"]
            self.training_steps = task_config["training_steps"]

            if stage_name in ["ca2_vdm", "os_ext"]: # 'os_ext' is a baseline similar to Ca2-VDM in terms of extendable condition
                self.chunk_length = task_config["chunk_length"]
                self.max_prefix_length = task_config["max_prefix_length"]
                self.max_train_video_length = task_config["max_train_video_length"]
                self.train_video_length = None
                self.prefix_length = None
            elif stage_name == "os_fix_baseline":
                os_fix_config = task_config["os_fix_baseline"]
                self.train_video_length = os_fix_config["train_video_length"]
                self.prefix_length = os_fix_config["prefix_length"]
                self.chunk_length = None
                self.max_prefix_length = None
                self.max_train_video_length = None
            else:
                raise ValueError(f"Unknown stage_name '{stage_name}' for task 'vp_skytimelapse'. "
                                 f"Expected 'ca2_vdm', 'os_ext', or 'os_fix_baseline'.")
        else:
            raise ValueError(f"Unknown task_name '{task_name}'. Expected 't2v_internvid' or 'vp_skytimelapse'.")

    def __str__(self) -> str:
        """Provides a string representation of the current configuration."""
        return f"Config(vae_model_name='{self.vae_model_name}', image_size={self.image_size}, " \
               f"learning_rate={self.learning_rate}, chunk_length={self.chunk_length}, " \
               f"batch_size={self.batch_size}, training_steps={self.training_steps}, ...)"

