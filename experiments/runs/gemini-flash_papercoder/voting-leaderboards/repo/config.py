import yaml
from typing import Dict, List, Any, Optional

class Config:
    """
    Centralized configuration class for the entire reproduction pipeline.
    Loads settings from a YAML file and makes them accessible as class attributes.
    """

    # General Configuration
    RANDOM_SEED: int
    DETECTOR_OUTPUT_TOKEN_LENGTH: int

    # LLM API Keys
    LLM_API_KEYS: Dict[str, str]

    # List of Models to Evaluate
    MODEL_LIST: List[Dict[str, str]]

    # Identity-Probing Detector Configuration
    DETECTOR_PROMPTS_IDENTITY: List[str]
    DETECTOR_NUM_QUERIES_PER_PROMPT_IDENTITY: int

    # Training-Based Detector Configuration
    PROMPT_SOURCES: Dict[str, str]
    DETECTOR_PROMPTS_TRAINING_CATEGORIES: List[str]
    DETECTOR_NUM_PROMPTS_PER_CATEGORY: int
    DETECTOR_RESPONSES_PER_MODEL_PER_PROMPT: int
    DETECTOR_FEATURES: List[str]
    DETECTOR_CLASSIFIER_TYPE: str
    DETECTOR_CLASSIFIER_HYPERPARAMETERS: Dict[str, Any]
    DETECTOR_TRAIN_TEST_SPLIT_RATIO: float
    DETECTOR_LOGISTIC_REGRESSION_RANDOM_STATE: int
    DETECTOR_PCA_VISUALIZATION_PROMPTS: List[str]

    # Leaderboard Simulator Configuration
    SIMULATOR_DETECTOR_ACCURACY: float
    SIMULATOR_FALSE_POSITIVE_RATE: float
    SIMULATOR_FALSE_NEGATIVE_RATE: float
    SIMULATOR_DEFAULT_NON_DETECTION_STRATEGY: str
    SIMULATOR_BRADLEY_TERRY_INITIAL_RATING: float
    SIMULATOR_BRADLEY_TERRY_SCALE_FACTOR: float
    SIMULATOR_INTERACTION_TRACKING_INTERVAL: int
    SIMULATOR_HISTORICAL_VOTES_PATH: str

    # Mitigation Analysis Configuration
    MITIGATION_DETECTOR_COST: float
    MITIGATION_MALICIOUS_DETECTION_ALPHA: float
    MITIGATION_MALICIOUS_DETECTION_SIM_SEQUENCES: int

    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initializes the Config object by loading settings from the specified YAML file.

        Args:
            config_path (str): The file path to the config.yaml configuration file.
        """
        try:
            with open(config_path, 'r') as f:
                _config_data = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file: {e}")

        # General Configuration
        general_config = _config_data.get('general', {})
        self.RANDOM_SEED = general_config.get('random_seed', 42)
        self.DETECTOR_OUTPUT_TOKEN_LENGTH = general_config.get('output_token_length', 512)

        # LLM API Keys
        self.LLM_API_KEYS = _config_data.get('llm_api_keys', {})

        # List of Models to Evaluate
        self.MODEL_LIST = _config_data.get('model_list', [])

        # Identity-Probing Detector Configuration
        detector_identity_config = _config_data.get('detector_identity', {})
        self.DETECTOR_PROMPTS_IDENTITY = detector_identity_config.get('prompts', [])
        self.DETECTOR_NUM_QUERIES_PER_PROMPT_IDENTITY = detector_identity_config.get('num_queries_per_prompt', 1000)

        # Training-Based Detector Configuration
        detector_training_config = _config_data.get('detector_training', {})
        self.PROMPT_SOURCES = detector_training_config.get('prompt_sources', {})
        # DETECTOR_PROMPTS_TRAINING_CATEGORIES is derived from the keys of PROMPT_SOURCES
        self.DETECTOR_PROMPTS_TRAINING_CATEGORIES = list(self.PROMPT_SOURCES.keys())
        self.DETECTOR_NUM_PROMPTS_PER_CATEGORY = detector_training_config.get('num_prompts_per_category', 200)
        self.DETECTOR_RESPONSES_PER_MODEL_PER_PROMPT = detector_training_config.get('responses_per_model_per_prompt', 50)
        self.DETECTOR_FEATURES = detector_training_config.get('features', ["BoW(R)"])
        self.DETECTOR_CLASSIFIER_TYPE = detector_training_config.get('classifier_type', "LogisticRegression")
        self.DETECTOR_CLASSIFIER_HYPERPARAMETERS = detector_training_config.get('classifier_hyperparameters', {
            "solver": "lbfgs",
            "penalty": "l2",
            "max_iter": 100
        })
        self.DETECTOR_TRAIN_TEST_SPLIT_RATIO = detector_training_config.get('train_test_split_ratio', 0.2)
        self.DETECTOR_LOGISTIC_REGRESSION_RANDOM_STATE = detector_training_config.get('logistic_regression_random_state', 42)
        self.DETECTOR_PCA_VISUALIZATION_PROMPTS = detector_training_config.get('pca_visualization_prompts', [])

        # Leaderboard Simulator Configuration
        simulator_config = _config_data.get('simulator', {})
        self.SIMULATOR_DETECTOR_ACCURACY = simulator_config.get('detector_accuracy', 0.95)
        self.SIMULATOR_FALSE_POSITIVE_RATE = simulator_config.get('false_positive_rate', 0.05)
        self.SIMULATOR_FALSE_NEGATIVE_RATE = simulator_config.get('false_negative_rate', 0.05)
        self.SIMULATOR_DEFAULT_NON_DETECTION_STRATEGY = simulator_config.get('default_non_detection_strategy', "do_nothing")
        self.SIMULATOR_BRADLEY_TERRY_INITIAL_RATING = simulator_config.get('bradley_terry_initial_rating', 1500.0)
        self.SIMULATOR_BRADLEY_TERRY_SCALE_FACTOR = simulator_config.get('bradley_terry_scale_factor', 173.7)
        self.SIMULATOR_INTERACTION_TRACKING_INTERVAL = simulator_config.get('interaction_tracking_interval', 1000)
        self.SIMULATOR_HISTORICAL_VOTES_PATH = simulator_config.get('historical_votes_path', "data/simulated_historical_votes.json")

        # Mitigation Analysis Configuration
        mitigation_config = _config_data.get('mitigation', {})
        self.MITIGATION_DETECTOR_COST = mitigation_config.get('detector_training_cost', 440.0)
        self.MITIGATION_MALICIOUS_DETECTION_ALPHA = mitigation_config.get('malicious_detection_alpha', 0.01)
        self.MITIGATION_MALICIOUS_DETECTION_SIM_SEQUENCES = mitigation_config.get('malicious_detection_sim_sequences', 1000)

    def get_model_api_config(self, model_name: str) -> Optional[Dict[str, str]]:
        """
        Retrieves the API configuration specific to a given model.

        Args:
            model_name (str): The identifier of the model for which to retrieve configuration.

        Returns:
            Optional[Dict[str, str]]: A dictionary containing model configuration
                                      (model_id, company, query_method) if found,
                                      otherwise None.
        """
        for model_info in self.MODEL_LIST:
            if model_info.get('model_id') == model_name:
                return model_info
        return None

