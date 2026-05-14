"""Configuration loader for voting-leaderboard experiments."""
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class DetectorConfig:
    identity_prompts: List[str] = field(default_factory=list)
    num_prompts_per_category: int = 200
    num_responses_per_model: int = 50
    train_test_split: float = 0.8
    random_state: int = 42
    output_tokens: int = 512
    classifier: str = "logistic_regression"
    prompt_categories: Dict = field(default_factory=dict)
    features: List[str] = field(default_factory=list)


@dataclass
class ModelConfig:
    name: str = ""
    provider: str = ""
    api_type: str = ""


@dataclass
class SimulationConfig:
    total_votes: int = 1670250
    total_users: int = 477322
    total_wins: int = 1093875
    total_ties: int = 576375
    unique_pairs: int = 6895
    detector_accuracy: float = 0.95
    false_positive_rate: float = 0.05
    false_negative_rate: float = 0.05
    steps_per_check: int = 1000
    bradley_terry_scale: float = 1.0
    attack_types: List[str] = field(default_factory=list)
    non_detection_strategies: List[str] = field(default_factory=list)


@dataclass
class CostModelConfig:
    detector_cost: float = 440.0
    action_cost: float = 0.01


@dataclass
class AuthenticationConfig:
    enabled: bool = False
    account_cost: float = 1.0


@dataclass
class RateLimitingConfig:
    enabled: bool = False
    max_actions_per_account: int = 100
    quantile: float = 0.5


@dataclass
class MaliciousDetectionScenario1Config:
    significance_level: float = 0.01
    num_simulations: int = 1000


@dataclass
class MaliciousDetectionScenario2Config:
    noise_scale: float = 0.0
    noise_scales_to_test: List[float] = field(default_factory=list)


@dataclass
class MaliciousDetectionConfig:
    scenario1: MaliciousDetectionScenario1Config = field(default_factory=MaliciousDetectionScenario1Config)
    scenario2: MaliciousDetectionScenario2Config = field(default_factory=MaliciousDetectionScenario2Config)


@dataclass
class CaptchaConfig:
    enabled: bool = False
    captcha_cost: float = 0.001


@dataclass
class PromptUniquenessConfig:
    enabled: bool = False
    cost_per_prompt: float = 20.0


@dataclass
class MitigationsConfig:
    cost_model: CostModelConfig = field(default_factory=CostModelConfig)
    authentication: AuthenticationConfig = field(default_factory=AuthenticationConfig)
    rate_limiting: RateLimitingConfig = field(default_factory=RateLimitingConfig)
    malicious_detection: MaliciousDetectionConfig = field(default_factory=MaliciousDetectionConfig)
    captcha: CaptchaConfig = field(default_factory=CaptchaConfig)
    prompt_uniqueness: PromptUniquenessConfig = field(default_factory=PromptUniquenessConfig)


@dataclass
class APIPricingConfig:
    proprietary_per_million_tokens: float = 5.00
    open_source_per_million_tokens: float = 1.80


@dataclass
class Config:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    models: List[ModelConfig] = field(default_factory=list)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    mitigations: MitigationsConfig = field(default_factory=MitigationsConfig)
    api_pricing: APIPricingConfig = field(default_factory=APIPricingConfig)


def load_config(path: str = "config.yaml") -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    config = Config()

    if "detector" in raw:
        d = raw["detector"]
        config.detector = DetectorConfig(
            identity_prompts=d.get("identity_prompts", []),
            num_prompts_per_category=d.get("training_based", {}).get("num_prompts_per_category", 200),
            num_responses_per_model=d.get("training_based", {}).get("num_responses_per_model", 50),
            train_test_split=d.get("training_based", {}).get("train_test_split", 0.8),
            random_state=d.get("training_based", {}).get("random_state", 42),
            output_tokens=d.get("training_based", {}).get("output_tokens", 512),
            classifier=d.get("training_based", {}).get("classifier", "logistic_regression"),
            prompt_categories=d.get("prompt_categories", {}),
            features=d.get("features", []),
        )

    if "models" in raw:
        config.models = [
            ModelConfig(name=m["name"], provider=m["provider"], api_type=m["api_type"])
            for m in raw["models"]
        ]

    if "simulation" in raw:
        s = raw["simulation"]
        config.simulation = SimulationConfig(
            total_votes=s.get("total_votes", 1670250),
            total_users=s.get("total_users", 477322),
            total_wins=s.get("total_wins", 1093875),
            total_ties=s.get("total_ties", 576375),
            unique_pairs=s.get("unique_pairs", 6895),
            detector_accuracy=s.get("detector_accuracy", 0.95),
            false_positive_rate=s.get("false_positive_rate", 0.05),
            false_negative_rate=s.get("false_negative_rate", 0.05),
            steps_per_check=s.get("steps_per_check", 1000),
            bradley_terry_scale=s.get("bradley_terry_scale", 1.0),
            attack_types=s.get("attack_types", []),
            non_detection_strategies=s.get("non_detection_strategies", []),
        )

    if "mitigations" in raw:
        m = raw["mitigations"]
        config.mitigations = MitigationsConfig(
            cost_model=CostModelConfig(
                detector_cost=m.get("cost_model", {}).get("detector_cost", 440.0),
                action_cost=m.get("cost_model", {}).get("action_cost", 0.01),
            ),
            authentication=AuthenticationConfig(
                enabled=m.get("authentication", {}).get("enabled", False),
                account_cost=m.get("authentication", {}).get("account_cost", 1.0),
            ),
            rate_limiting=RateLimitingConfig(
                enabled=m.get("rate_limiting", {}).get("enabled", False),
                max_actions_per_account=m.get("rate_limiting", {}).get("max_actions_per_account", 100),
                quantile=m.get("rate_limiting", {}).get("quantile", 0.5),
            ),
            malicious_detection=MaliciousDetectionConfig(
                scenario1=MaliciousDetectionScenario1Config(
                    significance_level=m.get("malicious_detection", {}).get("scenario1", {}).get("significance_level", 0.01),
                    num_simulations=m.get("malicious_detection", {}).get("scenario1", {}).get("num_simulations", 1000),
                ),
                scenario2=MaliciousDetectionScenario2Config(
                    noise_scale=m.get("malicious_detection", {}).get("scenario2", {}).get("noise_scale", 0.0),
                    noise_scales_to_test=m.get("malicious_detection", {}).get("scenario2", {}).get("noise_scales_to_test", []),
                ),
            ),
            captcha=CaptchaConfig(
                enabled=m.get("captcha", {}).get("enabled", False),
                captcha_cost=m.get("captcha", {}).get("captcha_cost", 0.001),
            ),
            prompt_uniqueness=PromptUniquenessConfig(
                enabled=m.get("prompt_uniqueness", {}).get("enabled", False),
                cost_per_prompt=m.get("prompt_uniqueness", {}).get("cost_per_prompt", 20.0),
            ),
        )

    if "api_pricing" in raw:
        p = raw["api_pricing"]
        config.api_pricing = APIPricingConfig(
            proprietary_per_million_tokens=p.get("proprietary_per_million_tokens", 5.00),
            open_source_per_million_tokens=p.get("open_source_per_million_tokens", 1.80),
        )

    return config
