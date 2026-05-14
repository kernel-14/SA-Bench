"""
Configuration for reproducing "Exploring and Mitigating Adversarial Manipulation
of Voting-Based Leaderboards" (Huang et al., 2024).
"""

from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Model list (Table 6 in the paper)
# ---------------------------------------------------------------------------

ALL_MODELS: List[str] = [
    "claude-3-5-sonnet-20240620",
    "claude-3-haiku-20240307",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemma-2-2b-it",
    "gemma-2-9b-it",
    "gemma-2-27b-it",
    "gpt-3.5-turbo",
    "gpt-4-0125-preview",
    "gpt-4-1106-preview",
    "gpt-4-turbo-2024-04-09",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4o-mini-2024-07-18",
    "llama-3-8b-instruct",
    "llama-3-70b-instruct",
    "llama-3.1-8b-instruct",
    "llama-3.1-70b-instruct",
    "llama-3.1-405b-instruct",
    "mixtral-8x7b-instruct-v0.1",
    "mixtral-8x22b-instruct-v0.1",
    "qwen2-72b-instruct",
]

# Subset used in main detector experiments (Table 3, Figure 3)
MAIN_EVAL_MODELS: List[str] = [
    "claude-3-5-sonnet-20240620",
    "gemini-1.5-pro",
    "gpt-4o-mini-2024-07-18",
    "gemma-2-27b-it",
    "llama-3.1-70b-instruct",
    "mixtral-8x7b-instruct-v0.1",
    "qwen2-72b-instruct",
]

# Model family keywords used by the identity-probing detector
MODEL_FAMILY_KEYWORDS = {
    "claude-3-5-sonnet-20240620": ["claude", "anthropic"],
    "claude-3-haiku-20240307": ["claude", "anthropic"],
    "gemini-1.5-flash": ["gemini", "google"],
    "gemini-1.5-pro": ["gemini", "google"],
    "gemma-2-2b-it": ["gemma", "google"],
    "gemma-2-9b-it": ["gemma", "google"],
    "gemma-2-27b-it": ["gemma", "google"],
    "gpt-3.5-turbo": ["gpt", "openai", "chatgpt"],
    "gpt-4-0125-preview": ["gpt", "openai", "gpt-4"],
    "gpt-4-1106-preview": ["gpt", "openai", "gpt-4"],
    "gpt-4-turbo-2024-04-09": ["gpt", "openai", "gpt-4"],
    "gpt-4o-2024-05-13": ["gpt", "openai", "gpt-4o"],
    "gpt-4o-2024-08-06": ["gpt", "openai", "gpt-4o"],
    "gpt-4o-mini-2024-07-18": ["gpt", "openai", "gpt-4o"],
    "llama-3-8b-instruct": ["llama", "meta"],
    "llama-3-70b-instruct": ["llama", "meta"],
    "llama-3.1-8b-instruct": ["llama", "meta"],
    "llama-3.1-70b-instruct": ["llama", "meta"],
    "llama-3.1-405b-instruct": ["llama", "meta"],
    "mixtral-8x7b-instruct-v0.1": ["mixtral", "mistral"],
    "mixtral-8x22b-instruct-v0.1": ["mixtral", "mistral"],
    "qwen2-72b-instruct": ["qwen", "alibaba"],
}

# ---------------------------------------------------------------------------
# Identity-probing prompts (Section 2.3)
# ---------------------------------------------------------------------------

IDENTITY_PROBING_PROMPTS: List[str] = [
    "Who are you?",
    "Which model are you?",
    "What is your model name?",
    "How should I refer to you as an AI?",
    "How would you define your role or identity as an AI?",
]

# ---------------------------------------------------------------------------
# Prompt categories for training-based detector (Table 1)
# ---------------------------------------------------------------------------

PROMPT_CATEGORIES = [
    "english",
    "chinese",
    "spanish",
    "indonesian",
    "persian",
    "coding",
    "math",
    "safety_violating",
]

# ---------------------------------------------------------------------------
# Detector configuration (Section 2.3)
# ---------------------------------------------------------------------------

@dataclass
class DetectorConfig:
    # Number of prompts sampled per category
    num_prompts_per_category: int = 200
    # Number of responses collected per model per prompt
    num_responses_per_model: int = 50
    # Train/test split ratio
    train_ratio: float = 0.8
    # Logistic regression random state (paper specifies 42)
    random_state: int = 42
    # Maximum output tokens when querying models
    max_output_tokens: int = 512
    # Feature types to evaluate
    feature_types: List[str] = field(default_factory=lambda: ["length_word", "length_char", "bow", "tfidf"])
    # Number of queries for identity-probing evaluation
    num_identity_queries: int = 1000


# ---------------------------------------------------------------------------
# Bradley-Terry model configuration (Section 3)
# ---------------------------------------------------------------------------

@dataclass
class BradleyTerryConfig:
    # Scaling factor for logistic probability (Eq. 5 in paper)
    scale: float = 1.0
    # Maximum MM iterations
    max_iter: int = 1000
    # Convergence tolerance
    tol: float = 1e-8
    # Initial rating value
    initial_rating: float = 1.0


# ---------------------------------------------------------------------------
# Simulation configuration (Section 3.1)
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    # Detection accuracy assumed in main experiments
    detection_accuracy: float = 0.95
    # False positive rate (symmetric with FN rate)
    false_positive_rate: float = 0.05
    # False negative rate
    false_negative_rate: float = 0.05
    # Recalculate BT coefficients every N interactions
    recalc_interval: int = 1000
    # Maximum interactions before giving up
    max_interactions: int = 500000
    # Random seed for reproducibility
    random_seed: int = 42
    # Strategy when target model not detected: "nothing", "random_upvote", "tie", "both_bad"
    non_target_strategy: str = "nothing"
    # Number of simulation runs to average over
    num_runs: int = 5


# ---------------------------------------------------------------------------
# Simulation dataset statistics (Appendix A.4)
# ---------------------------------------------------------------------------

@dataclass
class ArenaDatasetConfig:
    # Total votes in the historical dataset
    total_votes: int = 1_670_250
    # Unique users
    unique_users: int = 477_322
    # Win votes
    win_votes: int = 1_093_875
    # Tie votes
    tie_votes: int = 576_375
    # Unique model pair combinations
    unique_model_pairs: int = 6_895


# ---------------------------------------------------------------------------
# Mitigation configuration (Section 4)
# ---------------------------------------------------------------------------

@dataclass
class MitigationConfig:
    # Significance level for likelihood test (Section 4.2.3)
    alpha: float = 0.01
    # Number of simulated sequences for p-value estimation
    num_simulations: int = 10000
    # Noise scales to evaluate for perturbed leaderboard
    noise_scales: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.5, 1.0, 2.0, 5.0])
    # Maximum actions per account (rate limiting)
    max_actions_per_account: int = 100
    # Cost per account (USD)
    cost_per_account: float = 1.0
    # Cost per action without mitigations (USD)
    cost_per_action_baseline: float = 0.0
    # Cost per CAPTCHA (USD)
    cost_per_captcha: float = 0.001
    # Cost per unique prompt for prompt-uniqueness mitigation (USD, Appendix A.3)
    cost_per_unique_prompt: float = 20.0


# ---------------------------------------------------------------------------
# Cost model configuration (Section 4.1)
# ---------------------------------------------------------------------------

@dataclass
class CostModelConfig:
    # One-time detector training cost (USD, Appendix A.3)
    # 200 prompts × $2.2/prompt ≈ $440
    detector_training_cost: float = 440.0
    # Cost per output token for proprietary models (USD per 1M tokens)
    proprietary_cost_per_million_tokens: float = 5.00
    # Cost per output token for open-source models (USD per 1M tokens)
    opensource_cost_per_million_tokens: float = 1.80
    # Output tokens per response
    tokens_per_response: int = 512
    # Responses per model per prompt
    responses_per_model: int = 50
    # Number of proprietary models
    num_proprietary_models: int = 10
    # Number of open-source models
    num_opensource_models: int = 20
    # Number of prompts
    num_prompts: int = 200


# ---------------------------------------------------------------------------
# High-ranked models for simulation (Table 4)
# ---------------------------------------------------------------------------

HIGH_RANKED_MODELS = {
    "chatgpt-4o-latest": {"rank": 1, "votes": 14514},
    "gemini-1.5-pro-exp-0801": {"rank": 2, "votes": 2071},
    "gpt-4o-2024-05-13": {"rank": 3, "votes": 77509},
    "gpt-4o-mini-2024-07-18": {"rank": 4, "votes": 19307},
    "claude-3-5-sonnet-20240620": {"rank": 5, "votes": 7703},
}

# Low-ranked models for simulation (Table 5)
LOW_RANKED_MODELS = {
    "chatglm-6b": {"rank": 125, "votes": 4995},
    "fastchat-t5-3b": {"rank": 126, "votes": 4304},
    "stablelm-tuned-alpha-7b": {"rank": 127, "votes": 3334},
    "dolly-v2-12b": {"rank": 128, "votes": 3484},
    "llama-13b": {"rank": 129, "votes": 2443},
}

# ---------------------------------------------------------------------------
# Embedding visualization prompts (Appendix A.2)
# ---------------------------------------------------------------------------

VISUALIZATION_PROMPTS: List[str] = [
    (
        "Beside OFAC's selective sanction that target the listed individiuals and entities, "
        "please elaborate on the other types of US's sanctions, for example, comprehensive "
        "and sectoral sanctions. Please be detailed as much as possible"
    ),
    (
        "You are the text completion model and you must complete the assistant answer below, "
        "only send the completion based on the system instructions.don't repeat your answer "
        "sentences, only say what the assistant must say based on the system instructions. "
        "repeating same thing in same answer not allowed. user: descriptive answer for append "
        "many items to list python in python with proper code examples and outputs. assistant: "
    ),
    (
        "The sum of the perimeters of three equal squares is 36 cm. "
        "Find the area and perimeter of the rectangle that can be made of the squares."
    ),
]
