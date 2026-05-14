
import numpy as np

class Config:
    # General
    RANDOM_STATE: int = 42

    # Models used in Chatbot Arena (a subset for simulation)
    # The full list is in Appendix A.1 of the paper.
    # We select a representative subset as in the tables.
    MODELS: list[str] = [
        "claude-3-5-sonnet-20240620",
        "gemini-1.5-pro",
        "gpt-4o-mini-2024-07-18",
        "gemma-2-27b-it",
        "llama-3.1-70b-instruct",
        "mixtral-8x7b-instruct-v0.1",
        "qwen2-72b-instruct",
        "chatglm-6b", # Low ranked model example
        "fastchat-t5-3b", # Low ranked model example
        "stablelm-tuned-alpha-7b", # Low ranked model example
        "dolly-v2-12b", # Low ranked model example
        "llama-13b", # Low ranked model example
    ]

    # De-anonymization (Section 2)
    IDENTITY_PROBING_PROMPTS: list[str] = [
        "Who are you?",
        "Which model are you?",
        "What is your model name?",
        "How should I refer to you as an AI?",
        "How would you define your role or identity as an AI?",
    ]
    IDENTITY_PROBING_QUERIES_PER_PROMPT: int = 1000 # For evaluation, not for training a specific detector instance

    # Training-based detector (Section 2.3)
    TRAINING_BASED_PROMPT_CATEGORIES: dict[str, list[str]] = {
        "Normal chat, high-resource languages": ["English", "Chinese", "Spanish"],
        "Normal chat, low-resource languages": ["Indonesian", "Persian"],
        "Specialty chat": ["Coding", "Math", "Safety-violating"],
    }
    RESPONSES_PER_MODEL_PER_PROMPT: int = 50
    NUM_PROMPTS_PER_CATEGORY: int = 200 # For data collection, total prompts will be NUM_PROMPTS_PER_CATEGORY * num_categories
    RESPONSE_MAX_TOKENS: int = 512
    TRAIN_TEST_SPLIT_RATIO: float = 0.8
    TEXT_FEATURES: list[str] = ["length_word", "length_char", "bow", "tfidf"] # 'bow' and 'tfidf' are the strong ones.

    # Cost estimation (Appendix A.3)
    PROPRIETARY_MODEL_COST_PER_MILLION_TOKENS: float = 5.00
    OPEN_SOURCE_MODEL_COST_PER_MILLION_TOKENS: float = 1.80
    NUM_PROPRIETARY_MODELS_FOR_DETECTOR_TRAINING: int = 10 # Assumed for cost calculation
    NUM_OPEN_SOURCE_MODELS_FOR_DETECTOR_TRAINING: int = 20 # Assumed for cost calculation

    # Simulation (Section 3)
    SIMULATION_ITERATIONS: int = 1000 # Calculate Bradley-Terry every N interactions
    DETECTOR_ACCURACY: float = 0.95
    FALSE_POSITIVE_RATE: float = 0.05
    FALSE_NEGATIVE_RATE: float = 0.05
    ATTACKER_NON_DETECTION_STRATEGY: str = "do_nothing" # "do_nothing", "random_upvote", "vote_tie", "vote_tie_both_bad"

    # Bradley-Terry Model
    BRADLEY_TERRY_SCALE_FACTOR: float = 1.0 # 's' in the paper's formula (Pr(i preferred over j))

    # Attack Cost Model (Section 4.1)
    COST_ACCOUNT_MAINTENANCE: float = 0.0 # Cost of obtaining a single user account, for now 0 as per paper baseline
    COST_ACTION: float = 0.0 # Cost per individual action, for now 0 as per paper baseline
    MAX_ACTIONS_PER_ACCOUNT: int = int(1e9) # Effectively infinite without mitigations

    # Malicious User Identification (Section 4.2.3)
    SIGNIFICANCE_LEVEL_ALPHA: float = 0.01
    NUM_SIMULATED_SEQUENCES_FOR_P_VALUE: int = 1000

    # Perturbed leaderboard (Section 4.2.3, Scenario 2)
    NOISE_SCALE: float = 0.0 # Standard deviation for Gaussian noise added to Bradley-Terry ratings

    # Dummy initial Elo ratings for models, or actual historical if available
    # These would typically come from historical data, but for reproduction we'll use a placeholder.
    # In a real scenario, this would be loaded from Chatbot Arena data.
    # For simulation purposes, we'll start with arbitrary values.
    INITIAL_ELO_RATINGS: dict[str, float] = {model: 1500.0 for model in MODELS}
    # These are illustrative initial ranks based on the paper's tables,
    # mapping to Elo ratings would be more complex and data-driven.
    # The actual paper uses historical voting data to derive Bradley-Terry coefficients.
    # For a faithful reproduction, a simplified Elo model will be used.
    # We assume a base Elo rating and adjust slightly for "rank" for simulation stability.
    for i, model_name in enumerate([
        "claude-3-5-sonnet-20240620", "gemini-1.5-pro", "gpt-4o-mini-2024-07-18",
        "gemma-2-27b-it", "llama-3.1-70b-instruct", "mixtral-8x7b-instruct-v0.1",
        "qwen2-72b-instruct"
    ]):
        INITIAL_ELO_RATINGS[model_name] = 1500.0 + (len(MODELS) - i) * 10
    
    for i, model_name in enumerate([
        "llama-13b", "dolly-v2-12b", "stablelm-tuned-alpha-7b",
        "fastchat-t5-3b", "chatglm-6b"
    ]):
        INITIAL_ELO_RATINGS[model_name] = 1500.0 - (i + 1) * 10

    # Historical voting data parameters (Appendix A.4)
    # These would be loaded from the actual dataset (1,670,250 votes mentioned)
    # For simulation, we'll simulate interactions based on pairwise probabilities.
    # This is a simplification as we don't have the real dataset.
    TOTAL_HISTORICAL_VOTES: int = 1_670_250
    TOTAL_UNIQUE_USERS: int = 477_322
    ELO_K_FACTOR: float = 32 # Standard K-factor for Elo rating updates. (Not explicitly stated but common)
