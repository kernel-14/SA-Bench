"""Data loading and preprocessing for de-anonymization and voting simulation.

Handles:
  - Prompt datasets: LMSYS-Chat-1M, Alpaca Code, MATH, AdvBench (Table 1)
  - Model response collection and feature extraction
  - Chatbot Arena voting data for simulation (Section 3)
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle


PROMPT_CATEGORIES = {
    "english": {"source": "LMSYS-Chat-1M", "type": "normal_chat"},
    "chinese": {"source": "LMSYS-Chat-1M", "type": "normal_chat"},
    "spanish": {"source": "LMSYS-Chat-1M", "type": "normal_chat"},
    "indonesian": {"source": "LMSYS-Chat-1M", "type": "normal_chat"},
    "persian": {"source": "LMSYS-Chat-1M", "type": "normal_chat"},
    "coding": {"source": "Alpaca Code", "type": "specialty"},
    "math": {"source": "MATH", "type": "specialty"},
    "safety_violating": {"source": "AdvBench", "type": "specialty"},
}

IDENTITY_PROBING_PROMPTS = [
    "Who are you?",
    "Which model are you?",
    "What is your model name?",
    "How should I refer to you as an AI?",
    "How would you define your role or identity as an AI?",
]

MODEL_NAME_PATTERNS = [
    "claude", "gemini", "gemma", "gpt", "llama", "mixtral", "qwen",
    "anthropic", "google", "openai", "meta", "mistral", "alibaba",
    "chatgpt", "gpt-4", "gpt-3", "claude-3", "gemini-1",
]


def generate_synthetic_responses(
    model_name: str,
    prompt: str,
    num_responses: int = 50,
    output_tokens: int = 512,
    seed: int = 42,
) -> List[str]:
    """Generate synthetic model responses for offline simulation.

    In real experiments, these would come from API calls to the actual models.
    Here we generate synthetic data that exhibits model-specific patterns
    to enable demonstration of the detector training pipeline.

    The synthetic generator creates responses with model-specific characteristics
    (length distributions, vocabulary patterns) that emulate the clustering
    behavior observed in Figure 2 of the paper.
    """
    rng = np.random.RandomState(seed + hash(model_name) % 10000 + hash(prompt) % 10000)

    model_patterns = {
        "claude": {"base_len": 180, "std_len": 30, "words": ["detailed", "comprehensive", "analysis", "approach"]},
        "gemini": {"base_len": 150, "std_len": 25, "words": ["Google", "capable", "context", "understanding"]},
        "gemma": {"base_len": 140, "std_len": 28, "words": ["lightweight", "efficient", "open", "accessible"]},
        "gpt": {"base_len": 200, "std_len": 35, "words": ["advanced", "sophisticated", "nuanced", "complex"]},
        "llama": {"base_len": 160, "std_len": 30, "words": ["Meta", "community", "research", "transparent"]},
        "mixtral": {"base_len": 155, "std_len": 28, "words": ["mixture", "experts", "efficient", "French"]},
        "qwen": {"base_len": 145, "std_len": 25, "words": ["Alibaba", "Chinese", "comprehensive", "multilingual"]},
    }

    pattern = model_patterns.get("claude", {"base_len": 160, "std_len": 30, "words": ["helpful", "assistant", "response"]})
    for key, val in model_patterns.items():
        if key in model_name.lower():
            pattern = val
            break

    filler_words = pattern["words"]
    template_words = prompt.lower().split()[:10]

    responses = []
    for i in range(num_responses):
        length = max(20, int(rng.normal(pattern["base_len"], pattern["std_len"])))
        selected_words = rng.choice(filler_words, size=min(length // 3, len(filler_words)), replace=True)
        extra_words = rng.choice(template_words, size=min(length // 4, len(template_words)), replace=True) if template_words else []
        all_words = list(selected_words) + list(extra_words)
        rng.shuffle(all_words)
        words = []
        for j in range(length):
            if all_words:
                words.append(all_words[j % len(all_words)])
            else:
                words.append(f"word_{j}")

        text = " ".join(words)
        text += f" [{model_name}]"
        responses.append(text)

    return responses


def extract_features(
    responses: List[str],
    response_lengths_word: Optional[List[int]] = None,
    feature_type: str = "bow",
    vectorizer: Optional[object] = None,
    fit_vectorizer: bool = False,
) -> Tuple[np.ndarray, Optional[object]]:
    """Extract features from model responses.

    Implements the three feature types from Section 2.3:
      - Length(R): response length measured in words or characters
      - BoW(R): bag-of-words representations
      - TF-IDF(R): term frequency-inverse document frequency
    """
    if feature_type in ("length_word", "length_character"):
        if response_lengths_word is not None and feature_type == "length_word":
            return np.array(response_lengths_word).reshape(-1, 1), None
        elif feature_type == "length_character":
            lengths = [len(r) for r in responses]
            return np.array(lengths).reshape(-1, 1), None

    if feature_type == "bow":
        if fit_vectorizer:
            vectorizer = CountVectorizer(max_features=1000, binary=False)
            features = vectorizer.fit_transform(responses).toarray()
        else:
            features = vectorizer.transform(responses).toarray()
        return features, vectorizer

    if feature_type == "tfidf":
        if fit_vectorizer:
            vectorizer = TfidfVectorizer(max_features=1000)
            features = vectorizer.fit_transform(responses).toarray()
        else:
            features = vectorizer.transform(responses).toarray()
        return features, vectorizer

    raise ValueError(f"Unknown feature type: {feature_type}")


def prepare_detector_dataset(
    target_responses: List[str],
    other_responses: List[str],
    target_response_lengths: Optional[List[int]] = None,
    other_response_lengths: Optional[List[int]] = None,
    feature_type: str = "bow",
) -> Tuple[np.ndarray, np.ndarray, object]:
    """Prepare balanced dataset for training the binary detector classifier.

    Constructs balanced datasets containing responses from the target model
    (positive samples, class 1) and uniformly sampled responses from other
    models (negative samples, class 0), as described in Section 2.3.

    Returns features, labels, and fitted vectorizer.
    """
    all_responses = list(target_responses) + list(other_responses)
    labels = np.array([1] * len(target_responses) + [0] * len(other_responses))

    if feature_type in ("length_word",):
        all_lengths_word = (target_response_lengths or []) + (other_response_lengths or [])
        features, vec = extract_features(
            all_responses,
            response_lengths_word=all_lengths_word if all_lengths_word else None,
            feature_type=feature_type,
            fit_vectorizer=True,
        )
    elif feature_type == "length_character":
        features, vec = extract_features(all_responses, feature_type=feature_type, fit_vectorizer=True)
    else:
        features, vec = extract_features(all_responses, feature_type=feature_type, fit_vectorizer=True)

    return features, labels, vec


class VotingDataSimulator:
    """Simulates Chatbot Arena voting data for leaderboard manipulation experiments.

    Uses the anonymized and deduplicated dataset statistics from Appendix A.4:
      - 1,670,250 votes from 477,322 unique users
      - 1,093,875 wins, 576,375 ties
      - 6,895 unique combinations of model pairs
    """

    def __init__(
        self,
        model_names: List[str],
        total_votes: int = 1_670_250,
        total_wins: int = 1_093_875,
        total_ties: int = 576_375,
        seed: int = 42,
    ):
        self.model_names = list(model_names)
        self.n_models = len(self.model_names)
        self.total_votes = total_votes
        self.total_wins = total_wins
        self.total_ties = total_ties
        self.rng = np.random.RandomState(seed)

        self.model_to_idx = {name: i for i, name in enumerate(self.model_names)}

        self._initialize_bradley_terry_ratings()

    def _initialize_bradley_terry_ratings(self):
        """Initialize model ratings using a distribution similar to Chatbot Arena."""
        n = self.n_models
        self.ratings = self.rng.normal(1000, 200, n)
        self.ratings = np.sort(self.ratings)[::-1]

        self.vote_counts = self.rng.randint(2000, 80000, n)
        self.win_counts = np.zeros(n)
        self.loss_counts = np.zeros(n)
        self.tie_counts = np.zeros(n)

    def sample_pair(self) -> Tuple[int, int]:
        """Randomly sample two distinct models for a head-to-head comparison."""
        return tuple(self.rng.choice(self.n_models, size=2, replace=False))

    def simulate_vote(self, model_a: int, model_b: int) -> int:
        """Simulate a single vote between two models using Bradley-Terry model.

        Returns: 1 if A wins, 0 if tie, -1 if B wins.
        """
        s = 1.0
        diff = self.ratings[model_a] - self.ratings[model_b]
        prob_a_wins = 1.0 / (1.0 + np.exp(-diff / s))

        r = self.rng.random()
        tie_prob = self.total_ties / self.total_votes
        if r < (1 - tie_prob) * prob_a_wins:
            return 1
        elif r < (1 - tie_prob):
            return -1
        else:
            return 0

    def update_ratings(self, winner: Optional[int], loser: Optional[int], k_factor: float = 32.0):
        """Update Bradley-Terry ratings based on a vote outcome."""
        if winner is None and loser is None:
            return
        if winner is not None and loser is not None:
            expected = 1.0 / (1.0 + 10.0 ** ((self.ratings[loser] - self.ratings[winner]) / 400.0))
            delta = k_factor * (1.0 - expected)
            self.ratings[winner] += delta
            self.ratings[loser] -= delta

    def get_ranking(self) -> List[int]:
        """Return model indices sorted by rating (highest first)."""
        return list(np.argsort(-self.ratings))

    def get_rank(self, model_idx: int) -> int:
        """Get the current rank (1-indexed) of a model."""
        ranking = self.get_ranking()
        return ranking.index(model_idx) + 1
