
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import random
from typing import List, Dict, Tuple

class SyntheticDataGenerator:
    """
    Generates synthetic responses and extracts features for training-based detectors.
    Since we don't have access to real LLMs or the LMSYS-Chat-1M dataset,
    this class simulates responses based on the described features:
    length, TF-IDF, and BoW.
    """
    def __init__(self, models: List[str], num_prompts: int, responses_per_model_per_prompt: int, random_state: int = 42):
        self.models = models
        self.num_prompts = num_prompts
        self.responses_per_model_per_prompt = responses_per_model_per_prompt
        self.random_state = random_state
        np.random.seed(self.random_state)
        random.seed(self.random_state)

        # Dummy vocabulary for TF-IDF and BoW
        self.vocabulary = [f"word_{i}" for i in range(1000)]
        self.tfidf_vectorizer = TfidfVectorizer(vocabulary=self.vocabulary)
        self.bow_vectorizer = CountVectorizer(vocabulary=self.vocabulary)

        # Simulate distinct response characteristics for each model
        self.model_characteristics = {
            model: {
                "avg_len_word": np.random.randint(50, 200),
                "avg_len_char": np.random.randint(200, 1000),
                "vocab_preference": np.random.rand(len(self.vocabulary)) # For BoW/TF-IDF
            }
            for model in self.models
        }

    def _generate_synthetic_response(self, model_name: str) -> str:
        """Generates a single synthetic response for a given model."""
        characteristics = self.model_characteristics[model_name]
        
        # Simulate length
        len_word = max(10, int(np.random.normal(characteristics["avg_len_word"], 10)))
        
        # Simulate word choice based on vocab_preference
        words = random.choices(self.vocabulary, weights=characteristics["vocab_preference"], k=len_word)
        return " ".join(words)

    def generate_responses(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Generates synthetic responses for all models across multiple prompts.
        Returns:
            A dictionary: {prompt_id: {model_name: [response1, response2, ...]}}
        """
        all_responses = {}
        for i in range(self.num_prompts):
            prompt_id = f"prompt_{i}"
            all_responses[prompt_id] = {}
            for model_name in self.models:
                model_responses = []
                for _ in range(self.responses_per_model_per_prompt):
                    model_responses.append(self._generate_synthetic_response(model_name))
                all_responses[prompt_id][model_name] = model_responses
        return all_responses

    def extract_features(self, responses_by_prompt_and_model: Dict[str, Dict[str, List[str]]]) -> Dict[str, Dict[str, Dict[str, List[np.ndarray]]]]:
        """
        Extracts features (Length, BoW, TF-IDF) from generated responses.
        Returns:
            A nested dictionary: {
                prompt_id: {
                    model_name: {
                        "length_word": [feat1, feat2, ...],
                        "length_char": [feat1, feat2, ...],
                        "bow": [feat_vec1, feat_vec2, ...],
                        "tfidf": [feat_vec1, feat_vec2, ...]
                    }
                }
            }
        """
        features = {}
        for prompt_id, models_responses in responses_by_prompt_and_model.items():
            features[prompt_id] = {}
            all_responses_for_prompt = [resp for model_resps in models_responses.values() for resp in model_resps]

            # Fit vectorizers on all responses for the current prompt to ensure consistent feature space
            self.tfidf_vectorizer.fit(all_responses_for_prompt)
            self.bow_vectorizer.fit(all_responses_for_prompt)

            for model_name, model_responses in models_responses.items():
                model_features = {
                    "length_word": [],
                    "length_char": [],
                    "bow": [],
                    "tfidf": []
                }
                for response in model_responses:
                    model_features["length_word"].append(len(response.split()))
                    model_features["length_char"].append(len(response))
                    model_features["bow"].append(self.bow_vectorizer.transform([response]).toarray().flatten())
                    model_features["tfidf"].append(self.tfidf_vectorizer.transform([response]).toarray().flatten())
                features[prompt_id][model_name] = model_features
        return features

    def prepare_dataset_for_detector(self,
                                     features_by_prompt_and_model: Dict[str, Dict[str, Dict[str, List[np.ndarray]]]],
                                     target_model: str,
                                     prompt_id: str,
                                     feature_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepares a balanced dataset for a binary classifier for a specific target model, prompt, and feature type.
        Positive samples (class 1) are from the target_model, negative samples (class 0) from other models.
        """
        if prompt_id not in features_by_prompt_and_model:
            raise ValueError(f"Prompt ID {prompt_id} not found in features.")
        if target_model not in self.models:
            raise ValueError(f"Target model {target_model} not in the list of models.")
        if feature_type not in self.TEXT_FEATURES:
            raise ValueError(f"Feature type {feature_type} not supported.")

        target_model_features = features_by_prompt_and_model[prompt_id][target_model][feature_type]
        positive_samples = np.array(target_model_features)
        
        # Collect negative samples from other models
        other_models_features = []
        for model_name in self.models:
            if model_name != target_model:
                other_models_features.extend(features_by_prompt_and_model[prompt_id][model_name][feature_type])
        
        # Randomly sample negative examples to balance the dataset
        num_positive = len(positive_samples)
        if len(other_models_features) < num_positive:
            # If not enough unique negative samples, resample with replacement
            negative_samples = np.array(random.choices(other_models_features, k=num_positive))
        else:
            negative_samples = np.array(random.sample(other_models_features, num_positive))

        X = np.vstack((positive_samples, negative_samples))
        y = np.array([1] * num_positive + [0] * num_positive)

        # For length features, ensure they are 2D for scikit-learn
        if feature_type.startswith("length") and X.ndim == 1:
            X = X.reshape(-1, 1)

        return X, y

    @property
    def TEXT_FEATURES(self):
        return ["length_word", "length_char", "bow", "tfidf"]

