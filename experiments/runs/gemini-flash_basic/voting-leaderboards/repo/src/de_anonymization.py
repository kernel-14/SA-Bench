import json
import random
from collections import defaultdict

# --- Utility functions for simulated responses and features (simplified) ---

def _simulate_response_length(mean, std):
    """Simulate response length using a normal distribution."""
    return max(10, int(random.gauss(mean, std))) # Ensure minimum length

def _generate_words(num_words, vocab, model_specific_terms=None):
    """Generate a sequence of words, with some model-specific bias."""
    words = []
    if model_specific_terms:
        # Add some model-specific terms with higher probability
        for _ in range(num_words // 5): # 20% of words are model-specific
            words.append(random.choice(model_specific_terms))
    remaining_words = num_words - len(words)
    words.extend(random.choices(vocab, k=remaining_words))
    random.shuffle(words)
    return " ".join(words)

def _calculate_bow(text, vocab):
    """Simplified Bag-of-Words: counts word occurrences."""
    word_counts = defaultdict(int)
    for word in text.lower().split():
        if word in vocab: # Only count words in our simulated vocabulary
            word_counts[word] += 1
    return dict(word_counts)

def _calculate_tfidf(text, vocab, doc_frequencies):
    """Simplified TF-IDF: TF * IDF. IDF is pre-calculated from corpus."""
    tf = _calculate_bow(text, vocab)
    tfidf_vector = {}
    for word, count in tf.items():
        if word in doc_frequencies and doc_frequencies[word] > 0:
            # Simplified IDF: 1 / doc_frequency (log is usually applied, but keeping it simple)
            idf = 1 / doc_frequencies[word]
            tfidf_vector[word] = count * idf
    return tfidf_vector

# --- Detector Implementations ---

class IdentityProbingDetector:
    """A detector that tries to identify models based on their explicit self-identification.
    This is a conceptual implementation based on the paper's description.
    """
    def __init__(self, models, identity_probing_prompts, simulation_params):
        self.models = models
        self.identity_probing_prompts = identity_probing_prompts
        self.simulation_params = simulation_params
        self.model_keywords = simulation_params["model_names_in_responses"]

    def _simulate_model_response(self, model_name, prompt):
        """Simulate a model's response to an identity-probing prompt.
        A model might reveal its identity with a certain probability.
        """
        if prompt in self.identity_probing_prompts and            random.random() < self.simulation_params["identity_keyword_probability"]:
            # Simulate revealing identity
            keywords = self.model_keywords.get(model_name, [model_name.split('-')[0]])
            revealed_name = random.choice(keywords)
            return f"I am {revealed_name}, a large language model."                    f" I can assist with various tasks and provide information."
        else:
            # Generic response
            return "I am a helpful AI assistant, designed to answer your questions and assist with tasks."

    def detect(self, model_response, potential_target_model):
        """Detect if the response identifies the potential target model.
        Returns True if keywords for the target model are found in the response.
        """
        target_keywords = self.model_keywords.get(potential_target_model, [potential_target_model.split('-')[0]])
        for keyword in target_keywords:
            if keyword.lower() in model_response.lower():
                return True
        return False

    def evaluate_detector(self, target_model, num_queries=1000):
        """Simulate evaluation of the identity-probing detector for a target model.
        Returns accuracy based on simulated responses.
        """
        correct_detections = 0
        for _ in range(num_queries):
            prompt = random.choice(self.identity_probing_prompts)
            # Simulate a response from the target model
            target_response = self._simulate_model_response(target_model, prompt)
            
            # Check if our detector correctly identifies it
            if self.detect(target_response, target_model):
                correct_detections += 1
        
        return (correct_detections / num_queries) * 100


class TrainingBasedDetector:
    """A detector that uses supervised learning to differentiate between model responses.
    This implementation is conceptual and simplifies ML components due to environment constraints.
    """
    def __init__(self, models, prompts_config, simulation_params):
        self.models = models
        self.prompts_config = prompts_config
        self.simulation_params = simulation_params
        self.vocab = self._generate_simulated_vocabulary()
        self.doc_frequencies = self._generate_simulated_doc_frequencies() # For TF-IDF

    def _generate_simulated_vocabulary(self):
        """Generate a simulated vocabulary."""
        return [f"word_{i}" for i in range(self.simulation_params["vocab_size"])]

    def _generate_simulated_doc_frequencies(self):
        """Generate simulated document frequencies for IDF calculation."""
        # In a real scenario, this would come from a corpus. Here, we simulate it.
        doc_freq = {word: random.randint(1, 100) for word in self.vocab}
        return doc_freq

    def _simulate_model_response(self, model_name, prompt_text):
        """Simulate a model's response for the training-based detector.
        Responses will vary by model in length and word distribution.
        """
        mean_len = self.simulation_params["response_length_mean"]
        std_len = self.simulation_params["response_length_std"]
        
        # Introduce model-specific length variation
        model_len_adjustment = (self.models.index(model_name) - len(self.models)/2) *                                 self.simulation_params["model_specific_length_variation"]
        
        num_words = _simulate_response_length(mean_len + model_len_adjustment, std_len)

        # Simulate model-specific vocabulary bias
        model_specific_vocab = random.sample(self.vocab, k=int(len(self.vocab) * (1 - self.simulation_params["model_specific_vocab_overlap"]))) # Unique part
        common_vocab = random.sample(self.vocab, k=int(len(self.vocab) * self.simulation_params["model_specific_vocab_overlap"])) # Shared part
        
        # Combine common and model-specific words for this response generation
        current_model_vocab = list(set(model_specific_vocab + common_vocab))
        
        return _generate_words(num_words, current_model_vocab)

    def _extract_features(self, response_text):
        """Extract Length, BoW, and TF-IDF features from a response.
        Returns a dictionary of features.
        """
        features = {
            "length_word": len(response_text.split()),
            "length_character": len(response_text),
            "bow": _calculate_bow(response_text, self.vocab),
            "tfidf": _calculate_tfidf(response_text, self.vocab, self.doc_frequencies)
        }
        # Flatten BOW/TFIDF for a simple classifier if needed, here just store as dicts
        return features

    def _train_logistic_regression(self, X_train, y_train):
        """Conceptual Logistic Regression training.
        In a real scenario, this would use sklearn.linear_model.LogisticRegression.
        For this simulation, we return a 'dummy' classifier that makes predictions
        based on an extreme simplification of features.
        """
        # This is a highly simplified, non-functional placeholder.
        # A real implementation would involve actual parameter learning.
        class DummyClassifier:
            def predict_proba(self, X):
                # Simulate probabilities based on some feature, e.g., length
                # This is a heuristic, not actual LR.
                probs = []
                for x in X:
                    # Assume 'length_word' is the first feature if X is flattened
                    # Or if X is a dict of features, we'd access it.
                    # Here we just simulate a split based on feature values
                    if x['length_word'] > self.threshold:
                        probs.append([0.1, 0.9]) # High prob for class 1
                    else:
                        probs.append([0.9, 0.1]) # High prob for class 0
                return probs

            def fit(self, X, y):
                # In a real LR, we'd learn weights. Here, we set a heuristic threshold.
                # For demonstration, let's say target models tend to be longer.
                positive_lengths = [x['length_word'] for i, x in enumerate(X) if y[i] == 1]
                negative_lengths = [x['length_word'] for i, x in enumerate(X) if y[i] == 0]
                if positive_lengths and negative_lengths:
                    self.threshold = (sum(positive_lengths) / len(positive_lengths) + 
                                      sum(negative_lengths) / len(negative_lengths)) / 2
                else:
                    self.threshold = self.simulation_params["response_length_mean"]
                

            def predict(self, X):
                probs = self.predict_proba(X)
                return [1 if p[1] > p[0] else 0 for p in probs]

        classifier = DummyClassifier()
        classifier.fit(X_train, y_train)
        return classifier

    def evaluate_detector(self, target_model, num_prompts_per_category=200, responses_per_model=50):
        """Evaluate the training-based detector for a specific target model.
        This simulates data collection, feature extraction, training, and evaluation.
        Returns the average test accuracy.
        """
        all_accuracies = []

        for category_data in self.prompts_config["categories"]:
            for prompt_type_data in category_data["types"]:
                # Use example prompts or generate placeholder prompts
                prompts = prompt_type_data["examples"]
                if not prompts:
                    prompts = [f"A {prompt_type_data['type']} question {i}" for i in range(num_prompts_per_category)]
                
                # For each prompt, gather simulated responses
                for prompt_text in random.sample(prompts, k=min(len(prompts), num_prompts_per_category // len(category_data['types']))):
                    X, y = [], [] # Features and labels for this prompt-model pair

                    # Generate positive samples (target model)
                    for _ in range(responses_per_model):
                        response = self._simulate_model_response(target_model, prompt_text)
                        X.append(self._extract_features(response))
                        y.append(1)
                    
                    # Generate negative samples (other models)
                    other_models = [m for m in self.models if m != target_model]
                    for _ in range(responses_per_model):
                        negative_model = random.choice(other_models)
                        response = self._simulate_model_response(negative_model, prompt_text)
                        X.append(self._extract_features(response))
                        y.append(0)

                    # Simulate train-test split (80/20)
                    combined = list(zip(X, y))
                    random.shuffle(combined)
                    split_idx = int(0.8 * len(combined))
                    X_train, y_train = zip(*combined[:split_idx])
                    X_test, y_test = zip(*combined[split_idx:])

                    # Train and evaluate the classifier (conceptual)
                    classifier = self._train_logistic_regression(list(X_train), list(y_train))
                    predictions = classifier.predict(list(X_test))
                    
                    correct_predictions = sum(1 for p, t in zip(predictions, y_test) if p == t)
                    accuracy = (correct_predictions / len(y_test)) * 100 if len(y_test) > 0 else 0
                    all_accuracies.append(accuracy)

        if not all_accuracies: # Handle case where no prompts were processed
            return 50.0 # Default to random chance if no data

        return sum(all_accuracies) / len(all_accuracies)

# Example usage (for testing purposes, will be called from a main script)
if __name__ == "__main__":
    with open('/mnt/paper2any/pzw/proj/paperagent/hx/Research_space/SemanticAlign-Bench/experiments/runs/gemini-flash_basic/voting-leaderboards/voting-leaderboards-20260505-151946/workspace/repo/config.yaml', 'r') as f:
        config = json.load(f) # Using json.load for yaml, assuming simple structure
    with open('/mnt/paper2any/pzw/proj/paperagent/hx/Research_space/SemanticAlign-Bench/experiments/runs/gemini-flash_basic/voting-leaderboards/voting-leaderboards-20260505-151946/workspace/repo/data/prompts.json', 'r') as f:
        prompts_data = json.load(f)
    
    models = config["models"]
    identity_probing_prompts = config["identity_probing_prompts"]
    simulation_params = config["simulation_params"]

    print("
--- Identity-Probing Detector Simulation ---")
    id_detector = IdentityProbingDetector(models, identity_probing_prompts, simulation_params)
    for model in models:
        accuracy = id_detector.evaluate_detector(model, num_queries=100) # Reduced queries for quick test
        print(f"Identity-Probing Detector Accuracy for {model}: {accuracy:.2f}%")

    print("
--- Training-Based Detector Simulation ---")
    tb_detector = TrainingBasedDetector(models, prompts_data, simulation_params)
    for model in models:
        accuracy = tb_detector.evaluate_detector(model, num_prompts_per_category=5, responses_per_model=10) # Reduced for quick test
        print(f"Training-Based Detector Accuracy for {model}: {accuracy:.2f}%")
