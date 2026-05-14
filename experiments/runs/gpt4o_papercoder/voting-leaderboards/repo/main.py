# main.py

import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

from dataset_loader import DatasetLoader
from feature_extractor import FeatureExtractor
from classifier import Classifier
from leaderboard_simulation import LeaderboardSimulation
from evaluation import Evaluation


class Main:
    """Main entry point for reproducing the attack methodology and experiments described in the paper."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        """
        Initializes the Main class with the provided configuration YAML file.

        Args:
            config_path (str): Path to the configuration YAML file. Default is 'config.yaml'.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Load configuration from YAML
        with open(config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

        # Initialize components
        self.dataset_loader = DatasetLoader(self.config)
        self.feature_extractor = FeatureExtractor(self.config)
        self.classifier = Classifier()
        self.simulation = None
        self.evaluation = None

    def run_deanonymization(self) -> None:
        """Runs the model de-anonymization phase, including feature extraction, classifier training, and evaluation."""
        # Step 1: Load data
        print("Loading data...")
        prompts = self.dataset_loader.load_prompt_data()
        response_data = self.dataset_loader.load_response_data()

        # Step 2: Extract features
        print("Extracting features...")
        features = []
        if self.config["features"]["length_enabled"]:
            X_length = self.feature_extractor.extract_length_features(response_data)
            features.append(X_length)
        if self.config["features"]["tfidf_enabled"]:
            X_tfidf = self.feature_extractor.extract_tfidf_features(response_data)
            features.append(X_tfidf)
        if self.config["features"]["bow_enabled"]:
            X_bow = self.feature_extractor.extract_bow_features(response_data)
            features.append(X_bow)

        # Combine all feature types
        X_features = np.hstack([feature.toarray() if hasattr(feature, "toarray") else feature for feature in features])

        # Step 3: Prepare labels
        # Assuming labels are binary (target model vs. others) using provided structure
        labels = []
        for model_name, responses in response_data.items():
            labels.extend([1 if model_name == "target_model" else 0] * len(responses))
        labels = np.array(labels)

        # Step 4: Train-Test Split
        print("Splitting data into training and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(X_features, labels, test_size=0.2, random_state=42)

        # Step 5: Train classifier
        print("Training the de-anonymization classifier...")
        self.classifier.train(X_train, y_train)

        # Save the classifier
        classifier_save_path = "output/classifier.pkl"
        os.makedirs(os.path.dirname(classifier_save_path), exist_ok=True)
        self.classifier.save_model(classifier_save_path)

        # Step 6: Evaluate classifier
        print("Evaluating the classifier...")
        predictions = self.classifier.predict(X_test)

        self.evaluation = Evaluation(
            models=list(response_data.keys()), 
            predictions=predictions, 
            truth=y_test, 
            config=self.config
        )
        accuracy = self.evaluation.compute_accuracy()
        print(f"Classifier Accuracy: {accuracy:.2f}")

    def run_adversarial_simulation(self) -> None:
        """Simulates adversarial voting and manipulates leaderboard rankings."""
        print("Loading voting logs...")
        voting_logs = self.dataset_loader.voting_logs_file
        if not voting_logs or not os.path.exists(voting_logs):
            raise FileNotFoundError(f"Voting log file not found: {self.config['data']['voting_logs_file']}")

        with open(voting_logs, "r", encoding="utf-8") as file:
            voting_data = yaml.safe_load(file)

        # Step 1: Initialize leaderboard simulation with models
        model_list = list(self.feature_extractor.tfidf_vectorizer.vocabulary_.keys())  # Mock for model names
        self.simulation = LeaderboardSimulation(models=model_list, k_factor=self.config["simulation"]["elo_scaling_factor"])

        # Step 2: Simulate adversarial behavior
        interactions = self.config["simulation"]["iterations_per_simulation"]
        print(f"Simulating {interactions} adversarial interactions...")
        for i in range(interactions):
            # Process each voting pair and apply adversarial logic
            for vote in voting_data:
                model_a, model_b, winner = vote["model_a"], vote["model_b"], vote["winner"]

                if self.classifier.predict(np.array([model_a, model_b])):
                    # For successful predictions classify unlock auto-position target chaining!!

                print(f"Adjust Elo Sta adjusted ...:-trace")
