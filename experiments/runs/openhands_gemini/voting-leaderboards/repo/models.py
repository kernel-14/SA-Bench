
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple, Any

class IdentityProbingDetector:
    """
    Implements the Identity-Probing Detector described in Section 2.2 and 2.3.
    This detector tries to identify a model by asking it direct questions about its identity.
    """
    def __init__(self, target_model_name: str, associated_keywords: List[str]):
        """
        Args:
            target_model_name: The name of the model this detector is trying to identify.
            associated_keywords: A list of keywords (e.g., model name, organization name)
                                 that, if found in the response, indicate it's from the target model.
        """
        self.target_model_name = target_model_name
        self.associated_keywords = [kw.lower() for kw in associated_keywords]

    def detect(self, response: str) -> bool:
        """
        Predicts if a given response comes from the target model.
        Args:
            response: The text response from an LLM.
        Returns:
            True if the response contains any of the associated keywords, False otherwise.
        """
        response_lower = response.lower()
        for keyword in self.associated_keywords:
            if keyword in response_lower:
                return True
        return False

    def evaluate(self, model_responses: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Evaluates the detector's accuracy across different models.
        Args:
            model_responses: A dictionary where keys are model names and values are
                             lists of responses from that model for a specific prompt.
        Returns:
            A dictionary with accuracy for each model.
        """
        results = {}
        for model_name, responses in model_responses.items():
            correct_predictions = 0
            total_predictions = len(responses)
            for response in responses:
                is_target = (model_name == self.target_model_name)
                detection = self.detect(response)
                if (is_target and detection) or (not is_target and not detection):
                    correct_predictions += 1
            results[model_name] = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0.0
        return results


class TrainingBasedDetector:
    """
    Implements the Training-Based Detector described in Section 2.2 and 2.3.
    This detector uses supervised learning on text features (Length, BoW, TF-IDF)
    to classify if a response comes from a target model.
    """
    def __init__(self, random_state: int = 42):
        self.classifier: Any = LogisticRegression(random_state=random_state, max_iter=1000) # Increased max_iter for convergence
        self.random_state = random_state

    def train(self, X: np.ndarray, y: np.ndarray, train_test_split_ratio: float = 0.8) -> float:
        """
        Trains the logistic regression classifier and evaluates its accuracy.
        Args:
            X: Feature matrix.
            y: Labels (1 for target model, 0 for other models).
            train_test_split_ratio: Ratio for splitting training and testing data.
        Returns:
            Test accuracy of the trained classifier.
        """
        if len(X) == 0:
            return 0.0 # No data to train/test

        # Ensure X is 2D for scikit-learn
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim > 2:
            # Flatten features like BoW/TFIDF if they are still 3D (e.g., from List[np.ndarray])
            X = np.vstack(X) if isinstance(X[0], np.ndarray) and X[0].ndim > 0 else X # Handle case where X is list of arrays
        
        # Check if X contains non-numeric data or NaNs
        if not np.issubdtype(X.dtype, np.number):
             raise ValueError("Input features X must be numeric.")
        if np.isnan(X).any():
            raise ValueError("Input features X contain NaN values.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=(1 - train_test_split_ratio), random_state=self.random_state, stratify=y
        )
        
        if len(np.unique(y_train)) < 2:
            # Cannot train if only one class is present in training data
            return 0.0

        self.classifier.fit(X_train, y_train)
        accuracy = self.classifier.score(X_test, y_test)
        return accuracy

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the class labels for given features.
        Args:
            X: Feature matrix.
        Returns:
            Predicted labels.
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim > 2:
            X = np.vstack(X) if isinstance(X[0], np.ndarray) and X[0].ndim > 0 else X
            
        return self.classifier.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class probabilities for given features.
        Args:
            X: Feature matrix.
        Returns:
            Predicted probabilities.
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim > 2:
            X = np.vstack(X) if isinstance(X[0], np.ndarray) and X[0].ndim > 0 else X

        return self.classifier.predict_proba(X)
