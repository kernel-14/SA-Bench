# classifier.py

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import Optional

class Classifier:
    """Manages training, prediction, and persistence of the logistic regression classifier."""

    def __init__(self) -> None:
        """
        Initializes the Classifier instance and sets up the placeholder for training a logistic regression model.
        """
        self.model: Optional[LogisticRegression] = None  # Logistic regression model placeholder

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the logistic regression model on the given dataset.

        Args:
            X (np.ndarray): Feature matrix for training (shape: [n_samples, n_features]).
            y (np.ndarray): Binary labels for classification (shape: [n_samples]).
        
        Raises:
            ValueError: If input arrays X and y have incompatible dimensions.
        """
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("Both X and y must be numpy arrays.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Incompatible dimensions: X has {X.shape[0]} samples, while y has {y.shape[0]} labels.")

        # Initialize and fit the logistic regression model using default hyperparameters
        self.model = LogisticRegression(
            penalty='l2',                 # L2 regularization
            solver='lbfgs',               # Recommended solver for smaller datasets
            max_iter=1000,                # Allow up to 1,000 iterations for convergence
            random_state=42,              # Deterministic results
            class_weight="balanced"       # Handle class imbalance
        )
        self.model.fit(X, y)
        print("Model training completed successfully.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts labels for the given dataset using the trained logistic regression model.

        Args:
            X (np.ndarray): Feature matrix for inference (shape: [n_samples, n_features]).

        Returns:
            np.ndarray: Predicted binary labels (0 or 1).

        Raises:
            ValueError: If the classifier has not been trained yet.
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call 'train()' before making predictions.")

        predictions = self.model.predict(X)
        return predictions

    def save_model(self, path: str) -> None:
        """
        Saves the trained logistic regression model to the specified file path.

        Args:
            path (str): File path to save the trained model.
        
        Raises:
            ValueError: If the model has not been trained yet.
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Train the model before saving.")

        try:
            joblib.dump(self.model, path)
            print(f"Model successfully saved to '{path}'.")
        except Exception as e:
            raise IOError(f"Failed to save model to '{path}': {e}")

    def load_model(self, path: str) -> None:
        """
        Loads a trained logistic regression model from the specified file path.

        Args:
            path (str): File path to the saved model.
        
        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the loaded object does not match the expected classifier format.
        """
        try:
            self.model = joblib.load(path)
            if not isinstance(self.model, LogisticRegression):
                raise ValueError(f"The loaded object is not a valid LogisticRegression model.")
            print(f"Model successfully loaded from '{path}'.")
        except FileNotFoundError:
            raise FileNotFoundError(f"The model file '{path}' does not exist.")
        except Exception as e:
            raise ValueError(f"Failed to load model from '{path}': {e}")
