"""
Training-based detector for model response classification.
Implemented as described in Section 2.2 using scikit-learn.
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def create_features(responses, feature_type="bow"):
    """
    Generates text features for responses.
    Args:
        responses (list of str): Model responses.
        feature_type (str): Feature type ("bow" or "tfidf").
    Returns:
        sparse matrix: Feature matrix.
    """
    if feature_type == "bow":
        vectorizer = CountVectorizer()
    elif feature_type == "tfidf":
        vectorizer = TfidfVectorizer()
    else:
        raise ValueError("Invalid feature type.")
    return vectorizer.fit_transform(responses)

def train_detector(responses, labels, feature_type="bow"):
    """
    Trains a logistic regression detector.
    Args:
        responses (list of str): Model responses.
        labels (list of int): Binary labels (1 for target model, 0 otherwise).
        feature_type (str): Feature type ("bow" or "tfidf").
    Returns:
        sklearn model: Trained logistic regression model.
    """
    X = create_features(responses, feature_type)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"Test accuracy: {accuracy * 100:.2f}%")
    return model

# Example workflow (simulated data):
responses = ["Response from model A.", "Another response from model B.", "Model A output.", "Random response to question."]
labels = [1, 0, 1, 0]
train_detector(responses, labels, feature_type="bow")
