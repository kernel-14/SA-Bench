import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from typing import List

class TargetModelDetector:
    def __init__(self, feature_type: str = "tfidf"):
        self.feature_type = feature_type
        if feature_type == "tfidf":
            self.vectorizer = TfidfVectorizer()
        elif feature_type == "bow":
            self.vectorizer = None  # Placeholder for Bag-of-Words implementation
        else:
            raise ValueError("Unsupported feature type. Use 'tfidf' or 'bow'.")
        self.classifier = LogisticRegression(random_state=42)

    def fit(self, prompts: List[str], responses: List[str], labels: List[int]):
        if self.feature_type == "tfidf":
            features = self.vectorizer.fit_transform(responses)
        else:
            raise NotImplementedError("Only TF-IDF is implemented.")
        self.classifier.fit(features, labels)

    def predict(self, responses: List[str]) -> List[int]:
        if self.feature_type == "tfidf":
            features = self.vectorizer.transform(responses)
        else:
            raise NotImplementedError("Only TF-IDF is implemented.")
        return self.classifier.predict(features)

    def evaluate(self, responses: List[str], labels: List[int]) -> float:
        predictions = self.predict(responses)
        accuracy = (predictions == labels).mean()
        return accuracy

# Example usage
if __name__ == "__main__":
    detector = TargetModelDetector(feature_type="tfidf")
    prompts = ["What is your name?", "Describe yourself."]
    responses = ["I am GPT-4.", "I am a language model developed by OpenAI."]
    labels = [1, 0]

    detector.fit(prompts, responses, labels)
    accuracy = detector.evaluate(responses, labels)
    print(f"Accuracy: {accuracy}")