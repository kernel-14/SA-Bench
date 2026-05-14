import torch
import torch.nn as nn

class TextFeatureExtractor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(TextFeatureExtractor, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

class LogisticRegressionClassifier(nn.Module):
    def __init__(self, input_dim: int):
        super(LogisticRegressionClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        logits = self.fc(x)
        return self.sigmoid(logits)

# Example usage
if __name__ == "__main__":
    feature_extractor = TextFeatureExtractor(input_dim=512, output_dim=128)
    classifier = LogisticRegressionClassifier(input_dim=128)

    sample_input = torch.randn(1, 512)
    features = feature_extractor(sample_input)
    prediction = classifier(features)
    print(f"Prediction: {prediction.item()}")