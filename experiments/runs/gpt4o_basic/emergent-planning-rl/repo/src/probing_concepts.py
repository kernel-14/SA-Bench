import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearProbe, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

def train_probe(probe, data_loader, num_epochs=10, learning_rate=1e-3):
    optimizer = optim.Adam(probe.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        probe.train()
        for batch in data_loader:
            inputs, labels = batch
            outputs = probe(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

def evaluate_probe(probe, test_loader):
    probe.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            outputs = probe(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return f1_score(all_labels, all_preds, average='macro')
