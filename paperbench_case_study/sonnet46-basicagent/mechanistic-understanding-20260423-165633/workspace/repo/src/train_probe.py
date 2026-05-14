"""
Train a linear toxicity probe on the Jigsaw dataset.
The probe is trained on the residual stream of the last layer of GPT2-medium,
averaged across all timesteps.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Model, GPT2Tokenizer
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm


class ToxicityProbe(nn.Module):
    """Linear probe for toxicity classification.
    W_toxic is a matrix of shape [d_model, 2].
    W_toxic[:, 0] is for non-toxic, W_toxic[:, 1] is for toxic.
    """
    def __init__(self, d_model: int, num_classes: int = 2):
        super().__init__()
        self.linear = nn.Linear(d_model, num_classes)

    def forward(self, x):
        return self.linear(x)

    @property
    def W_toxic(self):
        """Returns the weight matrix of shape [d_model, num_classes]."""
        return self.linear.weight.T  # [d_model, num_classes]


class JigsawDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def collate_fn(batch, tokenizer, max_length=128):
    texts, labels = zip(*batch)
    encodings = tokenizer(
        list(texts),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )
    return encodings, torch.tensor(labels, dtype=torch.long)


def extract_residual_stream(model, input_ids, attention_mask, device):
    """Extract the last-layer residual stream averaged across timesteps."""
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            output_hidden_states=True
        )
    # Last hidden state: [batch, seq_len, d_model]
    last_hidden = outputs.hidden_states[-1]
    # Average across non-padding tokens
    mask = attention_mask.to(device).unsqueeze(-1).float()
    avg_hidden = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)
    return avg_hidden  # [batch, d_model]


def train_probe(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer and model
    print("Loading GPT2-medium...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    tokenizer.pad_token = tokenizer.eos_token
    gpt2 = GPT2Model.from_pretrained("gpt2-medium").to(device)
    gpt2.eval()

    d_model = gpt2.config.hidden_size  # 1024 for gpt2-medium

    # Load Jigsaw dataset
    print("Loading Jigsaw dataset...")
    dataset = load_dataset("thesofakillers/jigsaw-toxic-comment-classification-challenge")

    train_data = dataset["train"]
    # 90:10 split
    total = len(train_data)
    split_idx = int(total * 0.9)

    train_texts = train_data["comment_text"][:split_idx]
    train_labels = [int(l) for l in train_data["toxic"][:split_idx]]
    val_texts = train_data["comment_text"][split_idx:]
    val_labels = [int(l) for l in train_data["toxic"][split_idx:]]

    print(f"Train size: {len(train_texts)}, Val size: {len(val_texts)}")

    # Create datasets
    train_dataset = JigsawDataset(train_texts, train_labels, tokenizer)
    val_dataset = JigsawDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_length)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_length)
    )

    # Initialize probe
    probe = ToxicityProbe(d_model).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        probe.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for encodings, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            # Extract residual stream features
            features = extract_residual_stream(
                gpt2, encodings["input_ids"], encodings["attention_mask"], device
            )
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = probe(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        train_acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Train Acc={train_acc:.4f}")

        # Validation
        probe.eval()
        val_preds, val_labels_list = [], []
        with torch.no_grad():
            for encodings, labels in tqdm(val_loader, desc="Validation"):
                features = extract_residual_stream(
                    gpt2, encodings["input_ids"], encodings["attention_mask"], device
                )
                logits = probe(features)
                preds = logits.argmax(dim=-1).cpu().numpy()
                val_preds.extend(preds)
                val_labels_list.extend(labels.numpy())

        val_acc = accuracy_score(val_labels_list, val_preds)
        print(f"Validation Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(probe.state_dict(), os.path.join(args.output_dir, "probe_best.pt"))
            # Save W_toxic vector (toxic direction: column 1)
            W_toxic = probe.W_toxic.detach().cpu().numpy()  # [d_model, 2]
            np.save(os.path.join(args.output_dir, "W_toxic.npy"), W_toxic)
            print(f"Saved best probe with val acc: {val_acc:.4f}")

    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="checkpoints/probe")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()
    train_probe(args)
