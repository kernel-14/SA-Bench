import torch
from torch.optim import AdamW
from src.fmt import FlowMarchingTransformer
from src.dataset import load_data

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_fmt(data_dir, epochs=100, batch_size=64, embed_dim=256, num_heads=8, rnn_dim=256):
    train_loader = load_data(data_dir, batch_size, split='train')
    valid_loader = load_data(data_dir, batch_size, split='valid')
    
    model = FlowMarchingTransformer(input_dim=3, embed_dim=embed_dim, num_heads=num_heads, rnn_dim=rnn_dim).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    
    hidden_state = torch.zeros(1, batch_size, rnn_dim).to(DEVICE)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            outputs, hidden_state = model(batch, hidden_state)
            loss = torch.nn.MSELoss()(outputs, batch) # Placeholder for flow-matching loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss}')
    
        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            for batch in valid_loader:
                batch = batch.to(DEVICE)
                outputs, hidden_state = model(batch, hidden_state)
                loss = torch.nn.MSELoss()(outputs, batch) # Placeholder validation loss
                total_val_loss += loss.item()
            print(f'Validation Loss: {total_val_loss}')

if __name__ == '__main__':
    train_fmt(data_dir='/home/data')
