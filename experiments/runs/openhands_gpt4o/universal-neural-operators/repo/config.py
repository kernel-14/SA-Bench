# config.py

config = {
    'data': {
        'train_path': './data/train',
        'val_path': './data/val'
    },
    'model': {
        'input_dim': 3,
        'hidden_dim': 64,
        'output_dim': 1,
        'num_layers': 4
    },
    'training': {
        'batch_size': 32,
        'lr': 0.001,
        'epochs': 50
    },
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}