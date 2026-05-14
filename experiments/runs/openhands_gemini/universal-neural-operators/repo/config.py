
import argparse

def get_config():
    parser = argparse.ArgumentParser(description="Universal Neural Operators Configuration")

    # General
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use for training')

    # Data
    parser.add_argument('--dataset_name', type=str, default='Burgers', help='Name of the dataset to use')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--train_split', type=float, default=0.8, help='Training data split ratio')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation data split ratio')
    parser.add_argument('--test_split', type=float, default=0.1, help='Test data split ratio')
    parser.add_argument('--subsample_rate', type=int, default=1, help='Subsample rate for spatial dimensions')
    parser.add_argument('--output_res', type=int, default=64, help='Output resolution for models')

    # Model
    parser.add_argument('--model_type', type=str, default='FNO',
                        choices=['FNO', 'MambaFNO', 'PerceiverIONO', 'SwinV2NO', 'CoDANO'],
                        help='Type of neural operator model to use')
    parser.add_argument('--input_channels', type=int, default=0, help='Number of input channels (will be set dynamically by data.py)')
    parser.add_argument('--output_channels', type=int, default=0, help='Number of output channels (will be set dynamically by data.py)')
    parser.add_argument('--hidden_channels', type=int, default=64, help='Number of hidden channels in the model')
    parser.add_argument('--lifting_channels', type=int, default=256, help='Number of channels in lifting layer')
    parser.add_argument('--projection_channels', type=int, default=256, help='Number of channels in projection layer')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of operator layers')

    # FNO Specific
    parser.add_argument('--modes', type=int, default=12, help='Number of Fourier modes for FNO')

    # MambaFNO Specific
    parser.add_argument('--mamba_d_state', type=int, default=16, help='Mamba state dimension')
    parser.add_argument('--mamba_d_conv', type=int, default=4, help='Mamba convolution dimension')
    parser.add_argument('--mamba_expand', type=int, default=2, help='Mamba expansion factor')

    # PerceiverIONO Specific
    parser.add_argument('--num_latents', type=int, default=256, help='Number of latent vectors in Perceiver IO')
    parser.add_argument('--latent_dim', type=int, default=128, help='Dimension of latent vectors')
    parser.add_argument('--num_cross_attention_heads', type=int, default=1, help='Number of cross-attention heads')
    parser.add_argument('--num_self_attention_heads', type=int, default=8, help='Number of self-attention heads')
    parser.add_argument('--num_perceiver_blocks', type=int, default=3, help='Number of Perceiver IO blocks')

    # CoDANO Specific
    parser.add_argument('--codano_num_heads', type=int, default=8, help='Number of heads for codomain attention')

    # SwinV2NO Specific (Placeholder parameters)
    parser.add_argument('--swin_embed_dim', type=int, default=96, help='Swin Transformer embedding dimension')
    parser.add_argument('--swin_depths', type=int, nargs='+', default=[2, 2, 6, 2], help='Depths of each Swin Transformer stage')
    parser.add_argument('--swin_num_heads', type=int, nargs='+', default=[3, 6, 12, 24], help='Number of attention heads in different stages')
    parser.add_argument('--swin_window_size', type=int, default=7, help='Window size for Swin Transformer')

    # Training
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--scheduler_step_size', type=int, default=100, help='Step size for learning rate scheduler')
    parser.add_argument('--scheduler_gamma', type=float, default=0.5, help='Gamma for learning rate scheduler')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--pretrain_epochs', type=int, default=50, help='Number of epochs for pre-training')
    parser.add_argument('--finetune_epochs', type=int, default=50, help='Number of epochs for fine-tuning')
    parser.add_argument('--fine_tune_adapters_only', action='store_true', help='Only fine-tune adapter layers')

    # Logging and Checkpointing
    parser.add_argument('--log_interval', type=int, default=10, help='How many batches to wait before logging training status')
    parser.add_argument('--save_interval', type=int, default=10, help='How many epochs to wait before saving model')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results and checkpoints')

    # Adapter-based approach
    parser.add_argument('--num_physics_tasks', type=int, default=1, help='Number of distinct physics tasks for multi-physics pretraining')
    parser.add_argument('--current_physics_idx', type=int, default=0, help='Index of the current physics task being trained (for adapter selection)')

    args = parser.parse_args([]) # Passing an empty list to prevent argparse from reading sys.argv

    return args

