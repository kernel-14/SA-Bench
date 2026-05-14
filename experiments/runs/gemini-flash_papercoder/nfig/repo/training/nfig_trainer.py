import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from omegaconf import OmegaConf # For saving DictConfig

# Assuming these are defined in models/nfig_transformer.py, datasets.py, and config.py
from models.nfig_transformer import NFIGTransformer
from datasets import DatasetLoader
from config import Config
from utils import set_seed # Assuming set_seed is in utils.py


class NFIGTrainer:
    """
    Manages the training loop for the Next-Frequency Image Generation (NFIG) Transformer.
    This includes handling data loading, optimizer setup, the core training loop,
    loss calculation, backpropagation, and checkpointing.
    It incorporates the Classifier-Free Guidance (CFG) training strategy.
    """

    def __init__(self, nfig_transformer: NFIGTransformer, config: Config, token_data_loader: DatasetLoader):
        """
        Initializes the NFIGTrainer.

        Args:
            nfig_transformer (NFIGTransformer): The NFIG Transformer model.
            config (Config): Project configuration object.
            token_data_loader (DatasetLoader): Data loader utility providing tokenized sequences.
        """
        self.nfig_transformer = nfig_transformer
        self.config = config
        self.token_data_loader = token_data_loader # This is actually a DatasetLoader instance, not a DataLoader

        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nfig_transformer.to(self.device)

        # Optimizers
        optimizer_type: str = self.config.nfig_transformer_training.optimizer
        learning_rate: float = self.config.nfig_transformer_training.learning_rate

        if optimizer_type == "Adam":
            self.optimizer = optim.Adam(
                self.nfig_transformer.parameters(),
                lr=learning_rate
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

        # Loss function (Cross-Entropy Loss for token prediction)
        # NFIGTransformer's vocab_size is the number of possible token indices.
        # The 'null_class_token' is an input conditioning, not a target token to be ignored by CrossEntropyLoss.
        self.criterion = nn.CrossEntropyLoss()

        # Training parameters from config
        self.epochs: int = self.config.nfig_transformer_training.epochs
        self.batch_size: int = self.config.nfig_transformer_training.batch_size
        self.unconditional_training_probability: float = self.config.nfig_transformer.unconditional_training_probability
        
        # Null class token for CFG training. It's an index outside the range of actual classes.
        # The NFIGTransformer's class_embedding is initialized with num_classes + 1 capacity.
        self.null_class_token_idx: int = self.config.data.num_classes

        self.start_epoch: int = 0
        self.best_loss: float = float('inf') # Track best loss for saving model

        # Setup checkpoint directory
        self.checkpoint_dir: str = os.path.join("checkpoints", "nfig_transformer")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        print(f"NFIGTrainer initialized on device: {self.device}")
        print(f"NFIGTransformer training for {self.epochs} epochs with batch size {self.batch_size}.")
        print(f"Unconditional training probability for CFG: {self.unconditional_training_probability}.")

    def _train_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> dict:
        """
        Performs a single training step for one batch.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): A batch containing tokenized
                                                     sequences and their corresponding class labels.
                                                     - token_indices: (B, total_sequence_length)
                                                     - class_labels: (B,)

        Returns:
            dict: A dictionary of computed loss values for the current step.
        """
        token_indices, class_labels = batch
        token_indices = token_indices.to(self.device)
        class_labels = class_labels.to(self.device)

        # Prepare input and target sequences for autoregressive prediction
        # Input: all tokens except the last one
        input_tokens = token_indices[:, :-1] # (B, total_sequence_length - 1)
        # Target: all tokens except the first one
        target_tokens = token_indices[:, 1:]  # (B, total_sequence_length - 1)

        # Generate causal attention mask for the input sequence
        # The mask prevents attending to future tokens.
        # NFIGTransformer provides a helper for this.
        attn_mask = self.nfig_transformer._generate_square_subsequent_mask(
            input_tokens.size(1), self.device
        )

        # Implement Classifier-Free Guidance (CFG) training strategy
        # Randomly replace some class labels with a null token
        uncond_mask = (torch.rand(class_labels.shape, device=self.device) < self.unconditional_training_probability)
        class_labels_for_forward = class_labels.clone()
        class_labels_for_forward[uncond_mask] = self.null_class_token_idx

        # --- Forward Pass ---
        self.optimizer.zero_grad()
        self.nfig_transformer.train() # Ensure model is in training mode
        
        # Get logits from the NFIG Transformer
        # logits: (B, total_sequence_length - 1, vocab_size)
        logits = self.nfig_transformer(
            input_tokens, class_labels_for_forward, attn_mask
        )

        # --- Loss Calculation ---
        # Reshape logits and target_tokens for CrossEntropyLoss
        # logits: (B * (total_sequence_length - 1), vocab_size)
        # target_tokens: (B * (total_sequence_length - 1))
        loss = self.criterion(
            logits.view(-1, logits.size(-1)),
            target_tokens.view(-1)
        )

        # --- Backward Pass and Optimizer Step ---
        loss.backward()
        self.optimizer.step()
        
        return {"loss": loss.item()}

    def train(self) -> None:
        """
        Runs the main training loop for the NFIG Transformer.
        """
        print(f"Starting NFIG Transformer training for {self.epochs} epochs.")
        
        # Get the tokenized training data loader
        # This DataLoader is assumed to be prepared by DatasetLoader.tokenize_dataset
        # and then passed as `token_data_loader` during initialization.
        train_dataloader = self.token_data_loader.get_token_dataloader(
            token_dir=os.path.join(self.config.data.dataset_root, "tokenized_data"),
            batch_size=self.batch_size,
            sequence_length=self.config.nfig_transformer.total_sequence_length
        )

        for epoch in range(self.start_epoch, self.epochs):
            self.nfig_transformer.train() # Ensure model is in training mode
            
            total_epoch_loss: float = 0.0
            num_batches: int = 0

            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{self.epochs}")
            for batch_idx, batch in enumerate(pbar):
                step_metrics = self._train_step(batch)
                batch_loss = step_metrics['loss']
                
                total_epoch_loss += batch_loss
                num_batches += 1
                
                # Update progress bar description with current loss
                pbar.set_postfix({'loss': f"{batch_loss:.4f}"})

            avg_epoch_loss: float = total_epoch_loss / num_batches if num_batches > 0 else 0.0
            print(f"\nEpoch {epoch+1} finished. Average Loss: {avg_epoch_loss:.4f}")

            # Save checkpoint
            checkpoint_path = os.path.join(self.checkpoint_dir, f"nfig_transformer_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'nfig_transformer_state_dict': self.nfig_transformer.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_loss': self.best_loss, # Current best loss (to be updated if better)
                'config': OmegaConf.to_container(self.config._cfg, resolve=True) # Save raw config dict
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

            # Update best_loss and save best model if current epoch loss is better
            if avg_epoch_loss < self.best_loss:
                self.best_loss = avg_epoch_loss
                best_checkpoint_path = os.path.join(self.checkpoint_dir, "nfig_transformer_best.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'nfig_transformer_state_dict': self.nfig_transformer.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_loss': self.best_loss,
                    'config': OmegaConf.to_container(self.config._cfg, resolve=True)
                }, best_checkpoint_path)
                print(f"New best model saved at {best_checkpoint_path} with loss: {self.best_loss:.4f}")

        print("NFIG Transformer training complete.")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Loads a checkpoint to resume training or for evaluation.

        Args:
            checkpoint_path (str): Path to the checkpoint file.
        """
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found at {checkpoint_path}. Starting NFIG training from scratch.")
            return

        print(f"Loading NFIG Transformer checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.nfig_transformer.load_state_dict(checkpoint['nfig_transformer_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch']
        self.best_loss = checkpoint.get('best_loss', float('inf'))

        print(f"Loaded checkpoint from {checkpoint_path}. Resuming from epoch {self.start_epoch + 1}.")

