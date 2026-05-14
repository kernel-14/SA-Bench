import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm
import os
from typing import Any, Dict, List, Tuple, Optional, Union

# Attempt to import core dependencies
try:
    from config import Config
    from utils.logger import Logger
    from interpretability.probe_model import ProbeModel
except ImportError:
    # Dummy classes for standalone testing or if dependencies are not yet available
    print("Warning: Could not import core dependencies. Using dummy classes for ProbeTrainer.")

    class Config:
        """Dummy Config class for self-testing."""
        def __init__(self, data: Dict = None): self._data = data if data is not None else {}
        def get(self, key: str, default: Any = None) -> Any:
            keys = key.split('.')
            current = self._data
            for k in keys:
                if isinstance(current, dict) and k in current: current = current[k]
                else: return default
            return current
        def set(self, key: str, value: Any) -> None: pass
        def save(self, output_path: str) -> None: pass

    class Logger:
        def __init__(self, config: Config): pass
        def log_info(self, message: str) -> None: print(f"INFO: {message}")
        def log_metric(self, name: str, value: float, step: int = 0, tag: str = 'train') -> None: print(f"METRIC: {tag}/{name} @ {step}: {value}")
        def log_figure(self, name: str, fig: Any, step: int = 0) -> None: pass
        def save_model_weights(self, model: Any, path: str) -> None: pass
        def load_model_weights(self, model: Any, path: str) -> None: pass
        def close(self) -> None: pass

    class ProbeModel(nn.Module):
        def __init__(self, in_channels: int, num_classes: int, kernel_size: int, is_global: bool = False) -> None:
            super().__init__()
            self.is_global = is_global
            self.num_classes = num_classes
            if is_global:
                self.probe_layer = nn.Linear(in_channels, num_classes)
            else:
                self.probe_layer = nn.Conv2d(in_channels, num_classes, kernel_size, padding=kernel_size // 2, bias=True)
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.is_global:
                return self.probe_layer(x)
            else:
                return self.probe_layer(x) # Conv2d expects B, C, H, W


class ProbeDataset(Dataset):
    """
    A PyTorch Dataset for loading and preparing data for probe training.
    It extracts relevant activations and concept labels for specific layers/ticks/concepts
    from collected trajectories.
    """

    def __init__(self, labeled_data: List[Dict[str, Any]], concept_key: str, layer_idx: int,
                 tick_idx: int, is_baseline: bool,
                 input_spatial_dims: Tuple[int, int], num_input_channels: int, is_global_probe: bool) -> None:
        """
        Initializes the ProbeDataset.

        Args:
            labeled_data (List[Dict[str, Any]]): A list of dictionaries, each representing a timestep
                                                 from a collected trajectory, containing observations,
                                                 cell states, and concept labels.
            concept_key (str): The key for the concept to extract labels for (e.g., 'CA', 'CB').
            layer_idx (int): The 0-indexed layer from which to extract activations.
            tick_idx (int): The 0-indexed internal tick from which to extract activations.
                            Ignored for ResNet (feedforward) agents.
            is_baseline (bool): If True, use raw observations as input; otherwise, use agent's cell states.
            input_spatial_dims (Tuple[int, int]): (Height, Width) dimensions of the input (e.g., 8x8 for Sokoban).
            num_input_channels (int): The number of channels for the input feature map (e.g., 32 for cell states, 7 for observations).
            is_global_probe (bool): If True, inputs will be flattened for a global linear probe.
        """
        self.inputs: List[torch.Tensor] = []
        self.labels: List[torch.Tensor] = []
        self.is_global_probe: bool = is_global_probe
        
        H, W = input_spatial_dims
        
        for timestep_data in labeled_data:
            input_tensor_np: np.ndarray
            if is_baseline:
                # For baseline probes, input is raw observation (H, W, C_obs)
                input_tensor_np = timestep_data['observations']
            else:
                # For agent-based probes, input is cell state (H, W, C_cell)
                # Check if 'cell_states_HWC_tensors' exists and contains the layer/tick
                cell_states_at_layer = timestep_data.get('cell_states_HWC_tensors', {}).get(f'layer_{layer_idx}')
                if cell_states_at_layer is None:
                    # Fallback for ResNet or if tick_idx is not meaningful
                    input_tensor_np = timestep_data['cell_states_HWC_tensors'].get(f'layer_{layer_idx}', np.zeros((H, W, num_input_channels)))
                else:
                    input_tensor_np = cell_states_at_layer.get(f'tick_{tick_idx}', np.zeros((H, W, num_input_channels)))

            # Labels are (H, W) array of integer class IDs
            # For global probes, concept_labels will contain a single integer, not a spatial map
            labels_np: Union[np.ndarray, int] = timestep_data['concept_labels'].get(concept_key, np.zeros((H, W), dtype=int))
            
            if input_tensor_np is None:
                continue

            # Convert to PyTorch tensor and adjust dimensions
            input_tensor = torch.from_numpy(input_tensor_np).float()
            
            # Conv2d expects (C, H, W), so permute (H, W, C) -> (C, H, W)
            input_tensor = input_tensor.permute(2, 0, 1)

            # Labels are long for CrossEntropyLoss
            label_tensor: torch.Tensor
            if isinstance(labels_np, np.ndarray):
                label_tensor = torch.from_numpy(labels_np).long() # (H, W) for conv probes
            else: # Scalar label for global probe (e.g., ActionToTake_1)
                label_tensor = torch.tensor(labels_np, dtype=torch.long) # Scalar

            # For global probes, flatten the input tensor (C, H, W) -> (C*H*W)
            if self.is_global_probe:
                input_tensor = input_tensor.flatten()
                # For global probes, the label is usually a single value per timestep, not a grid.
                # If label_tensor is (H,W), take first element. If it's already scalar, it remains scalar.
                if label_tensor.ndim > 0: # Check if it's not scalar
                    label_tensor = label_tensor.flatten()[0] # Take the first element as the global label
            
            self.inputs.append(input_tensor)
            self.labels.append(label_tensor)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.labels[idx]


class ProbeTrainer:
    """
    Manages the training and evaluation of ProbeModel instances.
    It prepares ProbeDataset from labeled trajectories, sets up the AdamW optimizer,
    and runs the logistic regression (cross-entropy loss) training loop.
    It calculates and reports performance metrics such as Macro F1, precision,
    and recall per class, utilizing sklearn.metrics.
    """

    def __init__(self, probe_model: ProbeModel, config: Config, logger: Logger,
                 class_names: Optional[List[str]] = None) -> None:
        """
        Initializes the ProbeTrainer.

        Args:
            probe_model (ProbeModel): The ProbeModel instance to be trained.
            config (Config): The configuration object.
            logger (Logger): The logger instance.
            class_names (Optional[List[str]]): A list of strings for class names (e.g., ['NEVER', 'UP', ...]).
                                               Used for logging class-specific metrics.
        """
        self.probe_model: ProbeModel = probe_model
        self.config: Config = config
        self.logger: Logger = logger
        self.class_names: List[str] = class_names if class_names is not None else [str(i) for i in range(probe_model.num_classes)]
        
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.probe_model.to(self.device)

        # Optimizer setup
        optimizer_type: str = self.config.get('probing.optimizer', 'AdamW')
        learning_rate: float = self.config.get('probing.learning_rate', 0.001)
        weight_decay: float = self.config.get('probing.weight_decay', 0.001)

        if optimizer_type == 'AdamW':
            self.optimizer = optim.AdamW(self.probe_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer type for probes: {optimizer_type}")

        # Loss function: CrossEntropyLoss is suitable for multi-class classification from logits
        self.loss_fn = nn.CrossEntropyLoss()

        self.best_val_macro_f1: float = -1.0 # To track best model for saving
        self.best_model_path: Optional[str] = None
        
        self.logger.log_info(f"ProbeTrainer initialized for concept with {probe_model.num_classes} classes.")
        self.logger.log_info(f"Probe model on device: {self.device}")

    def train(self, dataloader: DataLoader, concept_name: str, layer_idx: int, tick_idx: int) -> None:
        """
        Executes the training loop for the probe_model.

        Args:
            dataloader (DataLoader): DataLoader for training data.
            concept_name (str): Name of the concept being probed (for logging).
            layer_idx (int): Index of the agent layer (for logging).
            tick_idx (int): Index of the agent tick (for logging).
        """
        self.probe_model.train() # Set probe model to training mode
        num_epochs: int = self.config.get('probing.epochs', 10)
        
        self.logger.log_info(f"Training probe for concept '{concept_name}' at Layer {layer_idx}, Tick {tick_idx} for {num_epochs} epochs.")

        for epoch in tqdm(range(num_epochs), desc=f"Training probe L{layer_idx}T{tick_idx}-{concept_name}"):
            total_loss: float = 0.0
            
            for batch_idx, (inputs, labels) in enumerate(dataloader):
                # Inputs: (B, C, H, W) for conv probes, (B, C*H*W) for global probes
                # Labels: (B, H, W) for conv probes, (B,) for global probes (single action label)
                
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.probe_model(inputs) # (B, num_classes, H, W) for conv, (B, num_classes) for global
                
                if not self.probe_model.is_global:
                    # For convolutional probes, flatten outputs and labels for CrossEntropyLoss
                    # Outputs (B, C_out, H, W) -> (B*H*W, C_out)
                    # Labels (B, H, W) -> (B*H*W)
                    outputs = outputs.permute(0, 2, 3, 1).reshape(-1, self.probe_model.num_classes)
                    labels = labels.flatten()
                
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_epoch_loss: float = total_loss / len(dataloader)
            self.logger.log_metric(
                name='loss',
                value=avg_epoch_loss,
                step=epoch,
                tag=f'probe/{concept_name}/L{layer_idx}T{tick_idx}/train'
            )
            self.logger.log_info(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_epoch_loss:.4f}")

    def evaluate(self, dataloader: DataLoader, concept_name: str, layer_idx: int, tick_idx: int,
                 is_validation: bool = True) -> Dict[str, float]:
        """
        Evaluates the probe_model's performance on a given dataset.

        Args:
            dataloader (DataLoader): DataLoader for evaluation data.
            concept_name (str): Name of the concept being probed (for logging).
            layer_idx (int): Index of the agent layer (for logging).
            tick_idx (int): Index of the agent tick (for logging).
            is_validation (bool): True if this is a validation set evaluation, False for test.

        Returns:
            Dict[str, float]: A dictionary containing calculated metrics.
        """
        self.probe_model.eval() # Set probe model to evaluation mode
        
        all_labels_flat: List[int] = []
        all_predictions_flat: List[int] = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.probe_model(inputs)
                
                if not self.probe_model.is_global:
                    # For convolutional probes, flatten outputs and labels
                    outputs = outputs.permute(0, 2, 3, 1).reshape(-1, self.probe_model.num_classes)
                    labels = labels.flatten()
                
                predictions = outputs.argmax(dim=1) # Get predicted class indices

                all_labels_flat.extend(labels.cpu().numpy())
                all_predictions_flat.extend(predictions.cpu().numpy())

        # Convert to numpy arrays for sklearn metrics
        all_labels_np = np.array(all_labels_flat)
        all_predictions_np = np.array(all_predictions_flat)

        metrics: Dict[str, float] = {}
        tag_prefix: str = 'probe_val' if is_validation else 'probe_test'

        # Macro F1 Score
        macro_f1: float = f1_score(all_labels_np, all_predictions_np, average='macro', zero_division=0)
        metrics['macro_f1'] = macro_f1
        self.logger.log_metric(
            name='macro_f1',
            value=macro_f1,
            step=self.config.get('global_step', 0), # Assume global_step is maintained in config or passed
            tag=f'{tag_prefix}/{concept_name}/L{layer_idx}T{tick_idx}'
        )

        # Accuracy
        accuracy: float = accuracy_score(all_labels_np, all_predictions_np)
        metrics['accuracy'] = accuracy
        self.logger.log_metric(
            name='accuracy',
            value=accuracy,
            step=self.config.get('global_step', 0),
            tag=f'{tag_prefix}/{concept_name}/L{layer_idx}T{tick_idx}'
        )
        
        # Class-Specific Metrics
        for k in range(self.probe_model.num_classes):
            class_name = self.class_names[k] if k < len(self.class_names) else f"Class_{k}"
            
            binary_true_k = (all_labels_np == k).astype(int)
            binary_pred_k = (all_predictions_np == k).astype(int)

            precision_k: float = precision_score(binary_true_k, binary_pred_k, zero_division=0)
            recall_k: float = recall_score(binary_true_k, binary_pred_k, zero_division=0)
            f1_k: float = f1_score(binary_true_k, binary_pred_k, zero_division=0)

            metrics[f'{class_name}_precision'] = precision_k
            metrics[f'{class_name}_recall'] = recall_k
            metrics[f'{class_name}_f1'] = f1_k

            self.logger.log_metric(
                name=f'{class_name}_precision',
                value=precision_k,
                step=self.config.get('global_step', 0),
                tag=f'{tag_prefix}/{concept_name}/L{layer_idx}T{tick_idx}'
            )
            self.logger.log_metric(
                name=f'{class_name}_recall',
                value=recall_k,
                step=self.config.get('global_step', 0),
                tag=f'{tag_prefix}/{concept_name}/L{layer_idx}T{tick_idx}'
            )
            self.logger.log_metric(
                name=f'{class_name}_f1',
                value=f1_k,
                step=self.config.get('global_step', 0),
                tag=f'{tag_prefix}/{concept_name}/L{layer_idx}T{tick_idx}'
            )

        self.logger.log_info(f"Evaluation for probe L{layer_idx}T{tick_idx}-{concept_name}: Macro F1={macro_f1:.4f}, Accuracy={accuracy:.4f}")

        # Optionally save best model
        if is_validation and macro_f1 > self.best_val_macro_f1:
            self.best_val_macro_f1 = macro_f1
            # Ensure path exists before saving
            chkpt_dir = self.logger.checkpoints_dir
            if not os.path.exists(chkpt_dir):
                os.makedirs(chkpt_dir)
            self.best_model_path = os.path.join(
                chkpt_dir,
                f"probe_{concept_name}_L{layer_idx}T{tick_idx}_best.pth"
            )
            self.logger.save_model_weights(self.probe_model, self.best_model_path)
            self.logger.log_info(f"New best probe model saved with Macro F1: {macro_f1:.4f}")

        return metrics


if __name__ == '__main__':
    print("--- Testing ProbeTrainer and ProbeDataset ---")

    # --- Dummy Config for testing ---
    dummy_config_data = {
        'experiment_name': 'test_probe_trainer',
        'paths': {
            'results_dir': 'test_results_probe_trainer',
            'log_dir': 'run_logs',
            'checkpoint_dir': 'models',
            'figures_dir': 'visuals'
        },
        'probing': {
            'optimizer': 'AdamW',
            'learning_rate': 0.001,
            'weight_decay': 0.001,
            'epochs': 2, # Small number for quick test
            'batch_size': 4 # Small batch size for quick test
        },
        'environment': {
            'name': 'Sokoban',
            'sokoban': {
                'grid_size': [8, 8],
                'observation_channels': 7,
            }
        },
        'agent': {
            'type': 'DRCAgent',
            'drc_agent': {
                'D': 3,
                'N': 3,
                'convlstm_channels': 32,
            }
        },
        'global_step': 0 # For logging
    }
    dummy_config = Config(dummy_config_data)
    dummy_logger = Logger(dummy_config) # Logger will create its own directories

    # --- Dummy Labeled Data for ProbeDataset ---
    num_timesteps_dummy = 10
    grid_h, grid_w = 8, 8
    num_obs_channels = dummy_config.get('environment.sokoban.observation_channels') # 7
    num_cell_channels = dummy_config.get('agent.drc_agent.convlstm_channels') # 32
    num_layers = dummy_config.get('agent.drc_agent.D') # 3
    num_ticks_plus_one = dummy_config.get('agent.drc_agent.N') + 1 # 4

    dummy_labeled_data: List[Dict[str, Any]] = []
    for _ in range(num_timesteps_dummy):
        cs_h_w_c = {}
        for l in range(num_layers):
            cs_h_w_c[f'layer_{l}'] = {}
            for t in range(num_ticks_plus_one):
                cs_h_w_c[f'layer_{l}'][f'tick_{t}'] = np.random.rand(grid_h, grid_w, num_cell_channels).astype(np.float32)
        
        dummy_labeled_data.append({
            'observations': np.random.rand(grid_h, grid_w, num_obs_channels).astype(np.float32),
            'cell_states_HWC_tensors': cs_h_w_c,
            'concept_labels': {
                'CA': np.random.randint(0, 5, size=(grid_h, grid_w)), # 5 classes for CA/CB
                'CB': np.random.randint(0, 5, size=(grid_h, grid_w)),
                'ActionToTake_1': np.random.randint(0, 5) # Single action for global probe
            }
        })
    
    # --- Test Case 1: Spatially-localized Probe (CA, layer 0, tick N-1) ---
    print("\n--- Testing Spatially-localized Probe (CA) ---")
    concept_key_conv = 'CA'
    layer_idx_conv = 0
    tick_idx_conv = num_ticks_plus_one - 1 # Final tick N
    num_classes_conv = 5
    kernel_size_conv = 1 # 1x1 probe
    
    probe_model_conv = ProbeModel(num_cell_channels, num_classes_conv, kernel_size_conv, is_global=False)
    probe_dataset_conv = ProbeDataset(
        dummy_labeled_data, concept_key_conv, layer_idx_conv, tick_idx_conv,
        is_baseline=False, input_spatial_dims=(grid_h, grid_w), num_input_channels=num_cell_channels,
        is_global_probe=False
    )
    probe_dataloader_conv = DataLoader(probe_dataset_conv, batch_size=dummy_config.get('probing.batch_size'), shuffle=True)
    
    probe_trainer_conv = ProbeTrainer(probe_model_conv, dummy_config, dummy_logger, class_names=['NEVER', 'UP', 'DOWN', 'LEFT', 'RIGHT'])
    probe_trainer_conv.train(probe_dataloader_conv, concept_key_conv, layer_idx_conv, tick_idx_conv)
    metrics_conv = probe_trainer_conv.evaluate(probe_dataloader_conv, concept_key_conv, layer_idx_conv, tick_idx_conv)
    print(f"Metrics for CA (conv): {metrics_conv}")
    
    # --- Test Case 2: Global Probe (ActionToTake_1) ---
    print("\n--- Testing Global Probe (ActionToTake_1) ---")
    concept_key_global = 'ActionToTake_1'
    layer_idx_global = 0 # Layer does not matter as much for global probe, but we still need to pick one to get output features
    tick_idx_global = num_ticks_plus_one - 1
    num_classes_global = 5 # Actions 0-4
    
    # For global probe, in_channels is flattened (C*H*W)
    in_channels_global = num_cell_channels * grid_h * grid_w
    probe_model_global = ProbeModel(in_channels_global, num_classes_global, kernel_size=1, is_global=True)
    probe_dataset_global = ProbeDataset(
        dummy_labeled_data, concept_key_global, layer_idx_global, tick_idx_global,
        is_baseline=False, input_spatial_dims=(grid_h, grid_w), num_input_channels=num_cell_channels,
        is_global_probe=True
    )
    probe_dataloader_global = DataLoader(probe_dataset_global, batch_size=dummy_config.get('probing.batch_size'), shuffle=True)
    
    probe_trainer_global = ProbeTrainer(probe_model_global, dummy_config, dummy_logger, class_names=['Up', 'Down', 'Left', 'Right', 'No-op'])
    probe_trainer_global.train(probe_dataloader_global, concept_key_global, layer_idx_global, tick_idx_global)
    metrics_global = probe_trainer_global.evaluate(probe_dataloader_global, concept_key_global, layer_idx_global, tick_idx_global)
    print(f"Metrics for ActionToTake_1 (global): {metrics_global}")

    dummy_logger.close()
    
    # Clean up generated directories
    import shutil
    if os.path.exists(dummy_config.get('paths.results_dir')):
        shutil.rmtree(dummy_config.get('paths.results_dir'))
        print(f"\nCleaned up '{dummy_config.get('paths.results_dir')}' directory.")
    print("\n--- ProbeTrainer testing complete ---")

