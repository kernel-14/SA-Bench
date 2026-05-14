import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Tuple, Union, Callable

from config import Config
from dataset_manager import DatasetManager
from models.neural_operator import NeuralOperatorModel
from models.base_operator import CoreOperator
from models.adapters import LiftingAdapter, ProjectionAdapter
from models.fno import FNO
from models.mamba_fno import MambaFNO
from models.perceiver_fno import PerceiverFNO
from models.swin_v2 import SwinV2
from models.codano import CoDANo
from utils import get_device, count_parameters, get_activation_fn


class Trainer:
    """
    Orchestrates the training process, including pre-training, fine-tuning, and training from scratch.
    """

    def __init__(self, config: Config, model_factory: Callable[..., NeuralOperatorModel], dataset_manager: DatasetManager):
        """
        Initializes the Trainer.

        Args:
            config (Config): The global configuration object.
            model_factory (Callable[..., NeuralOperatorModel]): A callable function that, given
                                                                 lifting_adapter, core_operator, and
                                                                 projection_adapter instances, returns
                                                                 a NeuralOperatorModel. This is typically
                                                                 the NeuralOperatorModel constructor itself.
            dataset_manager (DatasetManager): An instance of the DatasetManager for data access.
        """
        if not isinstance(config, Config):
            raise TypeError(f"Expected 'config' to be an instance of Config, but got {type(config)}.")
        if not callable(model_factory):
            raise TypeError(f"Expected 'model_factory' to be a callable, but got {type(model_factory)}.")
        if not isinstance(dataset_manager, DatasetManager):
            raise TypeError(f"Expected 'dataset_manager' to be an instance of DatasetManager, but got {type(dataset_manager)}.")

        self.config = config
        self.model_factory = model_factory
        self.dataset_manager = dataset_manager
        self.device = get_device(self.config.device)

        # Store model component classes for dynamic instantiation
        self.core_operator_classes = {
            "FNO": FNO,
            "MambaFNO": MambaFNO,
            "PerceiverFNO": PerceiverFNO,
            "SwinV2": SwinV2,
            "CoDANo": CoDANo,
        }
        
        # Ensure checkpoint directories exist
        os.makedirs(os.path.dirname(self.config.training_settings['pretrain_save_path']), exist_ok=True)
        os.makedirs(os.path.dirname(self.config.training_settings['finetune_save_path']), exist_ok=True)
        os.makedirs(os.path.dirname(self.config.training_settings['scratch_save_path']), exist_ok=True)

    def _create_lifting_adapter(self, input_dim: int) -> LiftingAdapter:
        """Helper to create a LiftingAdapter instance."""
        hidden_dim = self.config.model_settings['hidden_dim']
        num_mlp_layers = self.config.model_settings['num_mlp_layers']
        activation = self.config.model_settings['activation']
        return LiftingAdapter(input_dim=input_dim, hidden_dim=hidden_dim,
                              num_mlp_layers=num_mlp_layers, activation=activation)

    def _create_projection_adapter(self, output_dim: int) -> ProjectionAdapter:
        """Helper to create a ProjectionAdapter instance."""
        hidden_dim = self.config.model_settings['hidden_dim']
        num_mlp_layers = self.config.model_settings['num_mlp_layers']
        activation = self.config.model_settings['activation']
        return ProjectionAdapter(hidden_dim=hidden_dim, output_dim=output_dim,
                                 num_mlp_layers=num_mlp_layers, activation=activation)

    def _create_core_operator(self) -> CoreOperator:
        """Helper to create a CoreOperator instance based on config."""
        core_type = self.config.model_settings.get('core_operator_type', 'FNO') # Default to FNO
        CoreOpClass = self.core_operator_classes.get(core_type)
        if CoreOpClass is None:
            raise ValueError(f"Unknown core operator type specified in config: {core_type}")

        hidden_dim = self.config.model_settings['hidden_dim']
        core_config_key = f"{core_type.lower()}_config"
        core_config = self.config.model_settings.get(core_config_key, {})

        # Handle specific constructor arguments for each CoreOperator type
        if core_type == "FNO":
            return CoreOpClass(hidden_dim=hidden_dim, **core_config)
        elif core_type == "MambaFNO":
            # MambaFNO expects hidden_dim, fno_config, mamba_config
            fno_config = self.config.model_settings.get('fno_config', {})
            mamba_config = self.config.model_settings.get('mamba_config', {})
            return CoreOpClass(hidden_dim=hidden_dim, fno_config=fno_config, mamba_config=mamba_config)
        elif core_type == "PerceiverFNO":
            # PerceiverFNO expects hidden_dim, perceiver_config
            perceiver_config = self.config.model_settings.get('perceiver_config', {})
            return CoreOpClass(hidden_dim=hidden_dim, perceiver_config=perceiver_config)
        elif core_type == "SwinV2":
            # SwinV2 expects input_dim, output_dim, img_size, patch_size, embed_dim, depths, num_heads etc.
            # input_dim and output_dim for SwinV2's context here would be the hidden_dim of the overall NO.
            # img_size is derived from data resolution.
            swin_config = self.config.model_settings.get('swin_v2_config', {})
            spatial_resolution = self.config.data_settings.get('spatial_resolution', 64)
            img_size = (spatial_resolution, spatial_resolution) # Assuming square 2D input
            # Check for 1D case for SwinV2: current implementation assumes 2D image-like input
            # If 1D data is processed, it would need to be reshaped to 2D-like input for SwinV2
            # For now, let's assume 2D if SwinV2 is used, or handle 1D reshaping.
            # A simpler way is to assume `input_dim` and `output_dim` of SwinV2 should map to `hidden_dim`.
            return CoreOpClass(
                input_dim=hidden_dim,
                output_dim=hidden_dim, # SwinV2 maps C to C in the core
                img_size=img_size,
                patch_size=swin_config.get('patch_size', 4), # Assumed to be tuple or int for to_2tuple
                embed_dim=swin_config.get('embed_dim', 96),
                depths=swin_config.get('depths', [2, 2, 6, 2]),
                num_heads=swin_config.get('num_heads', [3, 6, 12, 24]),
                window_size=swin_config.get('window_size', 7),
                mlp_ratio=swin_config.get('mlp_ratio', 4.0),
                drop_path_rate=self.config.training_settings.get('drop_path_rate', 0.1) # from config.yaml training
            )
        elif core_type == "CoDANo":
            # CoDANo expects input_dim, output_dim, hidden_dim, num_attention_heads, num_layers
            codano_config = self.config.model_settings.get('codano_config', {})
            return CoreOpClass(
                input_dim=hidden_dim,
                output_dim=hidden_dim, # CoDANo maps C to C in the core
                hidden_dim=hidden_dim,
                num_attention_heads=codano_config.get('num_attention_heads', 8),
                num_layers=codano_config.get('num_layers', 4)
            )
        else:
            raise NotImplementedError(f"Core operator type '{core_type}' is not yet implemented in _create_core_operator.")


    def _train_epoch(self, model: NeuralOperatorModel, dataloader: DataLoader, optimizer: optim.Optimizer,
                     is_finetuning: bool) -> float:
        """
        Executes a single training epoch over the provided dataloader.

        Args:
            model (NeuralOperatorModel): The model to train.
            dataloader (DataLoader): DataLoader for the training data.
            optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
            is_finetuning (bool): Flag indicating if it's a fine-tuning epoch (affects logging).

        Returns:
            float: The average loss for the epoch.
        """
        model.train()
        total_loss = 0.0
        num_samples = 0
        start_time = time.time()

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            optimizer.zero_grad()
            predictions = model(inputs)
            loss = F.mse_loss(predictions, targets)
            loss.backward()

            gradient_clip_val = self.config.training_settings.get('gradient_clip_val')
            if gradient_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)

            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            num_samples += inputs.size(0)
        
        end_time = time.time()
        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        print(f"  Train Epoch Loss: {avg_loss:.6f} | Time: {end_time - start_time:.2f}s")
        return avg_loss

    def _validate(self, model: NeuralOperatorModel, dataloader: DataLoader) -> float:
        """
        Evaluates the model's performance on a validation set for one epoch.

        Args:
            model (NeuralOperatorModel): The model to evaluate.
            dataloader (DataLoader): DataLoader for the validation data.

        Returns:
            float: The average loss on the validation set.
        """
        model.eval()
        total_loss = 0.0
        num_samples = 0
        start_time = time.time()

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                predictions = model(inputs)
                loss = F.mse_loss(predictions, targets)
                total_loss += loss.item() * inputs.size(0)
                num_samples += inputs.size(0)
        
        end_time = time.time()
        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        print(f"  Validation Epoch Loss: {avg_loss:.6f} | Time: {end_time - start_time:.2f}s")
        return avg_loss

    def pretrain(self, multiphysics_datasets: Dict[str, Tuple[DataLoader, DataLoader, DataLoader]]) -> Dict[str, Any]:
        """
        Manages the pre-training loop over multiple PDEs, optimizing a shared core operator
        and individual adapters for each PDE.

        Args:
            multiphysics_datasets (Dict[str, Tuple[DataLoader, DataLoader, DataLoader]]):
                A dictionary where keys are PDE names and values are tuples of
                (train_dataloader, val_dataloader, test_dataloader).

        Returns:
            Dict[str, Any]: A dictionary containing the state_dict of the best pre-trained
                            core operator and its corresponding validation loss.
        """
        print("\n--- Starting Pre-training ---")
        
        # 1. Initialize the shared CoreOperator
        shared_core_operator = self._create_core_operator()
        shared_core_operator.to(self.device)
        print(f"Shared Core Operator: {shared_core_operator.__class__.__name__} Parameters: {count_parameters(shared_core_operator)}")

        # 2. Initialize problem-specific Lifting and Projection adapters for each PDE
        pde_specific_components: Dict[str, Tuple[LiftingAdapter, ProjectionAdapter]] = {}
        for pde_name, (train_dl, _, _) in multiphysics_datasets.items():
            # Get input_dim and output_dim from a sample in the dataloader
            sample_inputs, sample_targets = next(iter(train_dl))
            input_dim = sample_inputs.shape[-1]
            output_dim = sample_targets.shape[-1]

            lifting_adapter = self._create_lifting_adapter(input_dim)
            projection_adapter = self._create_projection_adapter(output_dim)
            
            lifting_adapter.to(self.device)
            projection_adapter.to(self.device)
            pde_specific_components[pde_name] = (lifting_adapter, projection_adapter)
            print(f"  PDE '{pde_name}': Lifting Params: {count_parameters(lifting_adapter)}, Projection Params: {count_parameters(projection_adapter)}")

        # 3. Collect all trainable parameters for the optimizer
        trainable_parameters = list(shared_core_operator.parameters())
        for lifting, projection in pde_specific_components.values():
            trainable_parameters.extend(lifting.parameters())
            trainable_parameters.extend(projection.parameters())
        
        optimizer_type = getattr(optim, self.config.training_settings.get('optimizer', 'Adam'))
        optimizer = optimizer_type(
            trainable_parameters,
            lr=self.config.training_settings['learning_rate'],
            weight_decay=self.config.training_settings['weight_decay']
        )

        # 4. Learning Rate Scheduler
        lr_scheduler_config = self.config.training_settings.get('lr_scheduler', {})
        scheduler = None
        if lr_scheduler_config and lr_scheduler_config.get('type'):
            scheduler_type = getattr(optim.lr_scheduler, lr_scheduler_config['type'])
            scheduler_params = lr_scheduler_config.get('params', {})
            if lr_scheduler_config['type'] == 'ReduceLROnPlateau':
                scheduler = scheduler_type(optimizer, mode='min', **scheduler_params)
            else:
                # Assuming T_max for CosineAnnealingLR might depend on epochs_pretrain
                if 'T_max' in scheduler_params:
                    scheduler_params['T_max'] = self.config.training_settings['epochs_pretrain']
                scheduler = scheduler_type(optimizer, **scheduler_params)


        best_val_loss = float('inf')
        patience_counter = 0
        best_core_state = None
        
        epochs_pretrain = self.config.training_settings['epochs_pretrain']
        early_stopping_patience = self.config.training_settings['early_stopping_patience']

        for epoch in range(1, epochs_pretrain + 1):
            print(f"\nPretrain Epoch {epoch}/{epochs_pretrain}")
            
            epoch_train_losses: List[float] = []
            epoch_val_losses: List[float] = []
            
            # Iterate through each PDE for training and validation
            for pde_name, (train_dl, val_dl, _) in multiphysics_datasets.items():
                print(f"  Training/Validating PDE: {pde_name}")
                lifting, projection = pde_specific_components[pde_name]
                current_model = self.model_factory(lifting, shared_core_operator, projection)

                train_loss = self._train_epoch(current_model, train_dl, optimizer, is_finetuning=False)
                val_loss = self._validate(current_model, val_dl)
                
                epoch_train_losses.append(train_loss)
                epoch_val_losses.append(val_loss)

            avg_epoch_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
            avg_epoch_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)

            print(f"Combined Average Train Loss: {avg_epoch_train_loss:.6f}")
            print(f"Combined Average Val Loss: {avg_epoch_val_loss:.6f}")

            # LR Scheduler step
            if scheduler:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(avg_epoch_val_loss)
                else:
                    scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                print(f"  Learning Rate: {current_lr:.6e}")


            # Early Stopping
            if avg_epoch_val_loss < best_val_loss:
                best_val_loss = avg_epoch_val_loss
                patience_counter = 0
                best_core_state = shared_core_operator.state_dict()
                print(f"  New best validation loss: {best_val_loss:.6f}. Saving core state.")
                self.save_model(shared_core_operator, os.path.dirname(self.config.training_settings['pretrain_save_path']), "pretrain_core.pth")
            else:
                patience_counter += 1
                print(f"  Patience: {patience_counter}/{early_stopping_patience}")
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break
        
        print("\n--- Pre-training Finished ---")
        if best_core_state is None:
            raise RuntimeError("Pre-training completed but no best core state was saved (e.g., due to no improvement).")

        return {'best_core_state': best_core_state, 'best_val_loss': best_val_loss}

    def finetune(self, pde_name: str, ft_dataloaders: Tuple[DataLoader, DataLoader, DataLoader],
                 pretrained_core_state: Dict[str, Any]) -> NeuralOperatorModel:
        """
        Fine-tunes the pre-trained core operator by training new, problem-specific adapter layers.

        Args:
            pde_name (str): The name of the PDE for fine-tuning.
            ft_dataloaders (Tuple[DataLoader, DataLoader, DataLoader]):
                Training, validation, and test DataLoaders for the fine-tuning PDE.
            pretrained_core_state (Dict[str, Any]): The state_dict of the pre-trained core operator.

        Returns:
            NeuralOperatorModel: The fine-tuned model.
        """
        print(f"\n--- Starting Fine-tuning for {pde_name} ---")

        train_dl, val_dl, _ = ft_dataloaders
        
        # Determine input and output dimensions from the fine-tuning dataset
        sample_inputs, sample_targets = next(iter(train_dl))
        input_dim = sample_inputs.shape[-1]
        output_dim = sample_targets.shape[-1]

        # 1. Initialize new Lifting and Projection adapters for the fine-tuning task
        lifting_adapter = self._create_lifting_adapter(input_dim)
        projection_adapter = self._create_projection_adapter(output_dim)

        # 2. Initialize the CoreOperator and load pre-trained state
        core_operator = self._create_core_operator()
        core_operator.load_state_dict(pretrained_core_state) # Load weights from pre-training

        # 3. Compose the NeuralOperatorModel
        model = self.model_factory(lifting_adapter, core_operator, projection_adapter)
        model.to(self.device)

        # 4. Freeze the core operator
        model.freeze_core_operator()
        print(f"Model Parameters (total): {count_parameters(model)}")
        print(f"Model Parameters (trainable - adapters only): {len(model.get_trainable_parameters(only_adapters=True))}")

        # 5. Optimizer Setup (only for adapters)
        optimizer_type = getattr(optim, self.config.training_settings.get('optimizer', 'Adam'))
        optimizer = optimizer_type(
            model.get_trainable_parameters(only_adapters=True),
            lr=self.config.training_settings['learning_rate'],
            weight_decay=self.config.training_settings['weight_decay']
        )

        # 6. Learning Rate Scheduler
        lr_scheduler_config = self.config.training_settings.get('lr_scheduler', {})
        scheduler = None
        if lr_scheduler_config and lr_scheduler_config.get('type'):
            scheduler_type = getattr(optim.lr_scheduler, lr_scheduler_config['type'])
            scheduler_params = lr_scheduler_config.get('params', {})
            if lr_scheduler_config['type'] == 'ReduceLROnPlateau':
                scheduler = scheduler_type(optimizer, mode='min', **scheduler_params)
            else:
                if 'T_max' in scheduler_params:
                    scheduler_params['T_max'] = self.config.training_settings['epochs_finetune']
                scheduler = scheduler_type(optimizer, **scheduler_params)


        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        epochs_finetune = self.config.training_settings['epochs_finetune']
        early_stopping_patience = self.config.training_settings['early_stopping_patience']

        for epoch in range(1, epochs_finetune + 1):
            print(f"\nFinetune Epoch {epoch}/{epochs_finetune}")
            train_loss = self._train_epoch(model, train_dl, optimizer, is_finetuning=True)
            val_loss = self._validate(model, val_dl)

            # LR Scheduler step
            if scheduler:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                print(f"  Learning Rate: {current_lr:.6e}")

            # Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict()
                print(f"  New best validation loss: {best_val_loss:.6f}. Saving model state.")
            else:
                patience_counter += 1
                print(f"  Patience: {patience_counter}/{early_stopping_patience}")
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break
        
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            self.save_model(model, os.path.dirname(self.config.training_settings['finetune_save_path']), 
                            self.config.training_settings['finetune_save_path'].format(pde_name=pde_name).split('/')[-1])
        else:
            print("Fine-tuning completed but no best model state was saved. Using the last trained state.")
            self.save_model(model, os.path.dirname(self.config.training_settings['finetune_save_path']), 
                            self.config.training_settings['finetune_save_path'].format(pde_name=pde_name).split('/')[-1])
        
        print(f"--- Fine-tuning for {pde_name} Finished ---")
        return model

    def train_from_scratch(self, pde_name: str, scratch_dataloaders: Tuple[DataLoader, DataLoader, DataLoader]) -> NeuralOperatorModel:
        """
        Trains a complete NeuralOperatorModel from random initialization on a single PDE's dataset.

        Args:
            pde_name (str): The name of the PDE for scratch training.
            scratch_dataloaders (Tuple[DataLoader, DataLoader, DataLoader]):
                Training, validation, and test DataLoaders for the scratch training PDE.

        Returns:
            NeuralOperatorModel: The trained model from scratch.
        """
        print(f"\n--- Starting Training From Scratch for {pde_name} ---")

        train_dl, val_dl, _ = scratch_dataloaders

        # Determine input and output dimensions from the scratch dataset
        sample_inputs, sample_targets = next(iter(train_dl))
        input_dim = sample_inputs.shape[-1]
        output_dim = sample_targets.shape[-1]
        
        # 1. Initialize all components from scratch
        lifting_adapter = self._create_lifting_adapter(input_dim)
        core_operator = self._create_core_operator()
        projection_adapter = self._create_projection_adapter(output_dim)

        # 2. Compose the NeuralOperatorModel
        model = self.model_factory(lifting_adapter, core_operator, projection_adapter)
        model.to(self.device)

        # Ensure all parameters are trainable
        model.unfreeze_core_operator()
        print(f"Model Parameters (total & trainable): {count_parameters(model)}")

        # 3. Optimizer Setup (for all parameters)
        optimizer_type = getattr(optim, self.config.training_settings.get('optimizer', 'Adam'))
        optimizer = optimizer_type(
            model.get_trainable_parameters(only_adapters=False),
            lr=self.config.training_settings['learning_rate'],
            weight_decay=self.config.training_settings['weight_decay']
        )

        # 4. Learning Rate Scheduler
        lr_scheduler_config = self.config.training_settings.get('lr_scheduler', {})
        scheduler = None
        if lr_scheduler_config and lr_scheduler_config.get('type'):
            scheduler_type = getattr(optim.lr_scheduler, lr_scheduler_config['type'])
            scheduler_params = lr_scheduler_config.get('params', {})
            # For scratch training, using epochs_pretrain as total epochs per task as per paper's description of scratch baseline.
            if lr_scheduler_config['type'] == 'ReduceLROnPlateau':
                scheduler = scheduler_type(optimizer, mode='min', **scheduler_params)
            else:
                if 'T_max' in scheduler_params:
                    scheduler_params['T_max'] = self.config.training_settings['epochs_pretrain'] # Use pretrain epochs as total for scratch
                scheduler = scheduler_type(optimizer, **scheduler_params)

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        epochs_scratch = self.config.training_settings['epochs_pretrain'] # As per task description
        early_stopping_patience = self.config.training_settings['early_stopping_patience']

        for epoch in range(1, epochs_scratch + 1):
            print(f"\nScratch Epoch {epoch}/{epochs_scratch}")
            train_loss = self._train_epoch(model, train_dl, optimizer, is_finetuning=False)
            val_loss = self._validate(model, val_dl)

            # LR Scheduler step
            if scheduler:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                print(f"  Learning Rate: {current_lr:.6e}")


            # Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict()
                print(f"  New best validation loss: {best_val_loss:.6f}. Saving model state.")
            else:
                patience_counter += 1
                print(f"  Patience: {patience_counter}/{early_stopping_patience}")
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            self.save_model(model, os.path.dirname(self.config.training_settings['scratch_save_path']), 
                            self.config.training_settings['scratch_save_path'].format(pde_name=pde_name).split('/')[-1])
        else:
            print("Scratch training completed but no best model state was saved. Using the last trained state.")
            self.save_model(model, os.path.dirname(self.config.training_settings['scratch_save_path']), 
                            self.config.training_settings['scratch_save_path'].format(pde_name=pde_name).split('/')[-1])

        print(f"--- Training From Scratch for {pde_name} Finished ---")
        return model

    def save_model(self, obj_to_save: Union[nn.Module, Dict[str, Any]], path: str, filename: str) -> None:
        """
        Saves the state dictionary of a trained PyTorch model or a dictionary.

        Args:
            obj_to_save (Union[nn.Module, Dict[str, Any]]): The object to save. If it's an nn.Module,
                                                            its state_dict is saved. Otherwise, the
                                                            dictionary itself is saved.
            path (str): The directory where the model should be saved.
            filename (str): The name of the file to save the model to.
        """
        os.makedirs(path, exist_ok=True)
        full_save_path = os.path.join(path, filename)
        
        if isinstance(obj_to_save, nn.Module):
            state_dict = obj_to_save.state_dict()
        elif isinstance(obj_to_save, dict):
            state_dict = obj_to_save
        else:
            raise TypeError(f"Expected obj_to_save to be nn.Module or dict, got {type(obj_to_save)}")

        torch.save(state_dict, full_save_path)
        print(f"Model state saved to: {full_save_path}")

