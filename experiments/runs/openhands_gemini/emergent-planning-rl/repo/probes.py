import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
from typing import List, Tuple, Dict

from config import AgentConfig, ProbeConfig

class LinearProbe(nn.Module):
    """
    Linear classifier probe to predict concept classes from agent activations.
    Supports 1x1, 3x3, 5x5, 7x7 probe sizes.
    """
    def __init__(self, input_dim: int, num_classes: int, probe_size: str, device: torch.device):
        super(LinearProbe, self).__init__()
        self.num_classes = num_classes
        self.probe_size_int = int(probe_size.replace('x', ''))
        self.device = device

        if self.probe_size_int == 1:
            self.linear = nn.Linear(input_dim, num_classes)
        else:
            # Convolutional layer for larger probe sizes
            # The input for convolutional probes will be patches of activations,
            # so the conv layer should effectively act as a linear projection on the flattened patch.
            # A 1x1 convolution over a patch of size probe_size_int x probe_size_int is effectively a linear layer
            # if we consider the input channels * probe_size_int * probe_size_int as the input_dim.
            # However, the paper describes probes taking input "cell state activations centered on (x,y)"
            # and that these "probes have 160 and 1440 parameters".
            # For 1x1 probe (input_dim=32, num_classes=5), 32*5 = 160 parameters. This matches.
            # For 3x3 probe (input_dim=32, num_classes=5), 32*3*3*5 = 1440 parameters. This also matches.
            # This confirms that for NxN probes, the convolution operates over the spatial patch,
            # and the `input_dim` to the Conv2d is the channel dimension.
            self.conv = nn.Conv2d(input_dim, num_classes, kernel_size=self.probe_size_int, padding=0)
            
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Activation tensor.
           For 1x1 probe: (batch_size * H * W, input_dim) representing activations at each (x,y)
           For NxN probe: (batch_size * H * W, input_dim, N, N) representing patches
        Returns:
            logits: (batch_size * H * W, num_classes)
        """
        if self.probe_size_int == 1:
            return self.linear(x)
        else:
            # Input `x` is already in (num_patches, channels, probe_size, probe_size) format.
            # The convolution should output (num_patches, num_classes, 1, 1).
            logits = self.conv(x) 
            return logits.squeeze(-1).squeeze(-1) # (num_patches, num_classes)

class ProbeTrainer:
    """
    Handles training and evaluation of linear probes.
    """
    def __init__(self, config: ProbeConfig, agent_config: AgentConfig, device: torch.device):
        self.config = config
        self.agent_config = agent_config
        self.device = device

    def train_probe(self, activations: torch.Tensor, targets: torch.Tensor, probe_type: str) -> LinearProbe:
        """
        Trains a single linear probe.
        
        Args:
            activations (torch.Tensor): Agent activations (e.g., cell states).
                                        Shape for 1x1: (num_samples, channels)
                                        Shape for NxN: (num_samples, channels, N, N)
            targets (torch.Tensor): Ground truth concept labels. Shape: (num_samples,)
            probe_type (str): Type of probe (e.g., "1x1", "3x3").
        
        Returns:
            LinearProbe: Trained probe model.
        """
        input_dim = activations.shape[1] # Channels dimension
        num_classes = int(targets.max().item() + 1) # Assumes labels are 0-indexed

        probe = LinearProbe(input_dim, num_classes, probe_type, self.device)
        
        if self.config.PROBE_OPTIMIZER == "AdamW":
            optimizer = torch.optim.AdamW(probe.parameters(), 
                                          lr=self.config.PROBE_LEARNING_RATE, 
                                          weight_decay=self.config.PROBE_WEIGHT_DECAY)
        else:
            raise ValueError(f"Optimizer {self.config.PROBE_OPTIMIZER} not supported.")

        criterion = nn.CrossEntropyLoss()

        # Create DataLoader
        dataset = torch.utils.data.TensorDataset(activations, targets)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.config.PROBE_BATCH_SIZE, shuffle=True)

        for epoch in range(self.config.PROBE_EPOCHS):
            probe.train()
            total_loss = 0
            for batch_activations, batch_targets in dataloader:
                batch_activations, batch_targets = batch_activations.to(self.device), batch_targets.to(self.device)
                
                optimizer.zero_grad()
                logits = probe(batch_activations)
                loss = criterion(logits, batch_targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            # print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}") # Optional: for debugging
        
        return probe

    def evaluate_probe(self, probe: LinearProbe, activations: torch.Tensor, targets: torch.Tensor, 
                       concept_classes: List[str]) -> Tuple[float, Dict[str, float]]:
        """
        Evaluates a trained probe and returns macro F1 score and class-specific metrics.
        """
        probe.eval()
        with torch.no_grad():
            activations, targets = activations.to(self.device), targets.to(self.device)
            logits = probe(activations)
            predictions = torch.argmax(logits, dim=1)

        # Convert to numpy for sklearn metrics
        predictions_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()

        # Macro F1
        macro_f1 = f1_score(targets_np, predictions_np, average='macro', zero_division=0)

        # Class-specific metrics
        class_metrics = {}
        # Ensure all possible classes are covered, not just those present in targets_np
        all_possible_classes = np.arange(len(concept_classes)) 

        for class_idx in all_possible_classes:
            class_name = concept_classes[class_idx]
            
            # Binary classification for each class
            binary_targets = (targets_np == class_idx).astype(int)
            binary_predictions = (predictions_np == class_idx).astype(int)

            f1 = f1_score(binary_targets, binary_predictions, zero_division=0)
            precision = precision_score(binary_targets, binary_predictions, zero_division=0)
            recall = recall_score(binary_targets, binary_predictions, zero_division=0)
            
            class_metrics[class_name] = {
                "F1": f1,
                "Precision": precision,
                "Recall": recall
            }
        
        return macro_f1, class_metrics

    def collect_activations_and_targets_drc(self, 
                                            agent_model: nn.Module, 
                                            dataloader: torch.utils.data.DataLoader, 
                                            concept_type: str, 
                                            probe_type: str,
                                            layer_idx: int,
                                            device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collects activations and corresponding ground truth targets for DRC agent.
        
        Args:
            agent_model: The DRC agent's model.
            dataloader: DataLoader providing (observation, previous_states, concept_CA_labels, concept_CB_labels)
            concept_type: "CA" for Agent Approach Direction, "CB" for Box Push Direction.
            probe_type: "1x1", "3x3", etc.
            layer_idx: Which ConvLSTM layer's cell state to extract.
            device: Device to run on.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Concatenated activations and targets.
        """
        all_activations = []
        all_targets = []
        probe_kernel_size = int(probe_type.replace('x',''))
        half_kernel = probe_kernel_size // 2

        agent_model.eval() # Set agent model to evaluation mode
        with torch.no_grad():
            for batch_obs, batch_prev_states_h, batch_prev_states_c, batch_CA_labels, batch_CB_labels in dataloader:
                batch_obs = batch_obs.to(device)
                
                # Reconstruct prev_states to be a list of tuples of tensors
                # batch_prev_states_h and batch_prev_states_c are (batch_size, D, C, H, W)
                actual_batch_prev_states = [
                    (batch_prev_states_h[:, d, :, :, :], batch_prev_states_c[:, d, :, :, :])
                    for d in range(self.agent_config.D_CONVLSTM_LAYERS)
                ]

                # Run agent model to get all intermediate cell states
                # policy_logits, value, new_states, all_cell_states_per_tick
                _, _, _, all_cell_states_per_tick = agent_model(batch_obs, actual_batch_prev_states)
                
                # We need the cell state (c_d_N) after the final internal tick N
                # all_cell_states_per_tick: list (N ticks) of list (D layers) of (h,c)
                final_tick_states = all_cell_states_per_tick[-1] # List of (h,c) for D layers
                # Paper (Section 2.3 and 4.1) states "cell state g_t^d", "cell state activations"
                # so we take c_state (index 1 in (h,c) tuple)
                target_cell_state_c = final_tick_states[layer_idx][1] # (batch_size, CHANNELS, H, W)

                # Use the cell state `c` as activations
                activations_to_probe = target_cell_state_c

                if concept_type == "CA":
                    batch_labels = batch_CA_labels # (batch_size, H, W)
                elif concept_type == "CB":
                    batch_labels = batch_CB_labels # (batch_size, H, W)
                else:
                    raise ValueError(f"Unknown concept type: {concept_type}")

                # Extract patches for NxN probes or flatten for 1x1
                batch_size, channels, H, W = activations_to_probe.shape
                
                if probe_kernel_size == 1:
                    # Flatten spatial dimensions: (B, C, H, W) -> (B*H*W, C)
                    flat_activations = activations_to_probe.permute(0, 2, 3, 1).reshape(-1, channels)
                    flat_targets = batch_labels.reshape(-1)
                    all_activations.append(flat_activations)
                    all_targets.append(flat_targets)
                else:
                    # For NxN probes, extract patches around each (x,y)
                    # We need to pad the activation map to extract patches for border elements.
                    # The paper states "our probes receive as input cell state activations centered on (x, y)"
                    # This implies padding should be handled consistently to allow patches centered on border elements.
                    
                    # Pad activations_to_probe
                    padding_val = half_kernel
                    padded_activations = F.pad(activations_to_probe, 
                                               (padding_val, padding_val, padding_val, padding_val), 
                                               'constant', 0)
                    
                    patches = []
                    spatial_targets = []
                    for b in range(batch_size):
                        for r in range(H):
                            for c in range(W):
                                # Extract patch centered at (r, c) from the original (non-padded) grid.
                                # In the padded tensor, the center of the patch for (r,c) should start at (r,c) in padded_activations
                                # and end at (r+probe_kernel_size, c+probe_kernel_size)
                                patch = padded_activations[b, :, r : r + probe_kernel_size, c : c + probe_kernel_size]
                                patches.append(patch)
                                spatial_targets.append(batch_labels[b, r, c])
                    
                    if patches:
                        all_activations.append(torch.stack(patches)) # (N_patches, C, K, K)
                        all_targets.append(torch.tensor(spatial_targets, device=device))
        
        if not all_activations:
            # Handle case where no data was collected (e.g., empty dataloader)
            dummy_activations_shape = (0, channels) if probe_kernel_size == 1 else (0, channels, probe_kernel_size, probe_kernel_size)
            return torch.empty(dummy_activations_shape, device=device), torch.empty(0, dtype=torch.long, device=device)

        return torch.cat(all_activations, dim=0), torch.cat(all_targets, dim=0)

    def collect_activations_and_targets_resnet(self,
                                               agent_model: nn.Module,
                                               dataloader: torch.utils.data.DataLoader,
                                               concept_type: str,
                                               probe_type: str,
                                               layer_idx: int,
                                               device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collects activations (hidden states after final ReLU of residual block) and corresponding ground truth targets for ResNet agent.
        
        Args:
            agent_model: The ResNet agent's model.
            dataloader: DataLoader providing (observation, concept_CA_labels, concept_CB_labels)
            concept_type: "CA" for Agent Approach Direction, "CB" for Box Push Direction.
            probe_type: "1x1", "3x3", etc.
            layer_idx: Which ResNet block's hidden state to extract (0-indexed).
            device: Device to run on.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Concatenated activations and targets.
        """
        all_activations = []
        all_targets = []
        probe_kernel_size = int(probe_type.replace('x',''))
        half_kernel = probe_kernel_size // 2

        agent_model.eval()
        with torch.no_grad():
            for batch_obs, batch_CA_labels, batch_CB_labels in dataloader:
                batch_obs = batch_obs.to(device)
                
                _, _, hidden_states_per_layer = agent_model(batch_obs)
                
                # ResNet hidden states are after final ReLU of residual block (batch, channels, H, W)
                # layer_idx maps to index in hidden_states_per_layer list
                activations_to_probe = hidden_states_per_layer[layer_idx]

                if concept_type == "CA":
                    batch_labels = batch_CA_labels
                elif concept_type == "CB":
                    batch_labels = batch_CB_labels
                else:
                    raise ValueError(f"Unknown concept type: {concept_type}")

                batch_size, channels, H, W = activations_to_probe.shape

                if probe_kernel_size == 1:
                    flat_activations = activations_to_probe.permute(0, 2, 3, 1).reshape(-1, channels)
                    flat_targets = batch_labels.reshape(-1)
                    all_activations.append(flat_activations)
                    all_targets.append(flat_targets)
                else:
                    padding_val = half_kernel
                    padded_activations = F.pad(activations_to_probe, 
                                               (padding_val, padding_val, padding_val, padding_val), 
                                               'constant', 0)
                    
                    patches = []
                    spatial_targets = []
                    for b in range(batch_size):
                        for r in range(H):
                            for c in range(W):
                                patch = padded_activations[b, :, r:r+probe_kernel_size, c:c+probe_kernel_size]
                                patches.append(patch)
                                spatial_targets.append(batch_labels[b, r, c])
                    
                    if patches:
                        all_activations.append(torch.stack(patches))
                        all_targets.append(torch.tensor(spatial_targets, device=device))

        if not all_activations:
            dummy_activations_shape = (0, channels) if probe_kernel_size == 1 else (0, channels, probe_kernel_size, probe_kernel_size)
            return torch.empty(dummy_activations_shape, device=device), torch.empty(0, dtype=torch.long, device=device)

        return torch.cat(all_activations, dim=0), torch.cat(all_targets, dim=0)

    def collect_activations_and_targets_baseline(self, 
                                                 dataloader: torch.utils.data.DataLoader, 
                                                 concept_type: str,
                                                 probe_type: str,
                                                 device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collects raw observation data as activations for baseline probes.
        
        Args:
            dataloader: DataLoader providing (observation, prev_states, concept_CA_labels, concept_CB_labels)
            concept_type: "CA" for Agent Approach Direction, "CB" for Box Push Direction.
            probe_type: "1x1", "3x3", etc.
            device: Device to run on.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Concatenated observations (as activations) and targets.
        """
        all_activations = []
        all_targets = []
        probe_kernel_size = int(probe_type.replace('x',''))
        half_kernel = probe_kernel_size // 2

        for batch_obs, _, _, batch_CA_labels, batch_CB_labels in dataloader:
            batch_obs = batch_obs.to(device) # (B, C, H, W)

            if concept_type == "CA":
                batch_labels = batch_CA_labels # (B, H, W)
            elif concept_type == "CB":
                batch_labels = batch_CB_labels # (B, H, W)
            else:
                raise ValueError(f"Unknown concept type: {concept_type}")

            batch_size, channels, H, W = batch_obs.shape

            if probe_kernel_size == 1:
                flat_activations = batch_obs.permute(0, 2, 3, 1).reshape(-1, channels)
                flat_targets = batch_labels.reshape(-1)
                all_activations.append(flat_activations)
                all_targets.append(flat_targets)
            else:
                padding_val = half_kernel
                padded_obs = F.pad(batch_obs, 
                                   (padding_val, padding_val, padding_val, padding_val), 
                                   'constant', 0)
                
                patches = []
                spatial_targets = []
                for b in range(batch_size):
                    for r in range(H):
                        for c in range(W):
                            patch = padded_obs[b, :, r:r+probe_kernel_size, c:c+probe_kernel_size]
                            patches.append(patch)
                            spatial_targets.append(batch_labels[b, r, c])
                
                if patches:
                    all_activations.append(torch.stack(patches))
                    all_targets.append(torch.tensor(spatial_targets, device=device))
        
        if not all_activations:
            dummy_activations_shape = (0, channels) if probe_kernel_size == 1 else (0, channels, probe_kernel_size, probe_kernel_size)
            return torch.empty(dummy_activations_shape, device=device), torch.empty(0, dtype=torch.long, device=device)

        return torch.cat(all_activations, dim=0), torch.cat(all_targets, dim=0)

    def get_concept_vectors(self, probe: LinearProbe) -> Dict[str, torch.Tensor]:
        """
        Extracts concept vectors (weights) from a trained probe.
        
        Returns:
            Dict[str, torch.Tensor]: A dictionary mapping class names to their concept vectors.
        """
        concept_vectors = {}
        if probe.probe_size_int == 1:
            weights = probe.linear.weight.data # (num_classes, input_dim)
            input_dim_effective = weights.shape[1]
        else:
            # For convolutional probe, reshape kernel (num_classes, input_dim, K, K)
            # to (num_classes, input_dim * K * K) for conceptual w_k^T g
            weights = probe.conv.weight.data.view(probe.num_classes, -1) 
            input_dim_effective = weights.shape[1]

        # Determine concept classes based on the number of classes the probe was trained for
        if probe.num_classes == len(self.config.CONCEPT_CA_CLASSES):
            concept_class_names = self.config.CONCEPT_CA_CLASSES
        elif probe.num_classes == len(self.config.CONCEPT_CB_CLASSES):
            concept_class_names = self.config.CONCEPT_CB_CLASSES
        else:
            raise ValueError("Number of probe classes does not match known concept class lists.")

        for i, class_name in enumerate(concept_class_names):
            concept_vectors[class_name] = weights[i].detach().cpu()
        return concept_vectors

