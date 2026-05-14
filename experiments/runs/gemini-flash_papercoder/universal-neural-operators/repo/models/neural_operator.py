import torch
import torch.nn as nn
from typing import List, Dict, Any

from models.adapters import LiftingAdapter, ProjectionAdapter
from models.base_operator import CoreOperator


class NeuralOperatorModel(nn.Module):
    """
    Encapsulates the entire neural operator architecture, comprising the
    problem-specific input adapter (Lifting), the generalizable core operator,
    and the problem-specific output adapter (Projection).

    This class orchestrates the forward pass and provides methods to manage
    the parameters during pre-training and fine-tuning by freezing/unfreezing
    the core operator.
    """

    def __init__(self,
                 lifting_adapter: LiftingAdapter,
                 core_operator: CoreOperator,
                 projection_adapter: ProjectionAdapter):
        """
        Initializes the NeuralOperatorModel.

        Args:
            lifting_adapter (LiftingAdapter): An instance of the LiftingAdapter module,
                                              responsible for mapping raw input functions
                                              to a higher-dimensional hidden representation.
            core_operator (CoreOperator): An instance of a CoreOperator module (e.g., FNO,
                                          MambaFNO), which performs the main operator learning
                                          on the hidden representation.
            projection_adapter (ProjectionAdapter): An instance of the ProjectionAdapter module,
                                                    responsible for mapping the core operator's
                                                    output back to the problem-specific output space.
        """
        super().__init__()
        if not isinstance(lifting_adapter, LiftingAdapter):
            raise TypeError(f"Expected lifting_adapter to be an instance of LiftingAdapter, got {type(lifting_adapter)}")
        if not isinstance(core_operator, CoreOperator):
            raise TypeError(f"Expected core_operator to be an instance of CoreOperator, got {type(core_operator)}")
        if not isinstance(projection_adapter, ProjectionAdapter):
            raise TypeError(f"Expected projection_adapter to be an instance of ProjectionAdapter, got {type(projection_adapter)}")

        self.lifting_adapter = lifting_adapter
        self.core_operator = core_operator
        self.projection_adapter = projection_adapter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the complete forward pass of the neural operator, following the
        Lifting-Operator-Projection sequence: P(F(L(x))).

        Args:
            x (torch.Tensor): The input tensor to the neural operator. Its shape
                              depends on the specific problem and the LiftingAdapter's
                              expected input format (e.g., (batch_size, H, W, input_dim)).

        Returns:
            torch.Tensor: The predicted output tensor from the neural operator. Its shape
                          depends on the problem and the ProjectionAdapter's output format
                          (e.g., (batch_size, H, W, output_dim)).
        """
        # 1. Apply the lifting adapter to the input
        lifted_features = self.lifting_adapter(x)

        # 2. Pass the lifted features through the core operator
        core_output = self.core_operator(lifted_features)

        # 3. Apply the projection adapter to get the final prediction
        output = self.projection_adapter(core_output)

        return output

    def get_trainable_parameters(self, only_adapters: bool = False) -> List[nn.Parameter]:
        """
        Returns a list of parameters that should be optimized by the trainer.

        This method supports two modes:
        - Training all parameters (for pre-training or training from scratch).
        - Training only the adapter parameters (for fine-tuning, with a frozen core operator).

        Args:
            only_adapters (bool): If True, only the parameters of the lifting and
                                  projection adapters are returned. If False, all
                                  trainable parameters of the entire model are returned.

        Returns:
            List[nn.Parameter]: A list of PyTorch parameters currently set to require gradients.
        """
        if not isinstance(only_adapters, bool):
            raise TypeError(f"Expected only_adapters to be a boolean, got {type(only_adapters)}")

        if only_adapters:
            # Collect parameters from lifting and projection adapters
            trainable_params: List[nn.Parameter] = []
            trainable_params.extend(self.lifting_adapter.parameters())
            trainable_params.extend(self.projection_adapter.parameters())
        else:
            # Collect all parameters from the entire model
            # This implicitly includes core_operator parameters if they are not frozen.
            trainable_params = list(self.parameters())

        # Filter to ensure only parameters with requires_grad=True are returned
        return [p for p in trainable_params if p.requires_grad]

    def freeze_core_operator(self) -> None:
        """
        Sets `requires_grad=False` for all parameters of the `core_operator`,
        effectively freezing it. This is typically used during the fine-tuning phase
        after pre-training, where only the adapters are updated.
        """
        for param in self.core_operator.parameters():
            param.requires_grad = False

    def unfreeze_core_operator(self) -> None:
        """
        Sets `requires_grad=True` for all parameters of the `core_operator`,
        making them trainable again. This might be useful if one intends to
        perform further full model training or a different fine-tuning strategy.
        """
        for param in self.core_operator.parameters():
            param.requires_grad = True

    def load_core_operator_state(self, state_dict: Dict[str, Any]) -> None:
        """
        Loads a saved state dictionary into the `core_operator`.

        This is used to load the pre-trained core operator weights before fine-tuning
        or for resuming training of the core operator.

        Args:
            state_dict (Dict[str, Any]): The state dictionary containing the parameters
                                         of the core operator.
        """
        if not isinstance(state_dict, dict):
            raise TypeError(f"Expected state_dict to be a dictionary, got {type(state_dict)}")
        self.core_operator.load_state_dict(state_dict, strict=True)

    def get_core_operator_state(self) -> Dict[str, Any]:
        """
        Returns the state dictionary of the `core_operator`.

        This is used to save the learned weights of the core operator after
        the pre-training phase, so they can be reused for fine-tuning.

        Returns:
            Dict[str, Any]: The state dictionary of the core operator.
        """
        return self.core_operator.state_dict()

