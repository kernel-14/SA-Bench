import abc
import torch
import torch.nn as nn

class CoreOperator(abc.ABC, nn.Module):
    """
    Abstract Base Class for all core neural operators.

    This class defines the common interface that all concrete neural operator
    implementations (e.g., FNO, MambaFNO, PerceiverFNO) must adhere to.
    It inherits from both `abc.ABC` to enforce abstract methods and `torch.nn.Module`
    to integrate with the PyTorch neural network framework.
    """

    def __init__(self):
        """
        Initializes the CoreOperator.

        While this base class does not have specific parameters, it calls the
        constructor of its parent `torch.nn.Module` to ensure proper PyTorch
        module initialization.
        """
        super().__init__()

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass for the core neural operator.

        This method must be implemented by all concrete subclasses. It takes
        an input tensor, processes it, and returns an output tensor.

        Args:
            x (torch.Tensor): The input tensor to the core operator.
                              This typically represents the hidden features
                              obtained after the Lifting layer.
                              The shape and content will depend on the specific
                              neural operator and the data being processed.

        Returns:
            torch.Tensor: The output tensor from the core operator,
                          representing the transformed hidden features.
                          The shape should be compatible with the subsequent
                          Projection layer.

        Raises:
            NotImplementedError: If a subclass fails to implement this method.
        """
        raise NotImplementedError("Subclasses must implement the 'forward' method.")

