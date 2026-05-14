import torch
import torch.nn as nn
from typing import Any, Tuple

class ProbeModel(nn.Module):
    """
    Represents a linear classifier (probe) capable of predicting concept classes from agent
    activations or raw observations. It can be configured as a spatially-localized
    convolutional probe or a global linear probe.
    """

    def __init__(self, in_channels: int, num_classes: int, kernel_size: int, is_global: bool = False) -> None:
        """
        Initializes the ProbeModel.

        Args:
            in_channels (int):
                If `is_global` is `False` (convolutional probe): Number of input feature channels.
                If `is_global` is `True` (linear probe): Total number of flattened input features.
            num_classes (int): The number of possible concept classes the probe should predict.
            kernel_size (int):
                If `is_global` is `False`: The kernel size for the `Conv2d` layer.
                If `is_global` is `True`: This parameter is ignored.
            is_global (bool, optional): A flag indicating whether the probe is a global linear probe (`True`)
                                        or a spatially-localized convolutional probe (`False`).
                                        Defaults to `False`.
        """
        super().__init__()
        self.is_global: bool = is_global
        self.num_classes: int = num_classes

        if self.is_global:
            # Global linear probe: expects a flattened input feature vector
            self.probe_layer = nn.Linear(in_features=in_channels, out_features=num_classes)
        else:
            # Spatially-localized convolutional probe: expects 4D input (batch, channels, H, W)
            # Padding ensures output spatial dimensions match input for odd kernel sizes.
            padding: int = kernel_size // 2
            self.probe_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_classes,
                kernel_size=kernel_size,
                padding=padding,
                bias=True  # Linear classifiers typically include a bias
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass through the probe.

        Args:
            x (torch.Tensor): The input tensor to the probe.
                If `self.is_global` is `True`: Expected shape `(batch_size, in_channels)`.
                If `self.is_global` is `False`: Expected shape `(batch_size, in_channels, height, width)`.

        Returns:
            torch.Tensor: The output logits from the probe layer.
                For convolutional probes: `(batch_size, num_classes, height, width)`.
                For global linear probes: `(batch_size, num_classes)`.
        """
        return self.probe_layer(x)


if __name__ == '__main__':
    print("--- Testing ProbeModel ---")

    # --- Test Case 1: Spatially-localized convolutional probe (1x1) ---
    print("\nTesting 1x1 convolutional probe:")
    in_channels_conv: int = 32 # e.g., ConvLSTM channel size
    num_classes_conv: int = 5  # e.g., UP, DOWN, LEFT, RIGHT, NEVER
    kernel_size_1x1: int = 1
    probe_1x1 = ProbeModel(in_channels=in_channels_conv, num_classes=num_classes_conv,
                           kernel_size=kernel_size_1x1, is_global=False)
    print(f"Probe 1x1 layer: {probe_1x1.probe_layer}")

    # Dummy input (batch_size, channels, H, W)
    batch_size_conv: int = 4
    height_conv: int = 8
    width_conv: int = 8
    dummy_input_conv = torch.randn(batch_size_conv, in_channels_conv, height_conv, width_conv)
    print(f"Dummy input conv shape: {dummy_input_conv.shape}")

    output_1x1 = probe_1x1.forward(dummy_input_conv)
    print(f"Output 1x1 probe shape: {output_1x1.shape}")
    expected_shape_1x1 = (batch_size_conv, num_classes_conv, height_conv, width_conv)
    assert output_1x1.shape == expected_shape_1x1, \
        f"1x1 probe output shape mismatch! Expected {expected_shape_1x1}, got {output_1x1.shape}"
    print("1x1 convolutional probe test PASSED.")

    # --- Test Case 2: Spatially-localized convolutional probe (3x3) ---
    print("\nTesting 3x3 convolutional probe:")
    kernel_size_3x3: int = 3
    probe_3x3 = ProbeModel(in_channels=in_channels_conv, num_classes=num_classes_conv,
                           kernel_size=kernel_size_3x3, is_global=False)
    print(f"Probe 3x3 layer: {probe_3x3.probe_layer}")

    output_3x3 = probe_3x3.forward(dummy_input_conv)
    print(f"Output 3x3 probe shape: {output_3x3.shape}")
    expected_shape_3x3 = (batch_size_conv, num_classes_conv, height_conv, width_conv)
    assert output_3x3.shape == expected_shape_3x3, \
        f"3x3 probe output shape mismatch! Expected {expected_shape_3x3}, got {output_3x3.shape}"
    print("3x3 convolutional probe test PASSED.")

    # --- Test Case 3: Global linear probe ---
    print("\nTesting global linear probe:")
    # For a global probe, in_features would be flattened (channels * H * W)
    in_features_linear: int = in_channels_conv * height_conv * width_conv # 32 * 8 * 8 = 2048
    num_classes_linear: int = 10 # e.g., Action To Take in N Time Steps
    probe_global = ProbeModel(in_channels=in_features_linear, num_classes=num_classes_linear,
                              kernel_size=1, is_global=True) # kernel_size is ignored here
    print(f"Probe global layer: {probe_global.probe_layer}")

    # Dummy input (batch_size, in_features)
    batch_size_linear: int = 2
    dummy_input_linear = torch.randn(batch_size_linear, in_features_linear)
    print(f"Dummy input linear shape: {dummy_input_linear.shape}")

    output_global = probe_global.forward(dummy_input_linear)
    print(f"Output global probe shape: {output_global.shape}")
    expected_shape_global = (batch_size_linear, num_classes_linear)
    assert output_global.shape == expected_shape_global, \
        f"Global probe output shape mismatch! Expected {expected_shape_global}, got {output_global.shape}"
    print("Global linear probe test PASSED.")

    print("\n--- All ProbeModel tests completed successfully! ---")

