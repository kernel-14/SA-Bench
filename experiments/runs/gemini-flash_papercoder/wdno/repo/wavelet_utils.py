```python
import torch
from typing import Tuple, List, Union, Dict

# Import pytorch_wavelets DWT and IDWT
# Note: DWTForward and DWTInverse are part of pytorch_wavelets.dwt module
from pytorch_wavelets import DWTForward, DWTInverse

# Import ptwt DWT and IDWT for 1D, 2D, 3D
from ptwt import DWT1D, IDWT1D, DWT2D, IDWT2D, DWT3D, IDWT3D


class WaveletTransformManager:
    """
    Manages wavelet transformations for WDNO, including forward/inverse transforms
    and handling of conditioning data with varying dimensionalities.
    """

    def __init__(self, wavelet_type: str, mode: str, wavelet_data_dim: int, device: str = "cpu"):
        """
        Initializes the WaveletTransformManager.

        Args:
            wavelet_type: Type of wavelet to use (e.g., 'bior2.4', 'bior1.3').
            mode: Wavelet transform mode (e.g., 'periodization', 'zero').
            wavelet_data_dim: Dimensionality of the primary data to be transformed (2 for 2D, 3 for 3D).
                              Corresponds to the number of spatial/temporal dimensions in the data tensor.
                              e.g., 1D Burgers' data is (Batch, Channels, Time, Space), so wavelet_data_dim=2.
                              e.g., 2D Fluid data is (Batch, Channels, Time, H, W), so wavelet_data_dim=3.
            device: The device to perform computations on ('cuda' or 'cpu').
        """
        self.wavelet_type: str = wavelet_type
        self.mode: str = mode
        self.wavelet_data_dim: int = wavelet_data_dim
        self.device: torch.device = torch.device(device)

        # Initialize DWT/IDWT for the primary data dimension
        # The paper implies single-level decomposition (l0=L, and d_L coefficients).
        # For pytorch_wavelets DWT, J=1 corresponds to one level of decomposition.
        # For ptwt DWTs, not specifying `level` usually implies a single full decomposition.
        if self.wavelet_data_dim == 2:
            # pytorch_wavelets handles (Batch, Channels, Height, Width)
            self.dwt_main = DWTForward(J=1, mode=self.mode, wave=self.wavelet_type).to(self.device)
            self.idwt_main = DWTInverse(mode=self.mode, wave=self.wavelet_type).to(self.device)
            # For pytorch_wavelets' 2D DWT with J=1, it returns one list element in yh.
            # This element is a single tensor combining 3 detail bands (HL, LH, HH) for each channel.
            self.num_detail_components_main = 3
        elif self.wavelet_data_dim == 3:
            # ptwt.dwt_3d handles (Batch, Channels, Depth, Height, Width)
            self.dwt_main = DWT3D(wave=self.wavelet_type, mode=self.mode).to(self.device)
            self.idwt_main = IDWT3D(wave=self.wavelet_type, mode=self.mode).to(self.device)
            # For ptwt's 3D DWT, it returns 7 detail components (LLH, LHL, LHH, HLL, HLH, HHL, HHH)
            self.num_detail_components_main = 7
        else:
            raise ValueError(f"Unsupported wavelet_data_dim: {wavelet_data_dim}. Must be 2 or 3.")

        # Initialize DWT/IDWT for conditioning data (1D and 2D components)
        # The paper states initial/target conditions for 1D problems are 1D,
        # and for 2D problems, IC is 2D and percentage of smoke is 1D.
        # These separate DWT/IDWT instances are used for `transform_conditioning_data`.
        self.dwt_1d = DWT1D(wave=self.wavelet_type, mode=self.mode).to(self.device)
        self.idwt_1d = IDWT1D(wave=self.wavelet_type, mode=self.mode).to(self.device)
        # For ptwt's 1D DWT, it returns 2 detail components (H, L)
        self.num_detail_components_1d = 2

        self.dwt_2d = DWT2D(wave=self.wavelet_type, mode=self.mode).to(self.device)
        self.idwt_2d = IDWT2D(wave=self.wavelet_type, mode=self.mode).to(self.device)
        # For ptwt's 2D DWT, it returns 3 detail components (HH, HL, LH)
        self.num_detail_components_2d = 3

    def forward(self, data: torch.Tensor) -> Tuple[torch.Tensor, List[List[torch.Tensor]]]:
        """
        Performs a forward wavelet transform (decomposition) on the input data.
        The output format is (approx_coeffs, [[detail_coeffs_level_1]]).

        Args:
            data: The input tensor. Expected shape:
                  (B, C, H, W) for wavelet_data_dim=2
                  (B, C, D, H, W) for wavelet_data_dim=3

        Returns:
            A tuple containing:
            - approx_coeffs (torch.Tensor): The approximation coefficients.
            - detail_coeffs_list (List[List[torch.Tensor]]): A list containing one list of detail
              coefficient tensors for the single decomposition level.
        """
        data = data.to(self.device)

        if self.wavelet_data_dim == 2:
            # pytorch_wavelets returns yl (approx) and yh (list of detail lists, one for each level)
            # Since J=1, yh will be a list with one element, which is a tensor.
            # This tensor bundles 3 detail bands (HL, LH, HH) for each channel.
            approx_coeffs, detail_coeffs_combined_list = self.dwt_main(data)
            
            # Split the combined detail tensor into individual detail components
            # For pytorch_wavelets DWTForward with J=1, detail_coeffs_combined_list[0] has shape (B, C*3, H_det, W_det)
            batch_size, num_channels_x3, *spatial_dims = detail_coeffs_combined_list[0].shape
            num_channels = num_channels_x3 // self.num_detail_components_main # Should be 3
            
            # Split along the channel dimension
            detail_coeffs_split = torch.split(detail_coeffs_combined_list[0], num_channels, dim=1)
            return approx_coeffs, [list(detail_coeffs_split)]
        elif self.wavelet_data_dim == 3:
            # ptwt.dwt_3d returns (approx_coeffs, detail_coeffs_list) directly
            # detail_coeffs_list will contain 7 tensors (LLH, LHL, LHH, HLL, HLH, HHL, HHH)
            approx_coeffs, detail_coeffs_list = self.dwt_main(data)
            return approx_coeffs, [detail_coeffs_list]
        else:
            raise ValueError(f"Invalid wavelet_data_dim: {self.wavelet_data_dim}")

    def inverse(self, coeffs: Tuple[torch.Tensor, List[List[torch.Tensor]]]) -> torch.Tensor:
        """
        Performs an inverse wavelet transform (reconstruction) from coefficients.

        Args:
            coeffs: A tuple (approx_coeffs, detail_coeffs_list) where
                    - approx_coeffs (torch.Tensor): Approximation coefficients.
                    - detail_coeffs_list (List[List[torch.Tensor]]): List containing one list of detail
                      coefficient tensors for the single decomposition level.

        Returns:
            torch.Tensor: The reconstructed data.
        """
        approx_coeffs, detail_coeffs_list = coeffs
        
        # Ensure coefficients are on the correct device
        approx_coeffs = approx_coeffs.to(self.device)
        # Ensure detail_coeffs_list is not empty before attempting to move its contents
        if detail_coeffs_list and detail_coeffs_list[0]:
            detail_coeffs_list = [[d.to(self.device) for d in level_details] for level_details in detail_coeffs_list]

        if self.wavelet_data_dim == 2:
            # Recombine the split detail tensors into the single combined tensor
            # expected by pytorch_wavelets DWTInverse (e.g., (B, C*3, H, W)).
            combined_details = torch.cat(detail_coeffs_list[0], dim=1)
            return self.idwt_main((approx_coeffs, [combined_details]))
        elif self.wavelet_data_dim == 3:
            # ptwt.idwt_3d expects (approx_coeffs, list_of_detail_coeffs) directly
            return self.idwt_main(approx_coeffs, detail_coeffs_list[0])
        else:
            raise ValueError(f"Invalid wavelet_data_dim: {self.wavelet_data_dim}")

    def flatten_coeffs(self, coeffs: Tuple[torch.Tensor, List[List[torch.Tensor]]]) -> torch.Tensor:
        """
        Flattens wavelet coefficients into a single tensor by concatenating
        approximation and all detail coefficients along the channel dimension.

        Args:
            coeffs: A tuple (approx_coeffs, detail_coeffs_list).

        Returns:
            torch.Tensor: A single tensor containing all coefficients.
        """
        approx_coeffs, detail_coeffs_list = coeffs
        # detail_coeffs_list[0] contains the actual list of detail tensors for the first (and only) level
        all_coeffs = [approx_coeffs] + detail_coeffs_list[0]
        return torch.cat(all_coeffs, dim=1)

    def unflatten_coeffs(self, flat_coeffs: torch.Tensor,
                         original_shapes: Dict[str, Tuple[int, ...]]) -> Tuple[torch.Tensor, List[List[torch.Tensor]]]:
        """
        Unflattens a single tensor of coefficients back into the
        (approx_coeffs, detail_coeffs_list) tuple.

        Args:
            flat_coeffs: A single tensor containing concatenated coefficients.
            original_shapes: A dictionary containing the original shapes of 'approx_shape'
                             and 'detail_shape_components' (a list of shapes for each detail component).

        Returns:
            Tuple[torch.Tensor, List[List[torch.Tensor]]]: Unflattened coefficients.
        """
        approx_shape = original_shapes['approx_shape']
        detail_shape_components = original_shapes['detail_shape_components'] # List of shapes for each detail component

        current_idx = 0
        
        # Extract approx_coeffs. It always has the same number of channels as the original input.
        approx_channels = approx_shape[1]
        approx_coeffs_slice = flat_coeffs.narrow(1, current_idx, approx_channels)
        approx_coeffs = approx_coeffs_slice.view(approx_shape)
        current_idx += approx_channels

        detail_coeffs_list_for_level = []
        for det_shape in detail_shape_components:
            # Each detail component also has the same number of channels as the original input for ptwt.
            # For pytorch_wavelets, this implies `channels` is C, and detail component shape is (B, C, H, W).
            # The splitting in forward() and combining in inverse() for pytorch_wavelets handles this difference.
            det_channels = det_shape[1]
            det_coeffs_slice = flat_coeffs.narrow(1, current_idx, det_channels)
            detail_coeffs_list_for_level.append(det_coeffs_slice.view(det_shape))
            current_idx += det_channels
        
        return approx_coeffs, [detail_coeffs_list_for_level]

    def _get_dwt_components_for_dim(self, input_data_dim: int) -> Tuple[Union[DWT1D, DWT2D], int]:
        """Helper to get appropriate DWT and number of detail components for conditioning data."""
        if input_data_dim == 1:
            return self.dwt_1d, self.num_detail_components_1d
        elif input_data_dim == 2:
            return self.dwt_2d, self.num_detail_components_2d
        else:
            raise ValueError(f"Conditioning input_data_dim {input_data_dim} not supported. Must be 1 or 2.")

    def _get_idwt_components_for_dim(self, output_data_dim: int) -> Union[IDWT1D, IDWT2D]:
        """Helper to get appropriate IDWT for conditioning data."""
        if output_data_dim == 1:
            return self.idwt_1d
        elif output_data_dim == 2:
            return self.idwt_2d
        else:
            raise ValueError(f"Conditioning output_data_dim {output_data_dim} not supported. Must be 1 or 2.")

    def transform_conditioning_data(self, cond_data: torch.Tensor,
                                    target_spatial_dims: Tuple[int, ...],
                                    input_data_dim: int) -> Tuple[torch.Tensor, Dict[str, Tuple[int, ...]]]:
        """
        Transforms conditioning data (e.g., initial condition, target state) using wavelets,
        and repeats coefficients to match the target dimensionality of the main data.

        Args:
            cond_data (torch.Tensor): The conditioning data. Shape e.g., (B, C, X) for 1D,
                                      (B, C, H, W) for 2D.
            target_spatial_dims (Tuple[int, ...]): The spatial dimensions of the target main data's
                                                    wavelet coefficients (e.g., (T_approx, X_approx)
                                                    for 1D Burgers', or (T_approx, H_approx, W_approx)
                                                    for 2D Fluid). This defines the target dimensions for repetition.
            input_data_dim (int): The actual dimensionality of the `cond_data` (1 for 1D, 2 for 2D).

        Returns:
            Tuple[torch.Tensor, Dict[str, Tuple[int, ...]]]:
            - A single flattened tensor of the wavelet coefficients, repeated to match target dims.
            - A dictionary containing 'approx_shape_orig' and 'detail_shape_components_orig'
              which are the shapes of the coefficients before repetition.
              Also includes `num_channels` from the input for unflattening.
        """
        cond_data = cond_data.to(self.device)
        dwt_cond, num_detail_components = self._get_dwt_components_for_dim(input_data_dim)

        approx_coeffs_orig: torch.Tensor
        detail_coeffs_orig: List[torch.Tensor]

        # DWT1D/DWT2D from ptwt returns (approx, [detail_H, detail_L]) or (approx, [detail_HH, detail_HL, detail_LH])
        approx_coeffs_orig, detail_coeffs_list_orig = dwt_cond(cond_data)
        detail_coeffs_orig = detail_coeffs_list_orig[0] # Get the actual list of detail tensors for the first level

        # Store original shapes before repetition
        original_shapes_dict = {
            'approx_shape_orig': approx_coeffs_orig.shape,
            'detail_shape_components_orig': [d.shape for d in detail_coeffs_orig],
            'num_channels': cond_data.shape[1]
        }

        num_target_spatial_dims = len(target_spatial_dims)
        num_cond_spatial_dims = cond_data.ndim - 2 # Exclude Batch and Channel (B, C, S1, S2...)

        # Calculate number of missing leading spatial/temporal dimensions
        missing_leading_dims_count = num_target_spatial_dims - num_cond_spatial_dims

        # Function to repeat a coefficient tensor
        def _repeat_coeffs(coeffs_tensor: torch.Tensor, target_dims: Tuple[int, ...]) -> torch.Tensor:
            current_coeffs_spatial_shape = coeffs_tensor.shape[2:]
            
            # Unsqueeze for missing leading spatial/temporal dimensions
            # (e.g., if cond_data is 1D (X), and target is 2D (T, X), we need to add T dimension)
            for _ in range(missing_leading_dims_count):
                coeffs_tensor = coeffs_tensor.unsqueeze(2) # Insert new dim after batch and channel
            
            # Prepare repetition factors
            repeat_factors = list(coeffs_tensor.shape) # Copy current shape
            
            # Set repetition