import torch
from torch.utils.data import DataLoader
from typing import Callable, Dict, List, Optional, Union, Tuple, Any
from tqdm import tqdm

# Local imports
from config import Config
from wdno_models import BaseResolutionModel, SuperResolutionModel
from wavelet_utils import WaveletTransformManager
from pde_solvers import PdeSolver # Base class, specific solvers will be passed
from utils import (
    normalize_data, denormalize_data, get_device,
    calculate_metrics as utils_calculate_metrics,
    interpolate_to_finest_resolution as utils_interpolate_to_finest_resolution
)


class Evaluator:
    """
    Handles evaluation of Base-Resolution Model (BRM) and Super-Resolution Model (SRM)
    on various tasks including simulation, control, and zero-shot super-resolution.
    It calculates performance metrics and manages data transformation.
    """

    def __init__(self,
                 config: Config,
                 wavelet_manager: WaveletTransformManager,
                 pde_solver: PdeSolver):
        """
        Initializes the Evaluator.

        Args:
            config: The global configuration object.
            wavelet_manager: An instance of WaveletTransformManager for wavelet operations.
            pde_solver: An instance of a concrete PdeSolver for ground truth and objective calculations.
        """
        self.config: Config = config
        self.wavelet_manager: WaveletTransformManager = wavelet_manager
        self.pde_solver: PdeSolver = pde_solver
        self.device: torch.device = get_device(self.config.device)
        self.problem_config: Dict = self.config.problems[self.config.problem_name]

    def _run_inference_and_denormalize(self,
                                        model: Union[BaseResolutionModel, SuperResolutionModel],
                                        initial_noise: torch.Tensor,
                                        conditions_wavelets_flat: torch.Tensor,
                                        data_stats_for_output: Dict[str, Any],
                                        guidance_scale: float = 1.0,
                                        ddim_steps: Optional[int] = None,
                                        ddim_eta: Optional[float] = None,
                                        control_objective_fn: Optional[Callable] = None, # For control tasks
                                        condition_info_for_control: Optional[Dict[str, Any]] = None # For control tasks
                                       ) -> torch.Tensor:
        """
        Helper method to perform model inference, inverse wavelet transform, and denormalization.

        Args:
            model: The diffusion model (BRM or SRM).
            initial_noise: The initial Gaussian noise for the DDIM process.
            conditions_wavelets_flat: Flattened wavelet coefficients of all conditioning data.
            data_stats_for_output: Dictionary containing mean and std for the output variable
                                   to be denormalized (e.g., {'mean': ..., 'std': ...} for 'u' or 'f').
            guidance_scale: Weight for classifier-free guidance.
            ddim_steps: Number of DDIM sampling steps.
            ddim_eta: Eta parameter for DDIM.
            control_objective_fn: Objective function for control guidance.
            condition_info_for_control: Dictionary with additional info for control guidance.

        Returns:
            The denormalized, real-space prediction from the model.
        """
        # Ensure model is in eval mode
        model.eval()

        with torch.no_grad():
            # Perform sampling
            predicted_wavelets_flat = model.sample(
                initial_noise=initial_noise,
                conditions_wavelets=conditions_wavelets_flat,
                guidance_scale=guidance_scale,
                ddim_steps=ddim_steps,
                ddim_eta=ddim_eta,
                control_objective_fn=control_objective_fn,
                condition_info_for_control=condition_info_for_control
            )

            # Unflatten wavelet coefficients
            # We need the original shapes of the wavelet coefficients (approx and detail components)
            # This meta-information should be stored during data preprocessing.
            # Assuming predicted_wavelets_flat corresponds to `x_0_wavelets_flat` and its metadata
            # is passed in `data_stats_for_output` (or directly in `conditions_wavelets_meta`).
            # For now, let's assume `data_stats_for_output` contains `output_wavelet_metadata`
            # which has 'approx_shape' and 'detail_shape_components' for the output variable.
            if 'output_wavelet_metadata' not in data_stats_for_output:
                raise ValueError("data_stats_for_output must contain 'output_wavelet_metadata' for unflattening.")
            
            output_wavelet_metadata = data_stats_for_output['output_wavelet_metadata']
            predicted_wavelets = self.wavelet_manager.unflatten_coeffs(predicted_wavelets_flat, output_wavelet_metadata)

            # Inverse wavelet transform to real space
            predicted_real_space = self.wavelet_manager.inverse(predicted_wavelets)

            # Denormalize
            denormalized_data = denormalize_data(predicted_real_space,
                                                 mean=data_stats_for_output['mean'],
                                                 std=data_stats_for_output['std'])
            return denormalized_data

    def calculate_metrics(self,
                          predictions: torch.Tensor,
                          ground_truth: torch.Tensor,
                          exclude_initial_condition: bool = False,
                          metrics: List[str] = ['mse', 'l2_relative', 'mae', 'l_inf']) -> Dict[str, float]:
        """
        Computes evaluation metrics, optionally excluding the initial condition.

        Args:
            predictions: The model's predictions (real-space, denormalized).
            ground_truth: The ground truth data (real-space, denormalized).
            exclude_initial_condition: If True, the first time step (index 0) is excluded
                                       from the metrics calculation.
            metrics: List of metrics to compute.

        Returns:
            A dictionary of computed metrics.
        """
        if exclude_initial_condition:
            # Assuming time dimension is the second dimension (index 1) after batch.
            # Shape: (Batch, Time, Spatial_Dims...) or (Batch, Channels, Time, Spatial_Dims...)
            # We want to exclude the first time step.
            if predictions.ndim > 2: # Check if there is a time dimension
                predictions = predictions.index_select(1, torch.arange(1, predictions.shape[1], device=predictions.device))
                ground_truth = ground_truth.index_select(1, torch.arange(1, ground_truth.shape[1], device=ground_truth.device))
            else: # If predictions are 1D (B, Time) or (B, Channels), this logic needs refinement based on data structure
                  # For current WDNO problems (u,f are (T,X) or (T,H,W)), time is often the first non-channel dim.
                  # Let's adjust based on the typical `u` (B, C, T, X) or (B, C, T, H, W)
                if predictions.ndim > 2: # Likely (B, C, T, X) or (B, C, T, H, W)
                    predictions = predictions[:, :, 1:, ...]
                    ground_truth = ground_truth[:, :, 1:, ...]
                elif predictions.ndim == 2: # Likely (B, T, X) or (B, T)
                     predictions = predictions[:, 1:, ...]
                     ground_truth = ground_truth[:, 1:, ...]
                else:
                    raise ValueError(f"Cannot exclude initial condition for data with shape {predictions.shape}. Expected at least 2 dimensions for time.")


        return utils_calculate_metrics(predictions, ground_truth, metrics)

    def evaluate_simulation(self,
                            model: BaseResolutionModel,
                            dataloader: DataLoader,
                            data_stats: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluates the BaseResolutionModel on a simulation task.

        Args:
            model: The BaseResolutionModel instance.
            dataloader: DataLoader for simulation test data.
            data_stats: Dictionary of normalization statistics for all relevant variables.

        Returns:
            A dictionary of computed metrics (e.g., MSE, L2 relative error).
        """
        all_predicted_u: List[torch.Tensor] = []
        all_ground_truth_u: List[torch.Tensor] = []
        
        # Determine DDIM steps based on problem config or default
        ddim_steps_sim = self.problem_config.get('inference', {}).get('ddim_sampling_iterations', self.config.ddim_steps)

        for batch_data in tqdm(dataloader, desc="Evaluating Simulation"):
            # Ensure tensors are on the correct device
            for key in ['x_0_wavelets_flat', 'x_0_wavelets_metadata', 'u0_wavelets_flat', 'u0_wavelets_metadata', 'f_wavelets_flat', 'f_wavelets_metadata', 'x_0_real_denormalized_gt']:
                if key in batch_data and isinstance(batch_data[key], torch.Tensor):
                    batch_data[key] = batch_data[key].to(self.device)
            
            # For simulation, conditions are u0 and f. Combine their flattened wavelets.
            # Assuming data_module ensures compatible flattening and concatenation.
            conditions_wavelets_flat = torch.cat([batch_data['u0_wavelets_flat'], batch_data['f_wavelets_flat']], dim=1)

            # Generate initial noise for the target output (x_0_wavelets_flat is the ground truth)
            initial_noise = torch.randn_like(batch_data['x_0_wavelets_flat']).to(self.device)

            # Define data_stats for the output variable (u in this case)
            output_data_stats = {
                'mean': data_stats['u']['mean'],
                'std': data_stats['u']['std'],
                'output_wavelet_metadata': batch_data['x_0_wavelets_metadata'] # Metadata for the full u trajectory
            }
            
            # Run inference
            predicted_u_denormalized = self._run_inference_and_denormalize(
                model=model,
                initial_noise=initial_noise,
                conditions_wavelets_flat=conditions_wavelets_flat,
                data_stats_for_output=output_data_stats,
                guidance_scale=1.0,
                ddim_steps=ddim_steps_sim,
                ddim_eta=self.config.ddim_eta
            )
            all_predicted_u.append(predicted_u_denormalized.cpu())
            all_ground_truth_u.append(batch_data['x_0_real_denormalized_gt'].cpu())

        # Aggregate predictions and ground truths
        final_predictions = torch.cat(all_predicted_u, dim=0)
        final_ground_truth = torch.cat(all_ground_truth_u, dim=0)

        # Calculate metrics, excluding the initial condition
        metrics_dict = self.calculate_metrics(
            predictions=final_predictions,
            ground_truth=final_ground_truth,
            exclude_initial_condition=True # As per paper for simulation tasks
        )
        return metrics_dict

    def evaluate_control(self,
                         model: BaseResolutionModel,
                         dataloader: DataLoader,
                         data_stats: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluates the BaseResolutionModel on a control task.

        Args:
            model: The BaseResolutionModel instance.
            dataloader: DataLoader for control test data.
            data_stats: Dictionary of normalization statistics for all relevant variables.

        Returns:
            A dictionary containing mean and std of the control objective.
        """
        if self.pde_solver is None:
            raise ValueError("PdeSolver must be initialized for control task evaluation.")
        if not self.problem_config['control_task']['enabled']:
            raise ValueError("Control task is not enabled in the configuration for this problem.")

        all_objective_values: List[float] = []
        guidance_lambda = self.problem_config['control_task'].get('guidance_lambda', self.config.guidance_lambda)
        
        # Determine DDIM steps based on problem config or default
        ddim_steps_control = self.problem_config.get('inference', {}).get('ddim_sampling_iterations', self.config.ddim_steps)


        for batch_data in tqdm(dataloader, desc="Evaluating Control"):
            # Ensure tensors are on the correct device
            for key in ['f_0_wavelets_flat', 'f_0_wavelets_metadata', 'u0_wavelets_flat', 'u0_wavelets_metadata', 'u_target_wavelets_flat', 'u_target_wavelets_metadata', 'u0_real_denormalized_gt', 'u_target_real_denormalized_gt']:
                 if key in batch_data and isinstance(batch_data[key], torch.Tensor):
                    batch_data[key] = batch_data[key].to(self.device)

            # For control, target output is f. Conditions are u0 and u_target.
            conditions_wavelets_flat = torch.cat([batch_data['u0_wavelets_flat'], batch_data['u_target_wavelets_flat']], dim=1)

            # Generate initial noise for the target output (f_0_wavelets_flat)
            initial_noise = torch.randn_like(batch_data['f_0_wavelets_flat']).to(self.device)

            # Prepare data_stats for the output variable (f in this case)
            output_data_stats = {
                'mean': data_stats['f']['mean'],
                'std': data_stats['f']['std'],
                'output_wavelet_metadata': batch_data['f_0_wavelets_metadata'] # Metadata for the full f trajectory
            }

            # Prepare condition_info for control guidance
            condition_info_for_control = {
                'u0_wavelets_flat': batch_data['u0_wavelets_flat'],
                'u0_wavelets_metadata': batch_data['u0_wavelets_metadata'],
                'u_target_wavelets_flat': batch_data['u_target_wavelets_flat'],
                'u_target_wavelets_metadata': batch_data['u_target_wavelets_metadata'],
                'u0_real_denormalized': batch_data['u0_real_denormalized_gt'],
                'u_target_real_denormalized': batch_data['u_target_real_denormalized_gt'],
                'u_stats': data_stats['u']
            }

            # Run inference for control force
            predicted_f_denormalized = self._run_inference_and_denormalize(
                model=model,
                initial_noise=initial_noise,
                conditions_wavelets_flat=conditions_wavelets_flat,
                data_stats_for_output=output_data_stats,
                guidance_scale=1.0, # Classifier-free guidance is usually separate from control guidance
                ddim_steps=ddim_steps_control,
                ddim_eta=self.config.ddim_eta,
                control_objective_fn=self.pde_solver.calculate_control_objective,
                condition_info_for_control=condition_info_for_control
            )
            
            # The PDE solver expects single sample inputs
            u0_real_denormalized_single = batch_data['u0_real_denormalized_gt'][0]
            predicted_f_denormalized_single = predicted_f_denormalized[0]
            u_target_real_denormalized_single = batch_data['u_target_real_denormalized_gt'][0]

            # Simulate the PDE with the predicted force to get the resulting trajectory
            simulated_u_real = self.pde_solver.solve(u0_real_denormalized_single, predicted_f_denormalized_single)
            
            # Calculate the control objective based on the simulated trajectory
            objective_value = self.pde_solver.calculate_control_objective(
                simulated_u_real.unsqueeze(0), # Add batch dim for consistency if needed by objective func
                u_target_real_denormalized_single.unsqueeze(0), # Add batch dim
                predicted_f_denormalized_single.unsqueeze(0) # Add batch dim
            )
            all_objective_values.append(objective_value.item())

        mean_objective = float(torch.mean(torch.tensor(all_objective_values)))
        std_objective = float(torch.std(torch.tensor(all_objective_values)))

        return {'mean_objective': mean_objective, 'std_objective': std_objective}

    def evaluate_super_resolution(self,
                                  brm: BaseResolutionModel,
                                  srm: SuperResolutionModel,
                                  dataloader: DataLoader,
                                  data_stats: Dict[str, Any],
                                  fno_baseline_data: Optional[Dict] = None, # Path to pre-computed FNO results
                                  wno_baseline_data: Optional[Dict] = None  # Path to pre-computed WNO results
                                 ) -> Dict[str, Any]:
        """
        Evaluates the Super-Resolution Model (SRM) for zero-shot super-resolution.

        Args:
            brm: The BaseResolutionModel instance (for 0x SR initial prediction).
            srm: The SuperResolutionModel instance.
            dataloader: DataLoader providing base-resolution inputs and finest-resolution ground truth.
            data_stats: Dictionary of normalization statistics for all relevant variables.
            fno_baseline_data: Pre-computed FNO baseline predictions (if available).
            wno_baseline_data: Pre-computed WNO baseline predictions (if available).

        Returns:
            A dictionary of metrics for each SR level (0x, 1x, etc.) and baselines.
        """
        brm.eval()
        srm.eval()

        # Get super-resolution config from problem config
        sr_config = self.problem_config['super_resolution_task']
        sr_target_resolutions = sr_config['sr_target_resolutions']
        multi_res_levels = self.config.multi_res_levels # Number of super-resolution steps
        train_resolution = sr_config['train_resolution'] # The base resolution for 0x SR

        # Store predictions and GT for metric calculation
        sr_results: Dict[str, Dict[str, List[torch.Tensor]]] = {
            'wdno_0x_sr': {'predictions': [], 'ground_truth': []}
        }
        for i in range(multi_res_levels):
            sr_results[f'wdno_{i+1}x_sr'] = {'predictions': [], 'ground_truth': []}
        
        # Prepare for baselines if provided
        baselines_to_eval = {}
        if fno_baseline_data:
            baselines_to_eval['fno'] = {'data': fno_baseline_data, 'predictions': [], 'ground_truth': []}
        if wno_baseline_data:
            baselines_to_eval['wno'] = {'data': wno_baseline_data, 'predictions': [], 'ground_truth': []}

        # Determine DDIM steps based on problem config or default
        ddim_steps_sim = self.problem_config.get('inference', {}).get('ddim_sampling_iterations', self.config.ddim_steps)


        for batch_data in tqdm(dataloader, desc="Evaluating Super-Resolution"):
            # Move batch data to device
            for key in ['u0_wavelets_base_flat', 'u0_wavelets_metadata_base', 'f_wavelets_base_flat', 'f_wavelets_metadata_base', 'a_h_finest_res_wavelets_flat', 'a_h_finest_res_wavelets_metadata', 'u_gt_finest_res_wavelets_flat', 'u_gt_finest_res_wavelets_metadata', 'u_gt_finest_res_real_denormalized']:
                if key in batch_data and isinstance(batch_data[key], torch.Tensor):
                    batch_data[key] = batch_data[key].to(self.device)

            u_gt_finest_real_denormalized = batch_data['u_gt_finest_res_real_denormalized']

            # --- 0x Super-Resolution (BRM Prediction) ---
            # Conditions for BRM (u0_base, f_base)
            conditions_brm_flat = torch.cat([batch_data['u0_wavelets_base_flat'], batch_data['f_wavelets_base_flat']], dim=1)
            # Noise for BRM's output (base resolution u)
            initial_noise_brm = torch.randn_like(batch_data['u_gt_finest_res_wavelets_flat']).to(self.device) # Shape needs to match the BRM's target output

            # Determine appropriate wavelet metadata for BRM's output (u at base resolution)
            # This requires some knowledge of how `u_gt_finest_res_wavelets_flat` relates to `u_base_res_wavelets_flat`.
            # For simplicity, if the dataloader provides metadata for base-res `u`, use that.
            # Otherwise, we might need to apply WM.forward to a dummy tensor of `train_resolution` shape.
            # Let's assume the metadata for `u_gt_finest_res` can be used to derive the metadata for base-res `u`
            # by scaling the spatial dimensions of the wavelet coefficients.
            
            # --- IMPORTANT ASSUMPTION ---
            # For 0x SR, the `initial_noise_brm` and `output_wavelet_metadata` should correspond to the
            # wavelet coefficients of a *base-resolution* u. However, the `_run_inference_and_denormalize`
            # is called once, and it denormalizes. So `predicted_u_0x_denormalized` is expected to be
            # base resolution real space `u`.
            #
            # The current `x_0_wavelets_flat` in batch_data for simulation is `u_gt_finest_res_wavelets_flat`.
            # But 0x SR result from BRM should be at `train_resolution`.
            # To fix this, the dataloader should also provide `u_gt_base_res_wavelets_flat` and its metadata.
            #
            # Re-read paper: "using the Base-Resolution Model, we first generate the wavelet coefficients of the base low resolution."
            # This implies the BRM's sample should directly output base-resolution coefficients.
            # So `initial_noise_brm` should be `randn_like` the base-resolution `u`'s wavelet coeffs.
            # The dataloader needs to provide `u_gt_base_res_wavelets_flat` and its metadata.
            # Let's assume `batch_data['u_gt_base_res_wavelets_flat']` and `batch_data['u_gt_base_res_wavelets_metadata']` exist.

            initial_noise_brm_base_res = torch.randn_like(batch_data['u_gt_base_res_wavelets_flat']).to(self.device)
            output_data_stats_brm = {
                'mean': data_stats['u']['mean'],
                'std': data_stats['u']['std'],
                'output_wavelet_metadata': batch_data['u_gt_base_res_wavelets_metadata']
            }

            predicted_u_0x_denormalized = self._run_inference_and_denormalize(
                model=brm,
                initial_noise=initial_noise_brm_base_res,
                conditions_wavelets_flat=conditions_brm_flat,
                data_stats_for_output=output_data_stats_brm,
                guidance_scale=1.0,
                ddim_steps=ddim_steps_sim,
                ddim_eta=self.config.ddim_eta
            )
            sr_results['wdno_0x_sr']['predictions'].append(
                utils_interpolate_to_finest_resolution(
                    predicted_u_0x_denormalized.cpu(),
                    u_gt_finest_real_denormalized.shape[2:], # Target spatial dims (T,H,W or T,X)
                    sr_config['evaluation_interpolation_method']
                )
            )
            sr_results['wdno_0x_sr']['ground_truth'].append(u_gt_finest_real_denormalized.cpu())

            # --- Iterative Super-Resolution (SRM Prediction) ---
            # The output of the previous step becomes the low-resolution input for the next.
            # The paper states: "Using the Base-Resolution Model, we first generate the wavelet coefficients
            # of the base low resolution. Subsequently, we utilize the Super-Resolution Model to generate the
            # data based on both the wavelet coefficients of lower-resolution results with size N × M
            # and the wavelet coefficients of ah at the post-super-resolution resolution 2N × 2M."

            current_low_res_u_wavelets_flat = self.wavelet_manager.flatten_coeffs(
                self.wavelet_manager.forward(predicted_u_0x_denormalized.to(self.device),
                                             data_stats['u']['mean'].to(self.device), # Pass mean/std for potentially correct normalization
                                             data_stats['u']['std'].to(self.device))
            )
            # The metadata for `current_low_res_u_wavelets_flat` is now its own.
            # This implies the `flatten_coeffs` should return metadata, or we pass its original shape
            # to unflatten later. For simplicity, `WaveletTransformManager.forward` returns coeffs as a tuple
            # and `flatten_coeffs` returns a tensor. The `_run_inference_and_denormalize`
            # requires `output_wavelet_metadata`. Let's augment `data_stats` with this metadata.
            
            current_low_res_metadata = self.wavelet_manager.get_wavelet_coeff_metadata(
                predicted_u_0x_denormalized.to(self.device),
                data_stats['u']['mean'].to(self.device),
                data_stats['u']['std'].to(self.device)
            )
            # Ensure the current low-res u is normalized before being flattened
            current_low_res_u_normalized = normalize_data(predicted_u_0x_denormalized.to(self.device),
                                                          data_stats['u']['mean'].to(self.device),
                                                          data_stats['u']['std'].to(self.device))
            current_low_res_u_wavelets_flat = self.wavelet_manager.flatten_coeffs(
                self.wavelet_manager.forward(current_low_res_u_normalized)
            )

            for i_level in range(multi_res_levels):
                level_label = f'wdno_{i_level+1}x_sr'
                target_spatial_dims_sr = sr_target_resolutions[i_level][1:] # Remove time dim for spatial only
                target_real_space_shape = sr_target_resolutions[i_level]
                
                # High-res condition a_h
                # The `a_h_finest_res_wavelets_flat` should be transformed to the *current target SR resolution* for `a_h`.
                # This requires a function to scale conditioning wavelets to arbitrary target resolution.
                # Assuming `wavelet_manager.transform_conditioning_data_to_target_resolution`
                # (which does not exist yet and is complex to implement generically without more details).
                # For now, let's assume `a_h_finest_res_wavelets_flat` is the one passed to SRM,
                # and SRM's internal attention mechanisms will handle resolution difference of `a_h`.
                # However, the paper explicitly says: "wavelet coefficients of ah at the post-super-resolution resolution 2N × 2M".
                # This implies `a_h` must match the target output resolution.

                # --- NEW ASSUMPTION FOR A_H ---
                # Dataloader should provide a_h wavelets for each target SR resolution,
                # or `a_h_finest_res_wavelets_flat` is passed, and SRM takes care of it.
                # Given the design, `conditions_wavelets_flat` for SRM combines `W_l` and `W_a_h`.
                # So `W_a_h` needs to be provided by the dataloader at appropriate resolution for each level.
                # Let's assume `batch_data[f'a_h_wavelets_level_{i_level}_flat']` exists.
                if f'a_h_wavelets_level_{i_level}_flat' not in batch_data:
                    raise KeyError(f"Missing a_h_wavelets for SR level {i_level}. Dataloader should provide it.")
                high_res_a_h_wavelets_flat = batch_data[f'a_h_wavelets_level_{i_level}_flat']
                
                conditions_srm_flat = torch.cat([current_low_res_u_wavelets_flat, high_res_a_h_wavelets_flat], dim=1)

                # Determine output wavelet metadata for the current target resolution
                # This is the metadata for `u` at `target_real_space_shape`.
                output_wavelet_metadata_srm = self.wavelet_manager.get_wavelet_coeff_metadata(
                    torch.zeros(1, data_stats['u']['mean'].shape[0], *target_real_space_shape, device=self.device),
                    data_stats['u']['mean'].to(self.device),
                    data_stats['u']['std'].to(self.device)
                )

                # Noise for SRM's output (high-res u)
                initial_noise_srm = torch.randn(output_wavelet_metadata_srm['approx_shape'][0],
                                                output_wavelet_metadata_srm['total_channels'],
                                                *output_wavelet_metadata_srm['approx_shape'][2:],
                                                device=self.device) # B, C_total, D, H, W (or H, W)
                
                predicted_u_sr_denormalized = self._run_inference_and_denormalize(
                    model=srm,
                    initial_noise=initial_noise_srm,
                    conditions_wavelets_flat=conditions_srm_flat,
                    data_stats_for_output={
                        'mean': data_stats['u']['mean'],
                        'std': data_stats['u']['std'],
                        'output_wavelet_metadata': output_wavelet_metadata_srm
                    },
                    guidance_scale=1.0,
                    ddim_steps=ddim_steps_sim,
                    ddim_eta=self.config.ddim_eta
                )
                
                sr_results[level_label]['predictions'].append(
                    utils_interpolate_to_finest_resolution(
                        predicted_u_sr_denormalized.cpu(),
                        u_gt_finest_real_denormalized.shape[2:],
                        sr_config['evaluation_interpolation_method']
                    )
                )
                sr_results[level_label]['ground_truth'].append(u_gt_finest_real_denormalized.cpu())

                # For the next iteration, the current SRM output becomes the low-res input
                current_low_res_u_normalized = normalize_data(predicted_u_sr_denormalized.to(self.device),
                                                              data_stats['u']['mean'].to(self.device),
                                                              data_stats['u']['std'].to(self.device))
                current_low_res_u_wavelets_flat = self.wavelet_manager.flatten_coeffs(
                    self.wavelet_manager.forward(current_low_res_u_normalized)
                )

            # --- Baselines (FNO, WNO) ---
            # Assuming baseline_data are dictionaries mapping SR level (e.g., '0x_sr', '1x_sr')
            # to lists of real-space denormalized tensors for each batch.
            for bl_name, bl_info in baselines_to_eval.items():
                for i_level in range(multi_res_levels + 1): # 0x, 1x, ..., Nx
                    if i_level == 0:
                        bl_level_key = '0x_sr'
                        original_bl_res = train_resolution
                    else:
                        bl_level_key = f'{i_level}x_sr'
                        original_bl_res = sr_target_resolutions[i_level-1] # Resolution of the baseline output itself
                    
                    if bl_level_key in bl_info['data']:
                        bl_pred_at_level = bl_info['data'][bl_level_key][batch_data['idx'].item()] # Assuming idx is in batch
                        bl_pred_tensor = bl_pred_at_level.unsqueeze(0).to(self.device) # Add batch dim

                        interpolated_bl_pred = utils_interpolate_to_finest_resolution(
                            bl_pred_tensor.cpu(),
                            u_gt_finest_real_denormalized.shape[2:],
                            sr_config['evaluation_interpolation_method']
                        )
                        sr_results.setdefault(f'{bl_name}_{bl_level_key}', {'predictions': [], 'ground_truth': []})
                        sr_results[f'{bl_name}_{bl_level_key}']['predictions'].append(interpolated_bl_pred)
                        sr_results[f'{bl_name}_{bl_level_key}']['ground_truth'].append(u_gt_finest_real_denormalized.cpu())


        # Calculate final metrics for all collected predictions
        final_metrics: Dict[str, Dict[str, float]] = {}
        for key, data_list in sr_results.items():
            if data_list['predictions']: # Only calculate if there are predictions
                final_predictions = torch.cat(data_list['predictions'], dim=0)
                final_ground_truth = torch.cat(data_list['ground_truth'], dim=0)
                # For SR, initial condition is usually not excluded, as the whole trajectory is super-resolved
                final_metrics[key] = self.calculate_metrics(
                    predictions=final_predictions,
                    ground_truth=final_ground_truth,
                    exclude_initial_condition=False
                )
        
        # Calculate metrics for baselines
        for bl_name, bl_info in baselines_to_eval.items():
             for i_level in range(multi_res_levels + 1):
                bl_level_key = '0x_sr' if i_level == 0 else f'{i_level}x_sr'
                full_key = f'{bl_name}_{bl_level_key}'
                if full_key in sr_results:
                    data_list = sr_results[full_key]
                    if data_list['predictions']:
                        final_predictions = torch.cat(data_list['predictions'], dim=0)
                        final_ground_truth = torch.cat(data_list['ground_truth'], dim=0)
                        final_metrics[full_key] = self.calculate_metrics(
                            predictions=final_predictions,
                            ground_truth=final_ground_truth,
                            exclude_initial_condition=False
                        )

        return final_metrics

