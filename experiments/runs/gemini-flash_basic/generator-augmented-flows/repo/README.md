# Reproducing "Improving Consistency Models with Generator-Augmented Flows"

This repository contains an attempt to reproduce the core contributions of the paper "Improving Consistency Models with Generator-Augmented Flows". The paper introduces Generator-Augmented Flows (GAF) and a novel Generator-Augmented Coupling (GC) to address the discrepancy between consistency training and consistency distillation in consistency models. The main contribution is a joint learning strategy that combines independent coupling (IC) and GC to improve performance and accelerate convergence of consistency models.

## Implemented Components:

1.  **Consistency Model Architecture**: The paper mentions `F_theta` as a neural network within the consistency model `f_theta`. I will implement a placeholder for this neural network, assuming a standard U-Net-like architecture commonly used in diffusion models, as specific details are not provided in the main text. The `c_skip` and `c_out` functions, which define the final output of `f_theta`, will be implemented according to Equation (3) in the paper.

2.  **Noise Schedule and Sampling**: The paper describes a diffusion process with `sigma_t` and mentions sampling `t_i` uniformly at random. I will implement a basic noise schedule and the sampling procedure for `x_star` and `z`.

3.  **Loss Functions**:
    *   **Consistency Training (CT) Loss (`L_CT`)**: Implemented based on Equation (6).
    *   **Generator-Augmented Consistency (GC) Loss (`L_GC`)**: Implemented based on Equation (15).
    *   **Joint Learning Loss (`L_GC-mu`)**: Implemented based on Equation (23), which combines `L_CT` and `L_GC` with a mixing factor `mu`.

4.  **Training Loop**: The training procedure outlined in Algorithm 1 (Appendix B) will be implemented, incorporating the joint learning strategy. This includes sampling `x_star`, `z`, and `i`, calculating `x_t_i` and `x_t_i_plus_1`, computing `hat_x_t_i` using a stop-gradient `f_theta`, mixing IC and GC trajectories based on `mu`, and then calculating the combined loss.

## Assumptions and Missing Details:

*   **Neural Network Architecture (`F_theta`)**: The exact architecture for `F_theta` is not specified in the main paper. A common choice like a U-Net will be assumed, but a concrete implementation will be a placeholder due to the static nature of the benchmark.
*   **Hyperparameters**: Specific hyperparameters for training (e.g., learning rate schedule, optimizer details, batch size, number of timesteps `N`, `sigma_0`, `sigma_T`, `lambda` function, `D` distance function) are not fully detailed in the main text. Default or commonly used values for similar models will be assumed where necessary.
*   **Distance Function (`D`)**: The paper mentions `D(x, y)` as a distance function. For implementation, a simple squared Euclidean distance (`torch.nn.MSELoss`) will be assumed, as its a common choice and aligns with the `alpha = 2` case mentioned in the theoretical analysis.
*   **Noise Distribution (`p_z`)**: Assumed to be a standard Gaussian distribution.
*   **Data Distribution (`p_star`)**: Assumed to be the empirical data distribution from the training dataset.
*   **`sigma_d`**: This parameter used in `c_skip` is not explicitly defined in the main text. It will be assumed to be related to the variance of the data, or a hyperparameter to be tuned.
*   **`p(sigma_t_0), ..., p(sigma_t_N)` for multinomial sampling**: The distribution `p` for sampling timesteps `i` is not explicitly defined. A uniform distribution over timesteps will be assumed.
*   **`sg` (stop-gradient)**: This operation will be applied explicitly where noted in the algorithm.
*   **Static Code**: This reproduction is purely static. The provided code is not runnable and serves as a blueprint based on the papers description. It will not include dataset loading, full model instantiation beyond class definitions, or actual training execution.
