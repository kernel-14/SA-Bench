
# Instance-dependent Convergence Theory for Diffusion Models

This repository contains a faithful reproduction of the numerical experiments described in the paper "Instance-dependent Convergence Theory for Diffusion Models". The codebase aims to replicate the core components, specifically the sampler, score functions, and evaluation metrics used to demonstrate the convergence rates, as presented in Appendix A (Figure 2).

## Project Structure

The project is organized as follows:

- `config.py`: Contains all configurable hyperparameters for the simulation, including data dimension, total iterations, number of rounds, and constants for the randomized schedule.
- `data.py`: Defines the target Gaussian distribution used in the numerical experiments, including its sampling and log-probability calculation.
- `models.py`: Implements the exact score function for the Gaussian target distribution, as derived in Appendix C.1 of the paper.
- `sampler.py`: Implements the core diffusion sampling algorithm from Section 2.2, which includes the randomized schedule, iterative updates, and noise injection steps.
- `metrics.py`: Provides functions to calculate evaluation metrics such as KL Divergence and Total Variation distance, essential for assessing the sampler's performance.
- `main.py`: The main script to orchestrate the numerical experiments. It runs the sampler, computes the output distribution's statistics, calculates KL divergence, and plots the results to reproduce Figure 2 from the paper.
- `requirements.txt`: Lists all Python dependencies required to run the project.

## Reproduction Details

The numerical experiments focus on validating the theoretical convergence rate of the proposed sampler for a Gaussian target distribution. The key aspects reproduced are:

- **Target Distribution**: A `d`-dimensional Gaussian distribution with a diagonal covariance matrix, where the first `k` diagonal entries are uniformly distributed within `[0, 10]` and the rest are zero (as per Appendix A).
- **Score Function**: The exact score function for this Gaussian distribution, used by the sampler, is implemented according to Appendix C.1.
- **Sampler Algorithm**: The randomized midpoint technique-based sampler is implemented as detailed in Section 2.2, including the randomized schedule (Eq. 8, 9), iterative updates (Eq. 10), and noise injection.
- **Convergence Analysis**: The KL divergence between the generated samples (`Y_K`) and the true target distribution (`q_K`) is computed for varying total iterations `T`.
- **Plotting**: The `main.py` script generates a plot comparing the empirical KL divergence with the theoretical rate `O(log^4 T / T^3)`, aiming to reproduce Figure 2.

## How to Run

To run the numerical experiments:

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Execute the Main Script**:
    ```bash
    python main.py
    ```

The script will print progress and save the generated plot in a `results/` directory.
