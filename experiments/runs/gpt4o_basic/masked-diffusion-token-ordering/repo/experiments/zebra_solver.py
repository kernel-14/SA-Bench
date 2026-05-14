'''
Experiment: Zebra (Einstein) Puzzle Solver
Description: Tests Masked Diffusion Model (MDM) and adaptive inference strategies for solving Zebra puzzles.
Author: Based on research paper.
'''

from src.mdm import MaskedDiffusionModel
import numpy as np

# Placeholder for Zebra dataset
def load_zebra_dataset():
    '''
    Simulates loading Zebra puzzles for testing.
    Replace with actual dataset logic.
    '''
    return np.zeros((10, 15))  # Example: 10 puzzles with 15 variables

# Experiment setup
def zebra_solver_experiment():
    '''Runs Zebra solving experiment using MDM.'''
    dataset = load_zebra_dataset()
    mdm_model = MaskedDiffusionModel(vocab_size=5, sequence_length=15, noise_schedule=np.linspace(1, 0, 10))

    results = []
    for puzzle in dataset:
        x_t = np.zeros(15)  # Fully masked input
        solution = mdm_model.adaptive_inference(x_t, strategy='top_margin', K=1)
        results.append(solution)
    
    print('Experiment completed. Results:', results)

if __name__ == '__main__':
    zebra_solver_experiment()

