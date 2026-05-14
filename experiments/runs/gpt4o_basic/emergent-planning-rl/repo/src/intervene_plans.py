import torch

def apply_intervention(cell_state, intervention_vector, target_positions):
    for position, vector in zip(target_positions, intervention_vector):
        x, y = position
        cell_state[:, x, y, :] += vector
    return cell_state

def intervention_example():
    # Example dummy intervention logic
    agent_cell_state = torch.zeros(1, 8, 8, 32)  # Example agent cell state
    target_positions = [(0, 0), (1, 1)]
    intervention_vector = [torch.ones(32), torch.ones(32) * -1]

    modified_state = apply_intervention(agent_cell_state, intervention_vector, target_positions)
    print(modified_state)

if __name__ == '__main__':
    intervention_example()
