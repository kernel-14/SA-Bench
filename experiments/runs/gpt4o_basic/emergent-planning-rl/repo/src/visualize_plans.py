import matplotlib.pyplot as plt
import numpy as np

def visualize_plan(grid, concept_A, concept_B):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(grid, cmap='gray', interpolation='none')

    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            direction_A = concept_A[x, y]
            direction_B = concept_B[x, y]

            if direction_A != 'NEVER':
                ax.arrow(y, x, dx=arrow_dx(direction_A), dy=arrow_dy(direction_A),
                         head_width=0.2, head_length=0.2, fc='teal', ec='teal')

            if direction_B != 'NEVER':
                ax.arrow(y, x, dx=arrow_dx(direction_B), dy=arrow_dy(direction_B),
                         head_width=0.2, head_length=0.2, fc='purple', ec='purple')

    plt.show()

def arrow_dx(direction):
    return {UP: 0, DOWN: 0, LEFT: -1, RIGHT: 1, NEVER: 0}.get(direction, 0)

def arrow_dy(direction):
    return {UP: -1, DOWN: 1, LEFT: 0, RIGHT: 0, NEVER: 0}.get(direction, 0)
