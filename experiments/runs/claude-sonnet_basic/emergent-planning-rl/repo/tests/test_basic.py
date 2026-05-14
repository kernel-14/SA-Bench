"""
Basic tests to verify the implementation is correct.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np


def test_sokoban_env():
    """Test Sokoban environment."""
    from environment.sokoban import SokobanEnv, WALL, EMPTY, BOX, AGENT, TARGET, BOX_ON_TARGET
    
    env = SokobanEnv(max_steps=120)
    
    # Create a simple level
    level = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 3, 1, 1, 1, 0],  # agent at (3,3)
        [0, 1, 1, 2, 1, 1, 1, 0],  # box at (4,3)
        [0, 1, 1, 6, 1, 1, 1, 0],  # target at (5,3)
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.int32)
    
    obs = env.reset(level)
    
    assert obs.shape == (8, 8, 7), f"Expected (8,8,7), got {obs.shape}"
    assert env.get_agent_pos() == (3, 3), f"Expected (3,3), got {env.get_agent_pos()}"
    
    # Move down to push box onto target
    obs, reward, done, info = env.step(2)  # DOWN
    assert env.get_agent_pos() == (4, 3), f"Expected (4,3), got {env.get_agent_pos()}"
    assert reward > 0, f"Expected positive reward, got {reward}"
    
    print("✓ Sokoban environment test passed")


def test_drc_agent():
    """Test DRC agent forward pass."""
    from agent.drc_agent import DRCAgent
    
    agent = DRCAgent(D=3, N=3, hidden_channels=32)
    
    # Test forward pass
    obs = torch.zeros(1, 8, 8, 7)
    logits, value, hidden_states, _ = agent.forward(obs)
    
    assert logits.shape == (1, 5), f"Expected (1,5), got {logits.shape}"
    assert value.shape == (1, 1), f"Expected (1,1), got {value.shape}"
    assert len(hidden_states) == 3, f"Expected 3 hidden states, got {len(hidden_states)}"
    
    # Test with return_cell_states
    logits, value, hidden_states, cell_states = agent.forward(obs, return_cell_states=True)
    
    assert cell_states is not None
    assert len(cell_states) == 3, f"Expected 3 ticks, got {len(cell_states)}"
    assert len(cell_states[0]) == 3, f"Expected 3 layers, got {len(cell_states[0])}"
    assert cell_states[0][0].shape == (1, 32, 8, 8), f"Expected (1,32,8,8), got {cell_states[0][0].shape}"
    
    print("✓ DRC agent test passed")


def test_linear_probe():
    """Test linear probe."""
    from probing.linear_probe import LinearProbe, compute_macro_f1
    
    # Test 1x1 probe
    probe = LinearProbe(hidden_channels=32, num_classes=5, probe_size=1)
    cell_state = torch.zeros(1, 32, 8, 8)
    logits = probe(cell_state)
    assert logits.shape == (1, 5, 8, 8), f"Expected (1,5,8,8), got {logits.shape}"
    
    # Test 3x3 probe
    probe_3x3 = LinearProbe(hidden_channels=32, num_classes=5, probe_size=3)
    logits_3x3 = probe_3x3(cell_state)
    assert logits_3x3.shape == (1, 5, 8, 8), f"Expected (1,5,8,8), got {logits_3x3.shape}"
    
    # Test class vectors
    vectors = probe.get_class_vectors()
    assert vectors.shape == (5, 32), f"Expected (5,32), got {vectors.shape}"
    
    # Test macro F1
    preds = np.array([0, 1, 2, 3, 4, 0, 1])
    labels = np.array([0, 1, 2, 3, 4, 1, 0])
    f1 = compute_macro_f1(preds, labels)
    assert 0 <= f1 <= 1, f"F1 should be in [0,1], got {f1}"
    
    print("✓ Linear probe test passed")


def test_concepts():
    """Test concept computation."""
    from probing.concepts import (
        ConceptClass, compute_agent_approach_direction, compute_box_push_direction
    )
    
    # Simple trajectory: agent moves right, then down
    trajectory = [(3, 3), (3, 4), (4, 4)]
    
    ca = compute_agent_approach_direction(trajectory)
    assert ca.shape == (8, 8), f"Expected (8,8), got {ca.shape}"
    
    # Agent moves right to (3,4) - approach direction is RIGHT
    assert ca[3, 4] == ConceptClass.RIGHT, f"Expected RIGHT, got {ca[3,4]}"
    # Agent moves down to (4,4) - approach direction is DOWN
    assert ca[4, 4] == ConceptClass.DOWN, f"Expected DOWN, got {ca[4,4]}"
    # Other squares should be NEVER
    assert ca[0, 0] == ConceptClass.NEVER, f"Expected NEVER, got {ca[0,0]}"
    
    print("✓ Concept computation test passed")


def test_intervention():
    """Test intervention framework."""
    from probing.linear_probe import LinearProbe
    from interventions.interventions import AgentShortcutIntervention
    from probing.concepts import ConceptClass
    
    probe = LinearProbe(hidden_channels=32, num_classes=5, probe_size=1)
    
    intervention = AgentShortcutIntervention(
        short_route_squares=[(3, 3), (3, 4)],
        long_route_squares_dirs=[((4, 3), ConceptClass.DOWN)],
        probe_ca=probe,
        layer=2,
        alpha=1.0,
        p=1,
    )
    
    # Create dummy cell states
    cell_states = [
        (torch.zeros(1, 32, 8, 8), torch.zeros(1, 32, 8, 8))
        for _ in range(3)
    ]
    
    # Apply intervention
    new_states = intervention.apply(cell_states, agent_pos=(3, 2))
    
    assert len(new_states) == 3, f"Expected 3 states, got {len(new_states)}"
    
    print("✓ Intervention test passed")


if __name__ == '__main__':
    print("Running basic tests...")
    test_sokoban_env()
    test_drc_agent()
    test_linear_probe()
    test_concepts()
    test_intervention()
    print("\nAll tests passed!")
