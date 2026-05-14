# Relevance Functions for Prioritized Generative Replay

import numpy as np

def return_based_relevance(q_function, state, action):
    """Relevance based on the state-action value function (return)."""
    return q_function(state, action)

def td_error_relevance(q_function, target_q_function, state, action, reward, next_state, gamma):
    """Relevance based on Temporal Difference (TD) error."""
    td_target = reward + gamma * np.max(target_q_function(next_state))
    td_error = td_target - q_function(state, action)
    return abs(td_error)

def curiosity_relevance(encoder, forward_model, state, action, next_state):
    """Relevance based on intrinsic curiosity modules."""
    state_features = encoder(state)
    next_state_features = encoder(next_state)
    predicted_features = forward_model(state_features, action)
    curiosity_score = 0.5 * np.sum((predicted_features - next_state_features)**2)
    return curiosity_score


