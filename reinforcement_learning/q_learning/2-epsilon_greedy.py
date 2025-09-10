#!/usr/bin/env python3
import numpy as np

def epsilon_greedy(Q, state, epsilon):
    """
    Select the next action using the epsilon-greedy policy.

    Args:
        Q (np.ndarray): The Q-table of shape (n_states, n_actions).
        state (int): Current state index.
        epsilon (float): Probability of choosing a random action (exploration).

    Returns:
        int: Index of the next action.
    """
    # Sample a random number between 0 and 1
    p = np.random.uniform()
    
    if p < epsilon:
        # Explore: choose a random action
        action = np.random.randint(Q.shape[1])
    else:
        # Exploit: choose the action with the highest Q-value
        action = np.argmax(Q[state])
    
    return action
