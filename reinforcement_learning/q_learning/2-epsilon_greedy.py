#!/usr/bin/env python3
"""
Module: 2-epsilon_greedy.py
Purpose: Select the next action in reinforcement learning using the
epsilon-greedy policy.

The epsilon-greedy policy is a method for balancing exploration and
exploitation:
- Exploration: the agent chooses a random action to discover new states.
- Exploitation: the agent chooses the action with the highest known
  reward based on the Q-table.

This module provides the `epsilon_greedy` function, which takes the
current Q-table, the current state, and an epsilon value to decide
whether to explore or exploit.
"""
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
