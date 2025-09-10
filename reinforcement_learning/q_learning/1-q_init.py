#!/usr/bin/env python3
import numpy as np

""" Module for initializing Q-table """


def q_init(env):
    """
    Initialize the Q-table with zeros.

    Args:
        env (np.ndarray or gym.Env):
            - If np.ndarray: transition matrix representing the environment.
                             Shape: (n_states, n_actions)
            - If gym.Env: environment with discrete observation and action spaces.

    Returns:
        np.ndarray: Zero-initialized Q-table of shape (n_states, n_actions).
    """
    if isinstance(env, np.ndarray):
        # Transition matrix case
        return np.zeros_like(env, dtype=float)

    # Gym-like environment case
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    return np.zeros((n_states, n_actions), dtype=float)
