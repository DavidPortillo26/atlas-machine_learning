#!/usr/bin/env python3
import numpy as np

""" Module for initializing Q-table """


def q_init(env):
    """
    Initialize the Q-table with zeros.

    Args:
        env (np.ndarray): Transition matrix representing the environment.
                          Shape: (n_states, n_actions)

    Returns:
        np.ndarray: Zero-initialized Q-table of the same shape as env.
    """
    return np.zeros_like(env, dtype=float)
