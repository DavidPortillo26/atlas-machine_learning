#!/usr/bin/env python3
"""
Module for initializing a Q-table for reinforcement learning.

A Q-table is a table used in Q-learning (a type of reinforcement learning)
to store "scores" (Q-values) for every possible action in every possible state.
These scores help an agent decide which action to take in each situation.

This module provides a function to create a Q-table initialized with zeros,
so the agent can start learning from scratch.
"""
import numpy as np


def q_init(env):
    """
    Initialize a Q-table with zeros.

    This function creates a table where each row represents a possible state
    of the environment, and each column represents a possible action.
    Initially,all the values are set to zero because the agent
    hasn't learned anything yet.

    Args:
        env (np.ndarray or gym.Env):
            - np.ndarray: a pre-defined transition matrix for the environment.
              Shape: (number of states, number of actions)
            - gym.Env: an environment object with discrete states and actions.

    Returns:
        np.ndarray: A 2D table (array) of zeros with shape
                    (number of states, number of actions). Each cell will
                    store the Q-value for taking that action in that state.
    """
    if isinstance(env, np.ndarray):
        # Transition matrix case
        return np.zeros_like(env, dtype=float)

    # Gym-like environment case
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    return np.zeros((n_states, n_actions), dtype=float)
