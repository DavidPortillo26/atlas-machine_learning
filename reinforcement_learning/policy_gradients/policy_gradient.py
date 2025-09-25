#!/usr/bin/env python3
"""
Policy Gradient Implementation

This module implements policy gradient methods for reinforcement learning.
It contains functions for computing policy probabilities
using softmax activation and related policy gradient algorithms.
"""

import numpy as np


def policy(matrix, weight):
    """
    Computes the policy with a weight of a matrix using softmax.

    Args:
        matrix: state matrix of shape (batch_size, state_dim)
        weight: weight matrix of shape (state_dim, action_dim)

    Returns:
        policy probabilities using softmax activation
    """
    logits = np.dot(matrix, weight)
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def policy_gradient(state, weight):
    """
    Computes the Monte-Carlo policy gradient based on a state and weight matrix.

    Args:
        state: matrix representing the current observation of the environment
        weight: matrix of random weight

    Returns:
        tuple: (action, gradient) - the selected action and the policy gradient
    """
    # Reshape state to be 2D if it's 1D
    if len(state.shape) == 1:
        state = state.reshape(1, -1)

    # Get policy probabilities using the policy function
    probs = policy(state, weight)

    # Sample action based on the probabilities
    action = np.random.choice(len(probs[0]), p=probs[0])

    # Compute the policy gradient
    # For Monte-Carlo policy gradient: ∇log(π(a|s)) = state * (indicator - prob)
    # Where indicator is 1 for the chosen action, 0 otherwise
    indicator = np.zeros_like(probs[0])
    indicator[action] = 1

    # Gradient computation: outer product of state and (indicator - probability)
    gradient = np.outer(state[0], indicator - probs[0])

    return action, gradient
