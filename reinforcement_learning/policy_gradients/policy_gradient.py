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
