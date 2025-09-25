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
    Computes the policy with a weight of a matrix using softmax activation.

    This function implements the softmax policy in reinforcement learning,
    which converts raw action logits into a probability distribution over actions.
    The softmax ensures all probabilities sum to 1 and are non-negative.

    Mathematical formulation:
        π(a|s) = exp(s^T * w_a) / Σ_b exp(s^T * w_b)

    where:
        - π(a|s) is the probability of taking action a in state s
        - s is the state vector
        - w_a is the weight vector for action a
        - The denominator normalizes across all possible actions b

    Args:
        matrix (np.ndarray): State matrix of shape (batch_size, state_dim)
                           representing the current environment observations
        weight (np.ndarray): Weight matrix of shape (state_dim, action_dim)
                           containing learnable parameters for each action

    Returns:
        np.ndarray: Policy probabilities of shape (batch_size, action_dim)
                   where each row sums to 1, representing the action probability
                   distribution for each state in the batch

    Example:
        >>> state = np.array([[1, 2, 3, 4]])  # Single state
        >>> weights = np.random.rand(4, 2)    # 4 state dims, 2 actions
        >>> probs = policy(state, weights)
        >>> print(probs.shape)  # (1, 2)
        >>> print(np.sum(probs))  # Should be ~1.0
    """
    # Compute raw logits: matrix multiplication of states and weights
    logits = np.dot(matrix, weight)

    # Apply numerical stability trick: subtract max to prevent overflow
    # This doesn't change the softmax output but prevents exp() overflow
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))

    # Normalize to get probabilities (softmax)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def policy_gradient(state, weight):
    """
    Computes the Monte-Carlo policy gradient based on a state and weight matrix.

    This function implements the REINFORCE algorithm's policy gradient computation.
    It samples an action from the current policy and computes the gradient of
    the log-policy with respect to the policy parameters (weights).

    The policy gradient theorem states:
        ∇_θ J(θ) = E[∇_θ log π_θ(a|s) * Q(s,a)]

    This function computes ∇_θ log π_θ(a|s), which will later be multiplied
    by the return (discounted cumulative reward) to get the full gradient.

    Mathematical derivation:
        For softmax policy π(a|s) = exp(s^T w_a) / Σ_b exp(s^T w_b)
        ∇_w log π(a|s) = s ⊗ (e_a - π(s))  [outer product]
        where e_a is the one-hot encoding of action a

    Args:
        state (np.ndarray): Vector representing the current observation of the
                          environment. Shape: (state_dim,) or (1, state_dim)
        weight (np.ndarray): Weight matrix containing policy parameters.
                           Shape: (state_dim, action_dim)

    Returns:
        tuple: A tuple containing:
            - action (int): The selected action sampled from the policy distribution
            - gradient (np.ndarray): The policy gradient ∇_w log π(a|s) with
                                   shape (state_dim, action_dim)

    Example:
        >>> state = np.array([1.0, -0.5, 0.2, 0.8])  # CartPole state
        >>> weight = np.random.rand(4, 2)             # 4 states, 2 actions
        >>> action, grad = policy_gradient(state, weight)
        >>> print(f"Selected action: {action}")       # 0 or 1
        >>> print(f"Gradient shape: {grad.shape}")     # (4, 2)

    Note:
        The returned gradient needs to be multiplied by the return (G_t) and
        learning rate (α) before updating the weights: w ← w + α * G_t * gradient
    """
    # Ensure state is 2D for consistent matrix operations
    # This handles both 1D state vectors and batched states
    if len(state.shape) == 1:
        state = state.reshape(1, -1)

    # Compute action probabilities using the softmax policy
    probs = policy(state, weight)

    # Sample action stochastically according to the policy distribution
    # This is crucial for exploration in policy gradient methods
    action = np.random.choice(len(probs[0]), p=probs[0])

    # Create one-hot indicator vector for the selected action
    # This represents e_a in the mathematical formulation
    indicator = np.zeros_like(probs[0])
    indicator[action] = 1

    # Compute the policy gradient: ∇_w log π(a|s)
    # Using the identity: ∇_w log π(a|s) = s ⊗ (e_a - π(s))
    # where ⊗ denotes the outer product
    gradient = np.outer(state[0], indicator - probs[0])

    return action, gradient
