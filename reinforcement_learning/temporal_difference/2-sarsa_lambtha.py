#!/usr/bin/env python3
"""
SARSA(λ) Algorithm Implementation

This module contains an implementation of the SARSA(λ) algorithm
for estimating the Q-value table in a reinforcement learning environment.
SARSA(λ) uses eligibility traces to combine benefits of TD learning and
Monte Carlo methods.
"""

import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1.0,
                  min_epsilon=0.1, epsilon_decay=0.05):
    """
    Estimate Q-values using SARSA(λ) with epsilon-greedy exploration.

    Args:
        env: OpenAI Gym-like environment.
        Q (np.ndarray): Initial Q-table of shape (states, actions).
        lambtha (float): Eligibility trace decay rate (λ).
        episodes (int): Number of training episodes.
        max_steps (int): Max steps per episode.
        alpha (float): Learning rate.
        gamma (float): Discount factor for future rewards.
        epsilon (float): Initial epsilon for epsilon-greedy policy.
        min_epsilon (float): Minimum epsilon after decay.
        epsilon_decay (float): Decay rate for epsilon.

    Returns:
        np.ndarray: Updated Q-table.
    """
    starting_epsilon = epsilon

    for ep in range(episodes):
        state = env.reset()[0]
        eligibility_trace = np.zeros_like(Q)

        # Select initial action using epsilon-greedy policy
        if np.random.rand() < epsilon:
            action = np.random.randint(Q.shape[1])
        else:
            action = np.argmax(Q[state])

        for _ in range(max_steps):
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Choose next action with epsilon-greedy policy
            if np.random.rand() < epsilon:
                next_action = np.random.randint(Q.shape[1])
            else:
                next_action = np.argmax(Q[next_state])

            # TD error calculation
            td_error = reward + gamma * Q[next_state, next_action] - Q[state, action]

            # Update eligibility traces
            eligibility_trace[state, action] += 1

            # Update Q-values
            Q += alpha * td_error * eligibility_trace

            # Decay eligibility traces
            eligibility_trace *= gamma * lambtha

            state, action = next_state, next_action

            if terminated or truncated:
                break

        # Update epsilon
        epsilon = min_epsilon + (starting_epsilon - min_epsilon) * np.exp(-epsilon_decay * ep)

    return Q
