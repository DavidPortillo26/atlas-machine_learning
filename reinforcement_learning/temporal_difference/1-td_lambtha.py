#!/usr/bin/env python3
"""
TD(λ) Value Estimation

This module implements the TD(λ) algorithm for estimating
state-value functions in a given environment. It uses eligibility
traces to blend Monte Carlo and temporal-difference methods.
"""

import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000,
               max_steps=100, alpha=0.1, gamma=0.99):
    """
    Estimate state values using TD(λ) algorithm.

    Args:
        env: Environment instance (OpenAI Gym-like).
        V (np.ndarray): Initial state-value estimates (shape = number of states).
        policy (callable): Function mapping state -> action.
        lambtha (float): Eligibility trace decay rate (λ).
        episodes (int): Number of episodes to run.
        max_steps (int): Maximum steps per episode.
        alpha (float): Step size (learning rate).
        gamma (float): Discount factor for future rewards.

    Returns:
        np.ndarray: Updated state-value function V.
    """
    for ep in range(episodes):
        current_state = env.reset()[0]
        eligibility = np.zeros_like(V, dtype=float)  # Eligibility traces

        for _ in range(max_steps):
            action = policy(current_state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            td_error = reward + gamma * V[next_state] - V[current_state]

            # Increment eligibility trace for the visited state
            eligibility[current_state] += 1

            # Update value estimates
            V += alpha * td_error * eligibility

            # Decay eligibility traces
            eligibility *= gamma * lambtha

            if terminated or truncated:
                break
            current_state = next_state

    return V
