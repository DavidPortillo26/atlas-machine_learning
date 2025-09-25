#!/usr/bin/env python3
"""
Monte Carlo Value Estimation

This module implements a Monte Carlo algorithm for estimating
state-value functions in a given environment. It performs
first-visit Monte Carlo updates over multiple episodes,
using a provided policy.
"""

import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Estimate state values using first-visit Monte Carlo method.

    Args:
        env: Environment instance (OpenAI Gym-like).
        V (np.ndarray): Initial state-value estimates (shape = number of states).
        policy (callable): Function mapping state -> action.
        episodes (int): Number of episodes to simulate.
        max_steps (int): Maximum steps per episode.
        alpha (float): Learning rate (step size).
        gamma (float): Discount factor for future rewards.

    Returns:
        np.ndarray: Updated state-value function V.
    """
    for ep in range(episodes):
        # Start new episode
        state = env.reset()[0]
        trajectory = []  # Stores (state, reward) pairs

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            trajectory.append((state, reward))

            if terminated or truncated:
                break
            state = next_state

        # Compute returns and update values
        G = 0.0
        visited_states = set()

        for s, r in reversed(trajectory):
            G = r + gamma * G
            if s not in visited_states:
                visited_states.add(s)
                V[s] += alpha * (G - V[s])

    return V
