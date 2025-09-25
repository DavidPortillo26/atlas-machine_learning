#!/usr/bin/env python3
"""Monte Carlo algorithm implementation"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000,
                max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm to estimate state values.

    Args:
        env: The environment instance
        V: numpy.ndarray, the value estimate
        policy: function(state) -> action
        episodes: total episodes to train
        max_steps: max steps per episode
        alpha: learning rate
        gamma: discount factor

    Returns:
        V: The updated value estimate.
    """
    for _ in range(episodes):
        # Generate an episode
        episode = []
        state, _ = env.reset()

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, reward))
            if terminated or truncated:
                break
            state = next_state

        # Monte Carlo return calculation
        G = 0
        visited = set()
        for state_t, reward_t in reversed(episode):
            G = reward_t + gamma * G
            if state_t not in visited:  # first visit check
                visited.add(state_t)
                V[state_t] += alpha * (G - V[state_t])

    return V
