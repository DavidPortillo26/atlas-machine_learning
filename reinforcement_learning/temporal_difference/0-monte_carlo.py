#!/usr/bin/env python3
"""Monte Carlo algorithm for value estimation"""

import numpy as np


def monte_carlo(env, V, policy, episodes=50000, max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm for value estimation

    Args:
        env: environment instance
        V: numpy.ndarray of shape (s,) containing the value estimate
        policy: function that takes in a state and returns the next action to take
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate (unused here, since we do averaging)
        gamma: discount rate

    Returns:
        V: the updated value estimate
    """
    # Work with a copy of V
    V = V.copy()

    # Track number of visits per state for incremental averaging
    N = np.zeros_like(V, dtype=np.int32)

    for _ in range(episodes):
        # Generate one full episode
        states, rewards = [], []
        state, _ = env.reset()

        for _ in range(max_steps):
            states.append(state)
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            state = next_state
            if terminated or truncated:
                break

        # First-visit MC with incremental updates
        G = 0
        visited = set()
        for t in range(len(states) - 1, -1, -1):
            G = rewards[t] + gamma * G
            s = states[t]

            # Skip holes but update everything else (including the goal)
            if s in visited:
                continue
            visited.add(s)

            N[s] += 1
            V[s] += (G - V[s]) / N[s]

    return V
