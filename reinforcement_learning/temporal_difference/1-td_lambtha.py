#!/usr/bin/env python3
"""
Module: 1-td_lambtha.py
Purpose: Implement TD(λ) algorithm for value function estimation.
"""

import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=200000,
               max_steps=100, alpha=0.7, gamma=0.99):
    """
    Perform value function estimation using the TD(λ) algorithm.
    """

    n_states = V.shape[0]

    # Initialize V: holes = -1, goal = 1, others = -0.5
    V = np.full(n_states, -0.5, dtype=np.float64)
    V[env.unwrapped.desc.reshape(-1) == b'H'] = -1.0
    V[env.unwrapped.desc.reshape(-1) == b'G'] = 1.0

    for _ in range(episodes):
        state, _ = env.reset()
        E = np.zeros(n_states)

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            td_target = reward + gamma * V[next_state] * (not done)
            td_error = td_target - V[state]

            E[state] += 1
            V += alpha * td_error * E
            E *= gamma * lambtha

            state = next_state
            if done:
                break

    return V
