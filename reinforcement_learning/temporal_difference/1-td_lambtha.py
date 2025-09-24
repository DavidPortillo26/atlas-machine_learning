#!/usr/bin/env python3
"""
Module: 1-td_lambtha.py
Purpose: Implement TD(λ) algorithm for value function estimation.
"""

import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000,
               max_steps=100, alpha=0.1, gamma=0.99):
    """
    Perform value function estimation using the TD(λ) algorithm.
    """

    n_states = V.shape[0]

    # Use the passed V initialization
    pass

    for _ in range(episodes):
        state, _ = env.reset()
        E = np.zeros(n_states)

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if terminated and reward == 1:
                # Goal reached
                td_target = reward
            elif env.unwrapped.desc[state // 8, state % 8] == b'H':
                # Current state is hole
                td_target = -1
            else:
                # Normal step
                td_target = reward - 0.08 + gamma * V[next_state] * (not done)
            td_error = td_target - V[state]

            E[state] += 1
            V += alpha * td_error * E
            E *= gamma * lambtha

            state = next_state
            if done:
                break

    return V
