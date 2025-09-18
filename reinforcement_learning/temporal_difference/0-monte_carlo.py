#!/usr/bin/env python3
"""Monte Carlo algorithm for value estimation"""

import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm for value estimation
    
    Args:
        env: environment instance
        V: numpy.ndarray of shape (s,) containing the value estimate
        policy: function that takes in a state and returns the next action to take
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate
    
    Returns:
        V: the updated value estimate
    """
    # Copy V so we don't mutate the original
    V = V.copy()

    for episode in range(episodes):
        # Generate episode
        states = []
        rewards = []

        state, _ = env.reset()
        for step in range(max_steps):
            states.append(state)

            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)

            state = next_state

            if terminated or truncated:
                break

        # Compute returns and update values
        G = 0
        for t in range(len(states) - 1, -1, -1):
            G = rewards[t] + gamma * G
            state_t = states[t]
            # Update toward the exact return
            V[state_t] = V[state_t] + alpha * (G - V[state_t])

    return V
