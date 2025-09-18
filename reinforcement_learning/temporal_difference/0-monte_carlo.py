#!/usr/bin/env python3
"""Monte Carlo algorithm for value estimation"""

import numpy as np


def monte_carlo(env, V, policy, episodes=50000, max_steps=100, alpha=0.1, gamma=0.99):

    """
    Performs the Monte Carlo algorithm for value estimation.

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
    # Work with a copy of V
    V = V.copy()

    for episode in range(episodes):
        # Generate episode
        states = []
        rewards = []

        # Reset environment
        state, _ = env.reset()

        # Run episode
        for step in range(max_steps):
            states.append(state)

            # Get action from policy
            action = policy(state)

            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)

            state = next_state

            if terminated or truncated:
                break

        # Every-visit Monte Carlo updates
        G = 0
        for t in range(len(states) - 1, -1, -1):
            G = rewards[t] + gamma * G
            state_t = states[t]

            # Skip terminal states (holes and goal)
            if env.unwrapped.desc[state_t // 8, state_t % 8] in (b'H', b'G'):
                continue

            # Incremental update
            V[state_t] = V[state_t] + alpha * (G - V[state_t])

    return V
