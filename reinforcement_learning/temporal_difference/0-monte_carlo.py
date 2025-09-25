#!/usr/bin/env python3
"""monte carlos algorithm implimentation"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000,
                max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm to estimate state values

    Args:
        env: The environment instance
        V: containing the value estimate
        policy: takes in a state and returns the next action to take
        episodes: episodes to train over
        max_steps: number of steps per episode
        alpha: The learning rate
        gamma: The discount rate

    Returns:
        V: The updated value estimate.
    """
    # Loop over episodes
    for episode in range(episodes):
        # Reset the environment
        state = env.reset()[0]
        # store (state, reward) pairs for each step
        data = []

        for step in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            data.append((state, reward))

            if terminated or truncated:
                break
            state = next_state

        G = 0
        data = np.array(data, dtype=int)
        # Calculate the return and update the value function
        for state, reward in reversed(data):
            G = reward + gamma * G
            if state not in data[:episode, 0]:
                V[state] += alpha * (G - V[state])
    return V
