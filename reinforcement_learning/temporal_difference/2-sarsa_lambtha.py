#!/usr/bin/env python3
"""SARSA(lambda) algorithm implementation"""

import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs SARSA(lambda) algorithm.

    Args:
        env: environment instance
        Q: numpy.ndarray of shape (s,a) containing the Q table
        lambtha: eligibility trace factor
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate
        epsilon: initial threshold for epsilon greedy
        min_epsilon: minimum value that epsilon should decay to
        epsilon_decay: decay rate for updating epsilon between episodes

    Returns:
        Q: the updated Q table
    """
    Q = Q.copy()

    def epsilon_greedy_policy(state, epsilon):
        """Epsilon-greedy policy for action selection"""
        if np.random.random() < epsilon:
            return env.action_space.sample()
        else:
            return np.argmax(Q[state])

    for episode in range(episodes):
        # Initialize eligibility traces
        E = np.zeros_like(Q)

        # Reset environment and choose initial action
        state, _ = env.reset()
        action = epsilon_greedy_policy(state, epsilon)

        for step in range(max_steps):
            # Take action and observe next state and reward
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Calculate TD error
            if terminated or truncated:
                td_error = reward - Q[state, action]
                # Update eligibility trace for current state-action pair
                E[state, action] += 1
                # Update Q-values and eligibility traces for all state-action pairs
                Q += alpha * td_error * E
                break
            else:
                # Choose next action using epsilon-greedy policy
                next_action = epsilon_greedy_policy(next_state, epsilon)

                td_error = reward + gamma * Q[next_state, next_action] - Q[state, action]

                # Update eligibility trace for current state-action pair
                E[state, action] += 1

                # Update Q-values and eligibility traces for all state-action pairs
                Q += alpha * td_error * E
                E *= gamma * lambtha

                # Move to next state and action
                state = next_state
                action = next_action

        # Decay epsilon
        if episode < episodes - 1:  # Don't decay on last episode
            epsilon = max(min_epsilon, epsilon - epsilon_decay)

    return Q