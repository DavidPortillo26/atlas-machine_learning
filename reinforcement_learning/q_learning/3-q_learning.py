#!/usr/bin/env python3
"""
Module: 3-q_learning.py
Purpose: Train a Q-learning agent on a FrozenLake environment.

This module provides a `train` function that runs Q-learning over multiple
episodes to learn the optimal actions in each state. The agent uses the
epsilon-greedy policy to balance exploration (trying new actions) and
exploitation (choosing the best-known actions). 

Special handling:
- If the agent falls in a hole, the reward is set to -1.
- Epsilon decays after each episode to reduce exploration over time.

The function returns the updated Q-table and a list of rewards obtained
per episode.
"""
import numpy as np


epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy

def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Train a Q-learning agent on the given environment.

    Args:
        env: FrozenLakeEnv instance.
        Q (np.ndarray): Q-table of shape (n_states, n_actions).
        episodes (int): Number of episodes to train over.
        max_steps (int): Max steps per episode.
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        epsilon (float): Initial epsilon for epsilon-greedy.
        min_epsilon (float): Minimum epsilon after decay.
        epsilon_decay (float): Decay rate per episode.

    Returns:
        Q (np.ndarray): Updated Q-table.
        total_rewards (list): Reward obtained in each episode.
    """
    total_rewards = []

    for ep in range(episodes):
        state = env.reset()[0]  # env.reset() returns a tuple in Gymnasium
        reward_sum = 0

        for step in range(max_steps):
            # Choose next action using epsilon-greedy
            action = epsilon_greedy(Q, state, epsilon)

            # Take the action in the environment
            new_state, reward, done, truncated, _ = env.step(action)

            # If agent falls in a hole, set reward to -1
            if reward == 0 and env.desc.flatten()[new_state] == b'H':
                reward = -1

            # Q-learning update rule
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[new_state]) - Q[state, action]
            )

            state = new_state
            reward_sum += reward

            if done:
                break

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))
        total_rewards.append(reward_sum)

    return Q, total_rewards
