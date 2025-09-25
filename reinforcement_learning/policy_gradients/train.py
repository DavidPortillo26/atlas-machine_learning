#!/usr/bin/env python3
"""
Policy Gradient Training Module

This module implements the training loop for policy gradient reinforcement learning.
"""

import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """
    Implements a full training using policy gradient method.

    Args:
        env: initial environment
        nb_episodes: number of episodes used for training
        alpha: the learning rate
        gamma: the discount factor

    Returns:
        all values of the score (sum of all rewards during one episode loop)
    """
    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Initialize weights randomly
    weight = np.random.rand(state_dim, action_dim)

    scores = []

    for episode in range(nb_episodes):
        state, _ = env.reset()
        states = []
        actions = []
        rewards = []
        gradients = []

        # Run one episode
        done = False
        while not done:
            action, gradient = policy_gradient(state, weight)
            next_state, reward, done, truncated, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            gradients.append(gradient)

            state = next_state
            done = done or truncated

        # Calculate discounted rewards (returns)
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + gamma * G
            returns.insert(0, G)

        # Normalize returns
        returns = np.array(returns)
        if len(returns) > 1:
            returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

        # Update weights using policy gradient
        for i, gradient in enumerate(gradients):
            weight += alpha * returns[i] * gradient

        # Calculate and store episode score
        episode_score = sum(rewards)
        scores.append(episode_score)

        # Print episode information
        print(f"Episode: {episode} Score: {episode_score}")

    return scores
