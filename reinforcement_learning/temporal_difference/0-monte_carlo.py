#!/usr/bin/env python3

import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm

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
    for episode in range(episodes):
        state_info = env.reset()
        if isinstance(state_info, tuple):
            state = state_info[0]
        else:
            state = state_info

        episode_states = []
        episode_rewards = []

        for step in range(max_steps):
            action = policy(state)
            step_result = env.step(action)

            if len(step_result) == 5:
                next_state, reward, done, truncated, _ = step_result
            else:
                next_state, reward, done, _ = step_result
                truncated = False

            episode_states.append(state)
            episode_rewards.append(reward)

            if done or truncated:
                break

            state = next_state

        G = 0
        for t in reversed(range(len(episode_states))):
            G = gamma * G + episode_rewards[t]
            state_t = episode_states[t]
            V[state_t] = V[state_t] + alpha * (G - V[state_t])

    return V