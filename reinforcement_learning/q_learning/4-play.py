#!/usr/bin/env python3
"""
Module: 4-play.py
Purpose: Let a trained Q-learning agent play a single episode on FrozenLake.

The agent always exploits the Q-table to select the best action.
Each step of the environment is rendered using "ansi" mode, and the final
state is also displayed.

Returns:
- total_rewards: Total reward obtained in the episode.
- rendered_outputs: List of string representations of the board at each step.
"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Play one episode using the trained Q-table, always exploiting.

    Args:
        env: FrozenLakeEnv instance.
        Q (np.ndarray): Trained Q-table.
        max_steps (int): Maximum steps in the episode.

    Returns:
        total_rewards (float): Total reward obtained.
        rendered_outputs (list): List of board states (strings) at each step.
    """
    rendered_outputs = []
    total_rewards = 0

    state = env.reset()[0]  # reset returns (obs, info)

    for step in range(max_steps):
        # Exploit: choose best action
        action = np.argmax(Q[state])

        # Take the action
        new_state, reward, done, truncated, info = env.step(action)

        # Render current board and append to outputs
        rendered_outputs.append(env.render())

        total_rewards += reward
        state = new_state

        if done:
            break

    # Render final state
    rendered_outputs.append(env.render())

    return total_rewards, rendered_outputs
