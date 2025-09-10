#!/usr/bin/env python3
"""
Module: 4-play.py
Purpose: Have a trained Q-learning agent play an episode on FrozenLake.
"""

import numpy as np

def play(env, Q, max_steps=100):
    """
    Play one episode using the trained Q-table and always exploit the Q-values.

    Args:
        env: FrozenLakeEnv instance (with render_mode="ansi").
        Q (np.ndarray): Trained Q-table.
        max_steps (int): Maximum steps for the episode.

    Returns:
        total_rewards (float): Sum of rewards obtained in the episode.
        rendered_outputs (list[str]): List of board states per step.
    """
    rendered_outputs = []
    total_rewards = 0

    state = env.reset()[0]
    ncol = env.unwrapped.ncol
    desc = env.unwrapped.desc.astype(str)

    # Highlight initial state
    board = []
    for i, row in enumerate(desc):
        row_str = ""
        for j, cell in enumerate(row):
            pos = i * ncol + j
            if pos == state:   # highlight initial position
                row_str += f'`{cell}`'
            else:
                row_str += cell
        board.append(row_str)
    rendered_outputs.append("\n".join(board))

    for step in range(max_steps):
        action = np.argmax(Q[state])
        new_state, reward, done, truncated, _ = env.step(action)
        total_rewards += reward

        board = []
        for i, row in enumerate(desc):
            row_str = ""
            for j, cell in enumerate(row):
                pos = i * ncol + j
                if pos == new_state:
                    row_str += f'`{cell}`'
                else:
                    row_str += cell
            board.append(row_str)

        rendered_outputs.append("\n".join(board))
        state = new_state
        if done:
            break


    return total_rewards, rendered_outputs
