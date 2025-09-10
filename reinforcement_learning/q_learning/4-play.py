#!/usr/bin/env python3
"""
Module: 4-play.py
Purpose: Let a trained Q-learning agent play one episode on FrozenLake.
"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Run one episode with a trained Q-table and return rewards and rendered states.

    Args:
        env: FrozenLakeEnv instance (with render_mode="ansi").
        Q (np.ndarray): Trained Q-table.
        max_steps (int): Maximum number of steps in the episode.

    Returns:
        total_reward (float): Total reward obtained in the episode.
        rendered_outputs (list of str): Board states at each step, with agent
                                        position highlighted and actions shown.
    """
    state = env.reset()[0]
    total_reward = 0
    rendered_outputs = []

    nrow, ncol = env.unwrapped.desc.shape

    for _ in range(max_steps):
        action = int(np.argmax(Q[state]))
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        # Get agent position AFTER move
        row, col = divmod(next_state, ncol)

        # Render board and highlight agent's tile with double quotes
        board_str = env.render()
        board_lines = board_str.strip().split('\n')
        highlighted_lines = []

        for r, line in enumerate(board_lines):
            tiles = list(line)
            if r == row:
                tiles[col] = f'"{tiles[col]}"'
            highlighted_lines.append(''.join(tiles))

        rendered_outputs.append('\n'.join(highlighted_lines))

        # Add action annotation
        action_str = {0: "(Left)", 1: "(Down)", 2: "(Right)", 3: "(Up)"}[action]
        rendered_outputs.append("  " + action_str)

        state = next_state
        if terminated or truncated:
            break

    # Final board state
    row, col = divmod(state, ncol)
    board_str = env.render()
    board_lines = board_str.strip().split('\n')
    highlighted_lines = []

    for r, line in enumerate(board_lines):
        tiles = list(line)
        if r == row:
            tiles[col] = f'"{tiles[col]}"'
        highlighted_lines.append(''.join(tiles))

    rendered_outputs.append('\n'.join(highlighted_lines))

    return total_reward, rendered_outputs
