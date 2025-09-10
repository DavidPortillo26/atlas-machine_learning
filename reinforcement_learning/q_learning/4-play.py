#!/usr/bin/env python3
"""
Module: 4-play.py
Purpose: Play one episode using a trained Q-table, showing agent's position.
"""
import numpy as np

def play(env, Q, max_steps=100):
    """
    Play a single episode with exploitation only, rendering each step
    with the agent highlighted using backticks.
    
    Args:
        env: FrozenLakeEnv instance (render_mode="ansi").
        Q (np.ndarray): Trained Q-table.
        max_steps (int): Maximum steps for the episode.
    
    Returns:
        total_rewards (float): Total reward accumulated.
        rendered_outputs (list): Board states with agent highlighted.
    """
    rendered_outputs = []
    total_rewards = 0

    state = env.reset()[0]  # reset returns (obs, info)
    
    for step in range(max_steps):
        action = np.argmax(Q[state])          # Exploit best action
        new_state, reward, done, truncated, _ = env.step(action)

        # Get the flattened board
        board = env.unwrapped.desc.copy().tolist()
        n = len(board)
        # Determine agent's row and column
        row, col = divmod(new_state, n)
        # Convert bytes to string if needed
        for r in range(n):
            board[r] = [c.decode() if isinstance(c, bytes) else c for c in board[r]]
        # Highlight agent's position
        board[row][col] = f"`{board[row][col]}`"

        # Convert board to string for display
        rendered_outputs.append("\n".join("".join(r) for r in board))
        rendered_outputs.append(f"  ({['Left','Down','Right','Up'][action]})")

        total_rewards += reward
        state = new_state

        if done:
            break

    # Show final state
    board = env.unwrapped.desc.copy().tolist()
    n = len(board)
    row, col = divmod(state, n)
    for r in range(n):
        board[r] = [c.decode() if isinstance(c, bytes) else c for c in board[r]]
    board[row][col] = f'"{board[row][col]}"'
    rendered_outputs.append("\n".join("".join(r) for r in board))

    return total_rewards, rendered_outputs
