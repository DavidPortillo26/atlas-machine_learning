#!/usr/bin/env python3
"""
Module: 4-play.py
Purpose: Play one episode using a trained Q-table, showing agent's position.
"""
import numpy as np

def play(env, Q, max_steps=100):
    rendered_outputs = []
    total_rewards = 0

    state = env.reset()[0]  # initial observation
    n = len(env.unwrapped.desc)

    def render_board(s):
        board = env.unwrapped.desc.copy().tolist()
        for r in range(n):
            board[r] = [c.decode() if isinstance(c, bytes) else c for c in board[r]]
        row, col = divmod(s, n)
        board[row][col] = f"`{board[row][col]}`"
        return "\n".join("".join(r) for r in board)

    for _ in range(max_steps):
        action = int(np.argmax(Q[state]))
        next_state, reward, done, truncated, _ = env.step(action)
        total_rewards += reward

        # Append board + action as one string
        rendered_outputs.append(f"{render_board(next_state)}\n  ({['Left','Down','Right','Up'][action]})")

        state = next_state
        if done or truncated:
            break

    return total_rewards, rendered_outputs
