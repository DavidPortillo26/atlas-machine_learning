#!/usr/bin/env python3

import gymnasium as gym
import numpy as np
import random

td_lambtha = __import__('1-td_lambtha').td_lambtha


def set_seed(env, seed=0):
    env.reset(seed=seed)
    np.random.seed(seed)
    random.seed(seed)


env = gym.make('FrozenLake8x8-v1')
set_seed(env, 0)

LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3


# Fully deterministic policy: always move toward the goal avoiding holes
def policy(s):
    row, col = divmod(s, 8)
    goal_row, goal_col = 7, 7

    # Move DOWN if possible
    if row < goal_row and env.unwrapped.desc[row + 1, col] != b'H':
        return DOWN
    # Move RIGHT if possible
    if col < goal_col and env.unwrapped.desc[row, col + 1] != b'H':
        return RIGHT
    # Move UP if necessary
    if row > goal_row and env.unwrapped.desc[row - 1, col] != b'H':
        return UP
    # Move LEFT if necessary
    if col > goal_col and env.unwrapped.desc[row, col - 1] != b'H':
        return LEFT

    # fallback random choice if stuck
    return np.random.choice([LEFT, RIGHT, UP, DOWN])


# V will be initialized inside td_lambtha
V = np.zeros(env.observation_space.n, dtype=np.float64)

np.set_printoptions(precision=4, suppress=True)

print(td_lambtha(env, V, policy, lambtha=0.9).reshape((8, 8)))
