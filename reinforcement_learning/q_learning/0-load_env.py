#!/usr/bin/env python3
import gymnasium as gym

""" Module for loading the FrozenLake environment from gymnasium """

def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Loads the FrozenLake environment from gymnasium.

    Args:
        desc (list[list[str]] | None):
        Custom map description as a list of lists.
        map_name (str | None): Pre-made map name (e.g., "4x4", "8x8").
        is_slippery (bool): Whether the ice is slippery.

    Returns:
        gym.Env: The FrozenLake environment.
    """
    env = gym.make(
        "FrozenLake-v1",
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery,
    )
    return env
