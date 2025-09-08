#!/usr/bin/env python3
"""
Module for loading the FrozenLake environment from gymnasium.

The FrozenLake environment is a simple gridworld where the agent
must navigate from the start (S) to the goal (G), avoiding holes (H)
while moving over frozen tiles (F). The environment can be either
deterministic or stochastic depending on the `is_slippery` flag.
"""

import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Loads the FrozenLake environment from gymnasium.

    Args:
        desc (list[list[str]] | None):
            Custom description of the map as a list of lists.
            Example:
                [['S', 'F', 'F'],
                 ['F', 'H', 'H'],
                 ['F', 'F', 'G']]
            Defaults to None.
        map_name (str | None):
            Name of a pre-made map (e.g., "4x4", "8x8").
            If both `desc` and `map_name` are None, a random 8x8 map is loaded.
        is_slippery (bool):
            Whether the ice is slippery (stochastic transitions).
            Defaults to False.

    Returns:
        gym.Env: A FrozenLake-v1 environment instance.
    """
    env = gym.make(
        "FrozenLake-v1",
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery,
    )
    return env
