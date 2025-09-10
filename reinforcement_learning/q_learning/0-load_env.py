#!/usr/bin/env python3
"""
Module: 0-load_env
------------------
This module provides a helper function for loading the
FrozenLake environment from the `gymnasium` library.

The FrozenLake environment is a gridworld game where the
agent must navigate from the starting position (S) to the
goal (G) while avoiding holes (H). Frozen tiles (F) may be
slippery depending on the `is_slippery` parameter, which
adds stochasticity to the agent’s movements.

Features:
- Load built-in maps (e.g., "4x4", "8x8")
- Load custom maps via a description matrix
- Control whether the ice is slippery or deterministic
"""

import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Create and return a FrozenLake environment.

    This function wraps `gymnasium.make("FrozenLake-v1")`
    with flexible parameters to allow loading either a
    built-in map, a custom map, or a randomly generated map.

    Args:
        desc (list[list[str]] | None, optional):
            A custom map description. It should be a square
            grid represented as a list of lists containing:
              - 'S' (start state, only one allowed)
              - 'F' (frozen tile, safe)
              - 'H' (hole, terminal bad state)
              - 'G' (goal, terminal success state)

            Example:
                desc = [
                    ['S', 'F', 'F'],
                    ['F', 'H', 'H'],
                    ['F', 'F', 'G']
                ]

            Defaults to None.

        map_name (str | None, optional):
            The name of a pre-defined map.
            Supported values include:
              - "4x4" : small board, good for testing
              - "8x8" : larger board, more challenging

            If both `desc` and `map_name` are None, the
            environment loads a randomly generated 8x8 map.

            Defaults to None.

        is_slippery (bool, optional):
            Whether the ice is slippery. If True, the agent's
            movements are stochastic (slips are possible).
            If False, the agent moves deterministically.

            Defaults to False.

    Returns:
        gym.Env:
            A `FrozenLake-v1` environment instance.

    Example:
        >>> env = load_frozen_lake(map_name="4x4", is_slippery=True)
        >>> print(env.unwrapped.desc)
    """
    # Create the FrozenLake environment with the given configuration
    env = gym.make(
        "FrozenLake-v1",
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery,
        render_mode="ansi"  # Render as text for easy visualization
    )
    return env
