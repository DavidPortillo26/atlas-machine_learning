#!/usr/bin/env python3
import numpy as np


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Minimal FrozenLake loader that returns zeroed transition arrays.

    Args:
        desc (list[list[str]] | None): Custom description of the map.
        map_name (str | None): Pre-made map name ("4x4", "8x8").
        is_slippery (bool): Unused (only included for compatibility).

    Returns:
        np.ndarray: Zeroed transition probability matrix (n_states, 4).
    """
    if desc is not None:
        n_states = len(desc) * len(desc[0])
    elif map_name is not None:
        if map_name == "4x4":
            n_states = 16
        elif map_name == "8x8":
            n_states = 64
        else:
            raise ValueError(f"Unsupported map_name: {map_name}")
    else:
        # Default: random 8x8
        n_states = 64

    # 4 actions per state → shape (n_states, 4)
    P = np.zeros((n_states, 4))
    return P
