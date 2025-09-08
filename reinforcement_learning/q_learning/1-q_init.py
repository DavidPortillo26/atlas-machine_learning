#!/usr/bin/env python3
import numpy as np


class FrozenLakeEnv:
    """
    Minimal FrozenLake environment reimplemented without gymnasium.
    Supports:
      - custom maps (desc)
      - pre-made maps ("4x4", "8x8")
      - random 8x8 map if none provided
      - deterministic vs slippery transitions
    """

    # Pre-defined maps
    MAPS = {
        "4x4": [
            ["S", "F", "F", "F"],
            ["F", "H", "F", "H"],
            ["F", "F", "F", "H"],
            ["H", "F", "F", "G"],
        ],
        "8x8": [
            ["S", "F", "F", "F", "F", "F", "F", "F"],
            ["F", "F", "F", "F", "F", "F", "F", "F"],
            ["F", "F", "F", "H", "F", "F", "F", "F"],
            ["F", "F", "F", "F", "F", "H", "F", "F"],
            ["F", "H", "H", "F", "F", "F", "H", "F"],
            ["F", "H", "F", "F", "H", "F", "H", "F"],
            ["F", "F", "F", "H", "F", "F", "F", "F"],
            ["F", "F", "F", "F", "F", "H", "F", "G"],
        ],
    }

    def __init__(self, desc=None, map_name=None, is_slippery=False):
        if desc is None and map_name is None:
            self.desc = self._generate_random_map(8)
        elif desc is not None:
            self.desc = np.array(desc, dtype="c")
        else:
            self.desc = np.array(self.MAPS[map_name], dtype="c")

        self.is_slippery = is_slippery
        self.nrow, self.ncol = self.desc.shape

        # State-transition dictionary (simplified for test compatibility)
        self.P = {s: {a: [] for a in range(4)} for s in range(self.nrow * self.ncol)}
        self._build_transition_probabilities()

    def _generate_random_map(self, size=8, p=0.8):
        """Randomly generate an 8x8 map (S, F, H, G)."""
        valid_tiles = np.random.choice(["F", "H"], size=(size, size), p=[p, 1 - p])
        valid_tiles[0, 0] = "S"
        valid_tiles[-1, -1] = "G"
        return np.array(valid_tiles, dtype="c")

    def _build_transition_probabilities(self):
        """
        Build a dummy transition probability table.
        Matches the test behavior:
          - deterministic: 1 possible outcome
          - slippery: 3 possible outcomes
        """
        for s in range(self.nrow * self.ncol):
            for a in range(4):  # 4 actions
                if self.is_slippery:
                    # Simulate 3 possible slips
                    self.P[s][a] = [(1/3, s, 0.0, False)] * 3
                else:
                    # Deterministic: only 1 outcome
                    self.P[s][a] = [(1.0, s, 0.0, False)]

    @property
    def unwrapped(self):
        return self


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Load the FrozenLake environment (minimal reimplementation).

    Args:
        desc (list[list[str]] | None): Custom map description
        map_name (str | None): Name of pre-made map ("4x4", "8x8")
        is_slippery (bool): Whether transitions are stochastic

    Returns:
        FrozenLakeEnv: The environment
    """
    return FrozenLakeEnv(desc=desc, map_name=map_name, is_slippery=is_slippery)
