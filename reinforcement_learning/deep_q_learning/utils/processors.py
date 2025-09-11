# utils/processors.py
import numpy as np
from rl.core import Processor

class StackDimProcessor(Processor):
    """
    Simple processor to convert observations to float32 and optionally normalize.
    - process_observation: called on single observation
    - process_state_batch: called on batched states before passing to model
    - process_reward: clip rewards (optional)
    """

    def __init__(self, normalize=True):
        self.normalize = normalize

    def process_observation(self, observation):
        # observation likely uint8 grayscale image shape (84,84,1) or (84,84)
        arr = np.array(observation)
        # Ensure last channel exists for conv net
        if arr.ndim == 2:
            arr = arr[..., None]
        arr = arr.astype("float32")
        if self.normalize:
            # Keep values in [0,1]
            arr /= 255.0
        return arr

    def process_state_batch(self, batch):
        batch = np.array(batch).astype("float32")
        if self.normalize:
            batch /= 255.0
        return batch

    def process_reward(self, reward):
        # Clip rewards similar to DQN paper
        return np.clip(reward, -1.0, 1.0)
