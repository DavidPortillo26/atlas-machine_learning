# utils/wrappers.py
import gymnasium as gym

class GymCompatibilityWrapper(gym.Wrapper):
    """
    Make Gymnasium envs compatible with keras-rl2 (older Gym API).
    - reset() -> returns observation (not (obs, info))
    - step() -> returns (obs, reward, done, info) where done = terminated or truncated
    """
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        done = bool(terminated or truncated)
        return obs, reward, done, info
