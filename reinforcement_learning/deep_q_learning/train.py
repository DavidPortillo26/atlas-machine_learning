#!/usr/bin/env python3
"""
Train a DQN agent on Atari Breakout using keras-rl2 and TensorFlow/Keras.
"""

import warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
import numpy as np

# Patch missing __version__ for keras-rl2 compatibility
if not hasattr(keras, "__version__"):
    keras.__version__ = tf.__version__

from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy
from rl.core import Processor


# ============================================================
# Inline replacement for utils/ files
# ============================================================

class GymCompatibilityWrapper(gym.Wrapper):
    """Make Gymnasium envs compatible with keras-rl2 (old Gym API)."""
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info


def model_template(input_shape, n_actions):
    """DeepMind-style ConvNet for Atari DQN."""
    inputs = keras.Input(shape=input_shape)

    x = keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation="relu")(inputs)
    x = keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation="relu")(x)
    x = keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation="relu")(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(512, activation="relu")(x)
    outputs = keras.layers.Dense(n_actions, activation="linear")(x)

    return keras.Model(inputs=inputs, outputs=outputs)


class StackDimProcessor(Processor):
    """Ensure observations/rewards are properly formatted for keras-rl2."""
    def process_observation(self, observation):
        return np.array(observation).astype("float32")

    def process_state_batch(self, batch):
        return batch.astype("float32")

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)


# ============================================================
# Training code
# ============================================================

def make_env(env_id="BreakoutNoFrameskip-v4"):
    """Create Atari environment with preprocessing + compatibility wrapper."""
    env = gym.make(env_id)
    env = AtariPreprocessing(
        env,
        noop_max=7,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=True,
        grayscale_obs=True,
        grayscale_newaxis=True,
        scale_obs=False,
    )
    env = GymCompatibilityWrapper(env)
    return env


if __name__ == "__main__":
    env = make_env()

    n_actions = env.action_space.n
    state_shape = (84, 84, 4)  # 4 stacked frames
    model = model_template(state_shape, n_actions)

    memory = SequentialMemory(limit=1000000, window_length=4)
    policy = EpsGreedyQPolicy()

    dqn = DQNAgent(
        model=model,
        nb_actions=n_actions,
        memory=memory,
        processor=StackDimProcessor(),
        policy=policy,
        nb_steps_warmup=50000,
        target_model_update=10000,
        train_interval=4,
        delta_clip=1.0,
    )

    dqn.compile(Adam(learning_rate=0.00025), metrics=["mae"])

    # Train agent
    dqn.fit(env, nb_steps=1000000, visualize=False, verbose=2)

    # Save trained weights
    dqn.save_weights("policy.h5", overwrite=True)

    env.close()
