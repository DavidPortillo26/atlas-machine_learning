#!/usr/bin/env python3
"""
Play Atari Breakout with a trained DQN agent.
"""

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

# Patch keras version for keras-rl2
from tensorflow import keras
if not hasattr(keras, "__version__"):
    keras.__version__ = tf.__version__

from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy

import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing

# Utils
from utils.wrappers import GymCompatibilityWrapper
from utils.models import model_template
from utils.processors import StackDimProcessor


def make_env(env_id="ALE/Breakout-v5", render_mode="human"):
    """Create evaluation environment with rendering enabled."""
    env = gym.make(env_id, render_mode=render_mode)

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
    processor = StackDimProcessor()
    env = GymCompatibilityWrapper(env, processor)
    return env


if __name__ == "__main__":
    env = make_env("ALE/Breakout-v5", render_mode="human")

    n_actions = env.action_space.n
    state_shape = (84, 84, 4)

    # Build model (must match training model)
    model = model_template(state_shape, n_actions)

    try:
        model = keras.models.load_model("policy.h5")
        print("Loaded full model from policy.h5")
    except Exception:
        print("Loading weights into model from policy.h5")
        model.load_weights("policy.h5")

    memory = SequentialMemory(limit=1_000_000, window_length=4)
    policy = GreedyQPolicy()

    dqn = DQNAgent(
        model=model,
        nb_actions=n_actions,
        memory=memory,
        policy=policy,
        processor=StackDimProcessor(),
    )

    dqn.compile(Adam(learning_rate=0.00025), metrics=["mae"])

    # Play for a few episodes
    dqn.test(env, nb_episodes=5, visualize=True)

    env.close()
