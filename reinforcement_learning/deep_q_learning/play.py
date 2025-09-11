#!/usr/bin/env python3
"""
Evaluate and display a trained Atari Breakout DQN agent
using keras-rl2.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam

# Patch missing __version__ for keras-rl2 compatibility
if not hasattr(keras, "__version__"):
    keras.__version__ = tf.__version__

from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing

# Import utilities
from utils.wrappers import GymCompatibilityWrapper
from utils.models import model_template
from utils.processors import StackDimProcessor


def make_env(env_id, render_mode="human"):
    """Create evaluation environment."""
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
    # Create environment
    env = make_env("BreakoutNoFrameskip-v4")

    # Recreate model architecture
    n_actions = env.action_space.n
    state_shape = (84, 84, 4)
    model = model_template(state_shape, n_actions)

    # Load saved model or weights
    try:
        model = keras.models.load_model("policy.h5")
        print("✅ Loaded full model from policy.h5")
    except Exception:
        print("⚠️ Could not load full model, loading weights instead")
        model.load_weights("policy.h5")

    # Wrap agent with greedy policy for evaluation
    memory = SequentialMemory(limit=1000000, window_length=4)
    policy = GreedyQPolicy()
    dqn = DQNAgent(
        model=model,
        nb_actions=n_actions,
        memory=memory,
        policy=policy,
        processor=StackDimProcessor(),
    )
    dqn.compile(Adam(learning_rate=0.00025), metrics=["mae"])

    # Play a few episodes
    dqn.test(env, nb_episodes=5, visualize=True)
    env.close()
