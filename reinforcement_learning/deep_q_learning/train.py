#!/usr/bin/env python3
"""
Train a DQN agent on Atari Breakout using keras-rl2 and Gymnasium.
"""

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

# Patch missing keras.__version__ (needed for keras-rl2)
from tensorflow import keras
if not hasattr(keras, "__version__"):
    keras.__version__ = tf.__version__

from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy

import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing

# Utils
from utils.wrappers import GymCompatibilityWrapper
from utils.models import model_template
from utils.processors import StackDimProcessor
from utils.callbacks import EpisodicTargetNetworkUpdate


def make_env(env_id="BreakoutNoFrameskip-v4", render_mode=None):
    """Create Atari environment with preprocessing + compatibility fixes."""
    env = gym.make(env_id, render_mode=render_mode) if render_mode else gym.make(env_id)

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


def train_agent(env, state_shape, n_actions, window_length=4, steps=100000):
    """Train a DQN agent with keras-rl2."""
    model = model_template(state_shape, n_actions)
    model.summary()

    memory = SequentialMemory(limit=1_000_000, window_length=window_length)

    # Annealed epsilon-greedy policy
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.1,
        value_test=0.05,
        nb_steps=1000000,
    )

    dqn = DQNAgent(
        model=model,
        nb_actions=n_actions,
        memory=memory,
        nb_steps_warmup=50_000,
        target_model_update=10_000,
        policy=policy,
        enable_double_dqn=True,
        processor=StackDimProcessor(),
    )

    dqn.compile(Adam(learning_rate=0.00025), metrics=["mae"])

    # Target net update every 30 episodes
    target_update_cb = EpisodicTargetNetworkUpdate(update_frequency=30, verbose=1)

    dqn.fit(
        env,
        nb_steps=steps,
        callbacks=[target_update_cb],
        visualize=False,
        verbose=2,
    )

    return dqn


if __name__ == "__main__":
    env = make_env("BreakoutNoFrameskip-v4")

    # Breakout setup
    n_actions = env.action_space.n
    state_shape = (84, 84, 4)

    dqn = train_agent(env, state_shape, n_actions, window_length=4, steps=50000)

    print("Training complete. Saving model weights...")
    dqn.save_weights("policy.h5", overwrite=True)

    env.close()
