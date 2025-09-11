#!/usr/bin/env python3
"""
Train weights to play Atari
"""

import warnings

# Suppress all urllib3 SSL warnings
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

# Suppress pkg_resources deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")

from PIL import Image
import numpy as np
import gymnasium as gym

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor

from keras import layers
import keras as K


class AtariProcessor(Processor):
    """Processor for Atari"""

    def process_observation(self, observation):
        INPUT_SHAPE = (84, 84)
        assert observation.ndim == 3
        img = Image.fromarray(observation).resize(INPUT_SHAPE).convert('L')
        processed_observation = np.array(img)
        assert processed_observation.shape == (84, 84)
        return processed_observation.astype('uint8')

    def process_state_batch(self, batch):
        return batch.astype('float32') / 255.

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)


if __name__ == '__main__':
    env = gym.make("ALE/Breakout-v5")
    env.reset()
    nb_actions = env.action_space.n

    INPUT_SHAPE = (84, 84)
    WINDOW_LENGTH = 4
    input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE

    # Build Conv2D model
    inputs = layers.Input(shape=input_shape)
    perm = layers.Permute((2, 3, 1))(inputs)
    layer = layers.Conv2D(32, 8, strides=(4, 4), activation='relu',
                          data_format="channels_last")(perm)
    layer = layers.Conv2D(64, 4, strides=(2, 2), activation='relu',
                          data_format="channels_last")(layer)
    layer = layers.Conv2D(64, 3, strides=(1, 1), activation='relu',
                          data_format="channels_last")(layer)
    layer = layers.Flatten()(layer)
    layer = layers.Dense(512, activation='relu')(layer)
    activation = layers.Dense(nb_actions, activation='linear')(layer)
    model = K.Model(inputs=inputs, outputs=activation)
    model.summary()

    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
    processor = AtariProcessor()

    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(),
                                  attr='eps',
                                  value_max=1.,
                                  value_min=.1,
                                  value_test=.05,
                                  nb_steps=1000000)

    dqn = DQNAgent(model=model, nb_actions=nb_actions,
                   policy=policy, memory=memory,
                   processor=processor, nb_steps_warmup=50000,
                   gamma=.99, target_model_update=10000,
                   train_interval=4, delta_clip=1.)

    dqn.compile(K.optimizers.Adam(lr=.00025), metrics=['mae'])

    # Training
    dqn.fit(env,
            nb_steps=1000000,
            log_interval=100000,
            visualize=False,
            verbose=2)

    # Save weights and model
    dqn.save_weights('policy.h5', overwrite=True)
    model.save("policy_model.h5")
