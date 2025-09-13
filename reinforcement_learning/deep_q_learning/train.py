#!/usr/bin/env python3
"""
Training script for Atari Breakout using DQN with keras-rl2 and gymnasium
"""

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
try:
    from gymnasium.wrappers import FrameStack
except ImportError:
    from gymnasium.wrappers.frame_stack import FrameStack
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


class AtariProcessor(Processor):
    """Processor for Atari games to handle observation preprocessing."""
    
    def process_observation(self, observation):
        """Process observation to ensure correct format."""
        # Ensure observation is float32 and normalized
        observation = np.array(observation).astype('float32') / 255.0
        return observation
    
    def process_state_batch(self, batch):
        """Process batch of states."""
        processed_batch = np.array(batch).astype('float32')
        return processed_batch


def create_env():
    """Create and configure the Atari Breakout environment."""
    # Create the base environment
    env = gym.make('ALE/Breakout-v5', render_mode='rgb_array')
    
    # Apply Atari preprocessing (grayscale, resize, etc.)
    env = AtariPreprocessing(env, 
                           noop_max=30,
                           frame_skip=4,
                           screen_size=84,
                           terminal_on_life_loss=True,
                           grayscale_obs=True,
                           grayscale_newaxis=False,
                           scale_obs=False)
    
    # Stack 4 frames together
    env = FrameStackObservation(env, stack_size=4)
    
    return env


def build_model(input_shape, nb_actions):
    """Build the CNN model for DQN."""
    model = Sequential()
    
    # Add input layer explicitly for newer TensorFlow versions
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    
    # Permute dimensions to match expected input format (channels_last to channels_first)
    model.add(Permute((2, 3, 1)))
    
    # Convolutional layers
    model.add(Convolution2D(32, (8, 8), strides=(4, 4), activation='relu'))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Convolution2D(64, (3, 3), strides=(1, 1), activation='relu'))
    
    # Flatten and fully connected layers
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))
    
    print(model.summary())
    return model


def main():
    """Main training function."""
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Create environment
    env = create_env()
    
    # Get environment info
    nb_actions = env.action_space.n
    input_shape = env.observation_space.shape
    
    print(f"Number of actions: {nb_actions}")
    print(f"Input shape: {input_shape}")
    
    # Build the model
    model = build_model(input_shape, nb_actions)
    
    # Configure and compile DQN agent
    processor = AtariProcessor()
    memory = SequentialMemory(limit=1000000, window_length=1)
    policy = EpsGreedyQPolicy(eps=1.0)
    
    dqn = DQNAgent(model=model, 
                   nb_actions=nb_actions,
                   memory=memory,
                   nb_steps_warmup=50000,
                   target_model_update=10000,
                   policy=policy,
                   processor=processor,
                   enable_dueling_network=True,
                   enable_double_dqn=True,
                   gamma=0.99)
    
    dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])
    
    # Create callbacks
    callbacks = [
        ModelIntervalCheckpoint('dqn_breakout_weights_{step}.h5f', interval=50000),
        FileLogger('dqn_breakout_log.json', interval=10000)
    ]
    
    # Train the agent
    print("Starting training...")
    dqn.fit(env, 
            nb_steps=1000000,
            visualize=False,
            verbose=1,
            nb_max_episode_steps=10000,
            log_interval=10000,
            callbacks=callbacks)
    
    # Save the final policy network
    print("Saving final model...")
    dqn.model.save('policy.h5')
    dqn.save_weights('dqn_breakout_weights_final.h5f', overwrite=True)
    
    # Test the agent
    print("Testing trained agent...")
    dqn.test(env, nb_episodes=5, visualize=False, nb_max_episode_steps=10000)
    
    env.close()
    print("Training completed!")


if __name__ == '__main__':
    main()
