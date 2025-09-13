#!/usr/bin/env python3
"""
Play script for Atari Breakout using trained DQN agent with keras-rl2 and gymnasium
"""

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
try:
    from gymnasium.wrappers import FrameStack
except ImportError:
    from gymnasium.wrappers.frame_stack import FrameStack
import tensorflow as tf
try:
    # Try TensorFlow 2.x imports first
    from tensorflow.keras.models import load_model
except ImportError:
    # Fallback to standalone Keras
    from keras.models import load_model
from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
import time


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


def create_env(render_mode='human'):
    """Create and configure the Atari Breakout environment."""
    # Create the base environment with human rendering
    env = gym.make('ALE/Breakout-v5', render_mode=render_mode)
    
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


def load_trained_agent(model_path='policy.h5'):
    """Load the trained DQN agent."""
    try:
        # Load the saved model
        model = load_model(model_path)
        print(f"Successfully loaded model from {model_path}")
        print(model.summary())
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def play_game(agent, env, nb_episodes=5, delay=0.02):
    """Play the game with the trained agent."""
    total_reward = 0
    
    for episode in range(nb_episodes):
        print(f"\nEpisode {episode + 1}/{nb_episodes}")
        
        # Reset environment
        observation, info = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        step_count = 0
        
        while not done and not truncated:
            # Render the environment
            env.render()
            time.sleep(delay)  # Add small delay for better visualization
            
            # Get action from agent
            action = agent.forward(observation)
            
            # Take action in environment
            observation, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            
            # Print progress periodically
            if step_count % 100 == 0:
                print(f"  Step: {step_count}, Episode Reward: {episode_reward}")
        
        print(f"Episode {episode + 1} finished:")
        print(f"  Total Steps: {step_count}")
        print(f"  Episode Reward: {episode_reward}")
        print(f"  Final Info: {info}")
        
        total_reward += episode_reward
        
        # Brief pause between episodes
        time.sleep(1)
    
    average_reward = total_reward / nb_episodes
    print(f"\nAverage reward over {nb_episodes} episodes: {average_reward:.2f}")


def main():
    """Main function to play the game."""
    print("Loading trained DQN agent for Atari Breakout...")
    
    # Create environment
    env = create_env(render_mode='human')
    
    # Get environment info
    nb_actions = env.action_space.n
    input_shape = env.observation_space.shape
    
    print(f"Number of actions: {nb_actions}")
    print(f"Input shape: {input_shape}")
    
    # Load the trained model
    model = load_trained_agent('policy.h5')
    if model is None:
        print("Failed to load model. Please ensure 'policy.h5' exists and was created by train.py")
        return
    
    # Create DQN agent with greedy policy
    processor = AtariProcessor()
    memory = SequentialMemory(limit=1000, window_length=1)  # Small memory for inference
    policy = GreedyQPolicy()  # Use greedy policy for playing (no exploration)
    
    dqn = DQNAgent(model=model,
                   nb_actions=nb_actions,
                   memory=memory,
                   nb_steps_warmup=0,
                   target_model_update=1,
                   policy=policy,
                   processor=processor)
    
    # Compile agent (required even for inference)
    dqn.compile(tf.keras.optimizers.Adam(learning_rate=0.00025), metrics=['mae'])
    
    print("Starting game visualization...")
    print("Press Ctrl+C to stop")
    
    try:
        # Play the game
        play_game(dqn, env, nb_episodes=5, delay=0.02)
    except KeyboardInterrupt:
        print("\nGame interrupted by user")
    finally:
        env.close()
        print("Game session ended")


if __name__ == '__main__':
    main()
