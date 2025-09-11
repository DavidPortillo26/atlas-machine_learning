# utils/callbacks.py
from rl.callbacks import Callback

class EpisodicTargetNetworkUpdate(Callback):
    """
    Update the agent's target network every `update_frequency` episodes.

    Usage:
        callback = EpisodicTargetNetworkUpdate(update_frequency=30, verbose=1)
        dqn.fit(..., callbacks=[callback])
    """#!/usr/bin/env python3
"""
Module containing custom callbacks for DQN training.
"""
from rl.callbacks import Callback


class EpisodicTargetNetworkUpdate(Callback):
    """
    Custom callback to update the target network after a specific number of episodes.
    This overrides the default step-based update mechanism in DQNAgent.
    """
    def __init__(self, update_frequency=10, verbose=0):
        """
        Args:
            update_frequency: Number of episodes between target network updates
            verbose: Verbosity level (0=silent, 1=progress bar, 2=one line per epoch)
        """
        super(EpisodicTargetNetworkUpdate, self).__init__()
        self.update_frequency = update_frequency
        self.episodes_since_update = 0
        self.verbose = verbose
        
    def on_episode_end(self, episode, logs={}):
        """Called at the end of each episode."""
        self.episodes_since_update += 1
        
        # Check if it's time to update the target network
        if self.episodes_since_update >= self.update_frequency:
            # Update target network by manually copying weights
            # In keras-rl2, we need to directly access and update the target model weights
            target_weights = self.model.target_model.get_weights()
            online_weights = self.model.model.get_weights()
            
            # Manual update
            for i in range(len(target_weights)):
                target_weights[i] = online_weights[i]
                
            # Set the updated weights
            self.model.target_model.set_weights(target_weights)
            
            # Also update reward scaler if processor has one
            if hasattr(self.model.processor, 'reward_scaler'):
                self.model.processor.reward_scaler.reset_on_target_update()
            
            # Reset counter
            self.episodes_since_update = 0
            
            if self.verbose >= 1:
                print(f"\nTarget network updated after {self.update_frequency} episodes")

    def __init__(self, update_frequency=1, verbose=0):
        super().__init__()
        self.update_frequency = int(update_frequency)
        self.verbose = int(verbose)
        self.episode_counter = 0

    def on_episode_end(self, episode, logs=None):
        self.episode_counter += 1
        if (self.episode_counter % self.update_frequency) == 0:
            # self.model is the agent (keras-rl sets this)
            try:
                # DQNAgent provides update_target_model() method
                self.model.update_target_model()
                if self.verbose:
                    print(f"[EpisodicTargetNetworkUpdate] target_model updated (episode {self.episode_counter})")
            except Exception as e:
                if self.verbose:
                    print(f"[EpisodicTargetNetworkUpdate] failed to update target model: {e}")
