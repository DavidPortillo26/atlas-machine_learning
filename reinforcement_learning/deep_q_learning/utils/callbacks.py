# utils/callbacks.py
from rl.callbacks import Callback

class EpisodicTargetNetworkUpdate(Callback):
    """
    Update the agent's target network every `update_frequency` episodes.

    Usage:
        callback = EpisodicTargetNetworkUpdate(update_frequency=30, verbose=1)
        dqn.fit(..., callbacks=[callback])
    """

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
