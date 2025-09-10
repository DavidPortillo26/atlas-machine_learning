#!/usr/bin/env python3
import gymnasium as gym
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Permute
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy


# Wrappers for gymnasium compatibility with keras-rl
class AtariProcessor(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)


def build_model(actions, input_shape):
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=input_shape))  # channels last
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation="relu"))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation="relu"))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dense(actions, activation="linear"))
    return model


def main():
    env = gym.make("ALE/Breakout-v5", render_mode="human", frameskip=1)
    env = AtariProcessor(env)

    nb_actions = env.action_space.n
    obs_shape = (1,) + env.observation_space.shape

    model = build_model(nb_actions, obs_shape)

    memory = SequentialMemory(limit=1000000, window_length=1)
    policy = GreedyQPolicy()

    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        memory=memory,
        nb_steps_warmup=0,
        target_model_update=10000,
        policy=policy,
    )
    dqn.compile(Adam(learning_rate=0.00025), metrics=["mae"])

    # Load weights from training
    dqn.load_weights("policy.h5")

    # Play 5 episodes
    dqn.test(env, nb_episodes=5, visualize=True)
    env.close()


if __name__ == "__main__":
    main()
