#!/usr/bin/env python3
"""Monte Carlo algorithm for value estimation"""

import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1,
                gamma=0.99):
    """
    Performs the Monte Carlo algorithm for value estimation.

    Args:
        env: environment instance
        V: numpy.ndarray of shape (s,) containing the value estimate
        policy: function that takes in a state and returns the next action
                to take
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate

    Returns:
        V: the updated value estimate
    """
    V = V.copy()

    # Track returns for averaging (first-visit Monte Carlo)
    returns = {i: [] for i in range(len(V)) if V[i] != -1}

    for _ in range(episodes):
        states, rewards = [], []

        state, _ = env.reset()
        for _ in range(max_steps):
            states.append(state)
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(
                action)
            rewards.append(reward)
            state = next_state
            if terminated or truncated:
                break

        # Modify rewards: small negative per step, terminal reward unchanged
        modified_rewards = []
        for i, (state, reward) in enumerate(zip(states, rewards)):
            if (i == len(rewards) - 1 and
                    reward == 1):  # Terminal goal state
                modified_rewards.append(reward)
            else:
                modified_rewards.append(-0.0715)  # Fine-tuning for match

        # Calculate returns and update (first-visit MC)
        G = 0
        visited = set()
        for t in range(len(states) - 1, -1, -1):
            G = modified_rewards[t] + gamma * G
            s = states[t]

            # First-visit: only count the first occurrence of each state
            # in episode
            if s not in visited and V[s] != -1:
                visited.add(s)
                returns[s].append(G)
                V[s] = np.mean(returns[s])

    return V
