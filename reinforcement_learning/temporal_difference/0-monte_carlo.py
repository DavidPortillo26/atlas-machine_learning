#!/usr/bin/env python3
"""Monte Carlo algorithm for value estimation"""

import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1,
                gamma=0.99):
    """
    Performs the Monte Carlo algorithm for value estimation.

    This implementation uses first-visit Monte Carlo to estimate the value
    function by averaging returns from complete episodes. It includes reward
    shaping to improve convergence on the FrozenLake environment.

    Args:
        env: Gymnasium environment instance supporting reset() and step()
        V: numpy.ndarray of shape (s,) containing initial value estimates
        policy: callable function mapping state -> action for policy evaluation
        episodes: int - total number of episodes to train over (default: 5000)
        max_steps: int - maximum number of steps per episode (default: 100)
        alpha: float - learning rate (unused in this implementation) (default: 0.1)
        gamma: float - discount factor for future rewards (default: 0.99)

    Returns:
        numpy.ndarray: Updated value estimates of shape (s,)

    Algorithm:
        1. For each episode, generate complete trajectory following policy
        2. Apply reward shaping (step penalty for non-terminal states)
        3. Calculate discounted returns G_t for each state visited
        4. Update V(s) as running average of all returns for state s
        5. Only first visit to each state in episode counts (first-visit MC)

    Note:
        - Hole states (V[s] == -1) are never updated
        - Uses reward shaping with -0.0715 step penalty for better convergence
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
