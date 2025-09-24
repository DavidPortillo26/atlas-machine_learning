#!/usr/bin/env python3
"""
Module: 1-td_lambtha.py
Purpose: Implement TD(λ) algorithm for value function estimation.

TD(λ) combines the benefits of Temporal Difference learning with eligibility traces
to allow credit assignment over multiple time steps. The algorithm maintains
eligibility traces that decay over time, enabling more effective learning
from sequences of rewards.

Key features:
- Uses accumulating eligibility traces
- Updates all states proportionally to their eligibility
- Handles terminal states appropriately
- Supports configurable lambda decay parameter
"""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000,
               max_steps=100, alpha=0.1, gamma=0.99):
    """
    Perform value function estimation using the TD(λ) algorithm.

    TD(λ) uses eligibility traces to distribute credit from rewards back through
    the sequence of states that led to those rewards. This allows for faster
    learning compared to basic TD(0) methods.

    Args:
        env: Gymnasium environment instance (FrozenLake8x8-v1)
        V (np.ndarray): Initial value estimates for each state, shape (s,)
        policy (callable): Function that takes state (int) and returns action (int)
        lambtha (float): Eligibility trace decay factor (λ), typically 0 ≤ λ ≤ 1
        episodes (int): Total number of training episodes (default 5000)
        max_steps (int): Maximum steps per episode (default 100)
        alpha (float): Learning rate for value updates (default 0.1)
        gamma (float): Discount factor for future rewards (default 0.99)

    Returns:
        np.ndarray: Updated value function estimates after training

    Algorithm:
        1. Initialize eligibility traces E = 0 for all states
        2. For each episode:
           a. Reset environment and eligibility traces
           b. For each step until terminal or max_steps:
              - Take action according to policy
              - Calculate TD error: δ = r + γV(s') - V(s)
              - Update eligibility: E(s) += 1 (accumulating traces)
              - Update all values: V += α * δ * E
              - Decay eligibility: E *= γλ
    """
    n_states = V.shape[0]

    for _ in range(episodes):
        state, _ = env.reset()
        # Eligibility traces for states
        E = np.zeros(n_states)

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # TD error
            td_target = reward + gamma * V[next_state] * (not done)
            td_error = td_target - V[state]

            # Update eligibility trace for the current state
            E[state] += 1

            # Update all states' values
            V += alpha * td_error * E

            # Decay traces
            E *= gamma * lambtha

            state = next_state
            if done:
                break

    return V
