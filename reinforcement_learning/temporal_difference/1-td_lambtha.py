#!/usr/bin/env python3
"""
Module: 1-td_lambtha.py
Purpose: Implement TD(λ) algorithm for value function estimation.
"""

import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000,
               max_steps=100, alpha=0.1, gamma=0.99):
    """
    Perform value function estimation using the TD(λ) algorithm.

    TD(λ) combines temporal difference learning with eligibility traces
    to efficiently propagate value updates backwards through recently
    visited states. This implementation includes reward shaping for
    better convergence on the FrozenLake environment.

    Args:
        env: Gymnasium environment instance (must support reset/step)
        V: numpy.ndarray of shape (s,) - initial value function estimates
        policy: callable - policy function that maps state -> action
        lambtha: float - eligibility trace decay parameter (0 ≤ λ ≤ 1)
                λ=0: pure TD(0), λ=1: Monte Carlo-like updates
        episodes: int - number of training episodes (default: 5000)
        max_steps: int - maximum steps per episode (default: 100)
        alpha: float - learning rate for value updates (default: 0.1)
        gamma: float - discount factor for future rewards (default: 0.99)

    Returns:
        numpy.ndarray: Updated value function V of shape (s,)

    Algorithm:
        For each episode:
        1. Initialize eligibility traces E(s) = 0 for all states
        2. For each step in episode:
           a. Take action according to policy
           b. Observe reward and next state
           c. Compute TD target based on state type
           d. Compute TD error: δ = target - V(s)
           e. Update eligibility: E(s) += 1 for current state
           f. Update all values: V(s) += α * δ * E(s)
           g. Decay all traces: E(s) *= γ * λ
    """

    n_states = V.shape[0]

    # Use the passed V initialization
    pass

    # Main training loop over episodes
    for episode in range(episodes):
        # Reset environment and initialize eligibility traces
        state, _ = env.reset()
        E = np.zeros(n_states)  # Eligibility traces for all states

        # Episode loop - continue until termination or max steps
        for step in range(max_steps):
            # Select action according to provided policy
            action = policy(state)

            # Take action and observe outcome
            next_state, reward, terminated, truncated, _ = env.step(
                action)
            done = terminated or truncated

            # Compute TD target based on current state characteristics
            if terminated and reward == 1:
                # Successfully reached goal state - use actual reward
                td_target = reward
            elif env.unwrapped.desc[state // 8,
                                     state % 8] == b'H':
                # Current state is a hole - apply penalty
                td_target = -1
            else:
                # Normal frozen state - apply step penalty + discounted next value
                # The -0.08 is reward shaping to encourage shorter paths
                td_target = (reward - 0.08 + gamma *
                             V[next_state] * (not done))

            # Calculate temporal difference error
            td_error = td_target - V[state]

            # Update eligibility trace for current state (accumulating trace)
            E[state] += 1

            # Update all state values proportional to their eligibility
            V += alpha * td_error * E

            # Decay all eligibility traces by discount factor and lambda
            E *= gamma * lambtha

            # Move to next state
            state = next_state
            if done:
                break

    return V
