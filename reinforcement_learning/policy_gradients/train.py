#!/usr/bin/env python3
"""
Policy Gradient Training Module

This module implements the complete REINFORCE (Monte-Carlo Policy Gradient)
training algorithm for reinforcement learning. REINFORCE is a policy gradient
method that learns a parameterized policy by directly optimizing the expected
cumulative reward.

The algorithm follows these key steps:
1. Initialize policy parameters (weights) randomly
2. For each episode:
   - Generate a complete episode using current policy
   - Calculate returns (discounted cumulative rewards)
   - Compute policy gradients for each step
   - Update policy parameters using gradient ascent

Mathematical Foundation:
    The policy gradient theorem shows that:
    ∇_θ J(θ) = E_τ[Σ_t ∇_θ log π_θ(a_t|s_t) * R_t]

    where:
    - θ represents policy parameters (weights)
    - J(θ) is the expected cumulative reward
    - τ represents a trajectory (episode)
    - R_t is the return from time step t
    - π_θ(a_t|s_t) is the policy (probability of action a_t in state s_t)

Author: Generated for reinforcement learning educational purposes
Date: 2024
"""

import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """
    Implements a full training loop using the REINFORCE policy gradient method.

    This function trains a neural network policy to maximize expected cumulative
    reward in a given environment using the Monte-Carlo Policy Gradient algorithm.
    The training process involves:

    1. Episode Generation: Run complete episodes using current policy
    2. Return Calculation: Compute discounted cumulative rewards (returns)
    3. Gradient Computation: Calculate policy gradients for each timestep
    4. Parameter Update: Update policy weights using gradient ascent
    5. Repeat until convergence or max episodes reached

    Algorithm Details:
        For each episode t = 1 to T:
            - Generate trajectory τ = (s_0, a_0, r_1, s_1, a_1, r_2, ...)
            - Calculate returns: R_t = Σ_{k=t}^T γ^{k-t} r_k
            - Normalize returns for stable learning
            - Update: θ ← θ + α Σ_t R_t ∇_θ log π_θ(a_t|s_t)

    Args:
        env (gym.Env): OpenAI Gymnasium environment instance (e.g., CartPole-v1)
                      Must have discrete action space and continuous state space
        nb_episodes (int): Number of episodes to train for. Typical range: 1000-10000
                         More episodes = better convergence but longer training
        alpha (float, optional): Learning rate for policy gradient updates.
                               Default: 0.000045. Typical range: 1e-6 to 1e-3
                               Higher values = faster learning but less stable
        gamma (float, optional): Discount factor for future rewards.
                               Default: 0.98. Range: [0, 1]
                               Higher values = more long-term thinking
        show_result (bool, optional): If True, renders the environment every 1000
                                    episodes for visualization. Default: False
                                    Useful for monitoring training progress

    Returns:
        list[float]: List of episode scores (sum of rewards per episode)
                    Length equals nb_episodes. Can be used to plot learning curves
                    and analyze training performance over time

    Example:
        >>> import gymnasium as gym
        >>> env = gym.make('CartPole-v1')
        >>> scores = train(env, nb_episodes=5000, alpha=1e-4, gamma=0.99)
        >>> print(f"Average score: {np.mean(scores[-100:])}")  # Last 100 episodes
        >>> env.close()

    Notes:
        - The function uses return normalization (z-score) for stable learning
        - Weights are initialized randomly from uniform distribution
        - Episode scores typically improve over time if hyperparameters are well-tuned
        - For CartPole-v1, scores of 200+ indicate good performance
        - Training time scales linearly with nb_episodes

    See Also:
        policy_gradient.policy_gradient: Function used to compute policy gradients
        policy_gradient.policy: Function used to compute action probabilities
    """
    # Extract environment dimensions for weight matrix initialization
    # state_dim: dimensionality of state space (e.g., 4 for CartPole)
    # action_dim: number of discrete actions (e.g., 2 for CartPole: left/right)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Initialize policy parameters (weights) randomly
    # Shape: (state_dim, action_dim) - each column represents weights for one action
    # Random initialization helps break symmetry and enables learning
    weight = np.random.rand(state_dim, action_dim)

    # Track episode scores for performance monitoring and learning curve analysis
    scores = []

    # Main training loop: iterate through episodes
    for episode in range(nb_episodes):
        # Reset environment to initial state for new episode
        # The second return value (_) contains additional info we don't need
        state, _ = env.reset()

        # Episode trajectory storage
        states = []      # State sequence: s_0, s_1, s_2, ...
        actions = []     # Action sequence: a_0, a_1, a_2, ...
        rewards = []     # Reward sequence: r_1, r_2, r_3, ...
        gradients = []   # Policy gradient sequence for parameter updates

        # Generate one complete episode using current policy
        done = False
        while not done:
            # Sample action from current policy and get policy gradient
            action, gradient = policy_gradient(state, weight)

            # Execute action in environment and observe results
            next_state, reward, done, truncated, _ = env.step(action)

            # Store trajectory components for later gradient computation
            states.append(state)        # Current state
            actions.append(action)      # Action taken
            rewards.append(reward)      # Reward received
            gradients.append(gradient)  # Policy gradient ∇_θ log π_θ(a|s)

            # Transition to next state
            state = next_state
            # Episode ends if done=True (goal reached) or truncated=True (time limit)
            done = done or truncated

        # Calculate discounted cumulative rewards (returns) using backward iteration
        # Return at time t: R_t = r_{t+1} + γ*r_{t+2} + γ^2*r_{t+3} + ...
        # This implementation is more efficient than forward computation
        returns = []
        G = 0  # Initialize return accumulator
        for reward in reversed(rewards):
            G = reward + gamma * G  # Accumulate discounted reward
            returns.insert(0, G)    # Prepend to maintain chronological order

        # Normalize returns to reduce variance and improve learning stability
        # This is a common technique in policy gradient methods
        returns = np.array(returns)
        if len(returns) > 1:  # Avoid division by zero for single-step episodes
            # Z-score normalization: (x - mean) / std
            # Small epsilon (1e-8) prevents division by zero when std=0
            returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

        # Apply policy gradient updates to weights
        # Update rule: θ ← θ + α * R_t * ∇_θ log π_θ(a_t|s_t)
        # This implements gradient ascent to maximize expected return
        for i, gradient in enumerate(gradients):
            # Multiply gradient by return (discounted reward) and learning rate
            # Higher returns reinforce the actions that led to them
            weight += alpha * returns[i] * gradient

        # Calculate and store episode performance metrics
        episode_score = sum(rewards)  # Total undiscounted reward for this episode
        scores.append(episode_score)

        # Optional visualization: render environment every 1000 episodes
        # Useful for monitoring training progress visually
        if show_result and episode % 1000 == 0:
            env.render()  # Display current environment state

        # Progress tracking: print episode statistics
        # Helps monitor learning progress and debug training issues
        print(f"Episode: {episode} Score: {episode_score}")

    # Return complete training history for analysis
    # Can be used to plot learning curves, compute statistics, etc.
    return scores
