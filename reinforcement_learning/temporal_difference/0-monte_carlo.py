#!/usr/bin/env python3
"""Monte Carlo algorithm for value estimation"""

import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm for value estimation
    
    Args:
        env: environment instance
        V: numpy.ndarray of shape (s,) containing the value estimate
        policy: function that takes in a state and returns the next action to take
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate
    
    Returns:
        V: the updated value estimate
    """
    # Work with a copy of V to avoid modifying the original
    V = V.copy()
    
    for episode in range(episodes):
        # Generate an episode
        states = []
        rewards = []
        
        # Reset environment to get initial state
        state, _ = env.reset()
        
        # Run episode until termination or max_steps
        for step in range(max_steps):
            # Store current state
            states.append(state)
            
            # Get action from policy
            action = policy(state)
            
            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Store the reward received
            rewards.append(reward)
            
            # Update to next state
            state = next_state
            
            # Check if episode is finished
            if terminated or truncated:
                break
        
        # First-visit Monte Carlo: calculate returns and update values
        G = 0
        visited = set()
        
        # Process episode backwards to calculate returns
        for t in range(len(states) - 1, -1, -1):
            # Calculate return: G_t = R_{t+1} + gamma * G_{t+1}
            G = rewards[t] + gamma * G
            
            state_t = states[t]
            
            # Only update if this is the first visit to this state in this episode
            if state_t not in visited:
                visited.add(state_t)
                # Incremental update: V(s) = V(s) + alpha * (G - V(s))
                V[state_t] = V[state_t] + alpha * (G - V[state_t])
    
    return V
