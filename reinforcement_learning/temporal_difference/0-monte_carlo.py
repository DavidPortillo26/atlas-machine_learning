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
    # Work with a copy of V
    V = V.copy()
    
    for episode in range(episodes):
        # Lists to store episode data
        states = []
        rewards = []
        
        # Reset environment and get initial state
        state, _ = env.reset()
        
        # Generate episode
        for step in range(max_steps):
            # Store current state
            states.append(state)
            
            # Get action from policy
            action = policy(state)
            
            # Take step in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Store reward received
            rewards.append(reward)
            
            # Move to next state
            state = next_state
            
            # Check if episode finished
            if terminated or truncated:
                break
        
        # Calculate returns using first-visit Monte Carlo
        G = 0
        visited = set()
        
        # Work backwards through episode
        for t in range(len(states) - 1, -1, -1):
            # Calculate discounted return
            G = rewards[t] + gamma * G
            
            state_t = states[t]
            
            # Only update on first visit to state in this episode
            if state_t not in visited:
                visited.add(state_t)
                
                # Incremental update: V(s) ← V(s) + α[G - V(s)]
                V[state_t] += alpha * (G - V[state_t])
    
    return V
