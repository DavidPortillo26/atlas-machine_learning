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
    # Make a copy of V to avoid modifying the original
    V = V.copy()
    
    for episode in range(episodes):
        # Generate an episode
        states = []
        rewards = []
        
        # Reset environment
        state, _ = env.reset()
        
        # Run episode
        done = False
        for step in range(max_steps):
            if done:
                break
                
            # Store current state
            states.append(state)
            
            # Get action from policy
            action = policy(state)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Store reward received after taking action in current state
            rewards.append(reward)
            
            # Update state
            state = next_state
            
            # Check if episode is done
            done = terminated or truncated
        
        # Calculate returns using first-visit Monte Carlo
        G = 0
        visited = set()
        
        # Work backwards through the episode
        for t in reversed(range(len(states))):
            # Update return: G = reward + gamma * G
            G = rewards[t] + gamma * G
            
            state_t = states[t]
            
            # First-visit: only update if we haven't seen this state before in this episode
            if state_t not in visited:
                visited.add(state_t)
                # Update value function with incremental average
                V[state_t] = V[state_t] + alpha * (G - V[state_t])
    
    return V
