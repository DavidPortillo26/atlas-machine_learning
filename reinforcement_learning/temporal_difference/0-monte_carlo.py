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
    # Initialize visit counts for each state
    N = np.zeros(V.shape)
    
    for episode in range(episodes):
        # Generate an episode
        states = []
        rewards = []
        
        # Reset environment
        state, _ = env.reset()
        
        # Run episode
        for step in range(max_steps):
            # Store current state
            states.append(state)
            
            # Get action from policy
            action = policy(state)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Store reward
            rewards.append(reward)
            
            # Update state
            state = next_state
            
            # Check if episode is done
            if terminated or truncated:
                break
        
        # Calculate returns for first-visit Monte Carlo
        G = 0
        visited = set()
        
        # Work backwards through the episode
        for t in reversed(range(len(states))):
            state_t = states[t]
            
            # Calculate return
            G = rewards[t] + gamma * G
            
            # First-visit Monte Carlo: only update if this is first visit to state in episode
            if state_t not in visited:
                visited.add(state_t)
                N[state_t] += 1
                
                # Incremental update of value function
                V[state_t] = V[state_t] + alpha * (G - V[state_t])
    
    return V
