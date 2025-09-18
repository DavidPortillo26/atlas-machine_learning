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
    # Work with a copy to avoid modifying original
    V = V.copy()
    
    # Keep track of returns for each state
    returns = {i: [] for i in range(len(V))}
    
    for episode in range(episodes):
        # Generate episode
        episode_states = []
        episode_rewards = []
        
        # Reset environment
        state, _ = env.reset()
        
        # Generate episode trajectory
        for step in range(max_steps):
            episode_states.append(state)
            
            # Get action from policy
            action = policy(state)
            
            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_rewards.append(reward)
            
            state = next_state
            
            if terminated or truncated:
                break
        
        # Calculate returns and update values (First-visit Monte Carlo)
        G = 0
        visited_states_in_episode = set()
        
        # Process episode backwards
        for t in reversed(range(len(episode_states))):
            G = episode_rewards[t] + gamma * G
            state_t = episode_states[t]
            
            # First-visit Monte Carlo
            if state_t not in visited_states_in_episode:
                visited_states_in_episode.add(state_t)
                returns[state_t].append(G)
                # Update value as average of returns
                V[state_t] = np.mean(returns[state_t])
    
    return V
