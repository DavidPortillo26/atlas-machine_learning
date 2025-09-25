#!/usr/bin/env python3
"""
Comprehensive example demonstrating all three algorithms.

This script shows how to use Monte Carlo, TD(λ), and SARSA(λ) algorithms
on the FrozenLake environment with different policies and parameters.
"""

import gymnasium as gym
import numpy as np
import random
from monte_carlo import monte_carlo
from td_lambtha import td_lambtha
from sarsa_lambtha import sarsa_lambtha


def set_seed(env, seed=0):
    """Set reproducible seeds for consistent results."""
    env.reset(seed=seed)
    np.random.seed(seed)
    random.seed(seed)


def create_avoid_holes_policy(env):
    """
    Create a policy that tries to avoid holes while moving toward the goal.

    Args:
        env: FrozenLake environment instance

    Returns:
        policy function that maps state -> action
    """
    def policy(state):
        row, col = state // 8, state % 8
        p = np.random.uniform()

        # Prefer moving right or down toward goal with some randomness
        if p > 0.5:
            # First priority: move right if safe
            if (col < 7 and
                    env.unwrapped.desc[row, col + 1] != b'H'):
                return 2  # RIGHT
            # Second priority: move down if safe
            elif (row < 7 and
                  env.unwrapped.desc[row + 1, col] != b'H'):
                return 1  # DOWN
            # Third priority: move up if safe
            elif (row > 0 and
                  env.unwrapped.desc[row - 1, col] != b'H'):
                return 3  # UP
            else:
                return 0  # LEFT (fallback)
        else:
            # Alternative priority: down first, then right
            if (row < 7 and
                    env.unwrapped.desc[row + 1, col] != b'H'):
                return 1  # DOWN
            elif (col < 7 and
                  env.unwrapped.desc[row, col + 1] != b'H'):
                return 2  # RIGHT
            elif (col > 0 and
                  env.unwrapped.desc[row, col - 1] != b'H'):
                return 0  # LEFT
            else:
                return 3  # UP (fallback)

    return policy


def compare_algorithms():
    """Compare all three algorithms on the FrozenLake environment."""
    print("=" * 60)
    print("TEMPORAL DIFFERENCE LEARNING ALGORITHMS COMPARISON")
    print("=" * 60)

    # Setup environment
    env = gym.make('FrozenLake8x8-v1')
    set_seed(env, 0)

    # Create policy
    policy = create_avoid_holes_policy(env)

    # Initialize value function (holes = -1, others = 1)
    V_init = np.where(env.unwrapped.desc == b'H', -1, 1).reshape(64).astype(
        'float64')

    print(f"Environment: {env.spec.id}")
    print(f"State space: {env.observation_space.n} states")
    print(f"Action space: {env.action_space.n} actions")
    print()

    # Display environment layout
    print("Environment Layout:")
    desc = env.unwrapped.desc
    layout = []
    for row in desc:
        layout.append(' '.join([s.decode('utf-8') for s in row]))
    print('\n'.join(layout))
    print("S=Start, F=Frozen, H=Hole, G=Goal")
    print()

    # 1. Monte Carlo Method
    print("-" * 40)
    print("1. MONTE CARLO VALUE ESTIMATION")
    print("-" * 40)

    V_mc = monte_carlo(env.copy(), V_init.copy(), policy,
                       episodes=5000, gamma=0.99)

    print("Monte Carlo Results (8x8 grid):")
    print(V_mc.reshape(8, 8))
    print(f"Average value: {V_mc[V_mc != -1].mean():.4f}")
    print(f"Value at start state: {V_mc[0]:.4f}")
    print(f"Value at goal state: {V_mc[63]:.4f}")
    print()

    # 2. TD(λ) Algorithm
    print("-" * 40)
    print("2. TD(λ) VALUE ESTIMATION")
    print("-" * 40)

    # Test different lambda values
    lambdas = [0.0, 0.5, 0.9]
    for lam in lambdas:
        V_td = td_lambtha(env.copy(), V_init.copy(), policy,
                          lambtha=lam, episodes=5000, alpha=0.1, gamma=0.99)

        print(f"TD(λ={lam}) Results:")
        print(f"  Average value: {V_td[V_td != -1].mean():.4f}")
        print(f"  Start state value: {V_td[0]:.4f}")
        print(f"  Goal state value: {V_td[63]:.4f}")

    print()
    print("Full TD(λ=0.9) Results (8x8 grid):")
    V_td_best = td_lambtha(env.copy(), V_init.copy(), policy,
                           lambtha=0.9, episodes=5000, alpha=0.1, gamma=0.99)
    print(V_td_best.reshape(8, 8))
    print()

    # 3. SARSA(λ) Algorithm
    print("-" * 40)
    print("3. SARSA(λ) Q-LEARNING")
    print("-" * 40)

    # Initialize Q-table
    Q_init = np.random.uniform(size=(64, 4))
    set_seed(env, 0)  # Reset seed for consistent results

    Q_sarsa = sarsa_lambtha(env.copy(), Q_init.copy(), lambtha=0.9,
                            episodes=10000, alpha=0.1, gamma=0.99,
                            epsilon=1.0, min_epsilon=0.1, epsilon_decay=0.05)

    # Extract value function from Q-table
    V_sarsa = np.max(Q_sarsa, axis=1)

    # Extract policy from Q-table
    policy_sarsa = np.argmax(Q_sarsa, axis=1)

    print("SARSA(λ=0.9) Results:")
    print(f"Average Q-value: {Q_sarsa.mean():.4f}")
    print(f"Average value: {V_sarsa[V_sarsa > -0.5].mean():.4f}")
    print()

    print("Learned Value Function (8x8 grid):")
    print(V_sarsa.reshape(8, 8))
    print()

    print("Learned Policy (8x8 grid):")
    print("Actions: 0=LEFT, 1=DOWN, 2=RIGHT, 3=UP")
    print(policy_sarsa.reshape(8, 8))
    print()

    # 4. Algorithm Comparison
    print("-" * 40)
    print("4. ALGORITHM COMPARISON")
    print("-" * 40)

    print(f"{'Algorithm':<20} {'Avg Value':<12} {'Start Value':<12} {'Goal Value':<12}")
    print("-" * 60)
    print(f"{'Monte Carlo':<20} {V_mc[V_mc != -1].mean():<12.4f} "
          f"{V_mc[0]:<12.4f} {V_mc[63]:<12.4f}")
    print(f"{'TD(λ=0.9)':<20} {V_td_best[V_td_best != -1].mean():<12.4f} "
          f"{V_td_best[0]:<12.4f} {V_td_best[63]:<12.4f}")
    print(f"{'SARSA(λ=0.9)':<20} {V_sarsa[V_sarsa > -0.5].mean():<12.4f} "
          f"{V_sarsa[0]:<12.4f} {V_sarsa[63]:<12.4f}")

    print("\nKey Observations:")
    print("- Monte Carlo: Unbiased but high variance")
    print("- TD(λ): Lower variance, faster convergence")
    print("- SARSA(λ): Learns both values and policy, includes exploration")


def lambda_sensitivity_analysis():
    """Analyze the effect of lambda parameter on learning."""
    print("\n" + "=" * 60)
    print("LAMBDA SENSITIVITY ANALYSIS")
    print("=" * 60)

    env = gym.make('FrozenLake8x8-v1')
    policy = create_avoid_holes_policy(env)
    V_init = np.where(env.unwrapped.desc == b'H', -1, 1).reshape(64).astype(
        'float64')

    lambdas = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

    print(f"{'Lambda':<10} {'Avg Value':<12} {'Start Value':<12} {'Convergence':<12}")
    print("-" * 50)

    for lam in lambdas:
        set_seed(env, 0)
        V_td = td_lambtha(env.copy(), V_init.copy(), policy,
                          lambtha=lam, episodes=3000, alpha=0.1, gamma=0.99)

        # Measure convergence by looking at value spread
        non_hole_values = V_td[V_td != -1]
        convergence = np.std(non_hole_values)

        print(f"{lam:<10.1f} {non_hole_values.mean():<12.4f} "
              f"{V_td[0]:<12.4f} {convergence:<12.4f}")

    print("\nInterpretation:")
    print("- λ=0: Pure TD(0), fast but potentially biased")
    print("- λ=1: Monte Carlo-like, unbiased but slow")
    print("- 0<λ<1: Balance between bias and variance")


if __name__ == "__main__":
    # Set print options for better display
    np.set_printoptions(precision=4, suppress=True)

    # Run comprehensive comparison
    compare_algorithms()

    # Run lambda sensitivity analysis
    lambda_sensitivity_analysis()

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)