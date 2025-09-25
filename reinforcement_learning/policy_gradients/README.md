# Policy Gradient Reinforcement Learning

A complete implementation of the REINFORCE (Monte-Carlo Policy Gradient) algorithm for reinforcement learning, designed for educational purposes and practical applications.

## 📋 Table of Contents

- [Overview](#overview)
- [Algorithm Details](#algorithm-details)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Documentation](#api-documentation)
- [Examples](#examples)
- [Mathematical Background](#mathematical-background)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Performance Analysis](#performance-analysis)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🎯 Overview

This repository implements the REINFORCE algorithm, a fundamental policy gradient method in reinforcement learning. The implementation is optimized for clarity, educational value, and practical usage with OpenAI Gymnasium environments.

### Key Features

- **Complete REINFORCE implementation** with Monte-Carlo policy gradients
- **Numerical stability** features including softmax numerical tricks and return normalization
- **Comprehensive documentation** with mathematical explanations
- **Visualization support** for training progress monitoring
- **Modular design** for easy extension and experimentation
- **Educational examples** demonstrating key concepts

### Supported Environments

- **CartPole-v1**: Classic control problem (primary test environment)
- **Any discrete action space environment** from OpenAI Gymnasium
- **Continuous state spaces** with finite action spaces

## 🧮 Algorithm Details

### REINFORCE Algorithm

The REINFORCE algorithm learns a parameterized policy by directly optimizing the expected cumulative reward using policy gradients.

**Core Concept**: Instead of learning value functions (like Q-learning), REINFORCE directly learns the policy that maps states to actions.

**Key Advantages**:
- Works with continuous state spaces
- Can handle stochastic policies naturally
- Direct optimization of the objective function
- No need for value function approximation

**Algorithm Steps**:
1. Initialize policy parameters θ (weights) randomly
2. For each episode:
   - Generate complete trajectory using current policy
   - Calculate discounted returns for each timestep
   - Compute policy gradients
   - Update parameters using gradient ascent
3. Repeat until convergence

## 🚀 Installation

### Prerequisites

```bash
# Python 3.7+ required
python --version  # Should be 3.7+

# Install required packages
pip install gymnasium numpy
```

### Optional Dependencies

```bash
# For CartPole visualization
pip install gymnasium[classic_control]

# For plotting (if you want to analyze results)
pip install matplotlib

# For enhanced environments
pip install gymnasium[all]
```

### Repository Setup

```bash
# Clone or download the repository
git clone <repository-url>
cd policy_gradients

# Verify installation
python 0-main.py  # Should run without errors
```

## ⚡ Quick Start

### Basic Usage

```python
#!/usr/bin/env python3
import gymnasium as gym
import numpy as np
from train import train

# Create environment
env = gym.make('CartPole-v1')

# Train the agent
scores = train(
    env=env,
    nb_episodes=5000,
    alpha=0.000045,
    gamma=0.98
)

# Analyze results
print(f"Average score (last 100 episodes): {np.mean(scores[-100:])}")
print(f"Best score: {max(scores)}")

env.close()
```

### Training with Visualization

```python
#!/usr/bin/env python3
import gymnasium as gym
from train import train

# Create environment with human rendering
env = gym.make('CartPole-v1', render_mode="human")

# Train with visualization every 1000 episodes
scores = train(
    env=env,
    nb_episodes=10000,
    alpha=0.000045,
    gamma=0.98,
    show_result=True  # Shows rendering every 1000 episodes
)

env.close()
```

## 📖 API Documentation

### Core Modules

#### `policy_gradient.py`

Contains the core policy gradient functions implementing the mathematical foundations of REINFORCE.

##### `policy(matrix, weight)`

Computes softmax policy probabilities for given states and weights.

**Parameters:**
- `matrix` (np.ndarray): State matrix of shape `(batch_size, state_dim)`
- `weight` (np.ndarray): Weight matrix of shape `(state_dim, action_dim)`

**Returns:**
- `np.ndarray`: Policy probabilities of shape `(batch_size, action_dim)`

**Mathematical Formula:**
```
π(a|s) = exp(s^T * w_a) / Σ_b exp(s^T * w_b)
```

**Example:**
```python
import numpy as np
from policy_gradient import policy

# Single state, 4 dimensions (like CartPole)
state = np.array([[1.0, -0.5, 0.2, 0.8]])
weights = np.random.rand(4, 2)  # 2 actions

probs = policy(state, weights)
print(f"Action probabilities: {probs[0]}")
print(f"Probabilities sum: {np.sum(probs[0])}")  # Should be ~1.0
```

##### `policy_gradient(state, weight)`

Computes the Monte-Carlo policy gradient for a single state.

**Parameters:**
- `state` (np.ndarray): Current state vector, shape `(state_dim,)` or `(1, state_dim)`
- `weight` (np.ndarray): Policy parameters, shape `(state_dim, action_dim)`

**Returns:**
- `tuple`: `(action, gradient)` where:
  - `action` (int): Selected action sampled from policy
  - `gradient` (np.ndarray): Policy gradient ∇_θ log π_θ(a|s), shape `(state_dim, action_dim)`

**Mathematical Formula:**
```
∇_θ log π_θ(a|s) = s ⊗ (e_a - π(s))
```
where `e_a` is one-hot encoding of action `a`, and `⊗` is outer product.

**Example:**
```python
import numpy as np
from policy_gradient import policy_gradient

state = np.array([1.0, -0.5, 0.2, 0.8])
weights = np.random.rand(4, 2)

action, gradient = policy_gradient(state, weights)
print(f"Selected action: {action}")
print(f"Gradient shape: {gradient.shape}")  # (4, 2)
```

#### `train.py`

Implements the complete REINFORCE training algorithm.

##### `train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False)`

Trains a policy using the REINFORCE algorithm.

**Parameters:**
- `env` (gym.Env): OpenAI Gymnasium environment
- `nb_episodes` (int): Number of training episodes
- `alpha` (float, optional): Learning rate, default `0.000045`
- `gamma` (float, optional): Discount factor, default `0.98`
- `show_result` (bool, optional): Render every 1000 episodes, default `False`

**Returns:**
- `list[float]`: Episode scores (sum of rewards per episode)

**Algorithm Flow:**
1. Initialize random weights
2. For each episode:
   - Generate trajectory using current policy
   - Calculate discounted returns
   - Normalize returns (for stability)
   - Update weights using policy gradients
3. Return training history

**Example:**
```python
import gymnasium as gym
from train import train

env = gym.make('CartPole-v1')

# Basic training
scores = train(env, nb_episodes=1000)

# Advanced training with custom parameters
scores = train(
    env=env,
    nb_episodes=5000,
    alpha=1e-4,        # Learning rate
    gamma=0.99,        # Discount factor
    show_result=True   # Visualization
)

print(f"Training completed. Final score: {scores[-1]}")
env.close()
```

### Test Scripts

The repository includes several test scripts demonstrating different aspects:

- **`0-main.py`**: Tests the `policy()` function with predefined inputs
- **`1-main.py`**: Tests the `policy_gradient()` function with random weights
- **`2-main.py`**: Demonstrates basic training with few episodes
- **`3-main.py`**: Full training example with visualization

## 🎮 Examples

### Example 1: Understanding Policy Gradients

```python
#!/usr/bin/env python3
"""
Demonstrate how policy gradients work step by step
"""
import numpy as np
import gymnasium as gym
from policy_gradient import policy, policy_gradient

# Create environment
env = gym.make('CartPole-v1')
state, _ = env.reset()

# Initialize random weights
np.random.seed(42)  # For reproducibility
weights = np.random.rand(4, 2)

print("=== Policy Gradient Demonstration ===")
print(f"Initial state: {state}")
print(f"Weight matrix shape: {weights.shape}")

# Compute policy probabilities
probs = policy(state.reshape(1, -1), weights)
print(f"Action probabilities: {probs[0]}")

# Sample action and compute gradient
action, gradient = policy_gradient(state, weights)
print(f"Sampled action: {action}")
print(f"Policy gradient shape: {gradient.shape}")

env.close()
```

### Example 2: Training Progress Monitoring

```python
#!/usr/bin/env python3
"""
Train an agent and monitor progress with statistics
"""
import gymnasium as gym
import numpy as np
from train import train

def analyze_training(scores, window=100):
    """Analyze training progress"""
    if len(scores) < window:
        window = len(scores)

    recent_avg = np.mean(scores[-window:])
    overall_avg = np.mean(scores)
    best_score = max(scores)
    worst_score = min(scores)

    print(f"=== Training Analysis ===")
    print(f"Total episodes: {len(scores)}")
    print(f"Recent average ({window} episodes): {recent_avg:.2f}")
    print(f"Overall average: {overall_avg:.2f}")
    print(f"Best score: {best_score:.2f}")
    print(f"Worst score: {worst_score:.2f}")

    # Check for learning (improvement over time)
    first_half = np.mean(scores[:len(scores)//2])
    second_half = np.mean(scores[len(scores)//2:])
    improvement = second_half - first_half

    print(f"Improvement (2nd half vs 1st half): {improvement:.2f}")
    print("Status:", "Learning!" if improvement > 10 else "Needs tuning")

# Train agent
env = gym.make('CartPole-v1')
print("Starting training...")

scores = train(
    env=env,
    nb_episodes=3000,
    alpha=0.0001,
    gamma=0.99
)

# Analyze results
analyze_training(scores)

env.close()
```

### Example 3: Hyperparameter Comparison

```python
#!/usr/bin/env python3
"""
Compare different hyperparameter settings
"""
import gymnasium as gym
import numpy as np
from train import train

def compare_hyperparameters():
    """Compare different learning rates"""

    # Different learning rates to test
    learning_rates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    results = {}

    for lr in learning_rates:
        print(f"\n=== Testing learning rate: {lr} ===")

        env = gym.make('CartPole-v1')

        # Set seed for fair comparison
        np.random.seed(42)
        env.reset(seed=42)

        scores = train(
            env=env,
            nb_episodes=1000,
            alpha=lr,
            gamma=0.98
        )

        # Store results
        final_performance = np.mean(scores[-100:])  # Average of last 100 episodes
        results[lr] = final_performance

        print(f"Final performance: {final_performance:.2f}")
        env.close()

    # Print comparison
    print("\n=== Hyperparameter Comparison ===")
    for lr, performance in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"α = {lr:>8}: {performance:>6.2f}")

    best_lr = max(results.items(), key=lambda x: x[1])
    print(f"\nBest learning rate: {best_lr[0]} (score: {best_lr[1]:.2f})")

if __name__ == "__main__":
    compare_hyperparameters()
```

## 🧮 Mathematical Background

### Policy Gradient Theorem

The foundation of REINFORCE is the policy gradient theorem, which provides an unbiased estimate of the gradient of the expected return with respect to policy parameters.

**Theorem**: For a parameterized policy π_θ(a|s), the gradient of the expected return J(θ) is:

```
∇_θ J(θ) = E_τ[Σ_t ∇_θ log π_θ(a_t|s_t) * R_t]
```

Where:
- `τ` represents a trajectory (episode)
- `R_t` is the return (discounted cumulative reward) from time step t
- `∇_θ log π_θ(a_t|s_t)` is the policy gradient (score function)

### Softmax Policy

We use a softmax policy to ensure valid probability distributions:

```
π_θ(a|s) = exp(s^T * θ_a) / Σ_{a'} exp(s^T * θ_{a'})
```

The gradient of the log-policy is:

```
∇_θ log π_θ(a|s) = s ⊗ (e_a - π_θ(s))
```

Where `e_a` is the one-hot encoding of action `a`.

### Return Calculation

Returns are calculated using the discounted sum of future rewards:

```
R_t = Σ_{k=t}^T γ^{k-t} * r_k
```

Where:
- `γ` (gamma) is the discount factor ∈ [0,1]
- `r_k` is the reward at time step k
- `T` is the episode termination time

### Variance Reduction

To reduce gradient variance and improve learning stability, we normalize returns:

```
R_normalized = (R - μ_R) / σ_R
```

This is a common technique in policy gradient methods.

## ⚙️ Hyperparameter Tuning

### Learning Rate (α)

**Range**: `1e-6` to `1e-3`
**Default**: `0.000045`

- **Too high**: Unstable learning, policy may oscillate or diverge
- **Too low**: Very slow learning, may not converge in reasonable time
- **Sweet spot**: Usually around `1e-5` to `1e-4` for most environments

**Tuning tips**:
- Start with `1e-4` and adjust based on performance
- If scores are unstable, reduce learning rate
- If learning is too slow, gradually increase

### Discount Factor (γ)

**Range**: `0.9` to `0.999`
**Default**: `0.98`

- **Lower values** (`0.9-0.95`): Agent focuses on immediate rewards
- **Higher values** (`0.98-0.999`): Agent considers long-term consequences
- **For CartPole**: `0.98-0.99` works well

### Episode Count

**Range**: `1000` to `50000`
**Recommended**: `5000-10000` for CartPole

- **Too few**: Agent doesn't have enough experience to learn
- **Too many**: Diminishing returns, longer training time
- **Monitor**: Plot scores to see when learning plateaus

### Environment-Specific Tips

#### CartPole-v1
- **Target score**: 200+ (maximum possible per episode)
- **Learning rate**: `1e-4` to `1e-5`
- **Episodes**: `3000-5000` usually sufficient
- **Discount**: `0.98-0.99`

#### Other Environments
- **Continuous control**: Lower learning rates (`1e-6` to `1e-5`)
- **High-dimensional spaces**: More episodes may be needed
- **Sparse rewards**: Higher discount factors help

## 📊 Performance Analysis

### Learning Curves

Track these metrics to evaluate training:

1. **Episode Scores**: Raw rewards per episode
2. **Moving Average**: Smoothed performance over window (e.g., 100 episodes)
3. **Best Score**: Peak performance achieved
4. **Convergence**: When performance stabilizes

### Expected Performance

#### CartPole-v1 Benchmarks
- **Random policy**: ~20-30 score
- **Decent learning**: 150+ score consistently
- **Good performance**: 180+ score
- **Excellent**: 200 score (max possible)

### Diagnostic Plots

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_training_progress(scores, window=100):
    """Plot training progress with moving average"""

    # Calculate moving average
    moving_avg = []
    for i in range(len(scores)):
        start_idx = max(0, i - window + 1)
        moving_avg.append(np.mean(scores[start_idx:i+1]))

    plt.figure(figsize=(12, 5))

    # Plot 1: Raw scores and moving average
    plt.subplot(1, 2, 1)
    plt.plot(scores, alpha=0.3, color='blue', label='Episode Scores')
    plt.plot(moving_avg, color='red', linewidth=2, label=f'Moving Average ({window})')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Score distribution
    plt.subplot(1, 2, 2)
    plt.hist(scores, bins=30, alpha=0.7, color='green')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title('Score Distribution')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# Usage after training
# plot_training_progress(scores)
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. Agent Not Learning (Flat Learning Curve)

**Symptoms**:
- Scores remain around random policy level
- No improvement over episodes
- Very flat learning curve

**Solutions**:
- **Increase learning rate**: Try `1e-4` or `5e-4`
- **Check environment**: Ensure rewards are meaningful
- **Verify implementation**: Run test scripts to check correctness
- **More episodes**: Some environments need more experience

#### 2. Unstable Learning (High Variance)

**Symptoms**:
- Scores fluctuate wildly
- Performance improves then degrades
- High variance in episode scores

**Solutions**:
- **Decrease learning rate**: Try `1e-5` or smaller
- **Return normalization**: Ensure it's enabled (should be by default)
- **Longer episodes**: Check if episodes are terminating too early
- **Environment stochasticity**: Some environments are naturally noisy

#### 3. Learning Then Forgetting (Catastrophic Forgetting)

**Symptoms**:
- Good performance early, then degradation
- Agent "forgets" good strategies

**Solutions**:
- **Lower learning rate**: Prevent overwriting good policies
- **Learning rate decay**: Reduce learning rate over time
- **Early stopping**: Stop training when performance peaks

#### 4. Very Slow Learning

**Symptoms**:
- Learning is happening but very slowly
- Need many episodes to see improvement

**Solutions**:
- **Increase learning rate**: Carefully, monitor stability
- **Check discount factor**: Ensure it's appropriate for problem
- **Baseline subtraction**: Consider implementing baseline for variance reduction

#### 5. Implementation Errors

**Common bugs**:
- **Shape mismatches**: Check array dimensions in matrix operations
- **Gradient calculation**: Verify policy gradient implementation
- **Return calculation**: Ensure proper discounting and normalization
- **Action sampling**: Make sure actions are sampled correctly

**Debugging steps**:
1. Run `0-main.py` - tests basic policy function
2. Run `1-main.py` - tests policy gradient computation
3. Run `2-main.py` - tests short training loop
4. Check array shapes at each step
5. Print intermediate values for verification

### Performance Optimization

#### For Faster Training
- **Vectorized operations**: Ensure all computations use NumPy vectorization
- **Reduce episode length**: For testing, use shorter episodes
- **Parallel environments**: Consider running multiple environments (advanced)

#### For Better Results
- **Baseline methods**: Implement actor-critic for variance reduction
- **Experience replay**: Store and reuse trajectories (advanced)
- **Parameter scheduling**: Decay learning rate over time

## 🤝 Contributing

### Code Style
- Follow PEP 8 guidelines
- Include comprehensive docstrings
- Add type hints where appropriate
- Maintain consistent naming conventions

### Testing
- Test new features with provided test scripts
- Ensure backward compatibility
- Add examples for new functionality

### Documentation
- Update README for significant changes
- Include mathematical explanations for algorithms
- Provide usage examples

## 📄 License

This project is intended for educational purposes. Feel free to use, modify, and distribute according to your institution's guidelines.

## 🙏 Acknowledgments

- **OpenAI Gymnasium**: For providing excellent RL environments
- **NumPy**: For efficient numerical computations
- **Classic RL Literature**: Sutton & Barto, "Reinforcement Learning: An Introduction"

---

**Happy Learning! 🚀**

For questions, issues, or contributions, please refer to the repository's issue tracker or contact the maintainers.