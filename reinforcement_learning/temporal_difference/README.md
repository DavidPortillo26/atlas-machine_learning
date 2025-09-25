# Temporal Difference Learning Algorithms

This project implements three fundamental reinforcement learning algorithms: Monte Carlo, TD(λ), and SARSA(λ). These algorithms are designed to solve value estimation and control problems in reinforcement learning environments, specifically tested on the FrozenLake environment from Gymnasium.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Algorithms Implemented](#algorithms-implemented)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Examples](#examples)
- [Environment Details](#environment-details)
- [Mathematical Background](#mathematical-background)
- [Contributing](#contributing)

## Overview

Temporal Difference (TD) learning is a central concept in reinforcement learning that combines ideas from Monte Carlo methods and dynamic programming. This project implements three key algorithms:

1. **Monte Carlo Method**: Model-free learning from complete episodes
2. **TD(λ)**: Temporal difference learning with eligibility traces for value function estimation
3. **SARSA(λ)**: State-Action-Reward-State-Action learning with eligibility traces for Q-function estimation

All algorithms are implemented with eligibility traces (λ parameter) to improve learning efficiency by providing better credit assignment to actions and states.

## Project Structure

```
temporal_difference/
├── README.md                 # This documentation file
├── 0-monte_carlo.py         # Monte Carlo algorithm implementation
├── 0-main.py                # Demo script for Monte Carlo
├── 1-td_lambtha.py          # TD(λ) algorithm implementation
├── 1-main.py                # Demo script for TD(λ)
├── 2-sarsa_lambtha.py       # SARSA(λ) algorithm implementation
└── 2-main.py                # Demo script for SARSA(λ)
```

## Algorithms Implemented

### 1. Monte Carlo Method (`0-monte_carlo.py`)

The Monte Carlo method learns value functions by averaging returns from complete episodes.

**Key Features:**
- First-visit Monte Carlo implementation
- Reward shaping for improved convergence
- Handles terminal and non-terminal states appropriately

**Mathematical Foundation:**
```
V(s) ← average of all returns following visits to state s
```

### 2. TD(λ) Algorithm (`1-td_lambtha.py`)

TD(λ) combines the benefits of Monte Carlo and temporal difference methods using eligibility traces.

**Key Features:**
- Eligibility traces for efficient credit assignment
- Handles goal states, holes, and normal steps differently
- Online learning (updates during episodes)

**Mathematical Foundation:**
```
δₜ = Rₜ₊₁ + γV(Sₜ₊₁) - V(Sₜ)
E(s) ← E(s) + 1 if s = Sₜ
V(s) ← V(s) + α·δₜ·E(s) for all s
E(s) ← γλE(s) for all s
```

### 3. SARSA(λ) Algorithm (`2-sarsa_lambtha.py`)

SARSA(λ) learns Q-values (state-action values) using eligibility traces and an epsilon-greedy policy.

**Key Features:**
- Q-learning with eligibility traces
- Epsilon-greedy exploration strategy
- On-policy learning algorithm
- Epsilon decay for balanced exploration/exploitation

**Mathematical Foundation:**
```
δₜ = Rₜ₊₁ + γQ(Sₜ₊₁,Aₜ₊₁) - Q(Sₜ,Aₜ)
E(s,a) ← E(s,a) + 1 if (s,a) = (Sₜ,Aₜ)
Q(s,a) ← Q(s,a) + α·δₜ·E(s,a) for all s,a
E(s,a) ← γλE(s,a) for all s,a
```

## Installation

### Requirements

```bash
pip install numpy gymnasium
```

### Dependencies

- Python 3.7+
- NumPy >= 1.18.0
- Gymnasium >= 0.26.0

## Usage

### Running the Algorithms

Each algorithm can be run using its corresponding main script:

```bash
# Monte Carlo
python3 0-main.py

# TD(λ)
python3 1-main.py

# SARSA(λ)
python3 2-main.py
```

### Basic Usage in Code

```python
import gymnasium as gym
import numpy as np
from monte_carlo import monte_carlo
from td_lambtha import td_lambtha
from sarsa_lambtha import sarsa_lambtha

# Create environment
env = gym.make('FrozenLake8x8-v1')

# Define a simple policy
def policy(state):
    # Your policy implementation
    return action

# Monte Carlo
V = np.zeros(64)
V = monte_carlo(env, V, policy, episodes=5000)

# TD(λ)
V = np.zeros(64)
V = td_lambtha(env, V, policy, lambtha=0.9, episodes=5000)

# SARSA(λ)
Q = np.random.uniform(size=(64, 4))
Q = sarsa_lambtha(env, Q, lambtha=0.9, episodes=5000)
```

## API Documentation

### monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99)

Performs Monte Carlo value estimation.

**Parameters:**
- `env`: Gymnasium environment instance
- `V`: numpy.ndarray of shape (s,) - initial value estimates
- `policy`: function(state) -> action - policy to evaluate
- `episodes`: int - number of episodes to run (default: 5000)
- `max_steps`: int - maximum steps per episode (default: 100)
- `alpha`: float - learning rate (default: 0.1)
- `gamma`: float - discount factor (default: 0.99)

**Returns:**
- `V`: numpy.ndarray - updated value estimates

### td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99)

Performs TD(λ) value estimation.

**Parameters:**
- `env`: Gymnasium environment instance
- `V`: numpy.ndarray of shape (s,) - initial value estimates
- `policy`: function(state) -> action - policy to evaluate
- `lambtha`: float - eligibility trace decay parameter (0 ≤ λ ≤ 1)
- `episodes`: int - number of episodes to run (default: 5000)
- `max_steps`: int - maximum steps per episode (default: 100)
- `alpha`: float - learning rate (default: 0.1)
- `gamma`: float - discount factor (default: 0.99)

**Returns:**
- `V`: numpy.ndarray - updated value estimates

### sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05)

Performs SARSA(λ) Q-learning.

**Parameters:**
- `env`: Gymnasium environment instance
- `Q`: numpy.ndarray of shape (s,a) - initial Q-table
- `lambtha`: float - eligibility trace decay parameter (0 ≤ λ ≤ 1)
- `episodes`: int - number of episodes to run (default: 5000)
- `max_steps`: int - maximum steps per episode (default: 100)
- `alpha`: float - learning rate (default: 0.1)
- `gamma`: float - discount factor (default: 0.99)
- `epsilon`: float - initial exploration rate (default: 1.0)
- `min_epsilon`: float - minimum exploration rate (default: 0.1)
- `epsilon_decay`: float - epsilon decay rate (default: 0.05)

**Returns:**
- `Q`: numpy.ndarray - updated Q-table

## Examples

### Example 1: Monte Carlo Value Estimation

```python
#!/usr/bin/env python3
import gymnasium as gym
import numpy as np
from monte_carlo import monte_carlo

# Set up environment
env = gym.make('FrozenLake8x8-v1')
env.reset(seed=0)

# Define a policy that avoids holes
def avoid_holes_policy(state):
    row, col = state // 8, state % 8

    # Try to move right if safe
    if col < 7 and env.unwrapped.desc[row, col + 1] != b'H':
        return 2  # RIGHT
    # Try to move down if safe
    elif row < 7 and env.unwrapped.desc[row + 1, col] != b'H':
        return 1  # DOWN
    else:
        return 0  # LEFT (fallback)

# Initialize value function
V = np.where(env.unwrapped.desc == b'H', -1, 1).reshape(64).astype('float64')

# Run Monte Carlo
V_mc = monte_carlo(env, V, avoid_holes_policy, episodes=10000)
print("Monte Carlo Value Estimates:")
print(V_mc.reshape(8, 8))
```

### Example 2: TD(λ) with Different Lambda Values

```python
#!/usr/bin/env python3
import gymnasium as gym
import numpy as np
from td_lambtha import td_lambtha

env = gym.make('FrozenLake8x8-v1')

def policy(state):
    # Random policy for demonstration
    return np.random.choice([0, 1, 2, 3])

# Compare different lambda values
lambdas = [0.0, 0.5, 0.9]
for lam in lambdas:
    V = np.zeros(64)
    V = td_lambtha(env, V, policy, lambtha=lam, episodes=5000)
    print(f"TD(λ={lam}) final values sum: {V.sum():.4f}")
```

### Example 3: SARSA(λ) with Policy Extraction

```python
#!/usr/bin/env python3
import gymnasium as gym
import numpy as np
from sarsa_lambtha import sarsa_lambtha

env = gym.make('FrozenLake8x8-v1')

# Learn Q-values
Q = np.random.uniform(size=(64, 4))
Q_learned = sarsa_lambtha(env, Q, lambtha=0.9, episodes=10000)

# Extract policy from Q-values
def extract_policy(Q):
    return np.argmax(Q, axis=1)

optimal_policy = extract_policy(Q_learned)
print("Learned Policy (action for each state):")
print(optimal_policy.reshape(8, 8))
```

## Environment Details

All algorithms are tested on the **FrozenLake8x8-v1** environment:

- **States**: 64 positions on an 8×8 grid
- **Actions**: 4 (LEFT=0, DOWN=1, RIGHT=2, UP=3)
- **Goal**: Reach the goal state (bottom-right corner)
- **Hazards**: Holes that terminate episodes with negative reward
- **Rewards**:
  - +1 for reaching the goal
  - 0 for regular steps
  - Episode terminates when reaching holes or goal

### Environment Layout

```
S F F F F F F F    S = Start
F F F H F F F F    F = Frozen (safe)
F F F F F F F F    H = Hole (hazardous)
F F F F F H F F    G = Goal
F F F H F F F F
F H H F F F H F
F H F F H F H F
F F F H F F F G
```

## Mathematical Background

### Eligibility Traces

Eligibility traces provide a mechanism for temporal credit assignment by maintaining a trace of recently visited states or state-action pairs:

- **λ = 0**: Only the current state/action is updated (pure TD)
- **λ = 1**: All states in the episode are updated (Monte Carlo-like)
- **0 < λ < 1**: Balance between TD and Monte Carlo

### Convergence Properties

1. **Monte Carlo**: Converges to true values with infinite samples
2. **TD(λ)**: Converges under certain conditions with lower variance than MC
3. **SARSA(λ)**: Converges to optimal Q* under appropriate exploration

### Key Hyperparameters

- **α (alpha)**: Learning rate controls update magnitude
- **γ (gamma)**: Discount factor determines importance of future rewards
- **λ (lambda)**: Eligibility trace parameter balances TD vs MC learning
- **ε (epsilon)**: Exploration rate in epsilon-greedy policies

## Performance Considerations

### Computational Complexity

- **Monte Carlo**: O(T) per episode, where T is episode length
- **TD(λ)**: O(S·T) per episode for S states
- **SARSA(λ)**: O(S·A·T) per episode for S states and A actions

### Memory Usage

- **Monte Carlo**: O(S) for value storage + episode storage
- **TD(λ)**: O(S) for values + O(S) for eligibility traces
- **SARSA(λ)**: O(S·A) for Q-table + O(S·A) for eligibility traces

## Contributing

When contributing to this project:

1. Follow PEP8 style guidelines
2. Add comprehensive docstrings
3. Include unit tests for new algorithms
4. Update documentation for API changes
5. Validate convergence properties of new implementations

## Author

Created by David Portillo for educational purposes.

---

*This implementation focuses on clarity and educational value while maintaining computational efficiency for the FrozenLake environment.*