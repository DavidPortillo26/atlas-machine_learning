#!/usr/bin/env python3

load_frozen_lake = __import__('0-load_env').load_frozen_lake

# Default: random 8x8 → 64 states
env = load_frozen_lake()
print(env.shape)

# Slippery: still 8x8 → 64 states
env = load_frozen_lake(is_slippery=True)
print(env.shape)

# Custom 3x3 → 9 states
desc = [['S', 'F', 'F'],
        ['F', 'H', 'H'],
        ['F', 'F', 'G']]
env = load_frozen_lake(desc=desc)
print(env.shape)
print(env)   # print full zero matrix

# Pre-made 4x4 → 16 states
env = load_frozen_lake(map_name='4x4')
print(env.shape)
