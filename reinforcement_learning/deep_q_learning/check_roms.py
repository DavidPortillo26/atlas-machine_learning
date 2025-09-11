import gymnasium as gym

try:
    env = gym.make("ALE/Breakout-v5")
    print("Breakout ROM is installed and ready!")
except gym.error.NamespaceNotFound:
    print("Namespace ALE not found. Make sure gymnasium[atari] and AutoROM are installed and AutoROM has run.")
except gym.error.NameNotFound:
    print("Breakout ROM is not installed.")
