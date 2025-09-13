#!/usr/bin/env python3
"""
Test script to verify that all dependencies are working correctly
"""

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import gymnasium as gym
        print("✓ Gymnasium imported successfully")
    except ImportError as e:
        print(f"✗ Gymnasium import failed: {e}")
        return False
    
    try:
        import tensorflow as tf
        try:
            version = tf.__version__
        except AttributeError:
            version = tf.version.VERSION
        print(f"✓ TensorFlow {version} imported successfully")
    except ImportError as e:
        print(f"✗ TensorFlow import failed: {e}")
        return False
    
    try:
        import ale_py
        print("✓ ALE-Py imported successfully")
    except ImportError as e:
        print(f"✗ ALE-Py import failed: {e}")
        return False
    
    try:
        from rl.agents.dqn import DQNAgent
        from rl.policy import EpsGreedyQPolicy, GreedyQPolicy
        from rl.memory import SequentialMemory
        print("✓ Keras-RL2 imported successfully")
    except ImportError as e:
        print(f"✗ Keras-RL2 import failed: {e}")
        print("  This is likely due to TensorFlow/Keras version compatibility issues")
        return False
    
    return True


def test_environment():
    """Test creating the Atari environment"""
    print("\nTesting environment creation...")
    
    try:
        import gymnasium as gym
        from gymnasium.wrappers import AtariPreprocessing, FrameStack
        
        # Create environment
        env = gym.make('ALE/Breakout-v5', render_mode='rgb_array')
        print("✓ Base environment created successfully")
        
        # Apply preprocessing
        env = AtariPreprocessing(env, 
                               noop_max=30,
                               frame_skip=4,
                               screen_size=84,
                               terminal_on_life_loss=True,
                               grayscale_obs=True,
                               grayscale_newaxis=False,
                               scale_obs=False)
        print("✓ Atari preprocessing applied successfully")
        
        # Apply frame stacking
        env = FrameStackObservation(env, stack_size=4)
        print("✓ Frame stacking applied successfully")
        
        # Test reset
        observation, info = env.reset()
        print(f"✓ Environment reset successful")
        print(f"  Observation shape: {observation.shape}")
        print(f"  Action space: {env.action_space}")
        print(f"  Number of actions: {env.action_space.n}")
        
        # Test a few steps
        for i in range(5):
            action = env.action_space.sample()
            observation, reward, done, truncated, info = env.step(action)
            print(f"  Step {i+1}: reward={reward}, done={done}")
            if done or truncated:
                break
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False


def test_model_creation():
    """Test creating a simple model"""
    print("\nTesting model creation...")
    
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Flatten, Conv2D, InputLayer
        
        # Create a simple model similar to our DQN
        model = Sequential()
        model.add(InputLayer(shape=(4, 84, 84)))  # 4 stacked frames, 84x84 pixels
        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', data_format='channels_first'))
        model.add(Flatten())
        model.add(Dense(4, activation='linear'))  # 4 actions for Breakout
        
        print("✓ Model created successfully")
        print(f"  Model input shape: {model.input_shape}")
        print(f"  Model output shape: {model.output_shape}")
        
        # Test model compilation
        model.compile(optimizer='adam', loss='mse')
        print("✓ Model compiled successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=== Setup Verification Test ===\n")
    
    tests_passed = 0
    total_tests = 3
    
    if test_imports():
        tests_passed += 1
    
    if test_environment():
        tests_passed += 1
    
    if test_model_creation():
        tests_passed += 1
    
    print(f"\n=== Test Results ===")
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✓ All tests passed! Your environment is ready for training.")
        return True
    else:
        print("✗ Some tests failed. Please check the error messages above.")
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
