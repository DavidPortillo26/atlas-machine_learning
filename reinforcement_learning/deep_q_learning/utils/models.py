# utils/models.py
from tensorflow import keras
from tensorflow.keras import layers

def model_template(input_shape, n_actions):
    """
    Build a DeepMind-style ConvNet for Atari DQN.
    input_shape: (H, W, C) - e.g. (84, 84, 4)
    n_actions: number of discrete actions
    """
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(32, (8, 8), strides=(4, 4), activation="relu")(inputs)
    x = layers.Conv2D(64, (4, 4), strides=(2, 2), activation="relu")(x)
    x = layers.Conv2D(64, (3, 3), strides=(1, 1), activation="relu")(x)

    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)
    outputs = layers.Dense(n_actions, activation="linear")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="dqn_atari")
    return model
