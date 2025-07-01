import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import GPyOpt
import matplotlib.pyplot as plt

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Directory for saving models
os.makedirs("checkpoints", exist_ok=True)

# Define the objective function
def model_train_eval(params):
    learning_rate = float(params[0][0])
    units = int(params[0][1])
    dropout = float(params[0][2])
    l2_weight = float(params[0][3])
    batch_size = int(params[0][4])

    # Define model
    model = keras.Sequential([
        layers.Dense(units, activation='relu', kernel_regularizer=regularizers.l2(l2_weight), input_shape=(X_train.shape[1],)),
        layers.Dropout(dropout),
        layers.Dense(1, activation='sigmoid')
    ])

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Set up checkpointing
    fname = f"lr{learning_rate:.4f}_u{units}_do{dropout:.2f}_l2{l2_weight:.5f}_bs{batch_size}.h5"
    checkpoint_path = os.path.join("checkpoints", fname)
    checkpoint = callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_accuracy', mode='max', verbose=0)
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)

    # Train
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        batch_size=batch_size,
                        epochs=100,
                        callbacks=[checkpoint, early_stop],
                        verbose=0)

    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    return -val_accuracy  # Because GPyOpt minimizes

# Define the bounds for hyperparameters
bounds = [
    {'name': 'learning_rate', 'type': 'continuous', 'domain': (1e-4, 1e-2)},
    {'name': 'units', 'type': 'discrete', 'domain': (32, 64, 128, 256)},
    {'name': 'dropout', 'type': 'continuous', 'domain': (0.1, 0.5)},
    {'name': 'l2_weight', 'type': 'continuous', 'domain': (1e-6, 1e-2)},
    {'name': 'batch_size', 'type': 'discrete', 'domain': (16, 32, 64, 128)}
]

# Create optimizer
opt = GPyOpt.methods.BayesianOptimization(f=model_train_eval,
                                          domain=bounds,
                                          acquisition_type='EI',
                                          maximize=False)

# Run optimization
opt.run_optimization(max_iter=30)

# Save report
with open("bayes_opt.txt", "w") as f:
    f.write("Best parameters:\n")
    for name, val in zip([b['name'] for b in bounds], opt.x_opt):
        f.write(f"{name}: {val}\n")
    f.write(f"\nBest validation accuracy: {-opt.fx_opt:.4f}\n")

# Plot convergence
opt.plot_convergence()
plt.title("Bayesian Optimization Convergence")
plt.savefig("convergence_plot.png")
plt.show()
