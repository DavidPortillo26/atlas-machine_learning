# Hyperparameter Tuning with Gaussian Processes

Lightweight reference implementation of Gaussian Processes and Bayesian Optimization, plus a practical GPyOpt example that tunes a small neural network on the Breast Cancer dataset.

## Files
- `0-gp.py` – base 1D Gaussian Process with the RBF kernel.
- `1-gp.py` – adds mean/variance prediction for new points.
- `2-gp.py` – adds incremental updates after sampling new points.
- `3-bayes_opt.py` – Bayesian Optimization scaffold (initialization only).
- `4-bayes_opt.py` – Expected Improvement acquisition on a GP surrogate.
- `5-bayes_opt.py` – full Bayesian Optimization loop with EI.
- `6-bayes_opt.py` – end-to-end hyperparameter tuning for a dense NN using GPyOpt; saves checkpoints, a convergence plot, and a text summary.

## Setup
Use Python 3.9+ and install the scientific stack:

```bash
pip install numpy scipy scikit-learn matplotlib tensorflow GPyOpt
```

GPyOpt depends on `GPy`; pip will pull it in, but if installation fails, install it manually (`pip install GPy`).

## Running the example optimizer
Execute the full tuning script:

```bash
python 6-bayes_opt.py
```

What happens:
- Loads `sklearn.datasets.load_breast_cancer` and splits train/val.
- Builds a small dense network with dropout and L2 regularization.
- Bayesian Optimization (EI) searches over learning rate, units, dropout, L2 weight, and batch size.
- Writes the best hyperparameters and validation accuracy to `bayes_opt.txt`, stores model checkpoints in `checkpoints/`, and plots convergence to `convergence_plot.png`.

Adjust `max_iter` in `6-bayes_opt.py` to change the number of BO iterations, or tweak the `bounds` list to search different ranges.

## Using the GP/BO implementations
The GP and BO classes are self-contained. For a minimal EI step:

```python
from math import sin
import numpy as np
from 5-bayes_opt import BayesianOptimization

f = lambda x: sin(x[0])  # black-box function
bo = BayesianOptimization(f,
                          X_init=np.array([[0.0], [2.0]]),
                          Y_init=np.array([[sin(0.0)], [sin(2.0)]]),
                          bounds=(0, 6),
                          ac_samples=50,
                          l=1,
                          sigma_f=1,
                          xsi=0.01,
                          minimize=False)
next_x, ei = bo.acquisition()
```

Extend this with `bo.optimize(iterations=100)` to run the full loop.
