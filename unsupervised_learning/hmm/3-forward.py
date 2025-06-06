#!/usr/bin/env python3
"""
Performs the forward algorithm for a Hidden Markov Model
"""

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Performs the forward algorithm for a Hidden Markov Model

    Parameters:
        Observation (np.ndarray): shape (T,)
            Index of each observation
        Emission (np.ndarray): shape (N, M)
            Emission probabilities:
            Emission[i, j] is the probability of observing j 
            from hidden state i
        Transition (np.ndarray): shape (N, N)
            Transition probabilities:
            Transition[i, j] is the probability of transitioning from i to j
        Initial (np.ndarray): shape (N, 1)
            Probability of starting in each hidden state

    Returns:
        P (float): Total likelihood of the observation sequence
        F (np.ndarray): shape (N, T) with forward path probabilities
        or (None, None) on failure
    """
    if (not isinstance(Observation, np.ndarray) or
            not isinstance(Emission, np.ndarray) or
            not isinstance(Transition, np.ndarray) or
            not isinstance(Initial, np.ndarray)):
        return None, None

    if Observation.ndim != 1:
        return None, None
    T = Observation.shape[0]

    if Emission.ndim != 2:
        return None, None
    N, M = Emission.shape

    if Transition.shape != (N, N):
        return None, None
    if Initial.shape != (N, 1):
        return None, None

    F = np.zeros((N, T))
    F[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]

    for t in range(1, T):
        for j in range(N):
            F[j, t] = np.dot(F[:, t - 1], Transition[:, j]) * \
                      Emission[j, Observation[t]]

    P = np.sum(F[:, -1])
    return P, F
