#!/usr/bin/env python3
"""
Defines function that performs the forward algorithm for a Hidden Markov Model
"""

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Performs the forward algorithm for a Hidden Markov Model

    Parameters:
        Observation (numpy.ndarray): shape (T,) containing the index of observations
        Emission (numpy.ndarray): shape (N, M) containing the emission probabilities
        Transition (numpy.ndarray): shape (N, N) containing transition probabilities
        Initial (numpy.ndarray): shape (N, 1) containing the starting probabilities

    Returns:
        P (float): the likelihood of the observations given the model
        F (numpy.ndarray): shape (N, T) containing the forward path probabilities
        or None, None on failure
    """

    # Validate input types and shapes
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

    # Initialize forward probability matrix
    F = np.zeros((N, T))
    F[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]

    # Compute forward probabilities using dynamic programming
    for t in range(1, T):
        for j in range(N):
            F[j, t] = np.dot(F[:, t - 1], Transition[:, j]) * Emission[j, Observation[t]]

    # Total probability of the observation sequence
    P = np.sum(F[:, -1])

    return P, F
