#!/usr/bin/env python3
"""Posterior"""

import numpy as np


def posterior(x, n, P, Pr):
    """
    Calculates the posterior probability of each hypothetical probability of
    developing severe side effects given the data

    Args:
        x: number of patients that develop severe side effects
        n: total number of patients observed
        P: 1D numpy.ndarray containing the hypothetical probabilities
        Pr: 1D numpy.ndarray containing the prior beliefs of P

    Returns:
        1D numpy.ndarray containing the posterior probabilities of P given x and n
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or P.shape != Pr.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if not np.all((P >= 0) & (P <= 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    if not np.all((Pr >= 0) & (Pr <= 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    factorial = 1
    for i in range(1, n + 1):
        factorial *= i
    factorial_x = 1
    for i in range(1, x + 1):
        factorial_x *= i
    factorial_n_x = 1
    for i in range(1, n - x + 1):
        factorial_n_x *= i
    binom_coeff = factorial / (factorial_x * factorial_n_x)

    likelihoods = binom_coeff * (P ** x) * ((1 - P) ** (n - x))
    intersection = likelihoods * Pr
    marginal = np.sum(intersection)

    posterior_prob = intersection / marginal

    return posterior_prob
