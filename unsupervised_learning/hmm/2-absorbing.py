#!/usr/bin/env python3
""" Absorbing Markov chain """

import numpy as np


def absorbing(P):
    """
    Determines if a Markov chain is absorbing
    Args:
        P: numpy.ndarray of shape (n, n) - the transition matrix
    Returns:
        True if absorbing, False if not, None on failure
    """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    n, m = P.shape
    if n != m:
        return None

    absorbing_states = []
    for i in range(n):
        if P[i, i] == 1 and np.sum(P[i]) == 1:
            absorbing_states.append(i)

    if not absorbing_states:
        return False

    # Build reachability matrix using powers of P
    reachable = np.copy(P)
    for _ in range(n):
        reachable = np.dot(reachable, P)

    for i in range(n):
        if i not in absorbing_states:
            if not any(reachable[i][j] > 0 for j in absorbing_states):
                return False

    return True
