#!/usr/bin/env python3
import numpy as np


class Neuron:
    def __init__(self, nx):
        """Initialize the neuron"""
        "Check if nx is an integer"
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")

        "Check if nx is a positive integer"
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        "Initialize the weights (W), bias (b), and activated output (A)"
        "Random normal distribution for weights"
        self.W = np.random.randn(1, nx)
        "Bias initialized to 0"
        self.b = 0
        "Activated output initialized to 0"
        self.A = 0
