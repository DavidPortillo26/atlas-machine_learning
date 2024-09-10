#!/usr/bin/env python3
"""
Script to calculate a Poisson distribution
"""


class Poisson():
    """
    Tye class to call methods of Poisson distribution
    CDF and PDF
    """

    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """
        Initialize method
        data: type list of given numbers
        lambtha: type lambda factor to calculate mean of data
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.lambtha = (sum(data) / len(data))
