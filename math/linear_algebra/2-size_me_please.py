#!/usr/bin/env python3
""" Recursion function to check the shape of the matrix"""


def matrix_shape(matrix):
    """ return the shape of a matrix """
    if isinstance(matrix[0], list):
        return [len(matrix)] + matrix_shape(matrix[0])
    else:
        return [len(matrix)]
