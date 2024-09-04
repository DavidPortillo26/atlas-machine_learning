#!/usr/bin/env python3

"""
Linear Algebra Module: Matrix Operations

This module provides functions for basic linear algebra operations,
particularly focusing on matrix manipulation and transformation.

Functions:
    matrix_transpose(matrix): Returns the transpose of a given matrix.
    matrix_shape(matrix): Determines the dimensions of a matrix.

Usage Examples:
    >>> import 3_flip_me_over as flip
    >>> matrix = [[1, 2, 3], [4, 5, 6]]
    >>> flipped = flip.matrix_transpose(matrix)
    >>> print(flipped)
    [[1, 4], [2, 5], [3, 6]]

    >>> shape = flip.matrix_shape(matrix)
    >>> print(shape)
    [2, 3]
"""


def matrix_transpose(matrix):
    """ return a new matrix transposed
    Args:
        matrix: given list
    Return:
        new_matrix: Transposed matrix
    """
    if isinstance(matrix[0], list):
        return [
            [matrix[j][i] for j in range(len(matrix))]
            for i in range(len(matrix[0]))
        ]
    else:
        return [len(matrix)]
