#!/usr/bin/env python3


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

        return [len(matrix)]
