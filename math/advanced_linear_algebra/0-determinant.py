#!/usr/bin/env python3
"""
Determinant function
"""


def multi_determinant(matrix):
    """
    Function that computes the determinant of a given matrix of
    dimension >= 2.

    Args:
        matrix: list of lists whose determinant should be calculated

    Returns:
        Determinant of matrix
    """
    mat_l = len(matrix)
    if mat_l == 2 and len(matrix[0]) == 2:
        return (matrix[0][0] * matrix[1][1] -
                matrix[1][0] * matrix[0][1])

    deter = 0
    for c in range(mat_l):
        mat_cp = [row[:] for row in matrix[1:]]
        for r in range(len(mat_cp)):
            mat_cp[r] = mat_cp[r][:c] + mat_cp[r][c + 1:]
        sign = (-1) ** c
        sub_det = multi_determinant(mat_cp)
        deter += sign * matrix[0][c] * sub_det
    return deter


def determinant(matrix):
    """
    Calculates the determinant of a matrix.

    Args:
        matrix: list of lists whose determinant should be calculated

    Returns:
        Determinant of matrix
    """
    mat_l = len(matrix)
    if not isinstance(matrix, list) or mat_l == 0:
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if matrix == [[]]:
        return 1
    if not all(mat_l == len(row) for row in matrix):
        raise
