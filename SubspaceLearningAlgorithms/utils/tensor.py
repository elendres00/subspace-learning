""" Tensor Utilities.
"""
# Copyright (c) 2022, Elias Endres;
# Copyright (c) 2016 The tensorly Developers.
# License: BSD 3 clause

import numpy as np


def unfold(tensor, mode=0):
    """Returns the mode-`mode` unfolding of the `tensor`
    with modes starting at `0`.

    Parameters
    ----------
    tensor : ndarray
        The input tensor to unfold along the mode-`mode`.

    mode : int, default=0
        The mode to unfold the tensor along.
        The indexing starts at 0,
        therefore mode is in ``range(0, tensor.ndim)``

    Returns
    -------
    unfolded_tensor: ndarray of shape (-1, tensor.shape[mode])
        The unfolded tensor along mode-`mode` which has
        the mode-`mode` vectors of the tensor in the rows.
    """
    return np.moveaxis(tensor, mode, -1).reshape(-1, tensor.shape[mode])

def fold(unfolded_tensor, mode, shape):
    """Refolds the mode-`mode` unfolding into a tensor of shape `shape`.

    In other words, refolds the mode-`mode` unfolded tensor
    into the original tensor of the specified shape.

    Parameters
    ----------
    unfolded_tensor : ndarray of shape (-1, tensor.shape[mode])
        The unfolded tensor along mode-`mode` which has
        the mode-`mode` vectors of the tensor in the rows.

    mode : int
        The mode the tensor was unfolded along.
        The indexing starts at 0,
        therefore mode is in ``range(0, len(shape))``

    shape : tuple
        Shape of the original tensor before unfolding.

    Returns
    -------
    tensor : ndarray of shape `shape`
        The refolded tensor.
    """
    full_shape = list(shape)
    mode_dim = full_shape.pop(mode)
    full_shape.append(mode_dim)
    return np.moveaxis(np.reshape(unfolded_tensor, full_shape), -1, mode)

def mode_dot(tensor, matrix, mode, transpose=False):
    """n-mode product of a tensor and a matrix at the specified mode.

    Parameters
    ----------
    tensor : ndarray of shape (i_1, ..., i_k, ..., i_N)
        The input tensor to multiply.

    matrix : ndarray of shape (i_k, J) or (J, i_k) if transpose=True
        The matrix with which to mode-`mode` multiply the tensor.

    mode : int
        The mode specifying the mode-`mode` multiplication of the
        tensor and the matrix.
        The indexing starts at 0, therefore
        mode is in ``range(0, tensor.ndim)``

    transpose : bool, default=False
        If True, the matrix is transposed for the multiplication.

    Returns
    -------
    tensor_new : ndarray of shape (i_1, ..., i_{k-1}, J, i_{k+1}, ..., i_N)
        Mode-`mode` tensor product of the input tensor and the matrix.
    """
    new_shape = list(tensor.shape)

    if len(matrix.shape) != 2:
        raise ValueError("Can only take n_mode_product with a matrix.")

    if transpose:
        matrix = matrix.T

    # Check that the dimensions of the tensor and the matrix match
    if matrix.shape[0] != tensor.shape[mode]:
        raise ValueError(
            "shapes {0} and {1} not aligned in mode-{2} multiplication:"
            " {3} (mode {2}) != {4} (dim of matrix)".format(
                tensor.shape, matrix.shape, mode,
                tensor.shape[mode], matrix.shape[0]
            )
        )

    tensor_new = np.dot(unfold(tensor, mode), matrix)

    new_shape[mode] = matrix.shape[1]

    return fold(tensor_new, mode, new_shape)

def multi_mode_dot(tensor, matrix_list, modes=None, skip=None, transpose=False):
    """n-mode product of a tensor and several matrices over several modes.

    Parameters
    ----------
    tensor : ndarray
        The input tensor to multiply.

    matrix_list : list of matrices of the same length as `modes`
        The matrices with which to multiply the tensor.

    modes : list of ints of the same length as `matrix_list`, default=None
        The modes specifying the mode-`mode` multiplication of the
        tensor and the corresponding matrix in the `matrix_list`.
        The indexing starts at 0, therefore each
        mode in modes is in ``range(0, tensor.ndim)``

        If modes is not set the modes are set in order starting at 0:

            modes == range(len(matrix_list))

    skip : int, default=None
        If not None, mode to skip.

    transpose : bool, default=False
        If True, the matrices in the list are transposed for multiplication.

    Returns
    -------
    tensor_new : ndarray
        Tensor product of the input tensor and the matrices
        in the list along the corresponding modes.
    """
    if modes is None:
        modes = range(len(matrix_list))

    tensor_new = tensor

    for i, (matrix, mode) in enumerate(zip(matrix_list, modes)):
        if (skip is not None) and (mode == skip):
            continue
        tensor_new = mode_dot(tensor_new, matrix, mode, transpose=transpose)

    return tensor_new
