"""Multilinear Principal Component Analysis.
"""

# License: BSD 3 clause

import numpy as np
from scipy import linalg
from .utils import tensor


class MultilinearPCA:
    """Multilinear Principal Component Analysis (MPCA).

    PCA based multilinear subspace learning method operating directly
    on the input tensor objects to project them to a lower dimensional
    tensor subspace, such that the variance of the projected tensors
    objects is maximized.
    The input tensors are centered before applying the projection.

    Let N denote the order of the tensor objects then:
    MPCA reduces to Principal Component Analysis for N=1 and
    Generalized Principal Component Analysis for N=2.

    Parameters
    ----------
    projection_shape : ndarray of shape (N,) of ints, default=None
        The is projection_shape is given by (p_1,...,p_N), which gives
        the dimension of the nth-mode of the projected tensor as p_n.
        If projection_shape is not set, we use:

            projection_shape[n] = min(n_samples*i_1*...*i_(n-1)*i_(n+1)*...*i_N, i_n)

    tol : float, default=0.0
        Convergence tolerance for the tensor variance
        in the local optimization.
        Must be of range [0.0, infinity).

    n_iterations : int, default=1
        Number of iterations in the local optimization.
        Must be of range [1, infinity).

    Attributes
    ----------
    projection_matrices_ : list of N ndarray's of shape (i_n, p_n)
        The N projection matrices containing p_n basis vectors
        of the n-mode space R^{i_n} for each mode n in 1,...,N,
        forming a tensor subspace capturing the most variation
        in the input tensors.
        The p_n vectors of each matrix are sorted by
        decreasing eigenvalue for each mode n in 1,...,N.

    mean_ : ndarray of shape (i_1,...,i_N)
        Per-feature empirical mean, estimated from the training set.
        Equal to `X.mean(axis=0)`.

    tensor_order_ : int
        The order or the number of dimensions N of the tensor objects.

    tensor_shape_ : ndarray of shape (N,)
        The shape of the tensor objects. That is given by (i_1,...,i_N).
        With i_n the n-mode dimension of the tensor.

    n_samples_ : int
        Number of samples in the training data.

    References
    ----------
    `Haiping Lu, K.N. Plataniotis, and A.N. Venetsanopoulos,
    "MPCA: Multilinear Principal Component Analysis of Tensor Objects",
    IEEE Transactions on Neural Networks,
    Vol. 19, No. 1, Page: 18-39, January 2008.`
    """

    def __init__(
            self,
            projection_shape=None,
            tol=10**-6,
            n_iterations=1
    ):
        self.projection_shape = projection_shape
        self.tol = tol
        self.n_iterations = n_iterations

    def fit(self, X):
        """Fit the MPCA model on the training data given in X

        This is done by finding a multilinear transformation
        that projects each tensor in X to a tensor subspace,
        such that the tensor variance is maximized.

        Parameters
        ----------
        X : ndarray of shape (n_samples, i_1,...,i_N)
            Training data.

        Returns
        -------
        X_new : ndarray of shape (n_samples, p_1,...,p_N)
            Projected tensor samples in the tensor subspace.
        """
        n_samples = X.shape[0]
        tensor_shape = X.shape[1:]
        tensor_order = len(tensor_shape)

        # Compute the upper boundary for projection_shape
        projection_shape_max = [
            min(np.prod(X.shape) / tensor_shape[n], tensor_shape[n])
            for n in range(tensor_order)
        ]

        # Handle projection_shape
        if (self.projection_shape is None):
            projection_shape = projection_shape_max
        else:
            projection_shape = self.projection_shape

        # Check projection_shape
        for n in range(tensor_order):
            if not (1 <= projection_shape[n] <= projection_shape_max[n]):
                raise ValueError(
                    "projection_shape[n] must be between 1 and "
                    "min(n_samples*i_1*...*i_(n-1)*i_(n+1)*...*i_N, i_n) "
                    "for each mode n"
                )

        # Center data
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_

        # Initialize the variance
        variance = 0
        projection_matrices = []

        # Local optimization of the projection_matrices
        for k in range(self.n_iterations):
            # Compute new projection_matrices
            projection_matrices_old = projection_matrices
            projection_matrices = []
            for n in range(tensor_order):
                # Compute the projection of all tensors in X on the tensor subspace
                # given by the old projection_matrices except the nth-mode component
                if k > 0:
                    X_n = tensor.multi_mode_dot(
                        X,
                        projection_matrices_old,
                        modes=range(1, tensor_order+1),
                        skip=n+1
                    )
                else:
                    X_n = X
                # Unfold each tensor sample in X along the n-th mode and store the
                # nth-mode vectors of each tensor in the rows of X_n
                X_n = tensor.unfold(X_n, n+1)
                # Compute the new projection matrix for the nth-mode
                U, S, VT = linalg.svd(X_n, full_matrices=False)
                projection_matrices.append(VT[:projection_shape[n]].T)

            # Compute the projection of all tensors in X onto the tensor subspace
            # given by the projection_matrices computed above
            X_transformed = tensor.multi_mode_dot(
                X,
                projection_matrices,
                modes=range(1, tensor_order+1)
            )

            variance_old = variance
            variance = np.sum(X_transformed ** 2)
            if (variance - variance_old < self.tol):
                break

        self.n_samples_ = n_samples
        self.tensor_shape_ = tensor_shape
        self.tensor_order_ = tensor_order
        self.projection_shape = projection_shape
        self.projection_matrices_ = projection_matrices

        return X_transformed

    def transform(self, X):
        """Project each tensor in X into the tensor subspace
        extracted from the training set.

        Parameters
        ----------
        X : ndarray of shape (n_samples, i_1,...,i_N)

        Returns
        -------
        X_new : ndarray of shape (n_samples, p_1,...,p_N)
            Projection of the tensors in X on the tensor subspace.
        """
        X = X - self.mean_

        return tensor.multi_mode_dot(
            X,
            self.projection_matrices_,
            modes=range(1, self.tensor_order_+1)
        )

    def inverse_transform(self, X):
        """Transform data back to its original space.

        In other words, return an input `X_original` whose transform would be X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, p_1,...,p_N)

        Returns
        -------
        X_original : ndarray of shape (n_samples, i_1,...,i_N)
        """
        X_original = tensor.multi_mode_dot(
            X,
            self.projection_matrices_,
            modes=range(1, self.tensor_order_+1),
            transpose=True
        )

        return X_original + self.mean_
