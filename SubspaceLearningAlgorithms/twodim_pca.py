"""Two Dimensional Principal Component Analysis.
"""

# Copyright (c) 2022, Elias Endres;
# Copyright (c) 2007-2022 The scikit-learn developers.
# License: BSD 3 clause

import numpy as np
from scipy import linalg


class TwoDimensionalPCA:
    """Two-Dimensional Principal Component Analysis (2DPCA).

    PCA based multilinear subspace learning method operating directly
    on images in their natural matrix representation. 2DPCA finds a
    linear transformation that is applied on the right side of the
    image matrices projecting them to a lower dimensional tensor
    subspace, such that the variance of the projected samples is
    maximized.
    The input matrices are centered before applying the projection.

    Parameters
    ----------
    n_components : int, default=None
        Number of components to keep.
        If n_components is not set, all components are kept:

            n_components = min(n_samples*n_rows, n_columns)

    Attributes
    ----------
    components_ : ndarray of shape (n_columns, n_components)
        The projection matrix consisting of the `n_components`
        eigenvectors of the covariance matrix of the image matrices
        corresponding to the `n_components` largest eigenvalues.
        Equivalently, the right singular vectors of the centered and
        stacked input matrices X of shape (n_samples*n_rows, n_columns).

    mean_ : ndarray of shape (n_rows, n_columns)
        Per-feature empirical mean, estimated from the training set.
        Equal to `X.mean(axis=0)`.

    n_rows_ : int
        Number of rows of each matrix in the training data.

    n_columns_ : int
        Number of columns of each matrix in the training data.

    n_samples_ : int
        Number of samples in the training data.

    References
    ----------
    `Jian Yang, D. Zhang, A. F. Frangi and Jing-yu Yang,
    "Two-dimensional PCA: a new approach to appearance-based face representation
    and recognition",
    IEEE Transactions on Pattern Analysis and Machine Intelligence,
    vol. 26, no. 1, pp. 131-137, Jan. 2004.`
    """

    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X):
        """Fit the 2DPCA model on the training data given in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_rows, n_columns)
            Training data.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_rows, n_components)
            Transformed values.
        """
        n_samples, n_rows, n_columns = X.shape

        # Handle n_components
        if (self.n_components is None):
            n_components = min(n_samples*n_rows, n_columns)
        else:
            n_components = self.n_components

        # Check n_components
        if not (1 <= n_components <= min(n_samples*n_rows, n_columns)):
            raise ValueError(
                "n_components={} must be between 1 and "
                "min(n_samples*n_rows, n_columns)="
                "{}".format(n_components, min(n_samples*n_rows, n_columns))
            )

        # Center data
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_

        # Stack the matrices in X and compute the Singular Value Decomposition
        U, S, VT = linalg.svd(X.reshape(-1, n_columns), full_matrices=False)

        components = VT.T

        self.n_samples_ = n_samples
        self.n_rows_, self.n_columns_ = n_rows, n_columns
        self.n_components = n_components
        self.components_ = components[:, :n_components]

        # X_stacked * V = U * S * VT * V = U * S
        U = U[:, :n_components]
        U *= S[:n_components]
        # Reshape U to get X_new
        X_new = U.reshape(n_samples, n_rows, n_components)

        return X_new

    def transform(self, X):
        """Apply the linear transformation extracted from the
        training set to each matrix in X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_rows, n_columns)

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_rows, n_components)
            Projection of the matrices in X
        """
        X = X - self.mean_
        X_transformed = np.matmul(X, self.components_)
        return X_transformed

    def inverse_transform(self, X):
        """Transform data back to its original space.

        In other words, return an input `X_original` whose transform would be X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_rows, n_components)

        Returns
        -------
        X_original : ndarray of shape (n_samples, n_rows, n_columns)
        """
        return np.matmul(X, self.components_.T) + self.mean_
