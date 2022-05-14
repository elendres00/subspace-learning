"""Principal Component Analysis.
"""

# Copyright (c) 2022, Elias Endres;
# Copyright (c) 2007-2022 The scikit-learn developers.
# License: BSD 3 clause

import numpy as np
from scipy import linalg


class PCA:
    """Principal Component Analysis (PCA).

    Linear subspace learning method using Singular Value Decomposition (SVD)
    to find a lower dimensional linear subspace, such that the variance of
    the projected vector samples is maximized.
    The input data is centered for each feature before applying the SVD.

    Parameters
    ----------
    n_components : int, default=None
        Number of components to keep.
        If n_components not set all components are kept:

            n_components = min(n_samples, n_features)

    Attributes
    ----------
    components_ : ndarray of shape (n_features, n_components)
        The projection matrix consisting of the `n_components`
        eigenvectors of the covariance matrix of the vector
        samples corresponding to the `n_components` largest
        eigenvalues.
        Equivalently, the right singular vectors of the centered
        input data X of shape (n_samples, n_features).

    mean_ : ndarray of shape (n_features,)
        Per-feature empirical mean, estimated from the training set.
        Equal to `X.mean(axis=0)`.

    n_features_ : int
        Number of features in each sample of the training data.

    n_samples_ : int
        Number of samples in the training data.
    """

    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X):
        """Fit the model with X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed values.
        """
        n_samples, n_features = X.shape

        # Handle n_components
        if (self.n_components is None):
            n_components = min(n_samples, n_features)
        else:
            n_components = self.n_components

        # Check n_components
        if not (1 <= n_components <= min(n_samples, n_features)):
            raise ValueError(
                "n_components={} must be between 1 and "
                "min(n_samples, n_features)="
                "{}".format(n_components, min(n_samples, n_features))
            )

        # Center data
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_

        U, S, VT = linalg.svd(X, full_matrices=False)

        components = VT.T

        self.n_samples_, self.n_features_ = n_samples, n_features
        self.n_components = n_components
        self.components_ = components[:, :n_components]

        U = U[:, :n_components]
        # X_new = X * V = U * S * VT * V = U * S
        U *= S[:n_components]

        return U

    def transform(self, X):
        """Apply the linear transformation extracted from the training
        set to each vector in X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Projection of X in the first principal components
        """
        X = X - self.mean_
        X_transformed = np.matmul(X, self.components_)
        return X_transformed

    def inverse_transform(self, X):
        """Transform data back to its original space.

        In other words, return an input `X_original` whose transform would be X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_components)

        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
        """
        return np.matmul(X, self.components_.T) + self.mean_
