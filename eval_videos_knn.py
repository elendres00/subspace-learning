"""Evaluate the subspace learning algorithms on videos.

The dataset is specified by the user and we randomly split it
in a train and test set. After that the subspace learning methods
are applied and then the K-Nearest-Neighbor classifier with K=1 is
used to predict the labels of the test set.
"""

# License: BSD 3 clause

from time import time
import argparse

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Classifier
from sklearn.neighbors import KNeighborsClassifier

# Datasets
from Datasets.avletters import fetch_avletters_averaged

# Feature Extraction Methods
from SubspaceLearningAlgorithms.pca import PCA
from SubspaceLearningAlgorithms.twodim_pca import TwoDimensionalPCA
from SubspaceLearningAlgorithms.multilinear_pca import MultilinearPCA


# Parse Arguments
parser = argparse.ArgumentParser(
    description="Evaluates the performance of different PCA based"
    "subspace learning methods on a video dataset chosen by the"
    "user using the KNN classifier."
)

parser.add_argument(
    '--dataset',
    choices=['avletters'],
    default='avletters',
    help='The dataset used (default: AVletters)'
)

args = parser.parse_args()


# Get the user specified dataset
if (args.dataset == 'avletters'):
    data, persons, target = fetch_avletters_averaged()
    n_samples, n_frames, n_rows, n_columns = data.shape
    p_dims = [1, 1, 3, 7, 13, 23]
    p1_dims = [2, 4, 6, 8, 10, 12]
    p2_dims = [2, 4, 6, 8, 10, 12]
    p3_dims = [2, 4, 6, 8, 10, 12]

print(
    "The data has {0} samples with shape {1}."
    "\n".format(data.shape[0], data.shape[1:])
)

# randomly split the data in train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.333, random_state=42,
    stratify=np.char.add(target.astype(np.str), persons)
)

X_train_n_samples, X_train_n_frames, X_train_rows, X_train_columns = X_train.shape
X_test_n_samples, X_test_frames, X_test_rows, X_test_columns = X_test.shape

# normalize the data
X_train_mean = X_train.mean(axis=0)
X_train = X_train - X_train_mean
X_test = X_test - X_train_mean


time_pca = []
time_2dpca = []
time_mpca = []

storage_pca = []
storage_2dpca = []
storage_mpca = []

reconstruction_error_pca = []
reconstruction_error_2dpca = []
reconstruction_error_mpca = []

accuracies_pca = []
accuracies_2dpca = []
accuracies_mpca = []

for p, p1, p2, p3 in zip(p_dims, p1_dims, p2_dims, p3_dims):
    # Apply PCA on the vectorized videos
    print("---------- PCA on videos ----------")
    print("Apply PCA to the data")
    t0 = time()
    pca = PCA(n_components=p)
    X_train_pca = pca.fit(X_train.reshape(X_train_n_samples, -1))
    X_test_pca = pca.transform(X_test.reshape(X_test_n_samples, -1))
    t_pca = time() - t0
    time_pca.append(t_pca)
    print("done in {:.3f}s\n".format(t_pca))

    print(
        "The number of principal components used is: "
        "{}\n".format(pca.n_components)
    )

    # Calculate the storage needed in the projected space
    storage = n_samples * p + n_frames * n_rows * n_columns * p
    storage_pca.append(storage)
    print("Scalars needed in projected space: {}\n".format(storage))

    # Calculate the reconstruction error on training data
    X_train_reconstructed = pca.inverse_transform(X_train_pca).reshape(X_train.shape)
    reconstruction_error = np.mean((X_train - X_train_reconstructed)**2)
    reconstruction_error_pca.append(reconstruction_error)
    print("Reconstruction Error: {}\n".format(reconstruction_error))

    # Use KNN to classify the projected test data
    knn = KNeighborsClassifier(n_neighbors=1, weights="distance")
    knn.fit(X_train_pca, y_train)
    y_pred = knn.predict(X_test_pca)

    accuracy = accuracy_score(y_test, y_pred)
    accuracies_pca.append(accuracy)
    print("Accuracy: {}\n".format(accuracy))


    # Apply 2DPCA on the image resulting from concatenating the
    # frames of each video sample
    print("---------- 2DPCA ----------")
    print("Apply 2DPCA to the data")
    t0 = time()
    twodpca = TwoDimensionalPCA(n_components=p3)
    X_train_2dpca = twodpca.fit(
        X_train.reshape(
            X_train_n_samples,
            X_train_n_frames*X_train_rows,
            X_train_columns
        )
    )
    X_test_2dpca = twodpca.transform(
        X_test.reshape(
            X_test_n_samples,
            X_test_frames*X_test_rows,
            X_test_columns
        )
    )
    t_2dpca = time() - t0
    time_2dpca.append(t_2dpca)
    print("done in {:.3f}s\n".format(t_2dpca))

    print(
        "The number of principal components used is: "
        "{}\n".format(twodpca.n_components)
    )

    # Calculate the storage needed in the projected space
    storage = n_samples * n_frames * n_rows * p3 + n_frames * n_rows * p3
    storage_2dpca.append(storage)
    print("Scalars needed in projected space: {}\n".format(storage))

    # Calculate the reconstruction error on training data
    X_train_reconstructed = twodpca.inverse_transform(X_train_2dpca).reshape(X_train.shape)
    reconstruction_error = np.mean((X_train - X_train_reconstructed)**2)
    reconstruction_error_2dpca.append(reconstruction_error)
    print("Reconstruction Error: {}\n".format(reconstruction_error))

    # Use KNN to classify the projected test data
    knn = KNeighborsClassifier(n_neighbors=1, weights="distance")
    knn.fit(X_train_2dpca.reshape(X_train_n_samples, -1), y_train)
    y_pred = knn.predict(X_test_2dpca.reshape(X_test_n_samples, -1))

    accuracy = accuracy_score(y_test, y_pred)
    accuracies_2dpca.append(accuracy)
    print("Accuracy: {}\n".format(accuracy))


    # Apply MPCA on the full data samples
    print("---------- MPCA on videos ----------")
    print("Apply MPCA to the data")
    t0 = time()
    mpca = MultilinearPCA(projection_shape=[p1, p2, p3], n_iterations=5)
    X_train_mpca = mpca.fit(X_train)
    X_test_mpca = mpca.transform(X_test)
    t_mpca = time() - t0
    time_mpca.append(t_mpca)
    print("done in {:.3f}s\n".format(t_mpca))

    print(
        "The number of components used in each mode is: "
        "{}\n".format(mpca.projection_shape)
    )

    # Calculate the storage needed in the projected space
    storage = n_samples * p1 * p2 * p3 + n_frames * p1 + n_rows * p2 + n_columns * p3
    storage_mpca.append(storage)
    print("Scalars needed in projected space: {}\n".format(storage))

    # Calculate the reconstruction error on training data
    X_train_reconstructed = mpca.inverse_transform(X_train_mpca)
    reconstruction_error = np.mean((X_train - X_train_reconstructed)**2)
    reconstruction_error_mpca.append(reconstruction_error)
    print("Reconstruction Error: {}\n".format(reconstruction_error))

    # Use KNN to classify the projected test data
    knn = KNeighborsClassifier(n_neighbors=1, weights="distance")
    knn.fit(X_train_mpca.reshape(X_train_n_samples, -1), y_train)
    y_pred = knn.predict(X_test_mpca.reshape(X_test_n_samples, -1))

    accuracy = accuracy_score(y_test, y_pred)
    accuracies_mpca.append(accuracy)
    print("Accuracy: {}\n".format(accuracy))

# Plot all results
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

# plot the reconstruction errors
ax1.plot(
    p3_dims,
    reconstruction_error_pca,
    linestyle='-',
    marker='s',
    label="PCA"
)
ax1.plot(
    p3_dims,
    reconstruction_error_2dpca,
    linestyle='--',
    marker='o',
    label="2DPCA"
)
ax1.plot(
    p3_dims,
    reconstruction_error_mpca,
    linestyle=':',
    marker='*',
    label="MPCA"
)
ax1.set_xticks(p3_dims)
ax1.set_xlabel("Values P1, P2 and P3 of MPCA")
ax1.set_ylabel("Mean Squared Reconstruction Error")
ax1.legend()

# plot the accuracies
ax2.plot(
    p3_dims,
    accuracies_pca,
    linestyle='-',
    marker='s',
    label="PCA"
)
ax2.plot(
    p3_dims,
    accuracies_2dpca,
    linestyle='--',
    marker='o',
    label="2DPCA"
)
ax2.plot(
    p3_dims,
    accuracies_mpca,
    linestyle=':',
    marker='*',
    label="MPCA"
)
ax1.set_xticks(p3_dims)
ax2.set_xlabel("Values P1, P2 and P3 of MPCA")
ax2.set_ylabel("Accuracy")
ax2.legend()

# plot the time
ax3.plot(
    p3_dims,
    time_pca,
    linestyle='-',
    marker='s',
    label="PCA"
)
ax3.plot(
    p3_dims,
    time_2dpca,
    linestyle='--',
    marker='o',
    label="2DPCA"
)
ax3.plot(
    p3_dims,
    time_mpca,
    linestyle=':',
    marker='*',
    label="MPCA"
)
ax1.set_xticks(p3_dims)
ax3.set_xlabel("Values P1, P2 and P3 of MPCA")
ax3.set_ylabel("Time in s")
ax3.legend()

# plot the storage
ax4.plot(
    p3_dims,
    storage_pca,
    linestyle='-',
    marker='s',
    label="PCA"
)
ax4.plot(
    p3_dims,
    storage_2dpca,
    linestyle='--',
    marker='o',
    label="2DPCA"
)
ax4.plot(
    p3_dims,
    storage_mpca,
    linestyle=':',
    marker='*',
    label="MPCA"
)
ax1.set_xticks(p3_dims)
ax4.set_xlabel("Values P1, P2 and P3 of MPCA")
ax4.set_ylabel("Storage (Numbers of scalars in reduced space)")
ax4.legend()

plt.show()
