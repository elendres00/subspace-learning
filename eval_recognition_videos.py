"""Evaluate the subspace learning algorithms on the
reconstruction error, the recognition accuracy and the
computation time on a given video dataset.

The dataset is specified by the user and we randomly split it
with a 70/30 split in a train and test set. After that the
subspace learning methods are applied and the K-Nearest-Neighbor
classifier with K=1 is used to predict the labels of the test set.
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
    description="Evaluate the subspace learning algorithms on the "
    "reconstruction error, the recognition accuracy and the "
    "computation time on a given video dataset."
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
    p_pca_dims = [4, 29, 96, 228, 446]
    p2_2dpca_dims = [1, 3, 10, 24, 47]
    p1_mpca_dims = [2, 4, 6, 8, 10]
    p2_mpca_dims = [10, 20, 30, 40, 50]
    p3_mpca_dims = [13, 27, 40, 53, 67]

print(
    "The data has {0} samples with shape {1}."
    "\n".format(data.shape[0], data.shape[1:])
)

# Number of repetitions of the experiment
repetitions = 10
# Number of dimensions to test
n_dims = len(p_pca_dims)


# Set the arrays to store the results
time_pca = np.empty((repetitions, n_dims))
time_2dpca = np.empty((repetitions, n_dims))
time_mpca = np.empty((repetitions, n_dims))

storage_pca = np.empty((repetitions, n_dims))
storage_2dpca = np.empty((repetitions, n_dims))
storage_mpca = np.empty((repetitions, n_dims))

reconstruction_error_pca = np.empty((repetitions, n_dims))
reconstruction_error_2dpca = np.empty((repetitions, n_dims))
reconstruction_error_mpca = np.empty((repetitions, n_dims))

accuracies_pca = np.empty((repetitions, n_dims))
accuracies_2dpca = np.empty((repetitions, n_dims))
accuracies_mpca = np.empty((repetitions, n_dims))

# Set the seed to get reproducable results
np.random.seed(42)

for repetition, random_state in enumerate(np.random.randint(400, size=repetitions)):
    # randomly split the data in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.333, random_state=random_state,
        stratify=np.char.add(target.astype(np.str), persons)
    )

    X_train_n_samples, X_train_n_frames, X_train_rows, X_train_columns = X_train.shape
    X_test_n_samples, X_test_frames, X_test_rows, X_test_columns = X_test.shape

    # normalize the data
    X_train_mean = X_train.mean(axis=0)
    X_train = X_train - X_train_mean
    X_test = X_test - X_train_mean

    dims = zip(p_pca_dims, p2_2dpca_dims, p1_mpca_dims, p2_mpca_dims, p3_mpca_dims)
    for i_dim, dim in enumerate(dims):
        p_pca, p2_2dpca, p1_mpca, p2_mpca, p3_mpca = dim
        # Apply PCA on the vectorized videos
        print("---------- PCA on videos ----------")
        print("Apply PCA to the data")
        t0 = time()
        pca = PCA(n_components=p_pca)
        X_train_pca = pca.fit(X_train.reshape(X_train_n_samples, -1))
        X_test_pca = pca.transform(X_test.reshape(X_test_n_samples, -1))
        t_pca = time() - t0
        time_pca[repetition, i_dim] = t_pca
        print("done in {:.3f}s\n".format(t_pca))

        print(
            "The number of principal components used is: "
            "{}\n".format(pca.n_components)
        )

        # Calculate the storage needed in the projected space
        storage = n_samples * p_pca + n_frames * n_rows * n_columns * p_pca
        storage_pca[repetition, i_dim] = storage
        print("Scalars needed in projected space: {}\n".format(storage))

        # Calculate the reconstruction error on training data
        X_train_reconstructed = pca.inverse_transform(X_train_pca).reshape(X_train.shape)
        reconstruction_error = np.mean((X_train - X_train_reconstructed)**2)
        reconstruction_error_pca[repetition, i_dim] = reconstruction_error
        print("Reconstruction Error: {}\n".format(reconstruction_error))

        # Use KNN to classify the projected test data
        knn = KNeighborsClassifier(n_neighbors=1, weights="distance")
        knn.fit(X_train_pca, y_train)
        y_pred = knn.predict(X_test_pca)

        accuracy = accuracy_score(y_test, y_pred)
        accuracies_pca[repetition, i_dim] = accuracy
        print("Accuracy: {}\n".format(accuracy))


        # Apply 2DPCA on the image resulting from concatenating the
        # frames of each video sample
        print("---------- 2DPCA ----------")
        print("Apply 2DPCA to the data")
        t0 = time()
        twodpca = TwoDimensionalPCA(n_components=p2_2dpca)
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
        time_2dpca[repetition, i_dim] = t_2dpca
        print("done in {:.3f}s\n".format(t_2dpca))

        print(
            "The number of principal components used is: "
            "{}\n".format(twodpca.n_components)
        )

        # Calculate the storage needed in the projected space
        storage = (n_samples * n_frames * n_rows * p2_2dpca
                + n_frames * n_rows * p2_2dpca)
        storage_2dpca[repetition, i_dim] = storage
        print("Scalars needed in projected space: {}\n".format(storage))

        # Calculate the reconstruction error on training data
        X_train_reconstructed = twodpca.inverse_transform(X_train_2dpca).reshape(X_train.shape)
        reconstruction_error = np.mean((X_train - X_train_reconstructed)**2)
        reconstruction_error_2dpca[repetition, i_dim] = reconstruction_error
        print("Reconstruction Error: {}\n".format(reconstruction_error))

        # Use KNN to classify the projected test data
        knn = KNeighborsClassifier(n_neighbors=1, weights="distance")
        knn.fit(X_train_2dpca.reshape(X_train_n_samples, -1), y_train)
        y_pred = knn.predict(X_test_2dpca.reshape(X_test_n_samples, -1))

        accuracy = accuracy_score(y_test, y_pred)
        accuracies_2dpca[repetition, i_dim] = accuracy
        print("Accuracy: {}\n".format(accuracy))


        # Apply MPCA on the full data samples
        print("---------- MPCA on videos ----------")
        print("Apply MPCA to the data")
        t0 = time()
        mpca = MultilinearPCA(
            projection_shape=[p1_mpca, p2_mpca, p3_mpca],
            n_iterations=5
        )
        X_train_mpca = mpca.fit(X_train)
        X_test_mpca = mpca.transform(X_test)
        t_mpca = time() - t0
        time_mpca[repetition, i_dim] = t_mpca
        print("done in {:.3f}s\n".format(t_mpca))

        print(
            "The number of components used in each mode is: "
            "{}\n".format(mpca.projection_shape)
        )

        # Calculate the storage needed in the projected space
        storage = (n_samples * p1_mpca * p2_mpca * p3_mpca
                + n_frames * p1_mpca + n_rows * p2_mpca + n_columns * p3_mpca)
        storage_mpca[repetition, i_dim] = storage
        print("Scalars needed in projected space: {}\n".format(storage))

        # Calculate the reconstruction error on training data
        X_train_reconstructed = mpca.inverse_transform(X_train_mpca)
        reconstruction_error = np.mean((X_train - X_train_reconstructed)**2)
        reconstruction_error_mpca[repetition, i_dim] = reconstruction_error
        print("Reconstruction Error: {}\n".format(reconstruction_error))

        # Use KNN to classify the projected test data
        knn = KNeighborsClassifier(n_neighbors=1, weights="distance")
        knn.fit(X_train_mpca.reshape(X_train_n_samples, -1), y_train)
        y_pred = knn.predict(X_test_mpca.reshape(X_test_n_samples, -1))

        accuracy = accuracy_score(y_test, y_pred)
        accuracies_mpca[repetition, i_dim] = accuracy
        print("Accuracy: {}\n".format(accuracy))


# Average the results
time_pca = np.mean(time_pca, axis=0)
time_2dpca = np.mean(time_2dpca, axis=0)
time_mpca = np.mean(time_mpca, axis=0)

storage_pca = np.mean(storage_pca, axis=0)
storage_2dpca = np.mean(storage_2dpca, axis=0)
storage_mpca = np.mean(storage_mpca, axis=0)

reconstruction_error_pca = np.mean(reconstruction_error_pca, axis=0)
reconstruction_error_2dpca = np.mean(reconstruction_error_2dpca, axis=0)
reconstruction_error_mpca = np.mean(reconstruction_error_mpca, axis=0)

accuracies_pca = np.mean(accuracies_pca, axis=0)
accuracies_2dpca = np.mean(accuracies_2dpca, axis=0)
accuracies_mpca = np.mean(accuracies_mpca, axis=0)

# Plot the results
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

# plot the reconstruction errors
ax1.plot(
    p2_mpca_dims,
    reconstruction_error_pca,
    linestyle='-',
    marker='s',
    label="PCA"
)
ax1.plot(
    p2_mpca_dims,
    reconstruction_error_2dpca,
    linestyle='--',
    marker='o',
    label="2DPCA"
)
ax1.plot(
    p2_mpca_dims,
    reconstruction_error_mpca,
    linestyle=':',
    marker='*',
    label="MPCA"
)
ax1.set_xticks(p2_mpca_dims)
ax1.set_xlabel("Value P2 of MPCA")
ax1.set_ylabel("Mean Squared Reconstruction Error")
ax1.legend()

# plot the accuracies
ax2.plot(
    p2_mpca_dims,
    accuracies_pca,
    linestyle='-',
    marker='s',
    label="PCA"
)
ax2.plot(
    p2_mpca_dims,
    accuracies_2dpca,
    linestyle='--',
    marker='o',
    label="2DPCA"
)
ax2.plot(
    p2_mpca_dims,
    accuracies_mpca,
    linestyle=':',
    marker='*',
    label="MPCA"
)
ax1.set_xticks(p2_mpca_dims)
ax2.set_xlabel("Value P2 of MPCA")
ax2.set_ylabel("Accuracy")
ax2.legend()

# plot the time
ax3.plot(
    p2_mpca_dims,
    time_pca,
    linestyle='-',
    marker='s',
    label="PCA"
)
ax3.plot(
    p2_mpca_dims,
    time_2dpca,
    linestyle='--',
    marker='o',
    label="2DPCA"
)
ax3.plot(
    p2_mpca_dims,
    time_mpca,
    linestyle=':',
    marker='*',
    label="MPCA"
)
ax1.set_xticks(p2_mpca_dims)
ax3.set_xlabel("Value P2 of MPCA")
ax3.set_ylabel("Time in s")
ax3.legend()

# plot the storage
ax4.plot(
    p2_mpca_dims,
    storage_pca,
    linestyle='-',
    marker='s',
    label="PCA"
)
ax4.plot(
    p2_mpca_dims,
    storage_2dpca,
    linestyle='--',
    marker='o',
    label="2DPCA"
)
ax4.plot(
    p2_mpca_dims,
    storage_mpca,
    linestyle=':',
    marker='*',
    label="MPCA"
)
ax1.set_xticks(p2_mpca_dims)
ax4.set_xlabel("Value P2 of MPCA")
ax4.set_ylabel("Storage (Numbers of scalars in reduced space)")
ax4.legend()

plt.show()
