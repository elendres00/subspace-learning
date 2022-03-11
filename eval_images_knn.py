"""Evaluate the subspace learning algorithms on images.

The dataset is specified by the user and we randomly split it
in a train and test set. After that the subspace learning methods
are applied and the K-Nearest-Neighbor classifier with K=1 is used
to predict the labels of the test set.
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
from sklearn.datasets import load_digits
from sklearn.datasets import fetch_olivetti_faces
from sklearn.datasets import fetch_lfw_people

# Feature Extraction Methods
from SubspaceLearningAlgorithms.pca import PCA
from SubspaceLearningAlgorithms.twodim_pca import TwoDimensionalPCA
from SubspaceLearningAlgorithms.multilinear_pca import MultilinearPCA


# Parse Arguments
parser = argparse.ArgumentParser(
    description="Evaluates the performance of different PCA based "
    "subspace learning methods on an image dataset chosen by the "
    "user using the KNN classifier."
)

parser.add_argument(
    '--dataset',
    choices=['digits', 'olivetti', 'lfw'],
    default='digits',
    help='The dataset used (default: digits)'
)

args = parser.parse_args()


# Get the user specified dataset
if (args.dataset == 'digits'):
    digits = load_digits()
    data = digits.images
    n_samples, n_rows, n_columns = data.shape
    target = digits.target
    p_dims = [1, 4, 9, 15, 24, 35, 47, 62]
    p1_dims = [1, 2, 3, 4, 5, 6, 7, 8]
    p2_dims = [1, 2, 3, 4, 5, 6, 7, 8]
elif (args.dataset == 'olivetti'):
    olivetti_faces = fetch_olivetti_faces(data_home='./Datasets/olivetti')
    data = olivetti_faces.images
    n_samples, n_rows, n_columns = data.shape
    target = olivetti_faces.target
    p_dims = [2, 6, 13, 23, 36, 52, 71, 91, 116, 143]
    p1_dims = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
    p2_dims = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
elif (args.dataset == 'lfw'):
    lfw_people = fetch_lfw_people(
        data_home='./Datasets/lfw', min_faces_per_person=70
    )
    data = lfw_people.images
    n_samples, n_rows, n_columns = data.shape
    target = lfw_people.target
    p_dims = [5, 20, 44, 79, 123, 177, 241, 315, 398, 491]
    p1_dims = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
    p2_dims = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]

print(
    "The data has {0} samples with shape {1}."
    "\n".format(data.shape[0], data.shape[1:])
)

# randomly split the data in train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.2, random_state=12, stratify=target
)
X_train_n_samples = X_train.shape[0]
X_test_n_samples = X_test.shape[0]

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

for p, p1, p2 in zip(p_dims, p1_dims, p2_dims):
    # Apply PCA on the vectorized images
    print("---------- PCA ----------")
    print("Apply PCA on the data")
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
    storage = n_samples * p + n_rows * n_columns * p
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


    # Apply 2DPCA on the images
    print("---------- 2DPCA ----------")
    print("Apply 2DPCA on the data")
    t0 = time()
    twodpca = TwoDimensionalPCA(n_components=p2)
    X_train_2dpca = twodpca.fit(X_train)
    X_test_2dpca = twodpca.transform(X_test)
    t_2dpca = time() - t0
    time_2dpca.append(t_2dpca)
    print("done in {:.3f}s\n".format(t_2dpca))

    print(
        "The number of principal components used is: "
        "{}\n".format(twodpca.n_components)
    )

    # Calculate the storage needed in the projected space
    storage = n_samples * n_rows * p2 + n_rows * p2
    storage_2dpca.append(storage)
    print("Scalars needed in projected space: {}\n".format(storage))

    # Calculate the reconstruction error on training data
    X_train_reconstructed = twodpca.inverse_transform(X_train_2dpca)
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


    # Apply MPCA on the images
    print("---------- MPCA ----------")
    print("Apply MPCA on the data")
    t0 = time()
    mpca = MultilinearPCA(projection_shape=[p1, p2], n_iterations=5)
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
    storage = n_samples * p1 * p2 + n_rows * p1 + n_columns * p2
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
    p2_dims,
    reconstruction_error_pca,
    linestyle='-',
    marker='s',
    label="PCA"
)
ax1.plot(
    p2_dims,
    reconstruction_error_2dpca,
    linestyle='--',
    marker='o',
    label="2DPCA"
)
ax1.plot(
    p2_dims,
    reconstruction_error_mpca,
    linestyle=':',
    marker='*',
    label="MPCA"
)
ax1.set_xticks(p2_dims)
ax1.set_xlabel("Values P1 and P2 of MPCA")
ax1.set_ylabel("Mean Squared Reconstruction Error")
ax1.legend()

# plot the accuracies
ax2.plot(
    p2_dims,
    accuracies_pca,
    linestyle='-',
    marker='s',
    label="PCA"
)
ax2.plot(
    p2_dims,
    accuracies_2dpca,
    linestyle='--',
    marker='o',
    label="2DPCA"
)
ax2.plot(
    p2_dims,
    accuracies_mpca,
    linestyle=':',
    marker='*',
    label="MPCA"
)
ax1.set_xticks(p2_dims)
ax2.set_xlabel("Values P1 and P2 of MPCA")
ax2.set_ylabel("Accuracy")
ax2.legend()

# plot the time
ax3.plot(
    p2_dims,
    time_pca,
    linestyle='-',
    marker='s',
    label="PCA"
)
ax3.plot(
    p2_dims,
    time_2dpca,
    linestyle='--',
    marker='o',
    label="2DPCA"
)
ax3.plot(
    p2_dims,
    time_mpca,
    linestyle=':',
    marker='*',
    label="MPCA"
)
ax1.set_xticks(p2_dims)
ax3.set_xlabel("Values P1 and P2 of MPCA")
ax3.set_ylabel("Time in s")
ax3.legend()

# plot the storage
ax4.plot(
    p2_dims,
    storage_pca,
    linestyle='-',
    marker='s',
    label="PCA"
)
ax4.plot(
    p2_dims,
    storage_2dpca,
    linestyle='--',
    marker='o',
    label="2DPCA"
)
ax4.plot(
    p2_dims,
    storage_mpca,
    linestyle=':',
    marker='*',
    label="MPCA"
)
ax1.set_xticks(p2_dims)
ax4.set_xlabel("Values P1 and P2 of MPCA")
ax4.set_ylabel("Storage (Numbers of scalars in reduced space)")
ax4.legend()

plt.show()
