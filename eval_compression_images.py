"""Evaluate the quality of the compressed images.
"""

# License: BSD 3 clause

from time import time
import argparse

import matplotlib.pyplot as plt
import numpy as np

# Datasets
from sklearn.datasets import load_digits
from sklearn.datasets import fetch_olivetti_faces
from sklearn.datasets import fetch_lfw_people

# Feature Extraction Methods
from SubspaceLearningAlgorithms.pca import PCA
from SubspaceLearningAlgorithms.twodim_pca import TwoDimensionalPCA
from SubspaceLearningAlgorithms.multilinear_pca import MultilinearPCA


parser = argparse.ArgumentParser(
    description="Evaluate the quality of the compressed "
    "images in the user specified dataset."
)

parser.add_argument(
    '--dataset',
    choices=['digits', 'lfw', 'olivetti'],
    default='digits'
)

args = parser.parse_args()


# Get the user specified dataset
if (args.dataset == 'digits'):
    digits = load_digits()
    data = digits.images
    n_samples, n_rows, n_columns = data.shape
    p_pca = 24
    p2_2dpca = 3
    p1_mpca = 5
    p2_mpca = 5
elif (args.dataset == 'olivetti'):
    olivetti_faces = fetch_olivetti_faces(data_home='./Datasets/olivetti')
    data = olivetti_faces.images
    n_samples, n_rows, n_columns = data.shape
    p_pca = 40
    p2_2dpca = 7
    p1_mpca = 21
    p2_mpca = 21
elif (args.dataset == 'lfw'):
    lfw_people = fetch_lfw_people(
        data_home='./Datasets/lfw', min_faces_per_person=70
    )
    data = lfw_people.images
    n_samples, n_rows, n_columns = data.shape
    p_pca = 103
    p2_2dpca = 5
    p1_mpca = 21
    p2_mpca = 16

print(
    "The data has {0} samples with shape {1}."
    "\n".format(data.shape[0], data.shape[1:])
)

# normalize the data
data_mean = data.mean(axis=0)
data = data - data_mean


# Reshape the data samples into vectors and apply PCA
print("----------PCA----------")
print("Applying PCA to the data")
pca = PCA(p_pca)
data_pca = pca.fit(data.reshape(n_samples, -1))

# Apply 2DPCA
print("----------2DPCA----------")
print("Applying 2DPCA to the data")
twodpca = TwoDimensionalPCA(p2_2dpca)
data_2dpca = twodpca.fit(data)

# Apply MPCA
print("----------MPCA----------")
print("Applying MPCA to the data")
mpca = MultilinearPCA([p1_mpca, p2_mpca], n_iterations=5)
data_mpca = mpca.fit(data)

# Reconstruct images
data = data + data_mean
reconstructed_data_pca = pca.inverse_transform(data_pca).reshape(data.shape) + data_mean
reconstructed_data_2dpca = twodpca.inverse_transform(data_2dpca) + data_mean
reconstructed_data_mpca = mpca.inverse_transform(data_mpca) + data_mean

# Plot some images and their reconstruction
fig, axs = plt.subplots(
    4, 5,
    subplot_kw=dict(xticks=[], yticks=[]),
    gridspec_kw=dict(hspace=0.1, wspace=0.1)
)
# For images plot the first sample images
for i in range(5):
    axs[0, i].imshow(data[i], cmap='binary_r')
    axs[1, i].imshow(reconstructed_data_pca[i], cmap='binary_r')
    axs[2, i].imshow(reconstructed_data_2dpca[i], cmap='binary_r')
    axs[3, i].imshow(reconstructed_data_mpca[i], cmap='binary_r')


axs[0,0].set_ylabel('original')
axs[1,0].set_ylabel('PCA')
axs[2,0].set_ylabel('2DPCA')
axs[3,0].set_ylabel('MPCA')

plt.show()
