"""Evaluate the quality of the compressed videos.
"""

# License: BSD 3 clause

from time import time
import argparse

import matplotlib.pyplot as plt
import numpy as np

# Datasets
from Datasets.avletters import fetch_avletters_averaged

# Feature Extraction Methods
from SubspaceLearningAlgorithms.pca import PCA
from SubspaceLearningAlgorithms.twodim_pca import TwoDimensionalPCA
from SubspaceLearningAlgorithms.multilinear_pca import MultilinearPCA


parser = argparse.ArgumentParser(
    description="Evaluate the quality of the compressed "
    "videos in the user specified dataset."
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
    p = 23
    p1 = 12
    p2 = 12
    p3 = 12

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
pca = PCA(p)
data_pca = pca.fit(data.reshape(n_samples, -1))

# Reshape the data samples into matrices and apply 2DPCA
print("----------2DPCA----------")
print("Applying 2DPCA to the data")
twodpca = TwoDimensionalPCA(p3)
data_2dpca = twodpca.fit(data.reshape(n_samples, n_frames*n_rows, n_columns))

# Apply MPCA
print("----------MPCA----------")
print("Applying MPCA to the data")
mpca = MultilinearPCA([p1, p2, p3], n_iterations=5)
data_mpca = mpca.fit(data)

# Reconstruct images
data = data + data_mean
reconstructed_data_pca = pca.inverse_transform(data_pca).reshape(data.shape) + data_mean
reconstructed_data_2dpca = twodpca.inverse_transform(data_2dpca).reshape(data.shape) + data_mean
reconstructed_data_mpca = mpca.inverse_transform(data_mpca) + data_mean

# Plot some images and their reconstruction
fig, axs = plt.subplots(
    4, n_frames,
    subplot_kw=dict(xticks=[], yticks=[]),
    gridspec_kw=dict(hspace=0.1, wspace=0.1)
)
# For images plot the first sample images
for i in range(n_frames):
    axs[0, i].imshow(data[4, i], cmap='binary_r')
    axs[1, i].imshow(reconstructed_data_pca[4, i], cmap='binary_r')
    axs[2, i].imshow(reconstructed_data_2dpca[4, i], cmap='binary_r')
    axs[3, i].imshow(reconstructed_data_mpca[4, i], cmap='binary_r')


axs[0,0].set_ylabel('original data\ninput')
axs[1,0].set_ylabel('reconstructed data\nPCA')
axs[2,0].set_ylabel('reconstructed data\n2DPCA')
axs[3,0].set_ylabel('reconstructed data\nMPCA')

plt.show()
