"""Evaluate the quality of the compressed videos.
"""

# License: BSD 3 clause

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
    p_pca = 29
    p2_2dpca = 3
    p1_mpca = 4
    p2_mpca = 20
    p3_mpca = 27

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

# Reshape the data samples into matrices and apply 2DPCA
print("----------2DPCA----------")
print("Applying 2DPCA to the data")
twodpca = TwoDimensionalPCA(p2_2dpca)
data_2dpca = twodpca.fit(data.reshape(n_samples, n_frames*n_rows, n_columns))

# Apply MPCA
print("----------MPCA----------")
print("Applying MPCA to the data")
mpca = MultilinearPCA([p1_mpca, p2_mpca, p3_mpca], n_iterations=5)
data_mpca = mpca.fit(data)

# Reconstruct videos
data = data + data_mean
reconstructed_data_pca = pca.inverse_transform(data_pca).reshape(data.shape) + data_mean
reconstructed_data_2dpca = twodpca.inverse_transform(data_2dpca).reshape(data.shape) + data_mean
reconstructed_data_mpca = mpca.inverse_transform(data_mpca) + data_mean

# Plot six frame of one video and its reconstruction
fig, axs = plt.subplots(
    4, 6,
    subplot_kw=dict(xticks=[], yticks=[]),
    gridspec_kw=dict(hspace=0.001, wspace=0.1),
)
video_sample = 42
# Plot every second frame of the chosen video sample and its reconstruction
for i, frame in enumerate(range(0, n_frames, 2)):
    axs[0, i].imshow(data[video_sample, frame], cmap='binary_r')
    axs[1, i].imshow(reconstructed_data_pca[video_sample, frame], cmap='binary_r')
    axs[2, i].imshow(reconstructed_data_2dpca[video_sample, frame], cmap='binary_r')
    axs[3, i].imshow(reconstructed_data_mpca[video_sample, frame], cmap='binary_r')


axs[0,0].set_ylabel('original')
axs[1,0].set_ylabel('PCA')
axs[2,0].set_ylabel('2DPCA')
axs[3,0].set_ylabel('MPCA')

plt.show()
