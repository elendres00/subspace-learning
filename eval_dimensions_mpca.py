"""Evaluate how different choices of the dimensions P1, ..., PN
for the reduced space of MPCA affect the reconstruction error.
"""

# License: BSD 3 clause

import argparse

import matplotlib.pyplot as plt
import numpy as np

# Datasets
from sklearn.datasets import load_digits
from sklearn.datasets import fetch_olivetti_faces
from sklearn.datasets import fetch_lfw_people
from Datasets.avletters import fetch_avletters_averaged

# Feature Extraction Method
from SubspaceLearningAlgorithms.multilinear_pca import MultilinearPCA


parser = argparse.ArgumentParser(
    description="In this experiment, we evaluate how different choices"
    "of the dimensions P1, ..., PN for the reduced space of MPCA affect"
    "the reconstruction error."
)

parser.add_argument(
    '--dataset',
    choices=['digits', 'olivetti', 'lfw', 'avletters'],
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
    p1_dims = [1, 2, 4]
    p2_dims = [4, 2, 1]
elif (args.dataset == 'olivetti'):
    olivetti_faces = fetch_olivetti_faces(data_home='./Datasets/olivetti')
    data = olivetti_faces.images
    n_samples, n_rows, n_columns = data.shape
    target = olivetti_faces.target
    p1_dims = [7, 14, 21, 28, 35]
    p2_dims = [62, 31, 21, 16, 13]
elif (args.dataset == 'lfw'):
    lfw_people = fetch_lfw_people(
        data_home='./Datasets/lfw', min_faces_per_person=70
    )
    data = lfw_people.images
    n_samples, n_rows, n_columns = data.shape
    target = lfw_people.target
    p1_dims = [8, 15, 21, 27, 33]
    p2_dims = [42, 22, 16, 12, 10]
elif (args.dataset == 'avletters'):
    data, persons, target = fetch_avletters_averaged()
    n_samples, n_frames, n_rows, n_columns = data.shape
    p1_dims = [2, 2, 2, 4, 4, 4, 6, 6, 6]
    p2_dims = [14, 20, 26, 14, 20, 26, 14, 20, 26]
    p3_dims = [77, 54, 41, 39, 27, 21, 26, 18, 14]

print(
    "The data has {0} samples with shape {1}."
    "\n".format(data.shape[0], data.shape[1:])
)

tensor_order = len(data.shape[1:])

# normalize the data
data_mean = data.mean(axis=0)
data = data - data_mean

# Apply MPCA with different dimensions P1, ..., PN with the
# same storage requirements
print("----------MPCA----------")
if tensor_order == 2:
    for p1, p2 in zip(p1_dims, p2_dims):
        print("Applying MPCA to the data with P1 = {0} and P2 = {1}".format(p1, p2))
        mpca = MultilinearPCA([p1, p2], n_iterations=4)
        data_mpca = mpca.fit(data)
        # Reconstruct videos
        reconstructed_data_mpca = mpca.inverse_transform(data_mpca)
        print("Reconstruction Error: {}".format(np.mean((data - reconstructed_data_mpca)**2)))
elif tensor_order == 3:
    for p1, p2, p3 in zip(p1_dims, p2_dims, p3_dims):
        print("Applying MPCA to the data with P1 = {0}, P2 = {1} and P3 = {2}".format(p1, p2, p3))
        mpca = MultilinearPCA([p1, p2, p3], n_iterations=4)
        data_mpca = mpca.fit(data)
        # Reconstruct videos
        reconstructed_data_mpca = mpca.inverse_transform(data_mpca)
        print("Reconstruction Error: {}".format(np.mean((data - reconstructed_data_mpca)**2)))
