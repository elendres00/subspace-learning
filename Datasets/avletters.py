"""AVLetters lip dataset.

The original dataset is available from

    http://www.ee.surrey.ac.uk/Projects/LILiR/datasets/avletters1/index.html

This dataset consists of three repetitions by each of 10 talkers,
five male (two with moustaches) and five female,
of the isolated letters A-Z, a total of 780 utterances

References
----------
I. Matthews, T.Cootes, J. Bangham, S. Cox, and R. Harvey.
Extraction of visual features for lipreading.
IEEE Trans. on Pattern Analysis and Machine Vision,
vol. 24, no. 2, pp. 198-213, 2002.
"""

# License: BSD 3 clause

import numpy as np
from string import ascii_uppercase
import random
from os import listdir
from os.path import dirname, exists, isfile, join
from scipy.io import loadmat

folderpath = join(dirname(__file__), './avletters/Lips/')

def fetch_avletters_averaged():
    """Load the AVLetters dataset with averaged frames

    ================   =======================
    Classes                                 26
    Samples total                          780
    Dimensionality                (12, 60, 80)
    Features           real, between 255 and 0
    ================   =======================

    Returns
    -------
     (lip_videos, label) : tuple
        lip_videos : ndarray of shape (780, 12, 60, 80)
            The lip videos with averaged frames.
            Each video consists of 12 60x80 image frames.

        persons : ndarray of shape (780,)
            The persons corresponding to the lip videos.

        label : ndarray of shape (780,)
            Labels corresponding to the lip videos.
            Those labels are ranging from 0-23 and
            correspond to the letters spoken in the lip video.
    """

    if not (exists(folderpath)):
        raise IOError("Data not found")

    lip_paths = []
    for f in listdir(folderpath):
        if isfile(join(folderpath, f)) and not f.endswith('Store'):
            lip_paths.append(f)

    n_samples = 780
    n_frames = 12
    n_rows = 60
    n_columns = 80

    people = ['Anya', 'Bill', 'Faye', 'John', 'Kate', 'Nicola', 'Stephen',
              'Steve', 'Verity', 'Yi']

    lip_videos = np.empty(shape=(n_samples, n_frames, n_rows, n_columns), dtype=float)
    persons = np.zeros(shape=(n_samples,), dtype='<U8')
    label = np.empty(shape=(n_samples,), dtype=int)

    # Save all lip videos in the preferred form
    for i, lip_path in enumerate(lip_paths):
        # Load the lip video
        lip_mat = loadmat(folderpath + lip_path)
        n_frames_curr = int(lip_mat['siz'][0,2])
        lip_video = lip_mat['vid'].reshape(n_columns, n_rows, n_frames_curr)
        lip_video = lip_video.transpose(2, 1, 0)

        # Average the video frames over a window of size
        # `n_frames_curr - n_frames + 1` so that the new video
        # has `n_frames` frames.
        window_size = n_frames_curr - n_frames + 1
        for j in range(n_frames):
            lip_videos[i, j] = lip_video[j:j+window_size].mean(axis=0)

        for p in people:
            if p in lip_path:
                persons[i] = p

        label[i] = ord(lip_path[0]) - ord('A')

    return (lip_videos, persons, label)

def fetch_avletters():
    """Load the AVLetters dataset.

    ================   =======================
    Classes                                 26
    Samples total                          780
    Dimensionality                (40, 60, 80)
    Features           real, between 255 and 0
    ================   =======================

    Returns
    -------
     (lip_videos, lip_videos_n_frames, persons, label) : tuple
        lip_video: ndarray of shape (780, 40, 60, 80)
            The lip videos.
            Each video has between 12 and 40 60x80 image frames.

        lip_video_n_frames : ndarray of shape (780,)
            Number of frames of each video sample.

        persons : ndarray of shape (780,)
            The persons corresponding to the lip videos.

        label : ndarray of shape (780,)
            Labels corresponding to the lip videos.
            Those labels are ranging from 0-23 and
            correspond to the letters spoken in the lip video.
    """

    if not (exists(folderpath)):
        raise IOError("Data not found")

    lip_paths = []
    for f in listdir(folderpath):
        if isfile(join(folderpath, f)) and not f.endswith('Store'):
            lip_paths.append(f)

    n_samples = 780
    max_n_frames = 40
    n_rows = 60
    n_columns = 80

    people = ['Anya', 'Bill', 'Faye', 'John', 'Kate', 'Nicola', 'Stephen',
              'Steve', 'Verity', 'Yi']

    lip_videos = np.zeros(shape=(n_samples, max_n_frames, n_rows, n_columns), dtype=float)
    lip_videos_n_frames = np.zeros(shape=(n_samples,), dtype=int)
    persons = np.zeros(shape=(n_samples,), dtype='<U8')
    label = np.zeros(shape=(n_samples,), dtype=int)

    # Save all lip videos in the preferred form
    for i, lip_path in enumerate(lip_paths):
        # Load the lip video
        lip_mat = loadmat(folderpath + lip_path)
        n_frames_curr = int(lip_mat['siz'][0,2])
        lip_video = lip_mat['vid'].reshape(n_columns, n_rows, n_frames_curr)
        lip_video = lip_video.transpose(2, 1, 0)

        for j in range(n_frames_curr):
            lip_videos[i, j] = lip_video[j]

        lip_videos_n_frames[i] = n_frames_curr

        for p in people:
            if p in lip_path:
                persons[i] = p

        label[i] = ord(lip_path[0]) - ord('A')

    return (lip_videos, lip_videos_n_frames, persons, label)
