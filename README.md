# subspace-learning
Experimental comparison of PCA based linear and multilinear subspace learning
algorithms PCA, 2DPCA and MPCA.

## Dependencies
`scikit-learn`

## Datasets
The UCI ML Digits, ORL and LFW datasets are automatically downloaded by the scikit-learn library
functions and then stored in the `/Datasets/` folder. 
The [AVLetters](http://www.ee.surrey.ac.uk/Projects/LILiR/datasets/avletters1/index.html)
dataset with the lip videos in Matlab's `.mat` format should be placed in the datasets
folder like this `/Datasets/avletters/Lips/lip_video.mat`.

## Experiments
There are 3 kinds of experiments:
1. Evaluate how different dimensions affect the reconstruction error of MPCA
`eval_dimensions_mpca.py`.

2. Evaluate the reconstruction error, the recognition accuracy and the
computation time and space needed for different subspace learning algorithms
PCA, 2DPCA and MPCA `eval_recognition_images.py`, `eval_recognition_videos.py`.

3. Evaluate the visual quality of the compressed images and videos by the 
subspace learning algorithms PCA, 2DPCA and MPCA 
`eval_compression_images.py`, `eval_compression_videos.py`.

## Running the experiments
The experiments can be run as follows
```
python experiment.py --dataset=dataset
```
by replacing `experiment.py` with one of the experiments above and `dataset` by one of the datasets below:
- UCI ML Digits dataset `--dataset=digits`
- ORL face dataset `--dataset=olivetti`
- LFW face dataset `--dataset=lfw`
- AVLetters lip reading dataset `--dataset=avletters`
