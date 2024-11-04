import math

import numpy as np
import struct
import time
import os
from utils import *
import argparse
from tqdm import tqdm
from sklearn.decomposition import PCA

source = './DATA'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='random projection')
    parser.add_argument('-d', '--dataset', help='dataset', default='msmarc')
    parser.add_argument('-b', '--bit', help='quantized bits', default=256)
    args = vars(parser.parse_args())
    dataset = args['dataset']
    bits = int(args['bit'])

    # path
    path = os.path.join(source, dataset)
    data_path = os.path.join(path, f'{dataset}_base.fvecs')

    X = fvecs_read(data_path)
    N, D = X.shape
    pca = PCA(n_components=D)
    if N < 1000000:
        pca.fit(X)
    else:
        pca.fit(X[:1000000])
    projection_matrix = pca.components_.T
    base = np.dot(X, projection_matrix)

    mean_ = np.mean(base[:1000000], axis=0)
    var_ = np.var(base[:1000000], axis=0)
    base -= mean_
    mean_var = np.vstack((mean_, var_))

    pca_data_path = os.path.join(path, f'{dataset}_proj.fvecs')
    matrix_save_path = os.path.join(path, f'{dataset}_pca.fvecs')
    mean_save_path = os.path.join(path, f'{dataset}_mean.fvecs')

    fvecs_write(pca_data_path, base)
    fvecs_write(matrix_save_path, projection_matrix)
    fvecs_write(mean_save_path, mean_var)
