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
    parser.add_argument('-d', '--dataset', help='dataset', default='gist')
    parser.add_argument('-b', '--bit', help='quantized bits', default=128)
    args = vars(parser.parse_args())
    dataset = args['dataset']
    bits = int(args['bit'])

    # path
    path = os.path.join(source, dataset)
    data_path = os.path.join(path, f'{dataset}_base.fvecs')

    X = fvecs_read(data_path)
    N, D = X.shape
    pca = PCA(n_components=D)
    pca.fit(X)
    projection_matrix = pca.components_.T
    base = np.dot(X, projection_matrix)

    pca_data_path = os.path.join(path, f'{dataset}_proj.fvecs')
    matrix_save_path = os.path.join(path, f'{dataset}_pca.fvecs')

    fvecs_write(pca_data_path, base)
    fvecs_write(matrix_save_path, projection_matrix)
