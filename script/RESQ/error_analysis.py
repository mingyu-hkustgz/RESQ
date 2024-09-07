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
    query_path = os.path.join(path, f'{dataset}_query.fvecs')

    X = fvecs_read(data_path)
    Q = fvecs_read(query_path)
    X = X[:100000]
    N, D = X.shape
    mean = np.mean(X, axis=0)
    X -= mean
    Q -= mean
    pca = PCA(n_components=D)
    pca.fit(X)
    projection_matrix = pca.components_.T
    base = np.dot(X, projection_matrix)
    query = np.dot(Q, projection_matrix)

    error = []
    for i in tqdm(range(100)):
        error.append(-2 * base[:, bits:] * query[i][bits:])

    np_error = np.vstack(error)
    print(3.0 * np.std(np_error))

# 0.0013919250341132283 128d
# 0.00048824171244632453