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
    begin_time = time.time()
    # path
    path = os.path.join(source, dataset)
    data_path = os.path.join(path, f'{dataset}_base.fvecs')

    X = fvecs_mmap(data_path)
    N, D = X.shape
    print(N, D)
    pca = PCA(n_components=D)
    if N < 1000000:
        pca.fit(X)
    else:
        pca.fit(X[:1000000])
    print("train finished")
    projection_matrix = pca.components_.T
    sample_base = np.dot(X[:1000000], projection_matrix)
    mean_ = np.mean(sample_base[:1000000], axis=0)
    var_ = np.var(sample_base[:1000000], axis=0)
    mean_var = np.vstack((mean_, var_))

    pca_data_path = os.path.join(path, f'{dataset}_proj.fvecs')
    matrix_save_path = os.path.join(path, f'{dataset}_pca.fvecs')
    mean_save_path = os.path.join(path, f'{dataset}_mean.fvecs')

    fvecs_write(matrix_save_path, projection_matrix)
    fvecs_write(mean_save_path, mean_var)
    end_time = time.time()

    print(f"Large PCA Train Time:: {end_time - begin_time}(s)")
