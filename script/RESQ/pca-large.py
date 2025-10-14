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
    print(N, D)
    begin_time = time.time()
    pca = PCA(n_components=D)
    if N < 1000000:
        pca.fit(X)
    else:
        pca.fit(X[:1000000])
    print("train finished")
    projection_matrix = pca.components_.T
    for i in range(0, N, 1000000):
        ends = min(N, i + 1000000)
        print(f"Processing ({i}, {ends})")
        X[i:ends] = np.dot(X[i:ends], projection_matrix)
    mean_ = np.mean(X[:1000000], axis=0)
    var_ = np.var(X[:1000000], axis=0)
    X -= mean_
    mean_var = np.vstack((mean_, var_))

    pca_data_path = os.path.join(path, f'{dataset}_proj.fvecs')
    matrix_save_path = os.path.join(path, f'{dataset}_pca.fvecs')
    mean_save_path = os.path.join(path, f'{dataset}_mean.fvecs')
    end_time = time.time()
    fvecs_write(pca_data_path, X)
    fvecs_write(matrix_save_path, projection_matrix)
    fvecs_write(mean_save_path, mean_var)

    time_log = f"./results/time-log/{dataset}/{dataset}-PCA-Time.log"
    f = open(time_log, "w")
    print(f"{dataset} PCA Time Cost {end_time - begin_time}(s)", file=f)
