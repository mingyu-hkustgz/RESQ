import time

import numpy as np
import faiss
import struct
import os
from utils import *
import argparse

source = './DATA'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='random projection')
    parser.add_argument('-d', '--dataset', help='dataset', default='msmarc')
    parser.add_argument('-b', '--bits', help='quantized bits', default=256)
    parser.add_argument('-c', '--centroids', help='IVF centroids', default=4096)
    args = vars(parser.parse_args())
    dataset = args['dataset']
    bits = int(args['bits'])
    K = int(args['centroids'])

    print(f"Clustering - {dataset}")
    # path
    path = os.path.join(source, dataset)
    data_path = os.path.join(path, f'{dataset}_proj.fvecs')
    X = fvecs_mmap(data_path)
    X = X[:, :bits]
    D = X.shape[1]
    centroids_path = os.path.join(path, f'p{dataset}_centroid_{K}.fvecs')
    dist_to_centroid_path = os.path.join(path, f'p{dataset}_dist_to_centroid_{K}.fvecs')
    cluster_id_path = os.path.join(path, f'p{dataset}_cluster_id_{K}.ivecs')

    # cluster data vectors
    start = time.time()
    index = faiss.index_factory(D, f"IVF{K},Flat")
    index.verbose = True
    index.train(X)
    centroids = index.quantizer.reconstruct_n(0, index.nlist)
    dist_to_centroid, cluster_id = index.quantizer.search(X, 1)
    dist_to_centroid = dist_to_centroid ** 0.5
    end = time.time()
    to_fvecs(dist_to_centroid_path, dist_to_centroid)
    I64vecs_write(cluster_id_path, cluster_id)
    to_fvecs(centroids_path, centroids)

    time_log = f"./results/time-log/{dataset}/{dataset}-IVF-Time.log"
    f = open(time_log, "a")
    print(f"{dataset} IVF Time Cost {end - start}(s)", file=f)
