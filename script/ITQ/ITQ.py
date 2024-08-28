import numpy as np
import struct
import time
import os
from utils import *
import argparse
from tqdm import tqdm
import torch

source = './DATA'


def Orthogonal(D):
    G = np.random.randn(D, D).astype('float32')
    Q, _ = np.linalg.qr(G)
    return Q


def pca_iterate_quantized(X_pad, max_iter=20):
    norms = np.linalg.norm(X_pad, axis=1)
    # remove zero vectors
    X = X_pad[norms != 0]
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    X = torch.from_numpy(X)
    N, D = X.shape
    R = torch.from_numpy(Orthogonal(D).T)
    for i in tqdm(range(max_iter)):
        X_tilde = X @ R
        B = X_tilde.sign()
        x0 = torch.sum(X * B / D ** 0.5, dim=(1,), keepdim=True)
        print(torch.mean(x0))
        [U, _, VT] = torch.svd(B.t() @ X)
        R = (VT.t() @ U.t())

    return bin_XP.numpy(), R.numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='random projection')
    parser.add_argument('-d', '--dataset', help='dataset', default='gist')
    args = vars(parser.parse_args())
    dataset = args['dataset']
    # path
    path                    = os.path.join(source, dataset)
    data_path               = os.path.join(path, f'{dataset}_base.fvecs')

    C = 4096
    centroids_path          = os.path.join(path, f'{dataset}_centroid_{C}.fvecs')
    dist_to_centroid_path   = os.path.join(path, f'{dataset}_dist_to_centroid_{C}.fvecs')
    cluster_id_path         = os.path.join(path, f'{dataset}_cluster_id_{C}.ivecs')

    X          = read_fvecs(data_path)
    centroids  = read_fvecs(centroids_path)
    cluster_id = read_ivecs(cluster_id_path)

    D = X.shape[1]
    B = (D + 63) // 64 * 64
    MAX_BD = max(D, B)

    projection_path          = os.path.join(path, f'P_C{C}_B{B}.fvecs')
    randomized_centroid_path = os.path.join(path, f'RandCentroid_C{C}_B{B}.fvecs')
    RN_path                  = os.path.join(path, f'RandNet_C{C}_B{B}.Ivecs')
    x0_path                  = os.path.join(path, f'x0_C{C}_B{B}.fvecs')

    X_pad         = np.pad(X, ((0, 0), (0, MAX_BD-D)), 'constant')
    centroids_pad = np.pad(centroids, ((0, 0), (0, MAX_BD-D)), 'constant')
    np.random.seed(0)

    cluster_id=np.squeeze(cluster_id)
    X_pad = X_pad - centroids_pad[cluster_id]
    bin_XP, R = pca_iterate_quantized(X_pad)

