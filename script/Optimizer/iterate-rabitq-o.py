import numpy as np
import struct
import time
import os
from utils import *
import argparse
from tqdm import tqdm
from sklearn.decomposition import PCA
source = './DATA'


def Orthogonal(D):
    G = np.random.randn(D, D).astype('float32')
    Q, _ = np.linalg.qr(G)
    return Q


def GenerateBinaryCode(X, P):
    XP = np.dot(X, P)
    binary_XP = (XP > 0)
    X0 = np.sum(XP * (2 * binary_XP - 1) / D ** 0.5, axis=1, keepdims=True) / np.linalg.norm(XP, axis=1, keepdims=True)
    return binary_XP, X0

def pca_iterate_quantized(X, max_iter=5):
    X_norm = np.linalg.norm(X, axis=1, keepdims=True)
    X_norm[X_norm == 0] = 1
    X = X / X_norm
    N, D = X.shape
    P = Orthogonal(D)
    P = P.T
    for i in range(max_iter):
        XP = np.dot(X, P)
        bin_XP = (XP > 0)
        bin_XP = (2 * bin_XP[:, :D] - 1) / D ** 0.5
        [U, _, VT] = np.linalg.svd(bin_XP.T @ X)
        P = (VT.T @ U.T)

    return P


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='random projection')
    parser.add_argument('-d', '--dataset', help='dataset', default='deep1M')
    args = vars(parser.parse_args())
    dataset = args['dataset']
    # path
    path = os.path.join(source, dataset)
    data_path = os.path.join(path, f'{dataset}_base.fvecs')

    X = read_fvecs(data_path)

    N, D = X.shape
    B = (D + 63) // 64 * 64
    MAX_BD = max(D, B)

    projection_path = os.path.join(path, f'P-o.fvecs')
    RN_path = os.path.join(path, f'Rand-o.Ivecs')
    centroid_path = os.path.join(path, f'RandCentroid-o.fvecs')
    x2_path = os.path.join(path, f'x2-o.fvecs')
    x0_path = os.path.join(path, f'x0-o.fvecs')

    X_pad = np.pad(X, ((0, 0), (0, MAX_BD - D)), 'constant')
    np.random.seed(0)

    # The inverse of an orthogonal matrix equals to its transpose.
    pca = PCA(n_components=MAX_BD)
    if X_pad.shape[0] < 1000000:
        pca.fit(X)
    else:
        pca.fit(X_pad[:1000000])
    PCA_matrix = pca.components_.T
    XP = np.dot(X_pad, PCA_matrix)
    X_mean = np.mean(XP, axis=0)
    XP -= X_mean

    P = pca_iterate_quantized(XP, 20)
    P = P.T
    XP = np.dot(X_pad, P)

    bin_XP = (XP > 0)
    # vector l2 norm
    x2 = np.linalg.norm(XP, axis=1, keepdims=True)
    # The inner product between the data vector and the quantized data vector, i.e., <\bar o, o>.
    x0 = np.sum(XP[:, :B] * (2 * bin_XP[:, :B] - 1) / B ** 0.5, axis=1, keepdims=True) / x2

    # To remove illy defined x0
    # np.linalg.norm(XP, axis=1, keepdims=True) = 0 indicates that its estimated distance based on our method has no error.
    # Thus, it should be good to set x0 as any finite non-zero number.
    x0[~np.isfinite(x0)] = 0.8

    print(np.mean(x0))

    bin_XP = bin_XP[:, :B].flatten()
    uint64_XP = np.packbits(bin_XP.reshape(-1, 8, 8)[:, ::-1]).view(np.uint64)
    uint64_XP = uint64_XP.reshape(-1, B >> 6)

    # Output
    to_Ivecs(RN_path, uint64_XP)
    to_fvecs(x0_path, x0)
    to_fvecs(x2_path, x2)
    to_fvecs(centroid_path, X_mean.reshape((1, D)))
    to_fvecs(projection_path, PCA_matrix)
