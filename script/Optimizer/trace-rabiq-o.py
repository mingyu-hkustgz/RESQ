import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm
import argparse
import os
from utils import *
import matplotlib.pyplot as plt

source = './DATA'
verbose = True


def Orthogonal(D):
    G = np.random.randn(D, D).astype('float32')
    Q, _ = np.linalg.qr(G)
    return Q


def lift_proj(X, iter=100):
    N, D = X.shape
    var = np.var(X, axis=0)
    mean_var = np.mean(var)
    P = Orthogonal(D)
    diag_var = np.diag(var)
    Z = P @ diag_var @ P.T
    for i in tqdm(range(iter)):
        T = Z
        for j in range(D):
            T[j][j] = mean_var
        _, P = np.linalg.eig(T)
        Z = P @ diag_var @ P.T

    if verbose:
        XP = np.dot(X, P.T)
        bin_XP = (XP > 0)
        bin_XP = (2 * bin_XP[:, :D] - 1) / D ** 0.5
        x0 = np.sum(XP[:, :D] * bin_XP, axis=1, keepdims=True)
        print(f"similarity to centroid {round(np.mean(x0), 4)} at iteration:: {iter}")

    return P


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='trace projection')
    parser.add_argument('-d', '--dataset', help='dataset', default='deep1M')
    parser.add_argument('-b', '--bits', help='quantized bits', default=128)

    args = vars(parser.parse_args())
    dataset = args['dataset']
    bits = int(args['bits'])
    # path
    path = os.path.join(source, dataset)
    data_path = os.path.join(path, f'{dataset}_proj.fvecs')

    X = read_fvecs(data_path)

    X = X[:, :bits]
    D = X.shape[1]
    B = (D + 63) // 64 * 64
    MAX_BD = max(D, B)

    projection_path = os.path.join(path, f'P-t.fvecs')
    centroid_path = os.path.join(path, f'RandCentroid-t.fvecs')
    RN_path = os.path.join(path, f'Rand-t.Ivecs')
    x2_path = os.path.join(path, f'x2-t.fvecs')
    x0_path = os.path.join(path, f'x0-t.fvecs')

    X_pad = np.pad(X, ((0, 0), (0, MAX_BD - D)), 'constant')
    np.random.seed(0)

    # The inverse of an orthogonal matrix equals to its transpose.

    X_mean = np.mean(X_pad, axis=0)
    X_pad -= X_mean
    P = lift_proj(X_pad)
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

    bin_XP = bin_XP[:, :B].flatten()
    uint64_XP = np.packbits(bin_XP.reshape(-1, 8, 8)[:, ::-1]).view(np.uint64)
    uint64_XP = uint64_XP.reshape(-1, B >> 6)

    print(f"similarity to centroid {round(np.mean(x0), 4)}")

    to_Ivecs(RN_path, uint64_XP)
    to_fvecs(x0_path, x0)
    to_fvecs(x2_path, x2)
    to_fvecs(centroid_path, X_mean.reshape((1, D)))
    to_fvecs(projection_path, P)
