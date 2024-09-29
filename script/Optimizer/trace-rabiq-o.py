import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm
import argparse
import os
from utils import *
import matplotlib.pyplot as plt

source = './DATA'
color = ["r", "b", "y", "g"]


def Orthogonal(D):
    G = np.random.randn(D, D).astype('float32')
    Q, _ = np.linalg.qr(G)
    return Q


def lift_proj(X, bits, iter=301):
    count = 0
    var = np.var(X, axis=0)
    mean_var = np.mean(var)
    R = Orthogonal(bits)
    diag_var = np.diag(var)
    Z = R @ diag_var @ R.T
    for i in tqdm(range(iter)):
        T = Z
        for j in range(bits):
            T[j][j] = mean_var
        _, R = np.linalg.eig(T)
        Z = R @ diag_var @ R.T

    return R


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='random projection')
    parser.add_argument('-d', '--dataset', help='dataset', default='deep1M')
    args = vars(parser.parse_args())
    dataset = args['dataset']
    # path
    path = os.path.join(source, dataset)
    data_path = os.path.join(path, f'{dataset}_base.fvecs')

    # read data vectors
    X = fvecs_read(data_path)

    N, D = X.shape
    B = (D + 63) // 64 * 64
    MAX_BD = max(D, B)

    projection_path = os.path.join(path, f'P-t.fvecs')
    RN_path = os.path.join(path, f'Rand-t.Ivecs')
    centroid_path = os.path.join(path, f'RandCentroid-t.fvecs')
    x2_path = os.path.join(path, f'x2-t.fvecs')
    x0_path = os.path.join(path, f'x0-t.fvecs')

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

    P = lift_proj(XP, MAX_BD)
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
    to_fvecs(projection_path, PCA_matrix)
