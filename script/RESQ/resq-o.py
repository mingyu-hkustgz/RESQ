import numpy as np
import struct
import time
import os
from utils import *
import argparse
from tqdm import tqdm

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='random projection')
    parser.add_argument('-d', '--dataset', help='dataset', default='gist')
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

    projection_path = os.path.join(path, f'RESP.fvecs')
    centroid_path = os.path.join(path, f'RESCentroid.fvecs')
    RN_path = os.path.join(path, f'RES_Rand.Ivecs')
    x2_path = os.path.join(path, f'RES_x2.fvecs')
    x0_path = os.path.join(path, f'RES_x0.fvecs')

    X_pad = np.pad(X, ((0, 0), (0, MAX_BD - D)), 'constant')
    np.random.seed(0)

    # The inverse of an orthogonal matrix equals to its transpose.
    P = Orthogonal(MAX_BD)
    P = P.T

    XP = np.dot(X_pad, P)
    X_mean = np.mean(XP, axis=0)
    XP -= X_mean
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
    to_fvecs(projection_path, P)
