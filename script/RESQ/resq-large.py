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

    C = 65536
    centroids_path = os.path.join(path, f'p{dataset}_centroid_{C}.fvecs')
    dist_to_centroid_path = os.path.join(path, f'p{dataset}_dist_to_centroid_{C}.fvecs')
    cluster_id_path = os.path.join(path, f'p{dataset}_cluster_id_{C}.ivecs')

    X = fvecs_mmap(data_path)
    centroids = read_fvecs(centroids_path)
    cluster_id = read_ivecs(cluster_id_path)
    N, _ = X.shape
    D = bits
    B = (D + 63) // 64 * 64
    MAX_BD = max(D, B)

    projection_path = os.path.join(path, f'RESP_C{C}_B{B}.fvecs')
    randomized_centroid_path = os.path.join(path, f'RESCentroid_C{C}_B{B}.fvecs')
    RN_path = os.path.join(path, f'RES_Rand_C{C}_B{B}.Ivecs')
    x0_path = os.path.join(path, f'RES_x0_C{C}_B{B}.fvecs')

    np.random.seed(0)

    # The inverse of an orthogonal matrix equals to its transpose.
    P = Orthogonal(MAX_BD)
    P = P.T

    cluster_id = np.squeeze(cluster_id)
    CP = np.dot(centroids, P)
    to_fvecs(randomized_centroid_path, CP)
    to_fvecs(projection_path, P)

    if N % 1000000 == 0:
        iter_count = int(N / 1000000)
    else:
        iter_count = int(N / 1000000) + 1
    pre = 0
    x0_list = []
    RN_list = []
    for i in tqdm(range(iter_count)):
        data_seg = X[pre:pre + 1000000, :bits]
        XP = np.dot(data_seg, P)
        XP = XP - CP[cluster_id[pre:pre + 1000000]]
        bin_XP = (XP > 0)

        # The inner product between the data vector and the quantized data vector, i.e., <\bar o, o>.
        x0 = np.sum(XP[:, :B] * (2 * bin_XP[:, :B] - 1) / B ** 0.5, axis=1, keepdims=True) / np.linalg.norm(XP, axis=1,
                                                                                                            keepdims=True)

        # To remove illy defined x0
        # np.linalg.norm(XP, axis=1, keepdims=True) = 0 indicates that its estimated distance based on our method has no error.
        # Thus, it should be good to set x0 as any finite non-zero number.
        x0[~np.isfinite(x0)] = 0.8

        print(np.mean(x0))

        bin_XP = bin_XP[:, :B].flatten()
        uint64_XP = np.packbits(bin_XP.reshape(-1, 8, 8)[:, ::-1]).view(np.uint64)
        uint64_XP = uint64_XP.reshape(-1, B >> 6)

        # Output
        RN_list.append(uint64_XP)
        x0_list.append(x0)
        pre += 1000000

    x0_final = np.vstack(x0_list)
    RN_final = np.vstack(RN_list)
    to_Ivecs(RN_path, RN_final)
    to_fvecs(x0_path, x0_final)
