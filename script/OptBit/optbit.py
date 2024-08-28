import numpy as np
import struct
import time
import math
import os
from utils import *
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from Net import DeepBit
from loss import *

source = './DATA'

config = {
    "epoch": 100,
    "batch_size": 1024,
    "lr": 0.005,
    "device": torch.device("cuda:0"),
    "dim": 960,
}

def nn_init(model):
    rand_matrix = np.random.randn(D, D).astype('float32')
    oth_rand_matrix, _ = np.linalg.qr(rand_matrix)
    oth_rand_matrix = torch.from_numpy(oth_rand_matrix)
    with torch.no_grad():
        model.map.weight = nn.Parameter(oth_rand_matrix)


def GenerateBinaryCode(X):
    X = torch.from_numpy(X).to(config['device'])
    vec_data = torch.utils.data.TensorDataset(X)
    vec_loader = torch.utils.data.DataLoader(vec_data, batch_size=config['batch_size'], shuffle=True)
    N, D = X.shape
    model = DeepBit(D, config['dim'])
    nn_init(model)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    model.to(config['device'])
    # Training loop
    model.scale = 1.0
    for epoch in tqdm(range(config['epoch'])):
        ave_loss = 0.0
        for vecs in vec_loader:
            vecs = vecs[0].to(config['device'])
            optimizer.zero_grad()
            # Compute loss
            loss = oth_sim_loss(vecs, model, config)
            # Backward pass
            loss.backward()
            # Update parameters
            optimizer.step()
            ave_loss += loss.item()
        ave_loss /= len(vec_loader)
        print(f'Epoch [{epoch}], Ave Loss: {ave_loss:.4f}, scale: {model.scale:.4f}')

        model.scale *= 1.05

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='random projection')
    parser.add_argument('-d', '--dataset', help='dataset', default='gist')
    args = vars(parser.parse_args())
    dataset = args['dataset']
    # path
    path = os.path.join(source, dataset)
    data_path = os.path.join(path, f'{dataset}_base.fvecs')

    C = 4096
    centroids_path = os.path.join(path, f'{dataset}_centroid_{C}.fvecs')
    dist_to_centroid_path = os.path.join(path, f'{dataset}_dist_to_centroid_{C}.fvecs')
    cluster_id_path = os.path.join(path, f'{dataset}_cluster_id_{C}.ivecs')

    X = read_fvecs(data_path)
    centroids = read_fvecs(centroids_path)
    cluster_id = read_ivecs(cluster_id_path)

    D = X.shape[1]
    B = (D + 63) // 64 * 64
    MAX_BD = max(D, B)

    projection_path = os.path.join(path, f'P_C{C}_B{B}.fvecs')
    randomized_centroid_path = os.path.join(path, f'RandCentroid_C{C}_B{B}.fvecs')
    RN_path = os.path.join(path, f'RandNet_C{C}_B{B}.Ivecs')
    x0_path = os.path.join(path, f'x0_C{C}_B{B}.fvecs')

    X_pad = np.pad(X, ((0, 0), (0, MAX_BD - D)), 'constant')
    centroids_pad = np.pad(centroids, ((0, 0), (0, MAX_BD - D)), 'constant')
    np.random.seed(0)

    cluster_id = np.squeeze(cluster_id)

    X_pad = X_pad - centroids_pad[cluster_id]

    X_pad = X_pad - np.mean(X_pad, axis=0)
    norms = np.linalg.norm(X_pad, axis=1)

    # remove zero vectors
    X_pad = X_pad[norms != 0]
    GenerateBinaryCode(X_pad)
