import math

import torch
import torch.nn as nn
import torch.optim as optim
from network import *
from loss import *
from tqdm import tqdm
import argparse
import os
from utils import fvecs_read, fvecs_write, ivecs_read, ivecs_write
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Set parameters
learning_rate = 0.001
num_epochs = 10
batch_size = 2048
source = '/home/BLD/mingyu/DATA/vector_data'
device = torch.device("cuda:0")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='random projection')
    parser.add_argument('-d', '--dataset', help='dataset', default='gist')
    args = vars(parser.parse_args())
    dataset = args['dataset']
    print(f"Learn Proj-{dataset}")
    # path
    path = os.path.join(source, dataset)
    base_path = os.path.join(path, f'{dataset}_base.fvecs')
    query_path = os.path.join(path, f'{dataset}_query.fvecs')
    query = fvecs_read(query_path)
    # read data vectors
    original_vectors = fvecs_read(base_path)
    original_vectors = torch.from_numpy(original_vectors).to(device)
    query_vectors = fvecs_read(query_path)
    query_vectors = torch.from_numpy(query_vectors).to(device)

    VecData = torch.utils.data.TensorDataset(original_vectors)
    origin_vector_loader = torch.utils.data.DataLoader(VecData, batch_size=batch_size, shuffle=True)
    N, D = original_vectors.shape
    input_dim = D
    projected_dim = 256

    # Initialize the model
    model = DeepBit(input_dim, projected_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.scale = 1.0
    for epoch in tqdm(range(num_epochs)):
        ave_loss = 0.0
        for vecs in origin_vector_loader:
            vecs = vecs[0].to(device)
            optimizer.zero_grad()
            # Forward pass
            projected_vectors = model(vecs)
            # Compute loss
            loss = reconstruct_loss(vecs, projected_vectors, model)
            # Backward pass
            loss.backward()
            # Update parameters
            optimizer.step()
            ave_loss += math.sqrt(loss.item())
        ave_loss /= len(origin_vector_loader)

        if (epoch + 1) % 2 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Ave Loss: {ave_loss:.4f}, scale: {model.scale:.4f}')
            print(f'model factor: {model.factor[0]}')
            # temperature *= temperature_increase_factor

        model.scale *= 1.05

    # Project vectors using the trained model
    with torch.no_grad():
        original_vectors = original_vectors[:50000]
        final_projected_vectors = model(original_vectors).cpu().numpy()
        final_query_vectors = model(query_vectors).cpu().numpy()
        original_vectors = original_vectors.cpu().numpy()
        query_vectors = query_vectors.cpu().numpy()
        precise_dot_product = (original_vectors * query_vectors[0]).sum(axis=1)
        dot_product = (final_projected_vectors * final_query_vectors[0]).sum(axis=1)
        differences = precise_dot_product - dot_product * model.factor.cpu().numpy()[0]
        print(f"Learn proj dimension: {projected_dim}")
        plt.hist(differences, bins=100)
        plt.savefig(f"./figure/{dataset}-learn-proj.png")
        plt.show()
        plt.hist(precise_dot_product, bins=100)
        plt.savefig(f"./figure/{dataset}-precise-proj.png")
        plt.show()
        plt.hist(dot_product * dot_product * model.factor.cpu().numpy()[0], bins=100)
        plt.savefig(f"./figure/{dataset}-app-proj.png")
        plt.show()

    print("Training completed.")
