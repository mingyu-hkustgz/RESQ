import torch
import torch.nn as nn
import torch.optim as optim


def projection_loss(original_vectors, projected_vectors):
    # Compute dot products of original vectors
    original_dot_products = torch.matmul(original_vectors, original_vectors.t())

    # Compute dot products of projected vectors
    projected_dot_products = torch.matmul(projected_vectors, projected_vectors.t())

    # Calculate square differences between dot products
    differences = (original_dot_products - projected_dot_products) ** 2

    # Compute average error
    loss = torch.sum(differences) / len(differences)

    return loss


def reconstruct_loss(original_vectors, projected_vectors, net):
    # Compute dot products of original vectors
    original_dot_products = torch.matmul(original_vectors, original_vectors.t())

    # Compute dot products of projected vectors
    projected_dot_products = torch.matmul(projected_vectors, projected_vectors.t())

    # Calculate square differences between dot products
    reconstruct = (original_dot_products - net.reconstruct(projected_dot_products)) ** 2

    return torch.sum(reconstruct) / (len(reconstruct) * len(reconstruct))