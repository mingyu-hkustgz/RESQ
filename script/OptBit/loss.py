import torch
import torch.nn as nn
import torch.optim as optim


def sim_loss(vecs, model, config):
    proj_vecs = model(vecs)
    proj_vecs = torch.tanh(proj_vecs * model.scale)
    sim = torch.sum(vecs * proj_vecs / config['dim'] ** 0.5, dim=(1, ), keepdim=True) / torch.linalg.norm(vecs, axis=1, keepdims=True)
    loss = 1.0 - sim
    ave_loss = torch.sum(loss) / len(loss)
    return ave_loss

def oth_sim_loss(vecs, model, config):
    I = torch.eye(config['dim'], device=config['device'])
    W = model.map.weight
    loss_ortho_ave = torch.sum(torch.abs(torch.matmul(W, W.t()) - I)) / (len(W) * len(W))

    proj_vecs = model(vecs)
    binary_proj_vecs = torch.tanh(proj_vecs * model.scale)
    sim = torch.sum(binary_proj_vecs * proj_vecs / config['dim'] ** 0.5, dim=(1, ), keepdim=True) / torch.linalg.norm(proj_vecs, axis=1, keepdims=True)
    sim_loss_sum = 1.0 - sim
    sim_loss_ave = torch.sum(sim_loss_sum) / len(sim_loss_sum)

    loss = loss_ortho_ave + sim_loss_ave
    return loss

