import torch
import torch.nn as nn
import torch.optim as optim


class DeepBit(nn.Module):
    def __init__(self, input_dim, map_dim):
        super(DeepBit, self).__init__()
        self.map = nn.Linear(input_dim, map_dim, bias=False)
        self.scale = 1

    def forward(self, x):
        return self.map(x)
