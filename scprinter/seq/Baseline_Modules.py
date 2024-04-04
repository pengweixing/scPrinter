import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm, trange

from .Functions import *


class BiasInjection(nn.Module):
    def __init__(self, layer, A_embedding=None, hidden_dim=None, **kwargs):
        super().__init__()
        self.layer = layer
        self.layer_dim_in = layer.in_channels
        self.layer_dim_out = layer.out_channels
        self.embedding_dim = A_embedding.embedding_dim
        self.embedding_nn = nn.Sequential(
            A_embedding,
            nn.Linear(
                in_features=self.embedding_dim,
                out_features=hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(in_features=hidden_dim, out_features=self.layer_dim_in),
        )

    def forward(self, X, cells, modes=None):
        bias = self.embedding_nn(cells)
        if modes is None:
            return self.layer(X + bias[..., None])
        else:
            X = X + bias[..., None]
            return F.conv1d(
                X,
                self.layer.weight[modes],
                self.layer.bias[modes],
                padding=self.layer.padding[0],
            )

    def forward_fixed(self, X, modes=None):
        X = X + self.bias[..., None]
        if modes is None:
            return self.layer(X)
        else:
            return F.conv1d(
                X,
                self.layer.weight[modes],
                self.layer.bias[modes],
                padding=self.layer.padding[0],
            )

    @torch.no_grad()
    def collapse_layer(self, cell):
        if type(cell) is not int:
            raise ValueError("cell must be an integer")
        self.bias = self.embedding_nn(
            torch.tensor([cell]).long().to(self.embedding_nn[0].weight.data.device)
        )
        # print (self.bias.shape)
        self.forward = self.forward_fixed
        return self
