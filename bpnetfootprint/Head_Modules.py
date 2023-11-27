import numpy as np
import torch
import torch.nn as nn


class Bindingscore_head(nn.Module):
    def __init__(self,
                 n_filters,
                 kernel_size=[75],
                 pool_size=10,
                 upsample_size=10,
                 pool_mode='avg'
                 ):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = [k // 2 for k in kernel_size]
        self.pool_size = pool_size
        self.upsample_size=upsample_size
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=n_filters,
                        out_channels=1,
                        kernel_size=k,
                        padding=p,
                        )
            for k, p in zip(kernel_size, self.padding)
        ])
        if pool_mode == 'avg':
            self.pool = nn.AvgPool1d(kernel_size=pool_size, stride=pool_size)
        elif pool_mode == 'max':
            self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
        self.upsampling = nn.Upsample(scale_factor=upsample_size, mode='nearest')

    def forward(self, X, output_len=None, upsample=True):
        X_bindingscore = torch.cat([conv(X) for conv in self.conv_layers], dim=1)
        if output_len is None:
            trim = 0
        else:
            output_len_needed_in_X = int(output_len / self.upsample_size * self.pool_size)
            trim = (X_bindingscore.shape[-1] - output_len_needed_in_X) // 2

        if trim > 0:
            X_bindingscore = X_bindingscore[:, :, trim:-trim]

        X_bindingscore = self.pool(X_bindingscore)
        if upsample:
            X_bindingscore = self.upsampling(X_bindingscore)

        return X_bindingscore

