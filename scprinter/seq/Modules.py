import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm, trange
import copy
from .Functions import *


class Residual(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, **kwargs):
        return torch.add(self.module(x, **kwargs), x)

# I assume the lr would be at 1e-3 scale, thus, the times 10 rescale the lr to 1e-2 scale
class ReZero(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.weight1 = nn.Parameter(torch.Tensor([0.001]))
    def forward(self, X, **kwargs):
        return torch.add(X, self.module(X, **kwargs) * (self.weight1 * 10))

# Do nothing just pass,
class Pass(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
    def forward(self, X, **kwargs):
        return self.module(X, **kwargs)

class DNA_CNN(nn.Module):
    """
    This is actually as simple as one CNN layer,
    It's used to extract the DNA sequence features (the first layer)
    just to keep the consistency using the Module way of construction
    """

    def __init__(self,
                 n_filters=64,
                 kernel_size=21,
                 activation=nn.ReLU(),
                 in_channels=4
                 ):
        super().__init__()

        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=n_filters,
                              kernel_size=kernel_size,
                              padding=kernel_size // 2)
        self.activation = copy.deepcopy(activation)

    def forward(self, X):
        X = self.conv(X)
        X = self.activation(X)
        return X

class DepthwiseSeparableConv1D(nn.Module):
    """
    This is actually as simple as one CNN layer,
    It's used to extract the DNA sequence features (the first layer)
    just to keep the consistency using the Module way of construction
    """

    def __init__(self,
                 n_filters=64,
                 kernel_size=21,
                 dilation=1,
                 padding=10,
                 in_channels=64,
                 out_channels=64,
                 activation_in_between=False,
                 groups=None
                 ):
        super().__init__()
        if groups is None:
            groups = in_channels
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=n_filters,
                              kernel_size=kernel_size,
                              dilation=dilation,
                              padding=padding,
                              groups=groups)
        self.pointwise = nn.Conv1d(in_channels=n_filters,
                                   out_channels=out_channels,
                                   kernel_size=1)
        self.activation_in_between = nn.GELU() if activation_in_between else nn.Identity()

    def forward(self, X):
        X = self.conv(X)
        X = self.activation_in_between(X)
        X = self.pointwise(X)
        return X

class InceptionModule(nn.Module):
    """
        This part only takes into account the Dilated CNN stack
        """

    def __init__(self,
                 n_filters=64,
                 bottleneck=None,
                 kernel_size=3,
                 dilation=1,
                 activation=nn.ReLU(),
                 batch_norm=False,
                 batch_norm_momentum=0.1,
                 groups=8,
                 inception_version=2,
                 ):
        super().__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.activation = activation
        self.batch_norm = batch_norm
        self.inception_version = inception_version


        bottleneck = n_filters // 2 if bottleneck is None else bottleneck

        if inception_version == 2:
            self.layers = nn.Sequential(
                nn.Conv1d(in_channels=n_filters,
                        out_channels=bottleneck,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        padding=dilation * (kernel_size // 2),
                        groups=groups, bias=False),
                nn.BatchNorm1d(bottleneck, momentum=batch_norm_momentum) if batch_norm else nn.Identity(),
                copy.deepcopy(activation),
                nn.Conv1d(in_channels=bottleneck,
                        out_channels=n_filters,
                        kernel_size=1, bias=False),
                nn.BatchNorm1d(n_filters, momentum=batch_norm_momentum) if batch_norm else nn.Identity(),
                copy.deepcopy(activation),
            )

            nn.init.kaiming_normal_(self.layers[0].weight.data, mode='fan_out')
            nn.init.kaiming_normal_(self.layers[3].weight.data, mode='fan_out')

            self.layers[1].weight.data[...] = 1
            self.layers[4].weight.data[...] = 1

            self.layers[1].bias.data[...] = 0
            self.layers[4].bias.data[...] = 0
        else:
            self.layers = nn.Sequential(
                nn.Conv1d(in_channels=n_filters,
                          out_channels=bottleneck,
                          kernel_size=1, bias=False),
                nn.BatchNorm1d(bottleneck, momentum=batch_norm_momentum) if batch_norm else nn.Identity(),
                copy.deepcopy(activation),
                nn.Conv1d(in_channels=bottleneck,
                          out_channels=bottleneck,
                          kernel_size=kernel_size,
                          dilation=dilation,
                          padding=dilation * (kernel_size // 2),
                          groups=groups, bias=False),
                nn.BatchNorm1d(bottleneck, momentum=batch_norm_momentum) if batch_norm else nn.Identity(),
                copy.deepcopy(activation),
                nn.Conv1d(in_channels=bottleneck,
                          out_channels=n_filters,
                          kernel_size=1, bias=False),
                nn.BatchNorm1d(n_filters, momentum=batch_norm_momentum) if batch_norm else nn.Identity(),
                copy.deepcopy(activation),
            )

            nn.init.kaiming_normal_(self.layers[0].weight.data, mode='fan_out')
            nn.init.kaiming_normal_(self.layers[3].weight.data, mode='fan_out')
            nn.init.kaiming_normal_(self.layers[6].weight.data, mode='fan_out')

            self.layers[1].weight.data[...] = 1
            self.layers[4].weight.data[...] = 1
            self.layers[7].weight.data[...] = 1

            self.layers[1].bias.data[...] = 0
            self.layers[4].bias.data[...] = 0
            self.layers[7].bias.data[...] = 0


    def forward(self, X):
        '''
        Parameters
        ----------
        self
        X: torch.tensor, shape=(batch_size, n_filters, seq_len)

        Returns
        -------

        '''
        return self.layers(X)


class DilatedConvModule(nn.Module):
    def __init__(self,
                 n_filters=64,
                 kernel_size=3,
                 activation=nn.ReLU(),
                 batch_norm=False,
                 batch_norm_momentum=0.1,
                 dilation=1,
                 ):
        super().__init__()
        self.model = nn.Sequential(
        nn.Conv1d(in_channels=n_filters,
                  out_channels=n_filters,
                  kernel_size=kernel_size,
                  dilation=dilation,
                  padding=dilation * (kernel_size // 2)),
        nn.BatchNorm1d(n_filters, momentum=batch_norm_momentum) if batch_norm else nn.Identity(),
        copy.deepcopy(activation))


    def forward(self, X):
        return self.model(X)

class DilatedCNN(nn.Module):
    """
    This part only takes into account the Dilated CNN stack
    """

    def __init__(self,
                 n_filters=64,
                 bottleneck=64,
                 n_layers=6,
                 kernel_size=3,
                 groups=8,
                 activation=nn.ReLU(),
                 batch_norm=False,
                 batch_norm_momentum=0.1,
                 rezero=False,
                 residual=True,
                 inception=True,
                 inception_version=2,
                 dilation_func=None,
                 ):
        super().__init__()
        if dilation_func is None:
            dilation_func = lambda x: 2 ** (x + 1)

        self.n_filters = n_filters
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.activation = activation
        self.batch_norm = batch_norm
        self.residual = residual
        self.inception = inception

        residual_class = Residual if residual else Pass
        if rezero:
            residual_class = ReZero
        self.layers = nn.ModuleList([
            residual_class(
            InceptionModule(n_filters=n_filters,
                            bottleneck=bottleneck,
                            kernel_size=kernel_size,
                            dilation=dilation_func(i),
                            activation=activation,
                            batch_norm=batch_norm,
                            batch_norm_momentum=batch_norm_momentum,
                            groups=groups,
                            inception_version=inception_version) if incpt
            else DilatedConvModule(
                n_filters=n_filters,
                kernel_size=kernel_size,
                activation=activation,
                batch_norm=batch_norm,
                batch_norm_momentum=batch_norm_momentum,
                dilation=dilation_func(i)
            )) for i, incpt in enumerate(inception)
        ])

    def forward(self, X):
        '''
        Parameters
        ----------
        self
        X: torch.tensor, shape=(batch_size, n_filters, seq_len)

        Returns
        -------

        '''
        for i in range(self.n_layers):
            X = self.layers[i](X)
        return X

class Footprints_head(nn.Module):
    def __init__(self,
                 n_filters,
                 kernel_size=5,
                 upsample_size=1,
                 n_scales=50,
                 per_peak_feats=1
                 ):
        super().__init__()
        self.conv_layer = nn.Conv1d(in_channels=n_filters,
                                     out_channels=n_scales,
                                     kernel_size=kernel_size,
                                        padding=kernel_size//2)
        self.upsample_size = upsample_size
        self.upsampling = nn.Upsample(scale_factor=upsample_size, mode='nearest') if upsample_size > 1 else nn.Identity()

        self.linear = nn.Linear(in_features=n_filters,
                                out_features=per_peak_feats,
                                bias=True)

    def forward(self, X, output_len=None, modes=None, upsample=True):
        if modes is None:
            X_bindingscore = self.conv_layer(X)
        else:
            X_bindingscore = F.conv1d(X, self.conv_layer.weight[modes],
                                      self.conv_layer.bias[modes], padding=self.conv_layer.padding[0])

        if output_len is None:
            trim = 0
        else:
            output_len_needed_in_X = int(output_len)
            trim = (X_bindingscore.shape[-1] - output_len_needed_in_X) // 2

        if trim > 0:
            X_bindingscore = X_bindingscore[..., trim:-trim]

        X_count = self.linear(X.detach().mean(dim=-1) if self.training else X.mean(dim=-1))

        return X_bindingscore, X_count[..., 0]


