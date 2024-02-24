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

    def forward(self, x, *args, **kwargs):
        return x + self.module(x, *args, **kwargs)

# I assume the lr would be at 1e-3 scale, thus, the times 10 rescale the lr to 1e-2 scale
class ReZero(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.weight1 = nn.Parameter(torch.Tensor([0.001]))
    def forward(self, X, *args, **kwargs):
        return torch.add(X, self.module(X, *args, **kwargs) * (self.weight1 * 10))

# Do nothing just pass,
class Pass(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
    def forward(self, X, *args, **kwargs):
        return self.module(X, *args, **kwargs)

class Conv1dWrapper(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv1d(*args, **kwargs)
        self.weight = self.conv.weight
        self.bias = self.conv.bias

    def forward(self, X, *args, modes=None, **kwargs):
        # The args, and kwargs are just placeholders
        # Use modes to select a subset of the weights
        return self.conv(X) if modes is None else F.conv1d(X, self.conv.weight[modes],
                                          self.conv.bias[modes], padding=self.conv.padding[0])

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
        """

        Parameters
        ----------
        n_filters: int
            number of filters
        kernel_size: int
            kernel size
        activation: nn.Module
        in_channels: int
            number of input channels
        """
        super().__init__()

        self.in_channels = in_channels
        self.n_filters = n_filters
        self.kernel_size = kernel_size

        self.conv = Conv1dWrapper(in_channels=in_channels,
                              out_channels=n_filters,
                              kernel_size=kernel_size,
                              padding=kernel_size // 2)
        self.activation = copy.deepcopy(activation)


    def forward(self, X, *args, **kwargs):
        X = self.conv(X, *args, **kwargs)
        X = self.activation(X)
        return X

class ConvBlockModule(nn.Module):
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
                 empty=False,
                 ):
        """

               Parameters
               ----------
               n_filters: int
                   number of kernels
               bottleneck: int
                   number of kernels in the bottleneck layer
               kernel_size: int
                   kernel size
               dilation: int
                   dilation rate
               activation: nn.Module
                   activation function
               batch_norm: bool
                   batch normalization in between layers
               batch_norm_momentum: float
                   batch normalization momentum
               groups: int
                   number of groups in the conv layer
               inception_version: int
                   2 or 3, 2 stands for 2 layers, 3 stands for 3 layers
        """

        super().__init__()
        if empty:
            return
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.activation = activation
        self.batch_norm = batch_norm
        self.inception_version = inception_version


        bottleneck = n_filters // 2 if bottleneck is None else bottleneck
        self.bottleneck = bottleneck

        if inception_version == 2:
            self.conv1 = Conv1dWrapper(in_channels=n_filters,
                        out_channels=bottleneck,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        padding=dilation * (kernel_size // 2),
                        groups=groups, bias=False)
            self.block1 = nn.Sequential(
                nn.BatchNorm1d(bottleneck, momentum=batch_norm_momentum) if batch_norm else nn.Identity(),
                copy.deepcopy(activation))
            self.conv2 = Conv1dWrapper(in_channels=bottleneck,
                        out_channels=n_filters,
                        kernel_size=1, bias=False)
            self.block2 = nn.Sequential(
                nn.BatchNorm1d(n_filters, momentum=batch_norm_momentum) if batch_norm else nn.Identity(),
                copy.deepcopy(activation),
            )

            nn.init.kaiming_normal_(self.conv1.weight.data, mode='fan_out')
            nn.init.kaiming_normal_(self.conv2.weight.data, mode='fan_out')

            self.block1[0].weight.data[...] = 1
            self.block2[0].weight.data[...] = 1

            self.block1[0].bias.data[...] = 0
            self.block2[0].bias.data[...] = 0
            self.block3 = None
        else:
            self.conv1 = Conv1dWrapper(in_channels=n_filters,
                        out_channels=bottleneck,
                        kernel_size=1, bias=False)

            self.block1 = nn.Sequential(
                nn.BatchNorm1d(bottleneck, momentum=batch_norm_momentum) if batch_norm else nn.Identity(),
                copy.deepcopy(activation))
            self.conv2 = Conv1dWrapper(in_channels=bottleneck,
                        out_channels=bottleneck,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        padding=dilation * (kernel_size // 2),
                        groups=groups, bias=False)

            self.block2 = nn.Sequential(
                nn.BatchNorm1d(bottleneck, momentum=batch_norm_momentum) if batch_norm else nn.Identity(),
                copy.deepcopy(activation))
            self.conv3 = Conv1dWrapper(in_channels=bottleneck,
                          out_channels=n_filters,
                          kernel_size=1, bias=False),
            self.block3 = nn.Sequential(
                nn.BatchNorm1d(n_filters, momentum=batch_norm_momentum) if batch_norm else nn.Identity(),
                copy.deepcopy(activation),
            )

            nn.init.kaiming_normal_(self.conv1.weight.data, mode='fan_out')
            nn.init.kaiming_normal_(self.conv2.weight.data, mode='fan_out')
            nn.init.kaiming_normal_(self.conv3.weight.data, mode='fan_out')

            self.block1[0].weight.data[...] = 1
            self.block2[0].weight.data[...] = 1
            self.block3[0].weight.data[...] = 1

            self.block1[0].bias.data[...] = 0
            self.block2[0].bias.data[...] = 0
            self.block3[0].bias.data[...] = 0

    @classmethod
    def from_inception_v1(cls, inception_v1):
        model = cls(empty=True)
        model.conv1 = inception_v1.layers[0]
        model.block1 = inception_v1.layers[1:3]
        model.conv2 = inception_v1.layers[3]
        model.block2 = inception_v1.layers[4:6]
        if len(inception_v1.layers) > 6:
            model.block3 = inception_v1.layers[6:]
        else:
            model.block3 = None
        return model

    def forward(self, X, *args, **kwargs):
        '''
        Parameters
        ----------
        self
        X: torch.tensor, shape=(batch_size, n_filters, seq_len)

        Returns
        -------

        '''
        X = self.conv1(X, *args, **kwargs)
        X = self.block1(X)
        X = self.conv2(X, *args, **kwargs)
        X = self.block2(X)
        if self.block3 is not None:
            X = self.conv3(X, *args, **kwargs)
            X = self.block3(X)
        return X

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
        self.conv = Conv1dWrapper(in_channels=n_filters,
                  out_channels=n_filters,
                  kernel_size=kernel_size,
                  dilation=dilation,
                  padding=dilation * (kernel_size // 2)),
        self.block = nn.Sequential(
        nn.BatchNorm1d(n_filters, momentum=batch_norm_momentum) if batch_norm else nn.Identity(),
        copy.deepcopy(activation))


    def forward(self, X, *args, **kwargs):
        X = self.conv(X, *args, **kwargs)
        X = self.block(X)
        return X

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
        self.botleneck = bottleneck
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.groups = groups
        self.activation = activation
        self.batch_norm = batch_norm
        self.batch_norm_momentum = batch_norm_momentum
        self.rezero = rezero
        self.residual = residual
        self.inception = inception
        self.inception_version = inception_version

        residual_class = Residual if residual else Pass
        if rezero:
            residual_class = ReZero
        self.layers = nn.ModuleList([
            residual_class(
            ConvBlockModule(n_filters=n_filters,
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

    def forward(self, X, *args, **kwargs):
        '''
        Parameters
        ----------
        self
        X: torch.tensor, shape=(batch_size, n_filters, seq_len)

        Returns
        -------

        '''
        for i in range(self.n_layers):
            X = self.layers[i](X, *args, **kwargs)
        return X

class Footprints_head(nn.Module):
    # This is the output head of the footprints model
    def __init__(self,
                 n_filters,
                 kernel_size=5,
                 n_scales=50,
                 per_peak_feats=1
                 ):
        '''

        Parameters
        ----------
        n_filters: int
            number of filters
        kernel_size: int
            kernel size
        n_scales: int
            number of footprints scales
        per_peak_feats: int
            number of features per peak (such as coverages)
        '''
        super().__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.n_scale = n_scales
        self.per_peak_feats = per_peak_feats
        self.conv_layer = Conv1dWrapper(in_channels=n_filters,
                                     out_channels=n_scales,
                                     kernel_size=kernel_size,
                                        padding=kernel_size//2)
        # self.linear = nn.Linear(in_features=n_filters,
        #                         out_features=per_peak_feats,
        #                         bias=True)
        self.linear = Conv1dWrapper(in_channels=n_filters,
                                     out_channels=per_peak_feats,
                                     kernel_size=1,
                                     padding=0,
                                     bias=True)

    def forward(self, X, *args, output_len=None, modes=None, **kwargs):
        X_bindingscore = self.conv_layer(X, *args, modes=modes, **kwargs)
        # # Empircally this normalization makes the training more stable
        # X_bindingscore = X_bindingscore - torch.sum(X_bindingscore, dim=(1,2), keepdims=True) / (X_bindingscore.shape[-1] * X_bindingscore.shape[-2])
        if output_len is None:
            trim = 0
        else:
            output_len_needed_in_X = int(output_len)
            trim = (X_bindingscore.shape[-1] - output_len_needed_in_X) // 2

        if trim > 0:
            X_bindingscore = X_bindingscore[..., trim:-trim]

        if isinstance(self.linear, nn.Linear):
            X_count = self.linear(X.detach().mean(dim=-1) if self.training else X.mean(dim=-1))[..., 0]
        else:
            X_count = self.linear(X.detach().mean(dim=-1, keepdims=True) if
                              self.training else X.mean(dim=-1, keepdims=True), **kwargs)[..., 0, 0]

        return X_bindingscore, X_count

class Conv1dLoRA(nn.Module):
    def __init__(self,
                 layer,
                 A_embedding=None,
                 B_embedding=None,
                 r=8,
                 alpha=None,
                 hidden_dim=None,
                 n_layers=0):
        super().__init__()
        assert isinstance(layer, Conv1dWrapper), "The layer must be a Conv1dWrapper layer"
        self.layer = layer
        self.pretrain_conv = layer.conv
        self.layer_dim_in = self.pretrain_conv.in_channels
        self.layer_dim_out = self.pretrain_conv.out_channels
        self.kernel_size = self.pretrain_conv.kernel_size[0]
        self.dilation = self.pretrain_conv.dilation[0]
        self.padding = self.pretrain_conv.padding[0]
        self.groups = self.pretrain_conv.groups

        self.A_embedding_dim = A_embedding.embedding_dim
        self.B_embedding_dim = B_embedding.embedding_dim

        if alpha is None:
            alpha = r

        self.scale = alpha / r
        self.r = r

        if hidden_dim is None:
            self.hidden_dim = self.A_embedding_dim
        else:
            self.hidden_dim = hidden_dim

        layers = [
            A_embedding,
            nn.Linear(
            in_features=self.A_embedding_dim,
            out_features=self.hidden_dim
            ),
            nn.BatchNorm1d(self.hidden_dim),
            nn.GELU(),

        ] + [
            nn.Linear(in_features=self.hidden_dim,
                       out_features=self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.GELU(),] * n_layers + [
            nn.Linear(
            in_features=self.hidden_dim,
            out_features=int(self.layer_dim_in * r / self.groups) # lead to a weight matrix of shape (r, layer_dim_in)
            ),
        ]
        self.A_embedding = nn.Sequential(*layers)

        layers = [
            B_embedding,
            nn.Linear(
            in_features=self.B_embedding_dim,
            out_features=self.hidden_dim
            ),
            nn.BatchNorm1d(self.hidden_dim),
            nn.GELU(),
        ] + [
            nn.Linear(in_features=self.hidden_dim,
                       out_features=self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.GELU(),] * n_layers + [
            nn.Linear(
            in_features=self.hidden_dim,
            out_features=int(self.layer_dim_out * r * self.kernel_size) # lead to a weight matrix of shape (layer_dim_out, r)
            ),
        ]
        self.B_embedding = nn.Sequential(*layers)

        # When combined, this will lead to a weight matrix of shape (layer_dim_out, layer_dim_in, kernel_size)

        ## Make sure B starts as all zeros:
        for i in range(len(self.B_embedding)):
            if isinstance(self.B_embedding[i], nn.Linear):
                self.B_embedding[i].bias.data[...] = 0
                self.B_embedding[i].weight.data[...] = 0

        # test A_output distribution
        with torch.no_grad():
            self.A_embedding.eval()
            self.A_embedding.cuda()
            A_output = self.A_embedding(torch.arange(64).long().cuda())
            mean, std = A_output.mean(), A_output.std()
            print ("A_output mean: {}, std: {}".format(mean, std))
            # self.scale *= 1 / (std * r)
            rescale_factor = 1 / (std)
            self.A_embedding[0].weight.data[...] *= rescale_factor # rescale the embedding matrix

    @torch.no_grad()
    def collapse_layer(self, cell):
        # return a constant conv1d layer with the weight matrix collapsed from the A and B embeddings at cell

        if type(cell) is not int:
            raise ValueError("cell must be an integer")

        A = self.A_embedding(torch.tensor([cell]).long().to(self.A_embedding[0].weight.data.device))
        B = self.B_embedding(torch.tensor([cell]).long().to(self.A_embedding[0].weight.data.device))
        if self.kernel_size == 1:
            A = A.reshape(
            (self.r, self.layer_dim_in))
            B = B.reshape(
            (self.layer_dim_out, self.r))
            weight = torch.matmul(B, A)[..., None]
        else:
            A = A.reshape(
            (int(self.layer_dim_in / self.groups), self.r))
            B = B.reshape(
            (self.r, self.layer_dim_out * self.kernel_size))
            weight = torch.matmul(A, B).reshape((int(self.layer_dim_in / self.groups),
                                              self.layer_dim_out, self.kernel_size)).contiguous().permute(1, 0, 2)
        weight_scaled = weight * self.scale
        new_layer = copy.deepcopy(self.layer)
        new_layer.conv.weight.data[...] = new_layer.conv.weight.data + weight_scaled
        return new_layer

    def forward(self, X, cells, modes=None):
        if self.kernel_size == 1:
            # When kernel_size == 1, the convolution is actually a linear layer, take a short path
            A = self.A_embedding(cells).reshape((-1, self.r, self.layer_dim_in))
            B = self.B_embedding(cells).reshape((-1, self.layer_dim_out, self.r))
            # x: (batch_size, layer_dim_in, seq_len)
            lora_x = torch.bmm(A, X)  # (batch_size, r, seq_len)
            if modes is not None:
                B = B[:, modes]
            lora_x = torch.bmm(B, lora_x)  # (batch_size, layer_dim_out, seq_len
            return lora_x * self.scale + (self.layer(X, modes=modes))
        else:
            # When kernel_size > 1, the convolution can be written as groupped convolutioni,
            # take a long path
            bs = X.shape[0] # batch_size
            A = self.A_embedding(cells).reshape((bs, int(self.layer_dim_in / self.groups), self.r))
            B = self.B_embedding(cells).reshape((bs, self.r, self.layer_dim_out, self.kernel_size))
            if modes is not None:
                B = B[:, modes]
            B = B.reshape((bs, self.r, self.layer_dim_out * self.kernel_size))
            weight = torch.bmm(A, B).reshape((bs, int(self.layer_dim_in / self.groups),
                                              self.layer_dim_out, self.kernel_size)).contiguous().permute(0, 2, 1, 3)
            # size of (batch_size, layer_dim_out, layer_dim_in / groups, kernel_size)


            # route 1
            weight = weight.reshape((-1, int(self.layer_dim_in / self.groups), self.kernel_size))
            # size of (batch_size * layer_dim_out, layer_dim_in / groups, kernel_size)
            # X after reshape (1, batch_size*layer_dim_in, seq_len)
            lora_x = F.conv1d(X.reshape((1, -1, X.shape[-1])), weight=weight,
                         bias=None, dilation=self.dilation,
                         groups=bs * self.groups, padding=self.padding) # each batch_size is a group
            # within each group, the convolution projects from (layer_dim_in, seq_len) to (layer_dim_out, seq_len)
            # This is equivalent to a for loop over each sample in the batch
            lora_x = lora_x.view(bs, self.layer_dim_out, -1)
            X =  lora_x * self.scale + self.layer(X, modes=modes)

            # # route 2
            # weight = weight + self.layer.conv.weight[modes][None]
            # weight = weight.reshape((-1, int(self.layer_dim_in / self.groups), self.kernel_size))
            # X = F.conv1d(X.reshape((1, -1, X.shape[-1])), weight=weight,
            #                 bias=None, dilation=self.dilation,
            #                 groups=bs * self.groups, padding=self.padding)
            # X = X.view(bs, self.layer_dim_out, -1)
            # X = X + self.layer.conv.bias[modes][None, :, None]

            return X

